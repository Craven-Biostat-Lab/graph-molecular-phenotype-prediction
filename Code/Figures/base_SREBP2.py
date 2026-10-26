from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.metrics import roc_curve, auc
import numpy as np
import argparse
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight
import json
import random
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import pickle
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
warnings.filterwarnings('ignore')
arial = fm.FontProperties(family="Arial", size=12)
arial_legend = fm.FontProperties(family="Arial", size=14)
# Arial for all plot text. Where Arial isn't installed (e.g. CHTC), send Arial.ttf with the job; it is loaded from the working directory.
if os.path.exists('Arial.ttf'):
    fm.fontManager.addfont('Arial.ttf')
plt.rcParams["font.family"] = "Arial"

def seed_everything(seed=123):
    """"
    Seed everything.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def move2last(feature_df,col_to_move):
    other_columns = [col for col in feature_df.columns if col != col_to_move]
    feature_df = feature_df.reindex(columns=[*other_columns, col_to_move])
    return feature_df

def split_val(df):
    all_indexes = df.index.tolist()
    label_0_indexes = df[df['label'] == 0].index.tolist()
    label_1_indexes = df[df['label'] == 1].index.tolist()
    val_index = list()
    test_index = list()
    seed_everything()
    val_index.extend(random.sample(label_0_indexes,len(label_0_indexes)//10)) # 10% of the training set would be the validation set
    seed_everything()
    val_index.extend(random.sample(label_1_indexes,len(label_1_indexes)//10)) # 10% of the training set would be the validation set
    train_index = [i for i in all_indexes if i not in val_index]
    return train_index, val_index

def load_val(pickle_file_path):
    with open(pickle_file_path, 'rb') as f:
        val_set = pickle.load(f)
    return val_set

class MLP_1_relu(nn.Module):
    def __init__(self, input_size, hidden_size,dropout_rate):
        super(MLP_1_relu, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(p = dropout_rate)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = self.dropout(x)
        x = self.output(x)
        return x

class MLP_2_relu(nn.Module):
    def __init__(self, input_size, hidden_size,dropout_rate):
        super(MLP_2_relu, self).__init__()
        self.hidden_1 = nn.Linear(input_size, hidden_size)
        self.hidden_2 = nn.Linear(hidden_size,hidden_size//4)
        self.dropout = nn.Dropout(p = dropout_rate)
        self.output = nn.Linear(hidden_size//4, 1)

    def forward(self, x):
        x = torch.relu(self.hidden_1(x))
        x = self.dropout(x)
        x = torch.relu(self.hidden_2(x))
        x = self.dropout(x)
        x = self.output(x)
        return x


class MLP_1_sigmoid(nn.Module):
    def __init__(self, input_size, hidden_size,dropout_rate):
        super(MLP_1_sigmoid, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(p = dropout_rate)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))
        x = self.dropout(x)
        x = self.output(x)
        return x

class MLP_2_sigmoid(nn.Module):
    def __init__(self, input_size, hidden_size,dropout_rate):
        super(MLP_2_sigmoid, self).__init__()
        self.hidden_1 = nn.Linear(input_size, hidden_size)
        self.hidden_2 = nn.Linear(hidden_size,hidden_size//4)
        self.dropout = nn.Dropout(p = dropout_rate)
        self.output = nn.Linear(hidden_size//4, 1)

    def forward(self, x):
        x = torch.sigmoid(self.hidden_1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.hidden_2(x))
        x = self.dropout(x)
        x = self.output(x)
        return x


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def cosine_similarity(vec_a,vec_b):
    dot_product = np.dot(vec_a,vec_b)
    magnitude_a = np.linalg.norm(vec_a)
    magnitude_b = np.linalg.norm(vec_b)
    similarity = dot_product / (magnitude_a * magnitude_b)
    return similarity

PREC_KS = [20, 100]

def precision_at_k(labels, scores, k):
    # fraction of true positives among the top-k highest-scored instances.
    # Instances tied with the k-th score share the remaining slots in proportion
    # (the average over every way of breaking the tie), so the result does not
    # depend on row order.
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    k = min(k, len(scores))
    kth_score = np.sort(scores)[::-1][k - 1]
    above = scores > kth_score
    tied = scores == kth_score
    n_left = k - above.sum()  # top-k slots left for the tied group
    return (labels[above].sum() + n_left * labels[tied].mean()) / k

def train_lg(data,n_folds,target,phenotype,start_ratio = 1):
    # data is a pandas dataframe
    target_data = data.loc[target,:]
    target_data['label'] = 1
    data = data.drop(target)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,random_state=123)  # stratified sampling return index
    train_folds = list()
    test_folds = list()
    test_pro = list()  # all instances in test, used for baseline roc calculation
    ratio_list = list()
    for i, (train_index, test_index) in enumerate(skf.split(data.iloc[:,:-1], data.iloc[:,-1])):
        print(train_index, test_index)
        train_index = data.iloc[train_index,:].index.tolist()
        test_index = data.iloc[test_index,:].index.tolist()
        print("Fold " + str(i + 1))
        train_folds.append(train_index)
        test_folds.append(test_index)
        test_pro.extend(data.loc[test_index,:].index.tolist())
    pool_auc = list()
    pool_prec = {k: [] for k in PREC_KS}
    for r in range(start_ratio,11):
        ratio = r/10
        ratio_list.append(ratio)
        # for each ratio : 0.1, 0.2,... 1.0
        # We want to calculate the pooled auroc
        pool_prediction = list()
        pool_truelabel = list()
        pool_index = list()  # instance id for each pooled prediction
        pool_fold = list()   # fold that each instance was tested in
        for i in range(n_folds):
            lr_config = json.load(open('../../config/'+phenotype+'/best_paras_'+phenotype+'_LogisticRegression_subset_False_fold_'+str(i)+'_ratio_'+str(ratio)+'_5fold.json'))
            # lr_config = json.load(open('../../config/'+phenotype+'/best_paras_'+phenotype+'_LogisticRegression_subset_False_fold_'+str(i)+'_ratio_'+str(ratio)+'_ROS_True_5fold.json'))
            fold_data = data.copy()
            target_fold = target_data.copy()
            target_fold = pd.DataFrame(target_fold)
            test_data = fold_data.loc[test_folds[i],:]
            train_data = fold_data.loc[train_folds[i],]
            train_data = train_data.sample(frac=ratio, random_state=123)
            # print(target_fold)
            train_data = pd.concat([train_data,target_fold])
            X_train,y_train = train_data.iloc[:,:-1],train_data.iloc[:,-1]
            # ros = RandomOverSampler(sampling_strategy=0.1, random_state=123)
            # X_train, y_train = ros.fit_resample(X_train, y_train)
            X_test,y_test = test_data.iloc[:,:-1],test_data.iloc[:,-1]
            # print(X_train,y_train)
            if lr_config != None:
                seed_everything()
                model = LogisticRegression(**lr_config,solver = 'saga',class_weight='balanced')
            else:
                seed_everything()
                model = LogisticRegression(max_iter = 5000, class_weight='balanced')
            print("Fold " + str(i + 1)+' training ratio '+ str(ratio))
            model.fit(X_train, y_train)
            test_score = model.predict_proba(X_test)[:, 1]
            pool_prediction.extend(test_score)
            pool_truelabel.extend(y_test)
            pool_index.extend(test_data.index.tolist())
            pool_fold.extend([i] * len(y_test))
        pool_fpr,pool_tpr,_ = roc_curve(pool_truelabel,pool_prediction)
        pool_auroc = auc(pool_fpr,pool_tpr)
        pool_auc.append(pool_auroc)
        for k in PREC_KS:
            pool_prec[k].append(precision_at_k(pool_truelabel, pool_prediction, k))
        # pooled predictions for this ratio; the final ratio (1.0) overwrites and is returned
        pool_pred_df = pd.DataFrame({'prediction': pool_prediction, 'label': pool_truelabel, 'fold': pool_fold}, index=pool_index)
    return pool_auc,ratio_list,pool_fpr,pool_tpr,test_pro,pool_pred_df,pool_prec

def train_rf(data,n_folds,target,phenotype,start_ratio = 1):
    # data = data.drop(['RWR_0.2_1.0','RWR_0.4_1.0','RWR_0.6_1.0'],axis = 1)
    target_data = data.loc[target,:]
    target_data['label'] = 1
    data = data.drop(target)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,random_state=123)  # stratified sampling return index
    train_folds = list()
    test_folds = list()
    test_pro = list()  # all instances in test, used for baseline roc calculation
    ratio_list = list()
    for i, (train_index, test_index) in enumerate(skf.split(data.iloc[:,:-1], data.iloc[:,-1])):
        print(train_index, test_index)
        train_index = data.iloc[train_index,:].index.tolist()
        test_index = data.iloc[test_index,:].index.tolist()
        print("Fold " + str(i + 1))
        train_folds.append(train_index)
        test_folds.append(test_index)
        test_pro.extend(data.loc[test_index,:].index.tolist())
    pool_auc = list()
    pool_prec = {k: [] for k in PREC_KS}
    for r in range(start_ratio,11):
        ratio = r/10
        ratio_list.append(ratio)
        # for each ratio : 0.1, 0.2,... 1.0
        # We want to calculate the pooled auroc
        pool_prediction = list()
        pool_truelabel = list()
        pool_index = list()  # instance id for each pooled prediction
        pool_fold = list()   # fold that each instance was tested in
        for i in range(n_folds):
            rf_config = json.load(open('../../config/'+phenotype+'/best_paras_'+phenotype+'_RandomForest_subset_False_fold_'+str(i)+'_ratio_'+str(ratio)+'_5fold.json'))
            # rf_config = json.load(open('../../config/'+phenotype+'/best_paras_'+phenotype+'_RandomForest_subset_False_fold_'+str(i)+'_ratio_'+str(ratio)+'_ROS_True_5fold.json'))
            fold_data = data.copy()
            target_fold = target_data.copy()
            target_fold = pd.DataFrame(target_fold)
            test_data = fold_data.loc[test_folds[i],:]
            train_data = fold_data.loc[train_folds[i],]
            train_data = train_data.sample(frac=ratio, random_state=123)
            train_data = pd.concat([train_data,target_fold])
            X_train,y_train = train_data.iloc[:,:-1],train_data.iloc[:,-1]
            # ros = RandomOverSampler(sampling_strategy=0.1, random_state=123)
            # X_train, y_train = ros.fit_resample(X_train, y_train)
            X_test,y_test = test_data.iloc[:,:-1],test_data.iloc[:,-1]
            if rf_config != None:
                seed_everything()
                model = RandomForestClassifier(**rf_config,class_weight='balanced')
            else:
                seed_everything()
                model = RandomForestClassifier(n_estimators=1000,class_weight='balanced')
            print("Fold " + str(i + 1)+' training ratio '+ str(ratio))
            model.fit(X_train, y_train)
            test_score = model.predict_proba(X_test)[:, 1]
            pool_prediction.extend(test_score)
            pool_truelabel.extend(y_test)
            pool_index.extend(test_data.index.tolist())
            pool_fold.extend([i] * len(y_test))
            # a,b,_ = roc_curve(y_test,test_score)
            # print(auc(a,b))
        pool_fpr,pool_tpr,_ = roc_curve(pool_truelabel,pool_prediction)
        pool_auroc = auc(pool_fpr,pool_tpr)
        pool_auc.append(pool_auroc)
        for k in PREC_KS:
            pool_prec[k].append(precision_at_k(pool_truelabel, pool_prediction, k))
        # pooled predictions for this ratio; the final ratio (1.0) overwrites and is returned
        pool_pred_df = pd.DataFrame({'prediction': pool_prediction, 'label': pool_truelabel, 'fold': pool_fold}, index=pool_index)
    return pool_auc,ratio_list,pool_fpr,pool_tpr,test_pro,pool_pred_df,pool_prec

def train_xgb(data,n_folds,target,phenotype,start_ratio = 1):
    # data = data.drop(['RWR_0.2_1.0','RWR_0.4_1.0','RWR_0.6_1.0'],axis = 1)
    target_data = data.loc[target,:]
    target_data['label'] = 1
    data = data.drop(target)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,random_state=123)  # stratified sampling return index
    train_folds = list()
    test_folds = list()
    test_pro = list()  # all instances in test, used for baseline roc calculation
    ratio_list = list()
    for i, (train_index, test_index) in enumerate(skf.split(data.iloc[:,:-1], data.iloc[:,-1])):
        print(train_index, test_index)
        train_index = data.iloc[train_index,:].index.tolist()
        test_index = data.iloc[test_index,:].index.tolist()
        print("Fold " + str(i + 1))
        train_folds.append(train_index)
        test_folds.append(test_index)
        test_pro.extend(data.loc[test_index,:].index.tolist())
    pool_auc = list()
    pool_prec = {k: [] for k in PREC_KS}
    for r in range(start_ratio,11):
        ratio = r/10
        ratio_list.append(ratio)
        # for each ratio : 0.1, 0.2,... 1.0
        # We want to calculate the pooled auroc
        pool_prediction = list()
        pool_truelabel = list()
        pool_index = list()  # instance id for each pooled prediction
        pool_fold = list()   # fold that each instance was tested in
        for i in range(n_folds):
            xgb_config = json.load(open('../../config/'+phenotype+'/best_paras_'+phenotype+'_XGBoost_subset_False_fold_'+str(i)+'_ratio_'+str(ratio)+'_5fold.json'))
            # xgb_config = json.load(open('../../config/'+phenotype+'/best_paras_'+phenotype+'_XGBoost_subset_False_fold_'+str(i)+'_ratio_'+str(ratio)+'_ROS_True_5fold.json'))
            fold_data = data.copy()
            target_fold = target_data.copy()
            target_fold = pd.DataFrame(target_fold)
            test_data = fold_data.loc[test_folds[i],:]
            train_data = fold_data.loc[train_folds[i],]
            train_data = train_data.sample(frac=ratio, random_state=123)
            train_data = pd.concat([train_data,target_fold])
            X_train,y_train = train_data.iloc[:,:-1],train_data.iloc[:,-1]
            X_test,y_test = test_data.iloc[:,:-1],test_data.iloc[:,-1]
            # ros = RandomOverSampler(sampling_strategy=0.1, random_state=123)
            # X_train, y_train = ros.fit_resample(X_train, y_train)
            # print(X_train,y_train)
            class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
            class_weight={0: class_weights[0], 1: class_weights[1]}
            if xgb_config != None:
                seed_everything()
                model = xgb.XGBClassifier(**xgb_config)
            else:
                seed_everything()
                model = xgb.XGBClassifier()
            model.fit(X_train, y_train,sample_weight = np.vectorize(class_weight.get)(y_train))
            print("Fold " + str(i + 1)+' training ratio '+ str(ratio))
            test_score = model.predict_proba(X_test)[:, 1]
            pool_prediction.extend(test_score)
            pool_truelabel.extend(y_test)
            pool_index.extend(test_data.index.tolist())
            pool_fold.extend([i] * len(y_test))
        pool_fpr,pool_tpr,_ = roc_curve(pool_truelabel,pool_prediction)
        pool_auroc = auc(pool_fpr,pool_tpr)
        pool_auc.append(pool_auroc)
        for k in PREC_KS:
            pool_prec[k].append(precision_at_k(pool_truelabel, pool_prediction, k))
        # pooled predictions for this ratio; the final ratio (1.0) overwrites and is returned
        pool_pred_df = pd.DataFrame({'prediction': pool_prediction, 'label': pool_truelabel, 'fold': pool_fold}, index=pool_index)
    return pool_auc,ratio_list,pool_fpr,pool_tpr,test_pro,pool_pred_df,pool_prec

def train_MLP(data,n_folds,target,phenotype,start_ratio = 1):
    target_data = data.loc[target,:]
    target_data['label'] = 1
    data = data.drop(target)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,random_state=123)  # stratified sampling return index
    train_folds = list()
    test_folds = list()
    test_pro = list()  # all instances in test, used for baseline roc calculation
    ratio_list = list()
    for i, (train_index, test_index) in enumerate(skf.split(data.iloc[:,:-1], data.iloc[:,-1])):
        print(train_index, test_index)
        train_index = data.iloc[train_index,:].index.tolist()
        test_index = data.iloc[test_index,:].index.tolist()
        print("Fold " + str(i + 1))
        train_folds.append(train_index)
        test_folds.append(test_index)
        test_pro.extend(data.loc[test_index,:].index.tolist())
    pool_auc = list()
    pool_prec = {k: [] for k in PREC_KS}
    for r in range(start_ratio,11):
        ratio = r/10
        ratio_list.append(ratio)
        # for each ratio : 0.1, 0.2,... 1.0
        # We want to calculate the pooled auroc
        pool_prediction = list()
        pool_truelabel = list()
        pool_index = list()  # instance id for each pooled prediction
        pool_fold = list()   # fold that each instance was tested in
        for i in range(n_folds):
            mlp_config = json.load(open('../../config/'+phenotype+'/best_paras_'+phenotype+'_MLP_subset_False_fold_'+str(i)+'_ratio_'+str(ratio)+'_5fold.json'))
            # mlp_config = json.load(open('../../config/'+phenotype+'/best_paras_'+phenotype+'_MLP_subset_False_fold_'+str(i)+'_ratio_'+str(ratio)+'_ROS_True_5fold.json'))
            fold_data = data.copy()
            target_fold = target_data.copy()
            target_fold = pd.DataFrame(target_fold)
            test_data = fold_data.loc[test_folds[i],:]
            train_data = fold_data.loc[train_folds[i],]
            train_data = train_data.sample(frac=ratio, random_state=123)
            train_data = pd.concat([train_data,target_fold])
            X_train,y_train = train_data.iloc[:,:-1],train_data.iloc[:,-1]
            # ros = RandomOverSampler(sampling_strategy=0.1, random_state=123)
            # X_train, y_train = ros.fit_resample(X_train, y_train)
            X_train = torch.tensor(np.array(X_train),dtype = torch.float32)
            y_train = torch.tensor(np.array(y_train),dtype = torch.float32).unsqueeze(1)
            X_test = torch.tensor(np.array(test_data.iloc[:,:-1]),dtype = torch.float32)
            y_test = np.array(test_data.iloc[:,-1])
            # print(X_train,y_train)
            # class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
            # class_weight={0: class_weights[0], 1: class_weights[1]}
            if mlp_config != None:
                batch_size = mlp_config['batch_size']
                num_epoch = mlp_config['num_epoch']
                learning_rate = mlp_config['learning_rate']
                hidden_size = mlp_config['hidden_size']
                num_layer = mlp_config['num_layer']
                activation_type = mlp_config['activation']
                if num_layer == 1:
                    if activation_type == 'sigmoid':
                        seed_everything()
                        model = MLP_1_sigmoid(X_train.shape[1],hidden_size=hidden_size,dropout_rate=0.3)
                    elif activation_type == 'relu':
                        seed_everything()
                        model = MLP_1_relu(X_train.shape[1],hidden_size=hidden_size,dropout_rate=0.3)
                elif num_layer == 2:
                    if activation_type == 'sigmoid':
                        seed_everything()
                        model = MLP_2_sigmoid(X_train.shape[1],hidden_size=hidden_size,dropout_rate=0.3)
                    elif activation_type == 'relu':
                        seed_everything()
                        model = MLP_2_relu(X_train.shape[1],hidden_size=hidden_size,dropout_rate=0.3)
                criterion = nn.BCEWithLogitsLoss()
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            else:
                seed_everything()
                model = MLP_1_relu(X_train.shape[1],hidden_size=128,dropout_rate=0.3)
                criterion = nn.BCEWithLogitsLoss()
                optimizer = optim.Adam(model.parameters(), lr=1e-3)
            train_dataset = CustomDataset(X_train,y_train)
            seed_everything()
            train_loader = DataLoader(dataset = train_dataset,batch_size = batch_size,shuffle=True)
            for epoch in range(num_epoch):
                for batch_feature,batch_label in train_loader:
                    # batch_feature, batch_label = batch_feature.to(device), batch_label.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_feature)
                    loss = criterion(outputs,batch_label)
                    loss.backward()
                    optimizer.step()
            model.eval()
            with torch.no_grad():
                logits = model(X_test).view(-1)
                predictions = torch.sigmoid(logits).tolist()
                # predictions = model(X_test).view(-1).tolist()
                pool_prediction.extend(predictions)
                pool_truelabel.extend(y_test)
                pool_index.extend(test_data.index.tolist())
                pool_fold.extend([i] * len(y_test))
            print("Fold " + str(i + 1)+' training ratio '+ str(ratio))
        pool_fpr,pool_tpr,_ = roc_curve(pool_truelabel,pool_prediction)
        pool_auroc = auc(pool_fpr,pool_tpr)
        pool_auc.append(pool_auroc)
        for k in PREC_KS:
            pool_prec[k].append(precision_at_k(pool_truelabel, pool_prediction, k))
        # pooled predictions for this ratio; the final ratio (1.0) overwrites and is returned
        pool_pred_df = pd.DataFrame({'prediction': pool_prediction, 'label': pool_truelabel, 'fold': pool_fold}, index=pool_index)
    return pool_auc,ratio_list,pool_fpr,pool_tpr,test_pro,pool_pred_df,pool_prec


def plot_heat(spl_data,phenotype,alpha_list,test_pro_list):
    ratio_list = [i/10 for i in range(1,11)]
    # spl_data = pd.read_csv(spl_file,index_col=0)
    label_data = spl_data['label']
    heat_roc = list()
    heat_prec = {k: [] for k in PREC_KS}
    last_fold_fpr = list()
    last_fold_tpr = list()
    # Calculate ROC curve and AUC
    for i in alpha_list:
        roc_score = list()
        prec_score = {k: [] for k in PREC_KS}
        heat_score = pd.read_csv('../../Data/'+phenotype+'/'+'/heat_positive_diffusion_'+phenotype+'_state_'+str(i)+'.csv',index_col = 0)
        heat_score = pd.merge(heat_score,pd.DataFrame(label_data),left_on='protein',right_on='protein')
        heat_score = heat_score.set_index('protein')
        heat_score = heat_score.loc[test_pro_list,:]
        cols = heat_score.columns
        cols = cols.drop('label')
        for j in cols:
            scores = heat_score[j]
            labels = heat_score['label']
            fpr, tpr, thresholds = roc_curve(labels, scores)
            roc_auc = auc(fpr, tpr)
            roc_score.append(roc_auc)
            for k in PREC_KS:
                prec_score[k].append(precision_at_k(labels, scores, k))
        last_fold_fpr.append(fpr)
        last_fold_tpr.append(tpr)
        heat_roc.append(roc_score) # length = len(alpha_list), each element is the list
        for k in PREC_KS:
            heat_prec[k].append(prec_score[k])
    return ratio_list,heat_roc,last_fold_fpr,last_fold_tpr,heat_prec

def plot_rw(spl_data,phenotype,k_steps,test_pro_list):
    ratio_list = [i/10 for i in range(1,11)]
    # spl_data = pd.read_csv(spl_file,index_col=0)
    label_data = spl_data['label']
    rw_roc = list()
    rw_prec = {k: [] for k in PREC_KS}
    last_fold_fpr = list()
    last_fold_tpr = list()
    # Calculate ROC curve and AUC
    for i in k_steps:
        roc_score = list()
        prec_score = {k: [] for k in PREC_KS}
        rw_score = pd.read_csv('../../Data/'+phenotype+'/'+'RW_positive_diffusion_'+phenotype+'_state_'+str(i)+'.csv',index_col = 0)
        rw_score = pd.merge(rw_score,pd.DataFrame(label_data),left_on='protein',right_on='protein')
        rw_score = rw_score.set_index('protein')
        rw_score = rw_score.loc[test_pro_list,:]
        cols = rw_score.columns
        cols = cols.drop('label')
        for j in cols:
            scores = rw_score[j]
            labels = rw_score['label']
            fpr, tpr, thresholds = roc_curve(labels, scores)
            roc_auc = auc(fpr, tpr)
            roc_score.append(roc_auc)
            for k in PREC_KS:
                prec_score[k].append(precision_at_k(labels, scores, k))
        last_fold_fpr.append(fpr)
        last_fold_tpr.append(tpr)
        rw_roc.append(roc_score) # length = len(alpha_list), each element is the list
        for k in PREC_KS:
            rw_prec[k].append(prec_score[k])
    # ax.plot(ratio_list,roc_score, linestyle='--',lw=1,alpha = 0.7)
    return ratio_list,rw_roc,last_fold_fpr,last_fold_tpr,rw_prec


def plot_rwr(spl_data,phenotype,restart_state,test_pro_list):
    ratio_list = [i/10 for i in range(1,11)]
    # spl_data = pd.read_csv(spl_file,index_col=0)
    label_data = spl_data['label']
    rwr_roc = list()
    rwr_prec = {k: [] for k in PREC_KS}
    last_fold_fpr = list()
    last_fold_tpr = list()
    # Calculate ROC curve and AUC
    for i in restart_state:
        roc_score = list()
        prec_score = {k: [] for k in PREC_KS}
        rwr_score = pd.read_csv('../../Data/'+phenotype+'/'+'RWR_positive_diffusion_'+phenotype+'_state_'+str(i)+'.csv',index_col = 0)
        rwr_score = pd.merge(rwr_score,pd.DataFrame(label_data),left_on='protein',right_on='protein')
        rwr_score = rwr_score.set_index('protein')
        rwr_score = rwr_score.loc[test_pro_list,:]
        cols = rwr_score.columns
        cols = cols.drop('label')
        for j in cols:
            scores = rwr_score[j]
            labels = rwr_score['label']
            fpr, tpr, thresholds = roc_curve(labels, scores)
            roc_auc = auc(fpr, tpr)
            roc_score.append(roc_auc)
            for k in PREC_KS:
                prec_score[k].append(precision_at_k(labels, scores, k))
        last_fold_fpr.append(fpr)
        last_fold_tpr.append(tpr)
        rwr_roc.append(roc_score) # length = len(alpha_list), each element is the list
        for k in PREC_KS:
            rwr_prec[k].append(prec_score[k])
    # ax.plot(ratio_list,roc_score, linestyle='--',lw=1,alpha = 0.7)
    return ratio_list,rwr_roc,last_fold_fpr,last_fold_tpr,rwr_prec


def plot_target(spl_data,phenotype,kernel_state,kernel_name,test_pro_list):
    # spl_data = pd.read_csv(spl_file,index_col=0)
    label_data = spl_data['label']
    fpr_list = list()
    tpr_list = list()
    target_prec = {k: [] for k in PREC_KS}
    data_path = '../../Data/'+phenotype + '/'+kernel_name+'_target_diffusion_'+phenotype+'.csv'
    score = pd.read_csv(data_path)
    # print(score)
    score = pd.merge(score,pd.DataFrame(label_data),left_on='protein',right_on='protein')
    # print(score)
    score = score.set_index('protein')
    score = score.loc[test_pro_list,:]
    for i in kernel_state:
        col_name = kernel_name+'_target_'+str(i)
        scores = score[col_name]
        labels = score['label']
        fpr, tpr, _ = roc_curve(labels, scores)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        for k in PREC_KS:
            target_prec[k].append(precision_at_k(labels, scores, k))
    return fpr_list,tpr_list,target_prec


def plot_learning_curve(data,n_folds,save_path,data_name,shortest_path_len_file,target,plot_title,
                        rw_positive_state,rwr_positive_state,heat_positive_state,rw_target_state,rwr_target_state,heat_target_state,phenotype,
                        start_ratio = 1,save_fig1 = True, save_fig2 = True, save_fig3 = True, save_fig4 = True):
    ## This function is used for generating AUROC/AUPRC/Loss for
    data = pd.read_csv(data,index_col = 0)
    print(data)
    pos = list(set(np.load('../../Data/Raw/'+phenotype+'/'+phenotype+'_pos_gene.npy')).intersection(set(data.index.tolist())))
    data.loc[pos,'label'] = 1
    all_ins = data.index.tolist()
    fig1, ax1 = plt.subplots(figsize = (8,6))
    fig2, ax2 = plt.subplots(figsize = (8,6))
    fig3, ax3 = plt.subplots(figsize = (8,6))  # Precision@20 learning curve
    fig4, ax4 = plt.subplots(figsize = (8,6))  # Precision@100 learning curve
    test_proteins = list(set(all_ins)-set(target))

    ## Shortest Path Len
    spl_data = pd.read_csv(shortest_path_len_file,index_col=0)
    # spl_data = spl_data.loc[all_ins,]
    # spl_data = spl_data.drop(target)
    spl_data = spl_data.loc[test_proteins,:]
    # print(spl_data)



    lr_pool_auc, ratio_list,pool_fpr,pool_tpr,test_proteins,lr_pred_df,lr_prec = train_lg(data,n_folds,target = target,phenotype = phenotype,start_ratio = start_ratio)
    lr_pred_df.to_csv(save_path+'pooled_predictions_LogisticRegression_'+data_name+'.csv')
    ax1.plot(ratio_list,lr_pool_auc,linestyle = '-',label = 'Logistic regression',lw = 3, alpha = 1)
    ax3.plot(ratio_list,lr_prec[20],linestyle = '-',label = 'Logistic regression',lw = 3, alpha = 1)
    ax4.plot(ratio_list,lr_prec[100],linestyle = '-',label = 'Logistic regression',lw = 3, alpha = 1)
    auroc = auc(pool_fpr,pool_tpr)
    ax2.plot(pool_fpr,pool_tpr,linestyle = '-',label = r"Logistic regression (AUROC = %0.2f )" % (auroc),lw = 2, alpha = 1)

    rf_pool_auc, ratio_list,pool_fpr,pool_tpr,test_proteins,rf_pred_df,rf_prec = train_rf(data,n_folds,target = target,phenotype = phenotype,start_ratio = start_ratio)
    rf_pred_df.to_csv(save_path+'pooled_predictions_RandomForest_'+data_name+'.csv')
    ax1.plot(ratio_list,rf_pool_auc,linestyle = '-',label = 'Random forest',lw=3, alpha=1)
    ax3.plot(ratio_list,rf_prec[20],linestyle = '-',label = 'Random forest',lw=3, alpha=1)
    ax4.plot(ratio_list,rf_prec[100],linestyle = '-',label = 'Random forest',lw=3, alpha=1)
    auroc = auc(pool_fpr,pool_tpr)
    ax2.plot(pool_fpr,pool_tpr,linestyle = '-',label = r"Random forest (AUROC = %0.2f )" % (auroc),lw = 2, alpha = 1)

    xgb_pool_auc, ratio_list,pool_fpr,pool_tpr,test_proteins,xgb_pred_df,xgb_prec = train_xgb(data,n_folds,target = target,phenotype = phenotype,start_ratio = start_ratio)
    xgb_pred_df.to_csv(save_path+'pooled_predictions_XGBoost_'+data_name+'.csv')
    ax1.plot(ratio_list,xgb_pool_auc,linestyle = '-',label = 'XGBoost',lw = 3, alpha = 1)
    ax3.plot(ratio_list,xgb_prec[20],linestyle = '-',label = 'XGBoost',lw = 3, alpha = 1)
    ax4.plot(ratio_list,xgb_prec[100],linestyle = '-',label = 'XGBoost',lw = 3, alpha = 1)
    auroc = auc(pool_fpr,pool_tpr)
    ax2.plot(pool_fpr,pool_tpr,linestyle = '-',label = r"XGBoost (AUROC = %0.2f )" % (auroc),lw = 2, alpha = 1)

    mlp_pool_auc, ratio_list,pool_fpr,pool_tpr,test_proteins,mlp_pred_df,mlp_prec = train_MLP(data,n_folds,target = target,phenotype = phenotype,start_ratio = start_ratio)
    mlp_pred_df.to_csv(save_path+'pooled_predictions_MLP_'+data_name+'.csv')
    ax1.plot(ratio_list,mlp_pool_auc,linestyle = '-',label = 'Neural network',lw = 3, alpha = 1)
    ax3.plot(ratio_list,mlp_prec[20],linestyle = '-',label = 'Neural network',lw = 3, alpha = 1)
    ax4.plot(ratio_list,mlp_prec[100],linestyle = '-',label = 'Neural network',lw = 3, alpha = 1)
    auroc = auc(pool_fpr,pool_tpr)
    ax2.plot(pool_fpr,pool_tpr,linestyle = '-',label = r"Neural network (AUROC = %0.2f )" % (auroc),lw = 2, alpha = 1)




    # Diffusion Scores
    # rw_positive_state = [10,20,30]
    # rwr_positive_state = [0.2,0.4,0.6]
    # heat_positive_state = [0.1,0.2,0.3]

    # rw_target_state = [10,20,30]
    # rwr_target_state = [0.2,0.4,0.6]
    # heat_target_state = [0.1,0.2,0.3]

    ## Plot only the best baselines
    # rw_state = [20]
    # rwr_state = [0.6]
    # heat_state = [0.1]

    ratio_list,rw_roc_list,rw_last_fold_fpr,rw_last_fold_tpr,rw_prec = plot_rw(spl_data,phenotype,rw_positive_state,test_proteins)
    ratio_list,rwr_roc_list,rwr_last_fold_fpr,rwr_last_fold_tpr,rwr_prec = plot_rwr(spl_data,phenotype,rwr_positive_state,test_proteins)
    ratio_list,heat_roc_list,heat_last_fold_fpr,heat_last_fold_tpr,heat_prec = plot_heat(spl_data,phenotype,heat_positive_state,test_proteins)
    rw_target_fpr, rw_target_tpr, rw_target_prec = plot_target(spl_data,phenotype,rw_target_state,'RW',test_proteins)
    rwr_target_fpr, rwr_target_tpr, rwr_target_prec = plot_target(spl_data,phenotype,rwr_target_state,'RWR',test_proteins)
    heat_target_fpr, heat_target_tpr, heat_target_prec = plot_target(spl_data,phenotype,heat_target_state,'Heat',test_proteins)
    for i in range(len(rw_positive_state)):
        rw_roc = rw_roc_list[i]
        rw_pool_fpr = rw_last_fold_fpr[i]
        rw_pool_tpr  = rw_last_fold_tpr[i]
        auroc = auc(rw_pool_fpr, rw_pool_tpr)
        # ax1.plot(ratio_list,rw_roc, marker = 'h',label = 'RW_positive_'+str(rw_positive_state[i]),lw=1,alpha = 0.7)
        ax1.plot(ratio_list,rw_roc, marker = 'h',label = 'RW positive diffusion',lw=1,alpha = 0.7)
        ax3.plot(ratio_list,rw_prec[20][i], marker = 'h',label = 'RW positive diffusion',lw=1,alpha = 0.7)
        ax4.plot(ratio_list,rw_prec[100][i], marker = 'h',label = 'RW positive diffusion',lw=1,alpha = 0.7)
        #ax2.plot(rw_pool_fpr,rw_pool_tpr,label = 'RW_'+str(rw_state[i]),lw=1,alpha=0.5)
    for i in range(len(rwr_positive_state)):
        rwr_roc = rwr_roc_list[i]
        rwr_pool_fpr = rwr_last_fold_fpr[i]
        rwr_pool_tpr  = rwr_last_fold_tpr[i]
        auroc = auc(rwr_pool_fpr, rwr_pool_tpr)
        # ax1.plot(ratio_list,rwr_roc, marker = '*',label = 'RWR_positive_'+str(rwr_positive_state[i]),lw=1,alpha = 0.7)
        ax1.plot(ratio_list,rwr_roc, marker = '*',label = 'RWR positive diffusion',lw=1,alpha = 0.7)
        ax3.plot(ratio_list,rwr_prec[20][i], marker = '*',label = 'RWR positive diffusion',lw=1,alpha = 0.7)
        ax4.plot(ratio_list,rwr_prec[100][i], marker = '*',label = 'RWR positive diffusion',lw=1,alpha = 0.7)
        #ax2.plot(rwr_pool_fpr,rwr_pool_tpr,label = 'RWR_'+str(rwr_state[i]),lw=1,alpha=0.5)
    for i in range(len(heat_positive_state)):
        heat_roc = heat_roc_list[i]
        heat_pool_fpr = heat_last_fold_fpr[i]
        heat_pool_tpr  = heat_last_fold_tpr[i]
        auroc = auc(heat_pool_fpr, heat_pool_tpr)
        # ax1.plot(ratio_list,heat_roc, marker = '^',label = 'Heat_positive_'+str(heat_positive_state[i]),lw=1,alpha = 0.7)
        ax1.plot(ratio_list,heat_roc, marker = '^',label = 'Heat positive diffusion',lw=1,alpha = 0.7)
        ax3.plot(ratio_list,heat_prec[20][i], marker = '^',label = 'Heat positive diffusion',lw=1,alpha = 0.7)
        ax4.plot(ratio_list,heat_prec[100][i], marker = '^',label = 'Heat positive diffusion',lw=1,alpha = 0.7)
        #ax2.plot(heat_pool_fpr,heat_pool_tpr,label = 'Heat_'+str(heat_state[i]),lw=1,alpha=0.5)

## Target Diffusion
    for i in range(len(rw_target_state)):
        auroc = auc(rw_target_fpr[i], rw_target_tpr[i])
        # ax1.plot(ratio_list,[auroc]*len(ratio_list), marker = 'h',linestyle = '--',label = 'RW_target_'+str(rw_target_state[i]),lw=1,alpha = 0.7)
        ax1.plot(ratio_list,[auroc]*len(ratio_list), marker = 'h',linestyle = '--',label = 'RW target diffusion',lw=1,alpha = 0.7)
        ax3.plot(ratio_list,[rw_target_prec[20][i]]*len(ratio_list), marker = 'h',linestyle = '--',label = 'RW target diffusion',lw=1,alpha = 0.7)
        ax4.plot(ratio_list,[rw_target_prec[100][i]]*len(ratio_list), marker = 'h',linestyle = '--',label = 'RW target diffusion',lw=1,alpha = 0.7)
        #ax2.plot(rw_target_fpr[i],rw_target_tpr[i],label = 'RW_target_'+str(rw_state[i]),lw=1,alpha=0.5)
    for i in range(len(rwr_target_state)):
        auroc = auc(rwr_target_fpr[i], rwr_target_tpr[i])
        # ax1.plot(ratio_list,[auroc]*len(ratio_list), marker = '*',linestyle = '--',label = 'RWR_target_'+str(rwr_target_state[i]),lw=1,alpha = 0.7)
        ax1.plot(ratio_list,[auroc]*len(ratio_list), marker = '*',linestyle = '--',label = 'RWR target diffusion',lw=1,alpha = 0.7)
        ax3.plot(ratio_list,[rwr_target_prec[20][i]]*len(ratio_list), marker = '*',linestyle = '--',label = 'RWR target diffusion',lw=1,alpha = 0.7)
        ax4.plot(ratio_list,[rwr_target_prec[100][i]]*len(ratio_list), marker = '*',linestyle = '--',label = 'RWR target diffusion',lw=1,alpha = 0.7)
        #ax2.plot(rwr_target_fpr[i],rwr_target_tpr[i],label = 'RWR_target_'+str(rwr_state[i]),lw=1,alpha=0.5)
    for i in range(len(heat_target_state)):
        auroc = auc(heat_target_fpr[i], heat_target_tpr[i])
        # ax1.plot(ratio_list,[auroc]*len(ratio_list), marker = '^',linestyle = '--',label = 'Heat_target_'+str(heat_target_state[i]),lw=1,alpha = 0.7)
        ax1.plot(ratio_list,[auroc]*len(ratio_list), marker = '^',linestyle = '--',label = 'Heat target diffusion',lw=1,alpha = 0.7)
        ax3.plot(ratio_list,[heat_target_prec[20][i]]*len(ratio_list), marker = '^',linestyle = '--',label = 'Heat target diffusion',lw=1,alpha = 0.7)
        ax4.plot(ratio_list,[heat_target_prec[100][i]]*len(ratio_list), marker = '^',linestyle = '--',label = 'Heat target diffusion',lw=1,alpha = 0.7)
        #ax2.plot(heat_target_fpr[i],heat_target_tpr[i],label = 'Heat_target_'+str(heat_state[i]),lw=1,alpha=0.5)

    scores = spl_data['shortest_path_length'].to_list()
    labels = spl_data['label'].to_list()
    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = auc(fpr, tpr)
    ax1.plot(ratio_list,[auroc]*len(ratio_list), linestyle=(0,(1,4)),label = 'Shortest path length',lw=1,alpha = 0.7)
    # path lengths are stored negated (closer to the target = higher score), so they rank directly
    spl_prec20 = precision_at_k(labels, scores, 20)
    spl_prec100 = precision_at_k(labels, scores, 100)
    ax3.plot(ratio_list,[spl_prec20]*len(ratio_list), linestyle=(0,(1,4)),label = 'Shortest path length',lw=1,alpha = 0.7)
    ax4.plot(ratio_list,[spl_prec100]*len(ratio_list), linestyle=(0,(1,4)),label = 'Shortest path length',lw=1,alpha = 0.7)
    # ax2.plot(fpr,tpr,label="Shortest_path_len",lw=1,alpha=0.5)

    #ax1.legend(bbox_to_anchor=(1, 1), borderaxespad=0.)
    ax1.set_xlabel("Fraction of the available training set used",fontproperties=arial,fontsize = 20)
    ax1.set_ylabel('AUROC',fontproperties=arial,fontsize = 20)
    ax1.tick_params(axis='x', labelsize = 16)
    ax1.tick_params(axis='y', labelsize = 16)
    ax1.set_ylim(0.5,1)
    #ax1.set_position([0.1, 0.1, 0.65, 0.8])
    ax1.legend(loc = 'best',mode = "expand", ncol = 2,prop=arial_legend)
    ax1.set_title(plot_title,fontproperties=arial,fontsize = 24)
    if save_fig1:
        fig1.savefig(save_path+'Baseline_Comparison_Learning_Curve_'+data_name+'.png',dpi = 800,bbox_inches='tight')

    # Only show fractions up to 0.5: truncate each curve's data at x <= 0.5 so
    # the lines end cleanly on their 0.5 point instead of being clipped
    # mid-segment (which happens if we only set_xlim and the data runs to 1.0).
    XMAX = 0.5
    for prec_ax in (ax3, ax4):
        for line in prec_ax.get_lines():
            xd = np.asarray(line.get_xdata(), dtype=float)
            yd = np.asarray(line.get_ydata(), dtype=float)
            mask = xd <= XMAX + 1e-9
            line.set_xdata(xd[mask])
            line.set_ydata(yd[mask])

    # Default precision@K = positive rate (prevalence) in the ranked test set.
    # A random ranker achieves this precision at any K, so it is the chance
    # baseline for precision@K (analogue of the diagonal Chance line in the ROC).
    positive_rate = spl_data['label'].mean()
    HATCH = '////'            # sparse -> light, unobtrusive
    HATCH_COLOR = '0.85'    # near-white grey -> shallow shadow
    BAND_ZORDER = 0
    for prec_ax in (ax3, ax4):
        prec_ax.axhspan(
            0, positive_rate,
            xmin=0, xmax=1,            # axes fraction -> always full width
            facecolor='none',          # transparent -> doesn't cover the plot
            edgecolor=HATCH_COLOR,
            hatch=HATCH,
            linewidth=0,               # no border rectangle
            zorder=BAND_ZORDER,        # behind the lines
        )
        prec_ax.axhline(
            positive_rate, color=HATCH_COLOR, linestyle=':', lw=1,
            zorder=BAND_ZORDER
        )

    # Precision y-axis: 0.4 by default, raised in 0.05 steps if a curve would be
    # clipped (0.01 headroom keeps markers at the top fully visible)
    def prec_ylim_top(ax):
        ymax = max(np.max(line.get_ydata()) for line in ax.get_lines())
        top = max(0.4, round(np.ceil((ymax + 0.01) / 0.05) * 0.05, 2))
        print('%s: highest value %.3f, y-axis top %.2f' % (ax.get_ylabel(), ymax, top))
        return top

    # fig3: Precision@20 learning curve (same layout as fig1)
    ax3.set_xlabel("Fraction of the available training set used",fontproperties=arial,fontsize = 20)
    ax3.set_ylabel('Precision@20',fontproperties=arial,fontsize = 20)
    ax3.tick_params(axis='x', labelsize = 16)
    ax3.tick_params(axis='y', labelsize = 16)
    ax3.set_ylim(0,prec_ylim_top(ax3))
    ax3.set_xlim(0.08,0.52)
    ax3.set_title(plot_title,fontproperties=arial,fontsize = 24)
    ax3.legend(loc = 'best',mode = "expand", ncol = 2,prop=arial_legend)
    if save_fig3:
        fig3.savefig(save_path+'Baseline_Comparison_Precision_at_20_'+data_name+'.png',dpi = 800,bbox_inches='tight')

    # fig4: Precision@100 learning curve (same layout as fig1)
    ax4.set_xlabel("Fraction of the available training set used",fontproperties=arial,fontsize = 20)
    ax4.set_ylabel('Precision@100',fontproperties=arial,fontsize = 20)
    ax4.tick_params(axis='x', labelsize = 16)
    ax4.tick_params(axis='y', labelsize = 16)
    ax4.set_ylim(0,prec_ylim_top(ax4))
    ax4.set_xlim(0.08,0.52)
    ax4.set_title(plot_title,fontproperties=arial,fontsize = 24)
    ax4.legend(loc = 'best',mode = "expand", ncol = 2,prop=arial_legend)
    if save_fig4:
        fig4.savefig(save_path+'Baseline_Comparison_Precision_at_100_'+data_name+'.png',dpi = 800,bbox_inches='tight')

    ax2.set_xlabel("False positive rate",fontproperties=arial,fontsize = 20)
    ax2.set_ylabel("True positive rate",fontproperties=arial,fontsize = 20)
    ax2.tick_params(axis='x', labelsize=16)
    ax2.tick_params(axis='y', labelsize=16)
    ax2.set_title(plot_title,fontproperties=arial,fontsize = 24)
    #ax2.set_position([0.1, 0.1, 0.65, 0.8])
    ax2.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.7)
    # ax2.legend(loc = 'lower right')
    ax2.legend(loc = 'best',prop=arial_legend)
    if save_fig2:
        fig2.savefig(save_path+'Baseline_Comparison_ROC_'+data_name+'.png',dpi = 800,bbox_inches='tight')


if __name__=='__main__':
    #mapper = HGNCMapper('../../Data/Graphs/9606.protein.info.v11.5.txt.gz', '../../Data/Graphs/9606.protein.aliases.v11.5.txt.gz')

    ## No subset , with target feature
    plot_learning_curve(data = '../../Data/SREBP2/train_node_attribute_SREBP2_with_target_feature.csv',
                    n_folds = 5,
                    save_path = '../../Plot/SREBP2/',
                    shortest_path_len_file = '../../Data/SREBP2/shortest_path_length_data.csv',
                    data_name = 'SREBP2',
                    target = list(np.load('../../Data/targets/SREBP2_target.npy')),
                    rw_positive_state = [10],
                    rwr_positive_state = [0.6],
                    heat_positive_state = [0.1],
                    rw_target_state = [10],
                    rwr_target_state = [0.6],
                    heat_target_state = [0.1],
                    start_ratio = 1,
                    save_fig1 = True,
                    save_fig2 = True,
                    phenotype = 'SREBP2',
                    plot_title = 'Cholesterol homeostasis'
                    )
