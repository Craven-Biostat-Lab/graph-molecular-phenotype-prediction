from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import sys
sys.path.insert(0, '../Dataset')  # merge_feature.py lives in Code/Dataset
import merge_feature as mf
import argparse
from sklearn.utils.class_weight import compute_class_weight
import random
import math
import pickle
import torch.optim as optim
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams["font.family"] = "Arial"

# DATA_DIREC = '../../Data/'
PLOT_DIREC = '../../Plot/'

def seed_everything(seed=123):
    """"
    Seed everything.
    """   
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)



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

def train_lg():
    all_sources = pd.read_csv('../../Data/Mitomics/train_node_attribute_Mitomics_with_target_feature.csv')['protein'].unique().tolist()
    pool_aucs = list()
    pool_fpr = list()
    pool_tpr = list()
    pool_prec = {k: [] for k in PREC_KS}
    for ratio in range(1,11):
        pool_prediction = list()
        pool_truelabel = list()
        pool_index = list()  # instance id for each pooled prediction
        pool_fold = list()   # fold that each instance was tested in
        for i in range(5):
            train_raw_data = pd.read_csv('../../Data/Mitomics/fold_'+str(i)+'_train_by_source_with_target_feature.csv',index_col = 0)
            test_source = np.load('../../Data/Mitomics/fold_'+str(i)+'_test_protein.npy')
            train_source = sorted(set(all_sources)-set(test_source))  # fixed order so random.sample is reproducible
            train_raw_data = train_raw_data.drop(['Target'],axis = 1)
            test_data = pd.read_csv('../../Data/Mitomics/fold_'+str(i)+'_test_by_source_with_target_feature.csv',index_col = 0)
            test_data = test_data.drop(['Target'],axis = 1)
            r = ratio/10
            lr_config = json.load(open('../../config/Mitomics/best_paras_Mitomics_LogisticRegression_fold_'+str(i)+'_ratio_'+str(r)+'_5fold.json'))
            if r != 1.0:
                sample_size = max(1,int(len(train_source)*r))
                random.seed(123)
                sampled_train_node = random.sample(train_source,sample_size)
                sampled_train_data = train_raw_data.loc[sampled_train_node,]
            else:
                sampled_train_data = train_raw_data.copy()
            train_data = mf.move2last(sampled_train_data,'label')
            # train_data = train_data.drop(['Target'],axis = 1)
            test_data = mf.move2last(test_data,'label')
            # test_data = test_data.drop(['Target'],axis = 1)
            # print(test_data)
            X_train, y_train = train_data.iloc[:,:-1],train_data.iloc[:,-1]
            # ros = RandomOverSampler(sampling_strategy=0.1, random_state=123)
            # X_train, y_train = ros.fit_resample(X_train, y_train)
            X_test, y_test = test_data.iloc[:,:-1],test_data.iloc[:,-1]
            if lr_config != None:
                seed_everything()
                model = LogisticRegression(**lr_config,solver = 'saga',class_weight='balanced')
            else:
                seed_everything()
                model = LogisticRegression(max_iter = 5000, class_weight='balanced')
            print("Fold " + str(i + 1)+' training ratio '+ str(r))
            model.fit(X_train, y_train)
            test_score = model.predict_proba(X_test)[:, 1]
            pool_prediction.extend(test_score)
            pool_truelabel.extend(y_test)
            pool_index.extend(test_data.index.tolist())
            pool_fold.extend([i] * len(y_test))
        fpr,tpr,_ = roc_curve(pool_truelabel,pool_prediction)
        pool_auroc = auc(fpr,tpr)
        pool_aucs.append(pool_auroc)
        for k in PREC_KS:
            pool_prec[k].append(precision_at_k(pool_truelabel, pool_prediction, k))
        # pooled predictions for this ratio; the final ratio (1.0) overwrites and is returned
        pool_pred_df = pd.DataFrame({'prediction': pool_prediction, 'label': pool_truelabel, 'fold': pool_fold}, index=pool_index)
        if ratio == 10:
            pool_fpr.extend(fpr)
            pool_tpr.extend(tpr)
    return pool_aucs,pool_fpr,pool_tpr,pool_pred_df,pool_prec

def train_rf():
    all_sources = pd.read_csv('../../Data/Mitomics/train_node_attribute_Mitomics_with_target_feature.csv')['protein'].unique().tolist()
    pool_aucs = list()
    pool_fpr = list()
    pool_tpr = list()
    pool_prec = {k: [] for k in PREC_KS}
    for ratio in range(1,11):
        pool_prediction = list()
        pool_truelabel = list()
        pool_index = list()  # instance id for each pooled prediction
        pool_fold = list()   # fold that each instance was tested in
        for i in range(5):
            train_raw_data = pd.read_csv('../../Data/Mitomics/fold_'+str(i)+'_train_by_source_with_target_feature.csv',index_col = 0)
            test_source = np.load('../../Data/Mitomics/fold_'+str(i)+'_test_protein.npy')
            train_source = sorted(set(all_sources)-set(test_source))  # fixed order so random.sample is reproducible
            train_raw_data = train_raw_data.drop(['Target'],axis = 1)
            test_data = pd.read_csv('../../Data/Mitomics/fold_'+str(i)+'_test_by_source_with_target_feature.csv',index_col = 0)
            test_data = test_data.drop(['Target'],axis = 1)
            r = ratio/10
            rf_config = json.load(open('../../config/Mitomics/best_paras_Mitomics_RandomForest_fold_'+str(i)+'_ratio_'+str(r)+'_5fold.json'))
            if r != 1.0:
                sample_size = max(1,int(len(train_source)*r))
                random.seed(123)
                sampled_train_node = random.sample(train_source,sample_size)
                sampled_train_data = train_raw_data.loc[sampled_train_node,]
            else:
                sampled_train_data = train_raw_data.copy()
            train_data = mf.move2last(sampled_train_data,'label')
            test_data = mf.move2last(test_data,'label')
            X_train, y_train = train_data.iloc[:,:-1],train_data.iloc[:,-1]
            X_test, y_test = test_data.iloc[:,:-1],test_data.iloc[:,-1]
            if rf_config != None:
                seed_everything()
                model = RandomForestClassifier(**rf_config,class_weight='balanced')
            else:
                seed_everything()
                model = RandomForestClassifier(n_estimators=1000,class_weight='balanced')
            print("Fold " + str(i + 1)+' training ratio '+ str(r))
            model.fit(X_train, y_train)
            test_score = model.predict_proba(X_test)[:, 1]
            pool_prediction.extend(test_score)
            pool_truelabel.extend(y_test)
            pool_index.extend(test_data.index.tolist())
            pool_fold.extend([i] * len(y_test))
        fpr,tpr,_ = roc_curve(pool_truelabel,pool_prediction)
        pool_auroc = auc(fpr,tpr)
        pool_aucs.append(pool_auroc)
        for k in PREC_KS:
            pool_prec[k].append(precision_at_k(pool_truelabel, pool_prediction, k))
        # pooled predictions for this ratio; the final ratio (1.0) overwrites and is returned
        pool_pred_df = pd.DataFrame({'prediction': pool_prediction, 'label': pool_truelabel, 'fold': pool_fold}, index=pool_index)
        if ratio == 10:
            pool_fpr.extend(fpr)
            pool_tpr.extend(tpr)
    return pool_aucs,pool_fpr,pool_tpr,pool_pred_df,pool_prec

def train_xgb():
    all_sources = pd.read_csv('../../Data/Mitomics/train_node_attribute_Mitomics_with_target_feature.csv')['protein'].unique().tolist()
    pool_aucs = list()
    pool_fpr = list()
    pool_tpr = list()
    pool_prec = {k: [] for k in PREC_KS}
    for ratio in range(1,11):
        pool_prediction = list()
        pool_truelabel = list()
        pool_index = list()  # instance id for each pooled prediction
        pool_fold = list()   # fold that each instance was tested in
        for i in range(5):
            train_raw_data = pd.read_csv('../../Data/Mitomics/fold_'+str(i)+'_train_by_source_with_target_feature.csv',index_col = 0)
            test_source = np.load('../../Data/Mitomics/fold_'+str(i)+'_test_protein.npy')
            train_source = sorted(set(all_sources)-set(test_source))  # fixed order so random.sample is reproducible
            train_raw_data = train_raw_data.drop(['Target'],axis = 1)
            test_data = pd.read_csv('../../Data/Mitomics/fold_'+str(i)+'_test_by_source_with_target_feature.csv',index_col = 0)
            test_data = test_data.drop(['Target'],axis = 1)
            r = ratio/10
            xgb_config = json.load(open('../../config/Mitomics/best_paras_Mitomics_XGBoost_fold_'+str(i)+'_ratio_'+str(r)+'_5fold.json'))
            if r != 1.0:
                sample_size = max(1,int(len(train_source)*r))
                random.seed(123)
                sampled_train_node = random.sample(train_source,sample_size)
                sampled_train_data = train_raw_data.loc[sampled_train_node,]
            else:
                sampled_train_data = train_raw_data.copy()
            train_data = mf.move2last(sampled_train_data,'label')
            test_data = mf.move2last(test_data,'label')
            X_train, y_train = train_data.iloc[:,:-1],train_data.iloc[:,-1]
            X_test, y_test = test_data.iloc[:,:-1],test_data.iloc[:,-1]
            class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
            class_weight={0: class_weights[0], 1: class_weights[1]}
            if xgb_config != None:
                seed_everything()
                model = xgb.XGBClassifier(**xgb_config)
            else:
                seed_everything()
                model = xgb.XGBClassifier()
            model.fit(X_train, y_train,sample_weight = np.vectorize(class_weight.get)(y_train))
            print("Fold " + str(i + 1)+' training ratio '+ str(r))
            test_score = model.predict_proba(X_test)[:, 1]
            pool_prediction.extend(test_score)
            pool_truelabel.extend(y_test)
            pool_index.extend(test_data.index.tolist())
            pool_fold.extend([i] * len(y_test))
        fpr,tpr,_ = roc_curve(pool_truelabel,pool_prediction)
        pool_auroc = auc(fpr,tpr)
        pool_aucs.append(pool_auroc)
        for k in PREC_KS:
            pool_prec[k].append(precision_at_k(pool_truelabel, pool_prediction, k))
        # pooled predictions for this ratio; the final ratio (1.0) overwrites and is returned
        pool_pred_df = pd.DataFrame({'prediction': pool_prediction, 'label': pool_truelabel, 'fold': pool_fold}, index=pool_index)
        if ratio == 10:
            pool_fpr.extend(fpr)
            pool_tpr.extend(tpr)
    return pool_aucs,pool_fpr,pool_tpr,pool_pred_df,pool_prec

def train_MLP():
    all_sources = pd.read_csv('../../Data/Mitomics/train_node_attribute_Mitomics_with_target_feature.csv')['protein'].unique().tolist()
    pool_aucs = list()
    pool_fpr = list()
    pool_tpr = list()
    pool_prec = {k: [] for k in PREC_KS}
    for ratio in range(1,11):
        pool_prediction = list()
        pool_truelabel = list()
        pool_index = list()  # instance id for each pooled prediction
        pool_fold = list()   # fold that each instance was tested in
        for i in range(5):
            train_raw_data = pd.read_csv('../../Data/Mitomics/fold_'+str(i)+'_train_by_source_with_target_feature.csv',index_col = 0)
            test_source = np.load('../../Data/Mitomics/fold_'+str(i)+'_test_protein.npy')
            train_source = sorted(set(all_sources)-set(test_source))  # fixed order so random.sample is reproducible
            train_raw_data = train_raw_data.drop(['Target'],axis = 1)
            test_data = pd.read_csv('../../Data/Mitomics/fold_'+str(i)+'_test_by_source_with_target_feature.csv',index_col = 0)
            test_data = test_data.drop(['Target'],axis = 1)
            r = ratio/10
            mlp_config = json.load(open('../../config/Mitomics/best_paras_Mitomics_MLP_fold_'+str(i)+'_ratio_'+str(r)+'_5fold.json'))
            if r != 1.0:
                sample_size = max(1,int(len(train_source)*r))
                random.seed(123)
                sampled_train_node = random.sample(train_source,sample_size)
                sampled_train_data = train_raw_data.loc[sampled_train_node,]
            else:
                sampled_train_data = train_raw_data.copy()
            train_data = mf.move2last(sampled_train_data,'label')
            test_data = mf.move2last(test_data,'label')
            X_train,y_train = train_data.iloc[:,:-1],train_data.iloc[:,-1]
            X_train = torch.tensor(np.array(X_train),dtype = torch.float32)
            y_train = torch.tensor(np.array(y_train),dtype = torch.float32).unsqueeze(1)
            X_test = torch.tensor(np.array(test_data.iloc[:,:-1]),dtype = torch.float32)
            y_test = np.array(test_data.iloc[:,-1])
            train_dataset = CustomDataset(X_train,y_train)
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
                model.train()
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
        fpr,tpr,_ = roc_curve(pool_truelabel,pool_prediction)
        pool_auroc = auc(fpr,tpr)
        pool_aucs.append(pool_auroc)
        for k in PREC_KS:
            pool_prec[k].append(precision_at_k(pool_truelabel, pool_prediction, k))
        # pooled predictions for this ratio; the final ratio (1.0) overwrites and is returned
        pool_pred_df = pd.DataFrame({'prediction': pool_prediction, 'label': pool_truelabel, 'fold': pool_fold}, index=pool_index)
        if ratio == 10:
            pool_fpr.extend(fpr)
            pool_tpr.extend(tpr)
    return pool_aucs,pool_fpr,pool_tpr,pool_pred_df,pool_prec


def plot_positive_diffusion(score_path):
    score = pd.read_csv(score_path,index_col = 0)
    print(score)
    score = score.drop(['Target'],axis = 1)
    cols = score.columns
    print(cols)
    roc_score = list()
    prec_score = {k: [] for k in PREC_KS}
    cols = cols.drop('label')
    for j in cols:
        scores = score[j]
        labels = score['label']
        fpr, tpr, thresholds = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        roc_score.append(roc_auc)
        for k in PREC_KS:
            prec_score[k].append(precision_at_k(labels, scores, k))
    return roc_score, prec_score


def plot_target_diffusion(score_path,kernel,state):
    target_score = pd.read_csv(score_path,index_col = 0)
    score = target_score[kernel+'_target_'+str(state)].tolist()
    label = target_score['label'].tolist()
    fpr,tpr,_ = roc_curve(label,score)
    auroc = auc(fpr,tpr)
    prec = {k: precision_at_k(label, score, k) for k in PREC_KS}
    return auroc, prec


def plot_learning_curve():
    ## This function is used for generating AUROC/AUPRC/Loss for 
    fig1, ax1 = plt.subplots(figsize = (8,6))
    fig2, ax2 = plt.subplots(figsize = (8,6))
    fig3, ax3 = plt.subplots(figsize = (8,6))  # Precision@20 learning curve
    fig4, ax4 = plt.subplots(figsize = (8,6))  # Precision@100 learning curve

    ratio_list = [i/10 for i in range(1,11)]
    lr_pool_auroc,lr_pool_fpr,lr_pool_tpr,lr_pred_df,lr_prec = train_lg()
    lr_pred_df.to_csv(PLOT_DIREC+'Mitomics/pooled_predictions_LogisticRegression_Mitomics_with_target.csv')
    ax1.plot(ratio_list,lr_pool_auroc,linestyle = '-',label = 'Logistic regression',lw = 3, alpha = 1)
    ax3.plot(ratio_list,lr_prec[20],linestyle = '-',label = 'Logistic regression',lw = 3, alpha = 1)
    ax4.plot(ratio_list,lr_prec[100],linestyle = '-',label = 'Logistic regression',lw = 3, alpha = 1)
    auroc = auc(lr_pool_fpr,lr_pool_tpr)
    ax2.plot(lr_pool_fpr,lr_pool_tpr,linestyle = '-',label = r"Logistic regression (AUROC = %0.2f )" % (auroc),lw = 2, alpha = 1)

    rf_pool_auroc,rf_pool_fpr,rf_pool_tpr,rf_pred_df,rf_prec = train_rf()
    rf_pred_df.to_csv(PLOT_DIREC+'Mitomics/pooled_predictions_RandomForest_Mitomics_with_target.csv')
    ax1.plot(ratio_list,rf_pool_auroc,linestyle = '-',label = 'Random forest',lw = 3, alpha = 1)
    ax3.plot(ratio_list,rf_prec[20],linestyle = '-',label = 'Random forest',lw = 3, alpha = 1)
    ax4.plot(ratio_list,rf_prec[100],linestyle = '-',label = 'Random forest',lw = 3, alpha = 1)
    auroc = auc(rf_pool_fpr,rf_pool_tpr)
    ax2.plot(rf_pool_fpr,rf_pool_tpr,linestyle = '-',label = r"Random forest (AUROC = %0.2f )" % (auroc),lw = 2, alpha = 1)
    
    xgb_pool_auroc,xgb_pool_fpr,xgb_pool_tpr,xgb_pred_df,xgb_prec = train_xgb()
    xgb_pred_df.to_csv(PLOT_DIREC+'Mitomics/pooled_predictions_XGBoost_Mitomics_with_target.csv')
    ax1.plot(ratio_list,xgb_pool_auroc,linestyle = '-',label = 'XGBoost',lw = 3, alpha = 1)
    ax3.plot(ratio_list,xgb_prec[20],linestyle = '-',label = 'XGBoost',lw = 3, alpha = 1)
    ax4.plot(ratio_list,xgb_prec[100],linestyle = '-',label = 'XGBoost',lw = 3, alpha = 1)
    auroc = auc(xgb_pool_fpr,xgb_pool_tpr)
    ax2.plot(xgb_pool_fpr,xgb_pool_tpr,linestyle = '-',label = r"XGBoost (AUROC = %0.2f )" % (auroc),lw = 2, alpha = 1)

    mlp_pool_auroc,mlp_pool_fpr,mlp_pool_tpr,mlp_pred_df,mlp_prec = train_MLP()
    mlp_pred_df.to_csv(PLOT_DIREC+'Mitomics/pooled_predictions_MLP_Mitomics_with_target.csv')
    ax1.plot(ratio_list,mlp_pool_auroc,linestyle = '-',label = 'Neural network',lw = 3, alpha = 1)
    ax3.plot(ratio_list,mlp_prec[20],linestyle = '-',label = 'Neural network',lw = 3, alpha = 1)
    ax4.plot(ratio_list,mlp_prec[100],linestyle = '-',label = 'Neural network',lw = 3, alpha = 1)
    auroc = auc(mlp_pool_fpr,mlp_pool_tpr)
    ax2.plot(mlp_pool_fpr,mlp_pool_tpr,linestyle = '-',label = r"Neural network (AUROC = %0.2f )" % (auroc),lw = 2, alpha = 1)


    ## Shortest Path Len
    spl_data = pd.read_csv('../../Data/Mitomics/shortest_path_length_data.csv',index_col=0)
    scores = spl_data['shortest_path_length'].to_list()
    labels = spl_data['label'].to_list()
    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = auc(fpr, tpr)
    # path lengths are stored negated (closer to the target = higher score), so they rank directly
    spl_prec20 = precision_at_k(labels, scores, 20)
    spl_prec100 = precision_at_k(labels, scores, 100)
    

    rw_with_target_score, rw_pos_prec = plot_positive_diffusion('../../Data/Mitomics/RW_positive_diffusion_Mitomics_state_30.csv')
    ax1.plot(ratio_list,rw_with_target_score, linestyle='-',marker = 'h',label = 'RW positive diffusion',lw=1,alpha = 0.7)
    ax3.plot(ratio_list,rw_pos_prec[20], linestyle='-',marker = 'h',label = 'RW positive diffusion',lw=1,alpha = 0.7)
    ax4.plot(ratio_list,rw_pos_prec[100], linestyle='-',marker = 'h',label = 'RW positive diffusion',lw=1,alpha = 0.7)

    rwr_with_target_score, rwr_pos_prec = plot_positive_diffusion('../../Data/Mitomics/RWR_positive_diffusion_Mitomics_state_0.2.csv')
    ax1.plot(ratio_list,rwr_with_target_score, linestyle='-',marker = '*',label = 'RWR positive diffusion',lw=1,alpha = 0.7)
    ax3.plot(ratio_list,rwr_pos_prec[20], linestyle='-',marker = '*',label = 'RWR positive diffusion',lw=1,alpha = 0.7)
    ax4.plot(ratio_list,rwr_pos_prec[100], linestyle='-',marker = '*',label = 'RWR positive diffusion',lw=1,alpha = 0.7)

    heat_with_target_score, heat_pos_prec = plot_positive_diffusion('../../Data/Mitomics/heat_positive_diffusion_Mitomics_state_0.3.csv')
    ax1.plot(ratio_list,heat_with_target_score, linestyle='-',marker = '^',label = 'Heat positive diffusion',lw=1,alpha = 0.7)
    ax3.plot(ratio_list,heat_pos_prec[20], linestyle='-',marker = '^',label = 'Heat positive diffusion',lw=1,alpha = 0.7)
    ax4.plot(ratio_list,heat_pos_prec[100], linestyle='-',marker = '^',label = 'Heat positive diffusion',lw=1,alpha = 0.7)



    #ax2.plot(fpr,tpr,label="Shortest_path_len",lw=1,alpha=0.5)
    ratio_list = [i/10 for i in range(1,11)]
    rw_target_auroc, rw_target_prec = plot_target_diffusion('../../Data/Mitomics/RW_target_diffusion_Mitomics.csv','RW',30)
    ax1.plot(ratio_list,[rw_target_auroc]*len(ratio_list), linestyle='--',marker = 'h',label = 'RW target diffusion',lw=1,alpha = 0.7)
    ax3.plot(ratio_list,[rw_target_prec[20]]*len(ratio_list), linestyle='--',marker = 'h',label = 'RW target diffusion',lw=1,alpha = 0.7)
    ax4.plot(ratio_list,[rw_target_prec[100]]*len(ratio_list), linestyle='--',marker = 'h',label = 'RW target diffusion',lw=1,alpha = 0.7)

    rwr_target_auroc, rwr_target_prec = plot_target_diffusion('../../Data/Mitomics/RWR_target_diffusion_Mitomics.csv','RWR',0.2)
    ax1.plot(ratio_list,[rwr_target_auroc]*len(ratio_list), linestyle='--',marker = '*',label = 'RWR target diffusion',lw=1,alpha = 0.7)
    ax3.plot(ratio_list,[rwr_target_prec[20]]*len(ratio_list), linestyle='--',marker = '*',label = 'RWR target diffusion',lw=1,alpha = 0.7)
    ax4.plot(ratio_list,[rwr_target_prec[100]]*len(ratio_list), linestyle='--',marker = '*',label = 'RWR target diffusion',lw=1,alpha = 0.7)


    heat_target_auroc, heat_target_prec = plot_target_diffusion('../../Data/Mitomics/heat_target_diffusion_Mitomics.csv','heat',0.3)
    ax1.plot(ratio_list,[heat_target_auroc]*len(ratio_list), linestyle='--',marker = '^',label = 'Heat target diffusion',lw=1,alpha = 0.7)
    ax3.plot(ratio_list,[heat_target_prec[20]]*len(ratio_list), linestyle='--',marker = '^',label = 'Heat target diffusion',lw=1,alpha = 0.7)
    ax4.plot(ratio_list,[heat_target_prec[100]]*len(ratio_list), linestyle='--',marker = '^',label = 'Heat target diffusion',lw=1,alpha = 0.7)


    
    #ax1.legend(bbox_to_anchor=(1, 1), borderaxespad=0.)
    ax1.set_xlabel("Fraction of the available training set used",fontsize = 20)
    ax1.set_ylabel('AUROC',fontsize = 20)
    ax1.set_ylim(0.5,1)
    ax1.tick_params(axis='y', labelsize=16)
    ax1.tick_params(axis='x', labelsize=16)
    #ax1.set_position([0.1, 0.1, 0.65, 0.8])
    ax1.plot(ratio_list,[auroc]*len(ratio_list), linestyle=(0,(1,4)),label = 'Shortest path length',lw=1,alpha = 0.7)
    ax3.plot(ratio_list,[spl_prec20]*len(ratio_list), linestyle=(0,(1,4)),label = 'Shortest path length',lw=1,alpha = 0.7)
    ax4.plot(ratio_list,[spl_prec100]*len(ratio_list), linestyle=(0,(1,4)),label = 'Shortest path length',lw=1,alpha = 0.7)
    # ax1.legend(loc = 'best',mode = "expand", ncol = 2,fontsize = 14)
    ax1.set_title('Mitochondrial protein abundance',fontsize = 24)
    fig1.savefig(PLOT_DIREC+'Mitomics/Baseline_Comparison_Learning_Curve_Mitomics_with_target.png',dpi = 800,bbox_inches='tight')

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
    HATCH = '//'            # sparse -> light, unobtrusive
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
            zorder=BAND_ZORDER, label='Positive rate (chance)',
        )

    # Precision y-axis: 0.4 by default, raised in 0.05 steps if a curve would be
    # clipped (0.01 headroom keeps markers at the top fully visible)
    def prec_ylim_top(ax):
        ymax = max(np.max(line.get_ydata()) for line in ax.get_lines())
        top = max(0.4, round(np.ceil((ymax + 0.01) / 0.05) * 0.05, 2))
        print('%s: highest value %.3f, y-axis top %.2f' % (ax.get_ylabel(), ymax, top))
        return top

    # fig3: Precision@20 learning curve (same layout as fig1)
    ax3.set_xlabel("Fraction of the available training set used",fontsize = 20)
    ax3.set_ylabel('Precision@20',fontsize = 20)
    ax3.set_ylim(0,prec_ylim_top(ax3))
    ax3.set_xlim(0.08,0.52)
    ax3.tick_params(axis='y', labelsize=16)
    ax3.tick_params(axis='x', labelsize=16)
    ax3.set_title('Mitochondrial protein abundance',fontsize = 24)
    fig3.savefig(PLOT_DIREC+'Mitomics/Baseline_Comparison_Precision_at_20_Mitomics_with_target.png',dpi = 800,bbox_inches='tight')

    # fig4: Precision@100 learning curve (same layout as fig1)
    ax4.set_xlabel("Fraction of the available training set used",fontsize = 20)
    ax4.set_ylabel('Precision@100',fontsize = 20)
    ax4.set_ylim(0,prec_ylim_top(ax4))
    ax4.set_xlim(0.08,0.52)
    ax4.tick_params(axis='y', labelsize=16)
    ax4.tick_params(axis='x', labelsize=16)
    ax4.set_title('Mitochondrial protein abundance',fontsize = 24)
    fig4.savefig(PLOT_DIREC+'Mitomics/Baseline_Comparison_Precision_at_100_Mitomics_with_target.png',dpi = 800,bbox_inches='tight')
    # plt.show()
    ax2.legend(bbox_to_anchor=(1, 1), borderaxespad=0.)
    
    ax2.set_xlabel("False positive rate",fontsize = 20)
    ax2.set_ylabel("True positive rate",fontsize = 20)
    ax2.tick_params(axis='y', labelsize=16)
    ax2.tick_params(axis='x', labelsize=16)
    #ax2.set_position([0.1, 0.1, 0.65, 0.8])
    ax2.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.7)
    # ax2.legend(loc = 'lower right')
    ax2.legend(loc = 'best',fontsize = 14)
    ax2.set_title('Mitochondrial protein abundance',fontsize = 24)
    fig2.savefig(PLOT_DIREC+'/Mitomics/Baseline_Comparison_ROC_Mitomics_with_target.png',dpi = 800,bbox_inches='tight')

if __name__=='__main__':
    plot_learning_curve()

    
    