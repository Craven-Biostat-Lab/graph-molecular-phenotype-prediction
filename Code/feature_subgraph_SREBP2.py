from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from sklearn.metrics import roc_curve, auc,average_precision_score,f1_score,roc_auc_score
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
import numpy as np
import statistics
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import MinMaxScaler
import argparse
import torch.optim as optim
# import merge_feature as mf 
# from stringdb_alias import HGNCMapper
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import json
import random
import os
import math
import warnings
warnings.filterwarnings('ignore')


def seed_everything(seed=123):
    """"
    Seed everything.
    """   
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
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


def train_lr(data,n_folds,target,lr_config_path = None):
    target_data = data.loc[target,:]
    target_data['label'] = 1
    data = data.drop(target)
    skf = StratifiedKFold(n_splits = n_folds, shuffle=True,random_state=123)
    all_pred,all_true,fold_auroc,fold_auprc = list(),list(),list(),list()
    for i, (train_index, test_index) in enumerate(skf.split(data.iloc[:,:-1], data.iloc[:,-1])):
        train_data = data.iloc[train_index,:]
        test_data = data.iloc[test_index,:]
        target_fold = target_data.copy()
        train_data = pd.concat([train_data,target_fold])
        train_data = move2last(train_data,'label')
        X_train = np.array(train_data.iloc[:,:-1])
        y_train = np.array(train_data.iloc[:,-1])
        X_test = np.array(test_data.iloc[:,:-1])
        y_test = np.array(test_data.iloc[:,-1])
        ros = RandomOverSampler(sampling_strategy=0.1, random_state=123)
        X_train, y_train = ros.fit_resample(X_train, y_train)
        if lr_config_path != None:
            lr_fold_config = lr_config_path+str(i)+'_ratio_1.0_5fold.json'
            lr_fold_config = json.load(open(lr_fold_config))
            if lr_fold_config['penalty'] == 'elasticnet':
                seed_everything()
                model = LogisticRegression(**lr_fold_config,solver = 'saga',class_weight='balanced')
            else:
                seed_everything()
                model = LogisticRegression(**lr_fold_config,class_weight='balanced')
        else:
            seed_everything()
            model = LogisticRegression(max_iter = 5000, class_weight='balanced')
        model.fit(X_train, y_train)
        test_score = model.predict_proba(X_test)[:, 1]
        all_true.extend(y_test)
        all_pred.extend(test_score)
        fold_auroc.append(roc_auc_score(y_test,test_score))
        fold_auprc.append(average_precision_score(y_test,test_score))
    return all_true,all_pred,fold_auroc,fold_auprc
        
def train_rf(data,n_folds,target,rf_config_path = None):
    target_data = data.loc[target,:]
    target_data['label'] = 1
    data = data.drop(target)
    skf = StratifiedKFold(n_splits = n_folds, shuffle=True,random_state=123)
    all_pred,all_true, fold_auroc, fold_auprc= list(),list(),list(),list()
    for i, (train_index, test_index) in enumerate(skf.split(data.iloc[:,:-1], data.iloc[:,-1])):
        train_data = data.iloc[train_index,:]
        test_data = data.iloc[test_index,:]
        target_fold = target_data.copy()
        train_data = pd.concat([train_data,target_fold])
        train_data = move2last(train_data,'label')
        X_train = np.array(train_data.iloc[:,:-1])
        y_train = np.array(train_data.iloc[:,-1])
        X_test = np.array(test_data.iloc[:,:-1])
        y_test = np.array(test_data.iloc[:,-1])
        ros = RandomOverSampler(sampling_strategy=0.1, random_state=123)
        X_train, y_train = ros.fit_resample(X_train, y_train)
        if rf_config_path != None:
            rf_fold_config = rf_config_path+str(i)+'_ratio_1.0_5fold.json'
            rf_fold_config = json.load(open(rf_fold_config))
            seed_everything()
            model = RandomForestClassifier(**rf_fold_config,class_weight='balanced')
        else:
            seed_everything()
            model = RandomForestClassifier(n_estimators=1000,class_weight='balanced')
        model.fit(X_train, y_train)
        test_score = model.predict_proba(X_test)[:, 1]
        all_true.extend(y_test)
        all_pred.extend(test_score)
        fold_auroc.append(roc_auc_score(y_test,test_score))
        fold_auprc.append(average_precision_score(y_test,test_score))
    return all_true,all_pred,fold_auroc,fold_auprc

def train_xgb(data,n_folds,target,xgb_config_path = None):
    target_data = data.loc[target,:]
    target_data['label'] = 1
    data = data.drop(target)
    skf = StratifiedKFold(n_splits = n_folds, shuffle = True,random_state = 123)
    all_pred,all_true, fold_auroc, fold_auprc= list(),list(),list(),list()
    for i, (train_index, test_index) in enumerate(skf.split(data.iloc[:,:-1], data.iloc[:,-1])):
        train_data = data.iloc[train_index,:]
        test_data = data.iloc[test_index,:]
        target_fold = target_data.copy()
        train_data = pd.concat([train_data,target_fold])
        train_data = move2last(train_data,'label')
        X_train = np.array(train_data.iloc[:,:-1])
        y_train = np.array(train_data.iloc[:,-1])
        ros = RandomOverSampler(sampling_strategy=0.1, random_state=123)
        X_train, y_train = ros.fit_resample(X_train, y_train)
        X_test = np.array(test_data.iloc[:,:-1])
        y_test = np.array(test_data.iloc[:,-1])
        class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
        class_weight={0: class_weights[0], 1: class_weights[1]}
        if xgb_config_path != None:
            xgb_fold_config = xgb_config_path+str(i)+'_ratio_1.0_5fold.json'
            xgb_fold_config = json.load(open(xgb_fold_config))
            seed_everything()
            model = xgb.XGBClassifier(**xgb_fold_config)
        else:
            seed_everything()
            model = xgb.XGBClassifier()
        model.fit(X_train, y_train,sample_weight = np.vectorize(class_weight.get)(y_train))
        test_score = model.predict_proba(X_test)[:, 1]
        all_true.extend(y_test)
        all_pred.extend(test_score)
        fold_auroc.append(roc_auc_score(y_test,test_score))
        fold_auprc.append(average_precision_score(y_test,test_score))
    return all_true,all_pred,fold_auroc,fold_auprc

def train_MLP(data,n_folds,target,mlp_config_path = None):
    target_data = data.loc[target,:]
    target_data['label'] = 1
    data = data.drop(target)
    skf = StratifiedKFold(n_splits = n_folds, shuffle = True,random_state = 123)
    all_pred,all_true, fold_auroc, fold_auprc= list(),list(),list(),list()
    for i, (train_index, test_index) in enumerate(skf.split(data.iloc[:,:-1], data.iloc[:,-1])):
        target_fold = target_data.copy()
        train_data = data.iloc[train_index,:]
        test_data = data.iloc[test_index,:]
        train_data = pd.concat([train_data,target_fold])
        train_data = move2last(train_data,'label')
        ros = RandomOverSampler(sampling_strategy=0.1, random_state=123)
        X_train,y_train = train_data.iloc[:,:-1],train_data.iloc[:,-1]
        X_train, y_train = ros.fit_resample(X_train, y_train)
        X_train = torch.tensor(np.array(X_train),dtype = torch.float32)
        y_train = torch.tensor(np.array(y_train),dtype = torch.float32).unsqueeze(1)
        X_test = torch.tensor(np.array(test_data.iloc[:,:-1]),dtype = torch.float32)
        y_test = np.array(test_data.iloc[:,-1])
        if mlp_config_path != None:
            mlp_fold_config = mlp_config_path+str(i)+'_ratio_1.0_5fold.json'
            mlp_fold_config = json.load(open(mlp_fold_config))
            batch_size = mlp_fold_config['batch_size']
            num_epoch = mlp_fold_config['num_epoch']
            learning_rate = mlp_fold_config['learning_rate']
            hidden_size = mlp_fold_config['hidden_size']
            num_layer = mlp_fold_config['num_layer']
            activation_type = mlp_fold_config['activation']
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
            n_pos = y_train.sum().item()
            n_neg = y_train.numel() - n_pos
            pos_w = torch.tensor([n_neg / n_pos])
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        else:
            seed_everything()
            model = MLP_1_relu(X_train.shape[1],hidden_size=128,dropout_rate=0.5)
            n_pos = y_train.sum().item()
            n_neg = y_train.numel() - n_pos
            pos_w = torch.tensor([n_neg / n_pos])
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
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
            all_pred.extend(predictions)
            all_true.extend(y_test)
            fold_auroc.append(roc_auc_score(y_test,predictions))
            fold_auprc.append(average_precision_score(y_test,predictions))
    return all_true,all_pred,fold_auroc,fold_auprc





def plot_ROC(per_true,per_pred,all_true,all_pred,model_name,data_name,save_path):
    fig,ax = plt.subplots()
    #mean_fpr = np.linspace(0,1,100)
    aucs = np.array([])
    n_folds = len(per_true)
    for i in range(n_folds):
        y_test = per_true[i]
        y_score = per_pred[i]
        fpr,tpr,thresholds = roc_curve(y_test,y_score)
        auc_val = auc(fpr,tpr)
        aucs = np.append(aucs,auc_val)
        # if n_folds < 8:
        #     ax.plot(fpr, tpr, lw=1, label="ROC fold {}".format(i) + ", AUC={:0.2f}".format(auc_val), alpha=0.5)
        # else:
        ax.plot(fpr, tpr, lw=1, alpha=0.5)
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    mean_fpr, mean_tpr, _ = roc_curve(all_true,all_pred) # pooled data
    mean_auc = auc(mean_fpr, mean_tpr)
    ax.plot(mean_fpr,mean_tpr,color="b",label=r"Pooled ROC (AUC = %0.2f )" % (mean_auc),lw=2,alpha=0.8)
    ax.legend(title=model_name, loc="lower right")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(model_name + "5_fold_ROC_"+data_name+".png")

    #plt.show()
    fig.savefig(save_path+model_name+"_ROC_"+data_name+".png")




def plot_fimportance(importances,std,feature_names,data_name,save_path):
    fig2,ax2 = plt.subplots()
    fig2.set_size_inches(30,8)
    forest_importances =pd.Series(importances,index = feature_names)
    print(forest_importances)
    #ax.boxplot(forest_importances.values())
    forest_importances.plot.bar(yerr=std, ax=ax2)
    ax2.grid(axis='y')
    ax2.set_xticklabels(forest_importances.keys(),rotation = 90)
    ax2.set_title("Feature importances using MDI")
    ax2.set_ylabel("Mean decrease in impurity")
    fig2.tight_layout()
    fig2.savefig(save_path+'feature_importance_'+data_name+'.pdf')
    plt.close()
    fig3,ax3 = plt.subplots()
    fig3.set_size_inches(30, 8)
    f_list = forest_importances.to_list()
    f_dict = forest_importances.to_dict()
    new_std = list()
    slist = sorted(feature_names, key=lambda x: x[-1])
    sorted_feature_dict = dict()
    for key in slist:
        index = feature_names.index(key)
        sorted_feature_dict[key] = f_dict[key]
        new_std.append(std[index])
    sforest_importances = pd.Series(sorted_feature_dict)
    sforest_importances.plot.bar(yerr=new_std, ax=ax3)
    ax3.grid(axis='y')
    ax3.set_xticklabels(sforest_importances.keys(), rotation=90)
    ax3.set_title("Feature importances using MDI")
    ax3.set_ylabel("Mean decrease in impurity")
    fig3.tight_layout()
    fig3.savefig(save_path+'feature_importance_'+data_name+'_sorted.pdf')


def drop_out_0_features(feature_df):
    column_name = list(feature_df.columns)
    col_with_oneval = list()
    for names in column_name:
        #print(len(feature_df[names].value_counts()))
        if len(feature_df[names].value_counts()) == 1:
            col_with_oneval.append(names)
    #print(col_with_oneval)
    #print(len(col_with_oneval))
    feature_df = feature_df.drop(columns=col_with_oneval)
    return feature_df
def thermo_encoding(dataframe,col_tobe_encode):
    df = dataframe.copy()
    max_num = dataframe[col_tobe_encode].max()
    for i in range(max_num):
        new_name = 'thermo'+str(i)
        df[new_name]=" "
    #print(df)
    for i in range(len(df)):
        num_one = dataframe.loc[i,col_tobe_encode]
        num_zero = max_num-num_one
        for j in range(max_num):
            col_name = 'thermo'+ str(j)
            if j < num_zero:
                df.loc[i,col_name] = 0
            else:
                df.loc[i,col_name] = 1
    df = df.drop(columns=col_tobe_encode)
    return df

def normalize_features(feature_df,cols_to_normalize):
    cols_to_normalize.append('num_paths')
    cols_to_normalize.append('max_path_score')
    cols_to_normalize.append('max_degree_score')
    scaler = MinMaxScaler()
    feature_df[cols_to_normalize] = scaler.fit_transform(feature_df[cols_to_normalize])
    return feature_df




def calc_Comparison_Metrics(feature_path_list,feature_list,n_fold,save_path,data_name,target,phenotype,
                            lr_config = None,rf_config=None,xgb_config = None,mlp_config = None):
    model_list = ['Logistic Regression','Random Forest','XGBoost','MLP']
    # model_list = ['MLP_2']
    for j in range(len(model_list)):
        pool_auc_df = pd.DataFrame(columns=feature_list,index=[phenotype])
        mean_auc_df = pd.DataFrame(columns=feature_list,index=[phenotype])
        sem_auc_df = pd.DataFrame(columns=feature_list,index=[phenotype])
        pool_prc_df = pd.DataFrame(columns=feature_list,index=[phenotype])
        mean_prc_df = pd.DataFrame(columns=feature_list,index=[phenotype])
        sem_prc_df = pd.DataFrame(columns=feature_list,index=[phenotype])
        model_name = model_list[j]
        print(model_name)
        for i in range(len(feature_path_list)):
            feature_path = feature_path_list[i]
            feature = feature_list[i]
            feature_data = pd.read_csv(feature_path,index_col='protein')
            if model_name == 'Logistic Regression':
                all_true,all_pred,fold_auroc,fold_auprc= train_lr(data = feature_data,n_folds = n_fold,target = target,lr_config_path = lr_config)
            elif model_name == 'Random Forest':
                all_true,all_pred,fold_auroc,fold_auprc = train_rf(data = feature_data,n_folds = n_fold,target = target,rf_config_path = rf_config)
            elif model_name == 'XGBoost':
                all_true,all_pred,fold_auroc,fold_auprc = train_xgb(data = feature_data,n_folds = n_fold,target = target,xgb_config_path = xgb_config)
            elif model_name == 'MLP':
                all_true,all_pred,fold_auroc,fold_auprc = train_MLP(data = feature_data,n_folds = n_fold,target = target,mlp_config_path = mlp_config)
            pool_fpr, pool_tpr, _ = roc_curve(all_true,all_pred) # pooled data
            pool_auc = auc(pool_fpr, pool_tpr)
            pool_auc_df.loc[phenotype,feature] = pool_auc
            mean_auc_df.loc[phenotype,feature] = statistics.mean(fold_auroc)
            std_dev_auroc = statistics.stdev(fold_auroc)
            sem_auroc = std_dev_auroc / math.sqrt(len(fold_auroc))
            sem_auc_df.loc[phenotype,feature] = sem_auroc
            pool_auprc= average_precision_score(all_true,all_pred)
            pool_prc_df.loc[phenotype,feature] = pool_auprc
            mean_prc_df.loc[phenotype,feature] = statistics.mean(fold_auprc)
            std_dev_auprc = statistics.stdev(fold_auprc)
            sem_auprc = std_dev_auprc / math.sqrt(len(fold_auprc))
            sem_prc_df.loc[phenotype,feature] = sem_auprc
        pool_auc_df.to_csv(save_path+model_name+'_'+data_name+'pool_auc_df.csv')
        mean_auc_df.to_csv(save_path+model_name+'_'+data_name+'mean_auc_df.csv')
        sem_auc_df.to_csv(save_path+model_name+'_'+data_name+'sem_auc_df.csv')
        pool_prc_df.to_csv(save_path+model_name+'_'+data_name+'pool_prc_df.csv')
        mean_prc_df.to_csv(save_path+model_name+'_'+data_name+'mean_prc_df.csv')
        sem_prc_df.to_csv(save_path+model_name+'_'+data_name+'sem_prc_df.csv')
if __name__=='__main__':
    DATA_DIREC = '../Data/'
    PLOT_DIREC = '../Plot/'
    feature_list = ['Source','Target','Source_Target_Relation','All']
    feature_path_list = [DATA_DIREC+'SREBP2/split_subgraph/source_feature_SREBP2.csv',
                        DATA_DIREC+'SREBP2/split_subgraph/target_feature_SREBP2.csv',
                        DATA_DIREC+'SREBP2/split_subgraph/ppi_feature_SREBP2.csv',
                        DATA_DIREC+'SREBP2/train_node_attribute_SREBP2_with_target_feature.csv']
    calc_Comparison_Metrics(feature_path_list,feature_list,n_fold=5,
                        save_path = PLOT_DIREC+'SREBP2/',
                        data_name = 'Subgraph_feature_comparison_SREBP2_',
                        phenotype = 'SREBP2',
                        target=list(np.load(DATA_DIREC+'targets/SREBP2_target.npy')),
                        lr_config = DATA_DIREC+'config/SREBP2/best_paras_SREBP2_LogisticRegression_subset_False_fold_',
                        rf_config = DATA_DIREC+'config/SREBP2/best_paras_SREBP2_RandomForest_subset_False_fold_',
                        xgb_config = DATA_DIREC+'config/SREBP2/best_paras_SREBP2_XGBoost_subset_False_fold_',
                        mlp_config = DATA_DIREC+'config/SREBP2/best_paras_SREBP2_MLP_subset_False_fold_'
                        )


   