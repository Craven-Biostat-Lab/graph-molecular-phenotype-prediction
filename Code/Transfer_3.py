
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc,precision_recall_curve,f1_score,average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report
import torch.optim as optim
import torch
import pickle
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import json
import random
import os
import warnings
warnings.filterwarnings('ignore')

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

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size,dropout_rate=0.5):
        super(MLP, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(p = dropout_rate)
        self.output = nn.Linear(hidden_size, 1) 

    def forward(self, x):
        x = torch.relu(self.hidden(x)) 
        x = torch.sigmoid(self.output(x)) 
        return x
    
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32) 

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]



##
DATA_DIREC = '../Datasets/'
PLOT_DIREC = '../Plot/'

## ST-relation features only
phenotype = 'SREBP2'
SREBP2_data = pd.read_csv(DATA_DIREC+phenotype+'/train_node_attribute_SREBP2_transfer_st_relation.csv',index_col=0)

phenotype = 'Influenza'
Influenza_data = pd.read_csv(DATA_DIREC+phenotype+'/train_node_attribute_Influenza_transfer_st_relation.csv',index_col=0)

phenotype = 'LDLR'
LDLR_data = pd.read_csv(DATA_DIREC+phenotype+'/train_node_attribute_LDLR_transfer_st_relation.csv',index_col=0)

phenotype = 'Mitomics'
Mitomics_data = pd.read_csv(DATA_DIREC+phenotype+'/train_node_attribute_Mitomics_transfer_st_relation.csv',index_col = 0)



train_list = [('Cholesterol homeostasis',SREBP2_data,'SREBP2'),('Cholesterol uptake',LDLR_data,'LDLR'),('Influenza A virus replication',Influenza_data,'Influenza'),('Mitochondrial protein abundance',Mitomics_data,'Mitomics')]
test_list = [('Cholesterol \nhomeostasis',SREBP2_data,'SREBP2'),('Cholesterol \nuptake',LDLR_data,'LDLR'),('Influenza A \nvirus replication',Influenza_data,'Influenza'),('Mitochondrial \nprotein abundance',Mitomics_data,'Mitomics')]
# XGBoost
xgb_auc_df = pd.DataFrame(index =[train_list[i][0]for i in range(len(train_list))],columns= [test_list[i][0]for i in range(len(test_list))])
xgb_prc_df = pd.DataFrame(index =[train_list[i][0]for i in range(len(train_list))],columns= [test_list[i][0]for i in range(len(test_list))])
xgb_f1_df = pd.DataFrame(index =[train_list[i][0]for i in range(len(train_list))],columns= [test_list[i][0]for i in range(len(test_list))])
for train_dataset in train_list:
    train_raw_data = train_dataset[1]
    train_name = train_dataset[2]
    train_fig_name = train_dataset[0]
    for test_dataset in test_list:
        test_fig_name = test_dataset[0]
        test_name = test_dataset[2]
        test_raw_data = test_dataset[1]
        features_2use = [col for col in train_raw_data.columns if col in test_raw_data.columns]
        train_pheno_data = train_raw_data.loc[:,features_2use]
        test_pheno_data = test_raw_data.loc[:,features_2use]
        if test_name == train_name:
            train_pheno_data = test_pheno_data.copy()
            if test_name != 'Mitomics':
                skf = StratifiedKFold(n_splits=5, shuffle=True,random_state=123)
                target = list(np.load(DATA_DIREC+'/targets/'+train_name+'_target.npy'))
                data = train_pheno_data.copy()
                target_data = data.loc[target,]
                target_data['label'] = 1
                data = data.drop(target)
                all_true = list()
                all_pred = list()
                for i, (train_index, test_index) in enumerate(skf.split(data.iloc[:,:-1], data.iloc[:,-1])):
                    xgb_config = json.load(open(DATA_DIREC+'/config/'+train_name+'/best_paras_'+train_name+'_XGBoost_subset_False_fold_'+str(i)+'_ratio_1.0_5fold.json'))
                    target_fold_feature = target_data.copy()
                    fold_feature = train_pheno_data.copy()
                    train_data = fold_feature.iloc[train_index,]
                    train_data = pd.concat([train_data,target_fold_feature])
                    X_train = np.array(train_data.iloc[:,:-1])
                    y_train = np.array(train_data.iloc[:,-1])
                    X_test = np.array(fold_feature.iloc[test_index,:-1])
                    y_test = np.array(fold_feature.iloc[test_index,-1])
                    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
                    class_weight={0: class_weights[0], 1: class_weights[1]}
                    seed_everything()
                    model = xgb.XGBClassifier(**xgb_config)
                    model.fit(X_train, y_train,sample_weight = np.vectorize(class_weight.get)(y_train))
                    score = model.predict_proba(X_test)[:, 1]
                    all_true.extend(y_test)
                    all_pred.extend(score)
                fpr,tpr,_ = roc_curve(all_true,all_pred)
                auc_score = auc(fpr,tpr)
                xgb_auc_df.loc[train_fig_name,test_fig_name] = auc_score
                precision,recall,_ = precision_recall_curve(all_true,all_pred)
                auc_prc = average_precision_score(all_true,all_pred)
                xgb_prc_df.loc[train_fig_name,test_fig_name] = auc_prc
            elif test_name == 'Mitomics':
                feature_columns = Mitomics_data.columns
                all_pred = list()
                all_true = list()
                for i in range(5):
                    xgb_config = json.load(open(DATA_DIREC+'/config/Mitomics/best_paras_mitomics_XGBoost_fold_'+str(i)+'_ratio_1.0_5fold.json'))
                    test_source = np.load(DATA_DIREC+'/Mitomics/Partition_Source/fold_'+str(i)+'_test_protein.npy')
                    train_data = train_raw_data[~train_raw_data.index.isin(test_source)]
                    test_data = train_raw_data[train_raw_data.index.isin(test_source)]
                    X_train, y_train = train_data.iloc[:,:-1],train_data.iloc[:,-1]
                    X_test, y_test = test_data.iloc[:,:-1],test_data.iloc[:,-1]
                    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
                    class_weight={0: class_weights[0], 1: class_weights[1]}
                    seed_everything()
                    model = xgb.XGBClassifier(**xgb_config)
                    model.fit(X_train, y_train,sample_weight = np.vectorize(class_weight.get)(y_train))
                    score = model.predict_proba(X_test)[:, 1]
                    all_true.extend(y_test)
                    all_pred.extend(score)
                fpr,tpr,_ = roc_curve(all_true,all_pred)
                auc_score = auc(fpr,tpr)
                xgb_auc_df.loc[train_fig_name,test_fig_name] = auc_score
                precision,recall,_ = precision_recall_curve(all_true,all_pred)
                auc_prc = average_precision_score(all_true,all_pred)
                xgb_prc_df.loc[train_fig_name,test_fig_name] = auc_prc
        else:
            X_train = train_pheno_data.iloc[:, :-1]
            y_train = train_pheno_data.iloc[:, -1]
            X_test = test_pheno_data.iloc[:, :-1]
            y_test = test_pheno_data.iloc[:, -1]
            class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
            class_weight={0: class_weights[0], 1: class_weights[1]}
            xgb_config = json.load(open(DATA_DIREC+'/config/'+train_name+'/best_paras_'+train_name+'_transfer_st_relation_XGBoost_transfer.json'))
            seed_everything()
            model = xgb.XGBClassifier(**xgb_config)
            model.fit(X_train, y_train,sample_weight = np.vectorize(class_weight.get)(y_train))
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = auc(fpr,tpr)
            xgb_auc_df.loc[train_fig_name,test_fig_name] = auc_score
            precision,recall,_ = precision_recall_curve(y_test,y_prob)
            auc_prc = average_precision_score(y_test,y_prob)
            xgb_prc_df.loc[train_fig_name,test_fig_name] = auc_prc

xgb_auc_df.to_csv(PLOT_DIREC+'/Transfer/Transfer_AUROC_XGBoost_st_relation.csv')
xgb_prc_df.to_csv(PLOT_DIREC+'/Transfer/Transfer_AUPRC_XGBoost_st_relation.csv')

# All features
phenotype = 'SREBP2'
SREBP2_data = pd.read_csv(DATA_DIREC+phenotype+'/train_node_attribute_SREBP2_transfer.csv',index_col=0)

phenotype = 'Influenza'
Influenza_data = pd.read_csv(DATA_DIREC+phenotype+'/train_node_attribute_Influenza_transfer.csv',index_col=0)

phenotype = 'LDLR'
LDLR_data = pd.read_csv(DATA_DIREC+phenotype+'/train_node_attribute_LDLR_transfer.csv',index_col=0)

phenotype = 'Mitomics'
Mitomics_data = pd.read_csv(DATA_DIREC+phenotype+'/train_node_attribute_Mitomics_transfer.csv',index_col = 0)


train_list = [('Cholesterol homeostasis',SREBP2_data,'SREBP2'),('Cholesterol uptake',LDLR_data,'LDLR'),('Influenza A virus replication',Influenza_data,'Influenza'),('Mitochondrial protein abundance',Mitomics_data,'Mitomics')]
test_list = [('Cholesterol \nhomeostasis',SREBP2_data,'SREBP2'),('Cholesterol \nuptake',LDLR_data,'LDLR'),('Influenza A \nvirus replication',Influenza_data,'Influenza'),('Mitochondrial \nprotein abundance',Mitomics_data,'Mitomics')]
# XGBoost
xgb_auc_df = pd.DataFrame(index =[train_list[i][0]for i in range(len(train_list))],columns= [test_list[i][0]for i in range(len(test_list))])
xgb_prc_df = pd.DataFrame(index =[train_list[i][0]for i in range(len(train_list))],columns= [test_list[i][0]for i in range(len(test_list))])
xgb_f1_df = pd.DataFrame(index =[train_list[i][0]for i in range(len(train_list))],columns= [test_list[i][0]for i in range(len(test_list))])
for train_dataset in train_list:
    train_raw_data = train_dataset[1]
    train_name = train_dataset[2]
    train_fig_name = train_dataset[0]
    for test_dataset in test_list:
        test_fig_name = test_dataset[0]
        test_name = test_dataset[2]
        test_raw_data = test_dataset[1]
        features_2use = [col for col in train_raw_data.columns if col in test_raw_data.columns]
        train_pheno_data = train_raw_data.loc[:,features_2use]
        test_pheno_data = test_raw_data.loc[:,features_2use]
        if test_name == train_name:
            train_pheno_data = test_pheno_data.copy()
            if test_name != 'Mitomics':
                skf = StratifiedKFold(n_splits=5, shuffle=True,random_state=123)
                target = list(np.load(DATA_DIREC+'/targets/'+train_name+'_target.npy'))
                data = train_pheno_data.copy()
                target_data = data.loc[target,]
                target_data['label'] = 1
                data = data.drop(target)
                all_true = list()
                all_pred = list()
                for i, (train_index, test_index) in enumerate(skf.split(data.iloc[:,:-1], data.iloc[:,-1])):
                    xgb_config = json.load(open(DATA_DIREC+'/config/'+train_name+'/best_paras_'+train_name+'_XGBoost_subset_False_fold_'+str(i)+'_ratio_1.0_5fold.json'))
                    target_fold_feature = target_data.copy()
                    fold_feature = train_pheno_data.copy()
                    train_data = fold_feature.iloc[train_index,]
                    train_data = pd.concat([train_data,target_fold_feature])
                    X_train = np.array(train_data.iloc[:,:-1])
                    y_train = np.array(train_data.iloc[:,-1])
                    X_test = np.array(fold_feature.iloc[test_index,:-1])
                    y_test = np.array(fold_feature.iloc[test_index,-1])
                    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
                    class_weight={0: class_weights[0], 1: class_weights[1]}
                    seed_everything()
                    model = xgb.XGBClassifier(**xgb_config)
                    model.fit(X_train, y_train,sample_weight = np.vectorize(class_weight.get)(y_train))
                    score = model.predict_proba(X_test)[:, 1]
                    all_true.extend(y_test)
                    all_pred.extend(score)
                fpr,tpr,_ = roc_curve(all_true,all_pred)
                auc_score = auc(fpr,tpr)
                xgb_auc_df.loc[train_fig_name,test_fig_name] = auc_score
                precision,recall,_ = precision_recall_curve(all_true,all_pred)
                auc_prc = average_precision_score(all_true,all_pred)
                xgb_prc_df.loc[train_fig_name,test_fig_name] = auc_prc
            elif test_name == 'Mitomics':
                feature_columns = Mitomics_data.columns
                all_pred = list()
                all_true = list()
                for i in range(5):
                    xgb_config = json.load(open(DATA_DIREC+'/config/Mitomics/best_paras_mitomics_XGBoost_fold_'+str(i)+'_ratio_1.0_5fold.json'))
                    test_source = np.load(DATA_DIREC+'/Mitomics/Partition_Source/fold_'+str(i)+'_test_protein.npy')
                    train_data = train_raw_data[~train_raw_data.index.isin(test_source)]
                    test_data = train_raw_data[train_raw_data.index.isin(test_source)]
                    X_train, y_train = train_data.iloc[:,:-1],train_data.iloc[:,-1]
                    X_test, y_test = test_data.iloc[:,:-1],test_data.iloc[:,-1]
                    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
                    class_weight={0: class_weights[0], 1: class_weights[1]}
                    seed_everything()
                    model = xgb.XGBClassifier(**xgb_config)
                    model.fit(X_train, y_train,sample_weight = np.vectorize(class_weight.get)(y_train))
                    score = model.predict_proba(X_test)[:, 1]
                    all_true.extend(y_test)
                    all_pred.extend(score)
                fpr,tpr,_ = roc_curve(all_true,all_pred)
                auc_score = auc(fpr,tpr)
                xgb_auc_df.loc[train_fig_name,test_fig_name] = auc_score
                precision,recall,_ = precision_recall_curve(all_true,all_pred)
                auc_prc = average_precision_score(all_true,all_pred)
                xgb_prc_df.loc[train_fig_name,test_fig_name] = auc_prc
        else:
            X_train = train_pheno_data.iloc[:, :-1]
            y_train = train_pheno_data.iloc[:, -1]
            X_test = test_pheno_data.iloc[:, :-1]
            y_test = test_pheno_data.iloc[:, -1]
            class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
            class_weight={0: class_weights[0], 1: class_weights[1]}
            xgb_config = json.load(open(DATA_DIREC+'/config/'+train_name+'/best_paras_'+train_name+'_transfer_XGBoost_transfer.json'))
            seed_everything()
            model = xgb.XGBClassifier(**xgb_config)
            # model = xgb.XGBClassifier()
            model.fit(X_train, y_train,sample_weight = np.vectorize(class_weight.get)(y_train))
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = auc(fpr,tpr)
            xgb_auc_df.loc[train_fig_name,test_fig_name] = auc_score
            precision,recall,_ = precision_recall_curve(y_test,y_prob)
            auc_prc = average_precision_score(y_test,y_prob)
            xgb_prc_df.loc[train_fig_name,test_fig_name] = auc_prc



xgb_auc_df.to_csv(PLOT_DIREC+'/Transfer/Transfer_AUROC_XGBoost_all_features.csv')
xgb_prc_df.to_csv(PLOT_DIREC+'/Transfer/Transfer_AUPRC_XGBoost_all_features.csv')



## PPI-network features only
phenotype = 'SREBP2'
SREBP2_data = pd.read_csv(DATA_DIREC+phenotype+'/train_node_attribute_SREBP2_transfer_ppi_only.csv',index_col=0)

phenotype = 'Influenza'
Influenza_data = pd.read_csv(DATA_DIREC+phenotype+'/train_node_attribute_Influenza_transfer_ppi_only.csv',index_col=0)

phenotype = 'LDLR'
LDLR_data = pd.read_csv(DATA_DIREC+phenotype+'/train_node_attribute_LDLR_transfer_ppi_only.csv',index_col=0)

phenotype = 'Mitomics'
Mitomics_data = pd.read_csv(DATA_DIREC+phenotype+'/train_node_attribute_Mitomics_transfer_ppi_only.csv',index_col = 0)


train_list = [('Cholesterol homeostasis',SREBP2_data,'SREBP2'),('Cholesterol uptake',LDLR_data,'LDLR'),('Influenza A virus replication',Influenza_data,'Influenza'),('Mitochondrial protein abundance',Mitomics_data,'Mitomics')]
test_list = [('Cholesterol \nhomeostasis',SREBP2_data,'SREBP2'),('Cholesterol \nuptake',LDLR_data,'LDLR'),('Influenza A \nvirus replication',Influenza_data,'Influenza'),('Mitochondrial \nprotein abundance',Mitomics_data,'Mitomics')]
# XGBoost
xgb_auc_df = pd.DataFrame(index =[train_list[i][0]for i in range(len(train_list))],columns= [test_list[i][0]for i in range(len(test_list))])
xgb_prc_df = pd.DataFrame(index =[train_list[i][0]for i in range(len(train_list))],columns= [test_list[i][0]for i in range(len(test_list))])
xgb_f1_df = pd.DataFrame(index =[train_list[i][0]for i in range(len(train_list))],columns= [test_list[i][0]for i in range(len(test_list))])
for train_dataset in train_list:
    train_raw_data = train_dataset[1]
    train_name = train_dataset[2]
    train_fig_name = train_dataset[0]
    for test_dataset in test_list:
        test_fig_name = test_dataset[0]
        test_name = test_dataset[2]
        test_raw_data = test_dataset[1]
        features_2use = [col for col in train_raw_data.columns if col in test_raw_data.columns]
        train_pheno_data = train_raw_data.loc[:,features_2use]
        test_pheno_data = test_raw_data.loc[:,features_2use]
        if test_name == train_name:
            train_pheno_data = test_pheno_data.copy()
            if test_name != 'Mitomics':
                skf = StratifiedKFold(n_splits=5, shuffle=True,random_state=123)
                target = list(np.load(DATA_DIREC+'/targets/'+train_name+'_target.npy'))
                data = train_pheno_data.copy()
                target_data = data.loc[target,]
                target_data['label'] = 1
                data = data.drop(target)
                all_true = list()
                all_pred = list()
                for i, (train_index, test_index) in enumerate(skf.split(data.iloc[:,:-1], data.iloc[:,-1])):
                    xgb_config = json.load(open(DATA_DIREC+'/config/'+train_name+'/best_paras_'+train_name+'_XGBoost_subset_False_fold_'+str(i)+'_ratio_1.0_5fold.json'))
                    target_fold_feature = target_data.copy()
                    fold_feature = train_pheno_data.copy()
                    train_data = fold_feature.iloc[train_index,]
                    train_data = pd.concat([train_data,target_fold_feature])
                    X_train = np.array(train_data.iloc[:,:-1])
                    y_train = np.array(train_data.iloc[:,-1])
                    X_test = np.array(fold_feature.iloc[test_index,:-1])
                    y_test = np.array(fold_feature.iloc[test_index,-1])
                    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
                    class_weight={0: class_weights[0], 1: class_weights[1]}
                    seed_everything()
                    model = xgb.XGBClassifier(**xgb_config)
                    model.fit(X_train, y_train,sample_weight = np.vectorize(class_weight.get)(y_train))
                    score = model.predict_proba(X_test)[:, 1]
                    all_true.extend(y_test)
                    all_pred.extend(score)
                fpr,tpr,_ = roc_curve(all_true,all_pred)
                auc_score = auc(fpr,tpr)
                xgb_auc_df.loc[train_fig_name,test_fig_name] = auc_score
                precision,recall,_ = precision_recall_curve(all_true,all_pred)
                auc_prc = average_precision_score(all_true,all_pred)
                xgb_prc_df.loc[train_fig_name,test_fig_name] = auc_prc
            elif test_name == 'Mitomics':
                feature_columns = Mitomics_data.columns
                all_pred = list()
                all_true = list()
                for i in range(5):
                    xgb_config = json.load(open(DATA_DIREC+'/config/Mitomics/best_paras_mitomics_XGBoost_fold_'+str(i)+'_ratio_1.0_5fold.json'))
                    test_source = np.load(DATA_DIREC+'/Mitomics/Partition_Source/fold_'+str(i)+'_test_protein.npy')
                    train_data = train_raw_data[~train_raw_data.index.isin(test_source)]
                    test_data = train_raw_data[train_raw_data.index.isin(test_source)]
                    X_train, y_train = train_data.iloc[:,:-1],train_data.iloc[:,-1]
                    X_test, y_test = test_data.iloc[:,:-1],test_data.iloc[:,-1]
                    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
                    class_weight={0: class_weights[0], 1: class_weights[1]}
                    seed_everything()
                    model = xgb.XGBClassifier(**xgb_config)
                    model.fit(X_train, y_train,sample_weight = np.vectorize(class_weight.get)(y_train))
                    score = model.predict_proba(X_test)[:, 1]
                    all_true.extend(y_test)
                    all_pred.extend(score)
                fpr,tpr,_ = roc_curve(all_true,all_pred)
                auc_score = auc(fpr,tpr)
                xgb_auc_df.loc[train_fig_name,test_fig_name] = auc_score
                precision,recall,_ = precision_recall_curve(all_true,all_pred)
                auc_prc = average_precision_score(all_true,all_pred)
                xgb_prc_df.loc[train_fig_name,test_fig_name] = auc_prc
        else:
            X_train = train_pheno_data.iloc[:, :-1]
            y_train = train_pheno_data.iloc[:, -1]
            X_test = test_pheno_data.iloc[:, :-1]
            y_test = test_pheno_data.iloc[:, -1]
            class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
            class_weight={0: class_weights[0], 1: class_weights[1]}
            xgb_config = json.load(open(DATA_DIREC+'/config/'+train_name+'/best_paras_'+train_name+'_transfer_ppi_only_XGBoost_transfer.json'))
            seed_everything()
            model = xgb.XGBClassifier(**xgb_config)
            # model = xgb.XGBClassifier()
            model.fit(X_train, y_train,sample_weight = np.vectorize(class_weight.get)(y_train))
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = auc(fpr,tpr)
            xgb_auc_df.loc[train_fig_name,test_fig_name] = auc_score
            precision,recall,_ = precision_recall_curve(y_test,y_prob)
            auc_prc = average_precision_score(y_test,y_prob)
            xgb_prc_df.loc[train_fig_name,test_fig_name] = auc_prc

xgb_auc_df.to_csv(PLOT_DIREC+'/Transfer/Transfer_AUROC_XGBoost_ppi_only.csv')
xgb_prc_df.to_csv(PLOT_DIREC+'/Transfer/Transfer_AUPRC_XGBoost_ppi_only.csv')

