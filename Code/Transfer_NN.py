# MLP
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
from imblearn.over_sampling import RandomOverSampler
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score,average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report
import torch.optim as optim
import torch
import torch.nn as nn
import pickle
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




##
DATA_DIREC = '../Data/'
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
mlp_auc_df = pd.DataFrame(index =[train_list[i][0]for i in range(len(train_list))],columns= [test_list[i][0]for i in range(len(test_list))])
mlp_prc_df = pd.DataFrame(index =[train_list[i][0]for i in range(len(train_list))],columns= [test_list[i][0]for i in range(len(test_list))])
mlp_f1_df = pd.DataFrame(index =[train_list[i][0]for i in range(len(train_list))],columns= [test_list[i][0]for i in range(len(test_list))])
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
                    mlp_config = json.load(open(DATA_DIREC+'/config/'+train_name+'/best_paras_'+train_name+'_MLP_subset_False_fold_'+str(i)+'_ratio_1.0_5fold.json'))
                    target_fold_feature = target_data.copy()
                    fold_feature = train_pheno_data.copy()
                    train_data = fold_feature.iloc[train_index,]
                    train_data = pd.concat([train_data,target_fold_feature])
                    X_train,y_train = train_data.iloc[:,:-1],train_data.iloc[:,-1]
                    X_train = torch.tensor(np.array(X_train),dtype = torch.float32)
                    y_train = torch.tensor(np.array(y_train),dtype = torch.float32).unsqueeze(1)
                    X_test =  torch.tensor(np.array(fold_feature.iloc[test_index, :-1]),dtype = torch.float32)
                    y_test = np.array(fold_feature.iloc[test_index,-1])
                    train_dataset = CustomDataset(X_train,y_train)
                    batch_size = mlp_config['batch_size']
                    num_epoch = mlp_config['num_epoch']
                    learning_rate = mlp_config['learning_rate']
                    hidden_size = mlp_config['hidden_size']
                    num_layer = mlp_config['num_layer']
                    activation_type = mlp_config['activation']
                    seed_everything()
                    train_loader = DataLoader(dataset = train_dataset,batch_size = batch_size,shuffle=True)
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
                    # n_pos = y_train.sum().item()
                    # n_neg = y_train.numel() - n_pos
                    # pos_w = torch.tensor([n_neg / n_pos])
                    criterion = nn.BCEWithLogitsLoss()
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                    for epoch in range(num_epoch):
                        model.train()
                        for batch_feature,batch_label in train_loader:
                            batch_feature, batch_label = batch_feature, batch_label
                            optimizer.zero_grad()
                            outputs = model(batch_feature)
                            loss = criterion(outputs,batch_label)
                            loss.backward()
                            optimizer.step()
                    logits = model(X_test).view(-1)
                    score = torch.sigmoid(logits).tolist()
                    all_true.extend(y_test)
                    all_pred.extend(score)
                fpr,tpr,_ = roc_curve(all_true,all_pred)
                auc_score = auc(fpr,tpr)
                mlp_auc_df.loc[train_fig_name,test_fig_name] = auc_score
                precision,recall,_ = precision_recall_curve(all_true,all_pred)
                auc_prc = average_precision_score(all_true,all_pred)
                mlp_prc_df.loc[train_fig_name,test_fig_name] = auc_prc
            elif test_name == 'Mitomics':
                feature_columns = Mitomics_data.columns
                all_pred = list()
                all_true = list()
                for i in range(5):
                    mlp_config = json.load(open(DATA_DIREC+'/config/Mitomics/best_paras_Mitomics_MLP_fold_'+str(i)+'_ratio_1.0_5fold.json'))
                    test_source = np.load(DATA_DIREC+'/Mitomics/Partition_Source/fold_'+str(i)+'_test_protein.npy')
                    train_data = train_raw_data[~train_raw_data.index.isin(test_source)]
                    test_data = train_raw_data[train_raw_data.index.isin(test_source)]
                    X_train,y_train = train_data.iloc[:,:-1],train_data.iloc[:,-1]
                    X_train = torch.tensor(np.array(X_train),dtype = torch.float32)
                    y_train = torch.tensor(np.array(y_train),dtype = torch.float32).unsqueeze(1)
                    X_test =  torch.tensor(np.array(test_data.iloc[:, :-1]),dtype = torch.float32)
                    y_test = np.array(test_data.iloc[:, -1])
                    train_dataset = CustomDataset(X_train,y_train)
                    batch_size = mlp_config['batch_size']
                    num_epoch = mlp_config['num_epoch']
                    learning_rate = mlp_config['learning_rate']
                    hidden_size = mlp_config['hidden_size']
                    num_layer = mlp_config['num_layer']
                    activation_type = mlp_config['activation']
                    seed_everything()
                    train_loader = DataLoader(dataset = train_dataset,batch_size = batch_size,shuffle=True)
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
                    # n_pos = y_train.sum().item()
                    # n_neg = y_train.numel() - n_pos
                    # pos_w = torch.tensor([n_neg / n_pos])
                    criterion = nn.BCEWithLogitsLoss()
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                    for epoch in range(num_epoch):
                        model.train()
                        for batch_feature,batch_label in train_loader:
                            batch_feature, batch_label = batch_feature, batch_label
                            optimizer.zero_grad()
                            outputs = model(batch_feature)
                            loss = criterion(outputs,batch_label)
                            loss.backward()
                            optimizer.step()
                    logits = model(X_test).view(-1)
                    score = torch.sigmoid(logits).tolist()
                    all_true.extend(y_test)
                    all_pred.extend(score)
                fpr,tpr,_ = roc_curve(all_true,all_pred)
                auc_score = auc(fpr,tpr)
                mlp_auc_df.loc[train_fig_name,test_fig_name] = auc_score
                precision,recall,_ = precision_recall_curve(all_true,all_pred)
                auc_prc = average_precision_score(all_true,all_pred)
                mlp_prc_df.loc[train_fig_name,test_fig_name] = auc_prc
        else:
            X_train = train_pheno_data.iloc[:, :-1]
            y_train = train_pheno_data.iloc[:, -1]
            X_test = test_pheno_data.iloc[:, :-1]
            y_test = test_pheno_data.iloc[:, -1]
            X_train = torch.tensor(np.array(X_train),dtype = torch.float32)
            y_train = torch.tensor(np.array(y_train),dtype = torch.float32).unsqueeze(1)
            X_test =  torch.tensor(np.array(test_pheno_data.iloc[:, :-1]),dtype = torch.float32)
            y_test = np.array(test_pheno_data.iloc[:, -1])
            train_dataset = CustomDataset(X_train,y_train)
            mlp_config = json.load(open(DATA_DIREC+'/config/'+train_name+'/best_paras_'+train_name+'_transfer_st_relation_MLP_transfer.json'))
            batch_size = mlp_config['batch_size']
            num_epoch = mlp_config['num_epoch']
            learning_rate = mlp_config['learning_rate']
            hidden_size = mlp_config['hidden_size']
            num_layer = mlp_config['num_layer']
            activation_type = mlp_config['activation']
            seed_everything()
            train_loader = DataLoader(dataset = train_dataset,batch_size = batch_size,shuffle=True)
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
            # n_pos = y_train.sum().item()
            # n_neg = y_train.numel() - n_pos
            # pos_w = torch.tensor([n_neg / n_pos])
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            for epoch in range(num_epoch):
                model.train()
                for batch_feature,batch_label in train_loader:
                    batch_feature, batch_label = batch_feature, batch_label
                    optimizer.zero_grad()
                    outputs = model(batch_feature)
                    loss = criterion(outputs,batch_label)
                    loss.backward()
                    optimizer.step()
            logits = model(X_test).view(-1)
            y_prob = torch.sigmoid(logits).tolist()
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = auc(fpr,tpr)
            print(train_fig_name,test_fig_name,auc_score)
            mlp_auc_df.loc[train_fig_name,test_fig_name] = auc_score
            precision,recall,_ = precision_recall_curve(y_test,y_prob)
            auc_prc = average_precision_score(y_test,y_prob)
            mlp_prc_df.loc[train_fig_name,test_fig_name] = auc_prc

    

mlp_auc_df.to_csv(PLOT_DIREC+'/Transfer/Transfer_AUROC_NN_st_relation.csv')
mlp_prc_df.to_csv(PLOT_DIREC+'/Transfer/Transfer_AUPRC_NN_st_relation.csv')



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
mlp_auc_df = pd.DataFrame(index =[train_list[i][0]for i in range(len(train_list))],columns= [test_list[i][0]for i in range(len(test_list))])
mlp_prc_df = pd.DataFrame(index =[train_list[i][0]for i in range(len(train_list))],columns= [test_list[i][0]for i in range(len(test_list))])
mlp_f1_df = pd.DataFrame(index =[train_list[i][0]for i in range(len(train_list))],columns= [test_list[i][0]for i in range(len(test_list))])
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
            train_pheno_data = train_pheno_data.copy()
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
                    mlp_config = json.load(open(DATA_DIREC+'/config/'+train_name+'/best_paras_'+train_name+'_MLP_subset_False_fold_'+str(i)+'_ratio_1.0_5fold.json'))
                    target_fold_feature = target_data.copy()
                    fold_feature = train_pheno_data.copy()
                    train_data = fold_feature.iloc[train_index,]
                    train_data = pd.concat([train_data,target_fold_feature])
                    X_train,y_train = train_data.iloc[:,:-1],train_data.iloc[:,-1]
                    X_train = torch.tensor(np.array(X_train),dtype = torch.float32)
                    y_train = torch.tensor(np.array(y_train),dtype = torch.float32).unsqueeze(1)
                    X_test =  torch.tensor(np.array(fold_feature.iloc[test_index, :-1]),dtype = torch.float32)
                    y_test = np.array(fold_feature.iloc[test_index,-1])
                    train_dataset = CustomDataset(X_train,y_train)
                    batch_size = mlp_config['batch_size']
                    num_epoch = mlp_config['num_epoch']
                    learning_rate = mlp_config['learning_rate']
                    hidden_size = mlp_config['hidden_size']
                    num_layer = mlp_config['num_layer']
                    activation_type = mlp_config['activation']
                    seed_everything()
                    train_loader = DataLoader(dataset = train_dataset,batch_size = batch_size,shuffle=True)
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
                    # n_pos = y_train.sum().item()
                    # n_neg = y_train.numel() - n_pos
                    # pos_w = torch.tensor([n_neg / n_pos])
                    criterion = nn.BCEWithLogitsLoss()
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                    for epoch in range(num_epoch):
                        model.train()
                        for batch_feature,batch_label in train_loader:
                            batch_feature, batch_label = batch_feature, batch_label
                            optimizer.zero_grad()
                            outputs = model(batch_feature)
                            loss = criterion(outputs,batch_label)
                            loss.backward()
                            optimizer.step()
                    logits = model(X_test).view(-1)
                    score = torch.sigmoid(logits).tolist()
                    all_true.extend(y_test)
                    all_pred.extend(score)
                fpr,tpr,_ = roc_curve(all_true,all_pred)
                auc_score = auc(fpr,tpr)
                mlp_auc_df.loc[train_fig_name,test_fig_name] = auc_score
                precision,recall,_ = precision_recall_curve(all_true,all_pred)
                auc_prc = average_precision_score(all_true,all_pred)
                mlp_prc_df.loc[train_fig_name,test_fig_name] = auc_prc
            elif test_name == 'Mitomics':
                feature_columns = Mitomics_data.columns
                all_pred = list()
                all_true = list()
                for i in range(5):
                    mlp_config = json.load(open(DATA_DIREC+'/config/Mitomics/best_paras_Mitomics_MLP_fold_'+str(i)+'_ratio_1.0_5fold.json'))
                    test_source = np.load(DATA_DIREC+'/Mitomics/Partition_Source/fold_'+str(i)+'_test_protein.npy')
                    train_data = train_raw_data[~train_raw_data.index.isin(test_source)]
                    test_data = train_raw_data[train_raw_data.index.isin(test_source)]
                    X_train,y_train = train_data.iloc[:,:-1],train_data.iloc[:,-1]
                    X_train = torch.tensor(np.array(X_train),dtype = torch.float32)
                    y_train = torch.tensor(np.array(y_train),dtype = torch.float32).unsqueeze(1)
                    X_test =  torch.tensor(np.array(test_data.iloc[:, :-1]),dtype = torch.float32)
                    y_test = np.array(test_data.iloc[:, -1])
                    train_dataset = CustomDataset(X_train,y_train)
                    batch_size = mlp_config['batch_size']
                    num_epoch = mlp_config['num_epoch']
                    learning_rate = mlp_config['learning_rate']
                    hidden_size = mlp_config['hidden_size']
                    num_layer = mlp_config['num_layer']
                    activation_type = mlp_config['activation']
                    seed_everything()
                    train_loader = DataLoader(dataset = train_dataset,batch_size = batch_size,shuffle=True)
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
                    # n_pos = y_train.sum().item()
                    # n_neg = y_train.numel() - n_pos
                    # pos_w = torch.tensor([n_neg / n_pos])
                    criterion = nn.BCEWithLogitsLoss()
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                    for epoch in range(num_epoch):
                        model.train()
                        for batch_feature,batch_label in train_loader:
                            batch_feature, batch_label = batch_feature, batch_label
                            optimizer.zero_grad()
                            outputs = model(batch_feature)
                            loss = criterion(outputs,batch_label)
                            loss.backward()
                            optimizer.step()
                    logits = model(X_test).view(-1)
                    score = torch.sigmoid(logits).tolist()
                    all_true.extend(y_test)
                    all_pred.extend(score)
                fpr,tpr,_ = roc_curve(all_true,all_pred)
                auc_score = auc(fpr,tpr)
                mlp_auc_df.loc[train_fig_name,test_fig_name] = auc_score
                precision,recall,_ = precision_recall_curve(all_true,all_pred)
                auc_prc = average_precision_score(all_true,all_pred)
                mlp_prc_df.loc[train_fig_name,test_fig_name] = auc_prc
        else:
            X_train = train_pheno_data.iloc[:, :-1]
            y_train = train_pheno_data.iloc[:, -1]
            X_test = test_pheno_data.iloc[:, :-1]
            y_test = test_pheno_data.iloc[:, -1]
            X_train = torch.tensor(np.array(X_train),dtype = torch.float32)
            y_train = torch.tensor(np.array(y_train),dtype = torch.float32).unsqueeze(1)
            X_test =  torch.tensor(np.array(test_pheno_data.iloc[:, :-1]),dtype = torch.float32)
            y_test = np.array(test_pheno_data.iloc[:, -1])
            train_dataset = CustomDataset(X_train,y_train)
            mlp_config = json.load(open(DATA_DIREC+'/config/'+train_name+'/best_paras_'+train_name+'_transfer_MLP_transfer.json'))
            batch_size = mlp_config['batch_size']
            num_epoch = mlp_config['num_epoch']
            learning_rate = mlp_config['learning_rate']
            hidden_size = mlp_config['hidden_size']
            num_layer = mlp_config['num_layer']
            activation_type = mlp_config['activation']
            seed_everything()
            train_loader = DataLoader(dataset = train_dataset,batch_size = batch_size,shuffle=True)
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
            # n_pos = y_train.sum().item()
            # n_neg = y_train.numel() - n_pos
            # pos_w = torch.tensor([n_neg / n_pos])
            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            for epoch in range(num_epoch):
                model.train()
                for batch_feature,batch_label in train_loader:
                    batch_feature, batch_label = batch_feature, batch_label
                    optimizer.zero_grad()
                    outputs = model(batch_feature)
                    loss = criterion(outputs,batch_label)
                    loss.backward()
                    optimizer.step()
            logits = model(X_test).view(-1)
            y_prob = torch.sigmoid(logits).tolist()
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = auc(fpr,tpr)
            print(train_fig_name,test_fig_name,auc_score)
            mlp_auc_df.loc[train_fig_name,test_fig_name] = auc_score
            precision,recall,_ = precision_recall_curve(y_test,y_prob)
            auc_prc = average_precision_score(y_test,y_prob)
            mlp_prc_df.loc[train_fig_name,test_fig_name] = auc_prc


mlp_auc_df.to_csv(PLOT_DIREC+'/Transfer/Transfer_AUROC_NN_all_features.csv')
mlp_prc_df.to_csv(PLOT_DIREC+'/Transfer/Transfer_AUPRC_NN_all_features.csv')

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
mlp_auc_df = pd.DataFrame(index =[train_list[i][0]for i in range(len(train_list))],columns= [test_list[i][0]for i in range(len(test_list))])
mlp_prc_df = pd.DataFrame(index =[train_list[i][0]for i in range(len(train_list))],columns= [test_list[i][0]for i in range(len(test_list))])
mlp_f1_df = pd.DataFrame(index =[train_list[i][0]for i in range(len(train_list))],columns= [test_list[i][0]for i in range(len(test_list))])
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
                    mlp_config = json.load(open(DATA_DIREC+'/config/'+train_name+'/best_paras_'+train_name+'_MLP_subset_False_fold_'+str(i)+'_ratio_1.0_5fold.json'))
                    target_fold_feature = target_data.copy()
                    fold_feature = train_pheno_data.copy()
                    train_data = fold_feature.iloc[train_index,]
                    train_data = pd.concat([train_data,target_fold_feature])
                    X_train,y_train = train_data.iloc[:,:-1],train_data.iloc[:,-1]
                    X_train = torch.tensor(np.array(X_train),dtype = torch.float32)
                    y_train = torch.tensor(np.array(y_train),dtype = torch.float32).unsqueeze(1)
                    X_test =  torch.tensor(np.array(fold_feature.iloc[test_index, :-1]),dtype = torch.float32)
                    y_test = np.array(fold_feature.iloc[test_index,-1])
                    train_dataset = CustomDataset(X_train,y_train)
                    batch_size = mlp_config['batch_size']
                    num_epoch = mlp_config['num_epoch']
                    learning_rate = mlp_config['learning_rate']
                    hidden_size = mlp_config['hidden_size']
                    num_layer = mlp_config['num_layer']
                    activation_type = mlp_config['activation']
                    seed_everything()
                    train_loader = DataLoader(dataset = train_dataset,batch_size = batch_size,shuffle=True)
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
                    # n_pos = y_train.sum().item()
                    # n_neg = y_train.numel() - n_pos
                    # pos_w = torch.tensor([n_neg / n_pos])
                    criterion = nn.BCEWithLogitsLoss() 
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                    for epoch in range(num_epoch):
                        model.train()
                        for batch_feature,batch_label in train_loader:
                            batch_feature, batch_label = batch_feature, batch_label
                            optimizer.zero_grad()
                            outputs = model(batch_feature)
                            loss = criterion(outputs,batch_label)
                            loss.backward()
                            optimizer.step()
                    logits = model(X_test).view(-1)
                    score = torch.sigmoid(logits).tolist()
                    all_true.extend(y_test)
                    all_pred.extend(score)
                fpr,tpr,_ = roc_curve(all_true,all_pred)
                auc_score = auc(fpr,tpr)
                mlp_auc_df.loc[train_fig_name,test_fig_name] = auc_score
                precision,recall,_ = precision_recall_curve(all_true,all_pred)
                auc_prc = average_precision_score(all_true,all_pred)
                mlp_prc_df.loc[train_fig_name,test_fig_name] = auc_prc
            elif test_name == 'Mitomics':
                feature_columns = Mitomics_data.columns
                all_pred = list()
                all_true = list()
                for i in range(5):
                    mlp_config = json.load(open(DATA_DIREC+'/config/Mitomics/best_paras_Mitomics_MLP_fold_'+str(i)+'_ratio_1.0_5fold.json'))
                    test_source = np.load(DATA_DIREC+'/Mitomics/Partition_Source/fold_'+str(i)+'_test_protein.npy')
                    train_data = train_raw_data[~train_raw_data.index.isin(test_source)]
                    test_data = train_raw_data[train_raw_data.index.isin(test_source)]
                    X_train,y_train = train_data.iloc[:,:-1],train_data.iloc[:,-1]
                    X_train = torch.tensor(np.array(X_train),dtype = torch.float32)
                    y_train = torch.tensor(np.array(y_train),dtype = torch.float32).unsqueeze(1)
                    X_test =  torch.tensor(np.array(test_data.iloc[:, :-1]),dtype = torch.float32)
                    y_test = np.array(test_data.iloc[:, -1])
                    train_dataset = CustomDataset(X_train,y_train)
                    batch_size = mlp_config['batch_size']
                    num_epoch = mlp_config['num_epoch']
                    learning_rate = mlp_config['learning_rate']
                    hidden_size = mlp_config['hidden_size']
                    num_layer = mlp_config['num_layer']
                    activation_type = mlp_config['activation']
                    seed_everything()
                    train_loader = DataLoader(dataset = train_dataset,batch_size = batch_size,shuffle=True)
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
                    # n_pos = y_train.sum().item()
                    # n_neg = y_train.numel() - n_pos
                    # pos_w = torch.tensor([n_neg / n_pos])
                    criterion = nn.BCEWithLogitsLoss()
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                    for epoch in range(num_epoch):
                        model.train()
                        for batch_feature,batch_label in train_loader:
                            batch_feature, batch_label = batch_feature, batch_label
                            optimizer.zero_grad()
                            outputs = model(batch_feature)
                            loss = criterion(outputs,batch_label)
                            loss.backward()
                            optimizer.step()
                    logits = model(X_test).view(-1)
                    score = torch.sigmoid(logits).tolist()
                    all_true.extend(y_test)
                    all_pred.extend(score)
                fpr,tpr,_ = roc_curve(all_true,all_pred)
                auc_score = auc(fpr,tpr)
                mlp_auc_df.loc[train_fig_name,test_fig_name] = auc_score
                precision,recall,_ = precision_recall_curve(all_true,all_pred)
                auc_prc = average_precision_score(all_true,all_pred)
                mlp_prc_df.loc[train_fig_name,test_fig_name] = auc_prc
        else:
            X_train = train_pheno_data.iloc[:, :-1]
            y_train = train_pheno_data.iloc[:, -1]
            X_test = test_pheno_data.iloc[:, :-1]
            y_test = test_pheno_data.iloc[:, -1]
            X_train = torch.tensor(np.array(X_train),dtype = torch.float32)
            y_train = torch.tensor(np.array(y_train),dtype = torch.float32).unsqueeze(1)
            X_test =  torch.tensor(np.array(test_pheno_data.iloc[:, :-1]),dtype = torch.float32)
            y_test = np.array(test_pheno_data.iloc[:, -1])
            train_dataset = CustomDataset(X_train,y_train)
            mlp_config = json.load(open(DATA_DIREC+'/config/'+train_name+'/best_paras_'+train_name+'_transfer_ppi_only_MLP_transfer.json'))
            batch_size = mlp_config['batch_size']
            num_epoch = mlp_config['num_epoch']
            learning_rate = mlp_config['learning_rate']
            hidden_size = mlp_config['hidden_size']
            num_layer = mlp_config['num_layer']
            activation_type = mlp_config['activation']
            seed_everything()
            train_loader = DataLoader(dataset = train_dataset,batch_size = batch_size,shuffle=True)
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
            # n_pos = y_train.sum().item()
            # n_neg = y_train.numel() - n_pos
            # pos_w = torch.tensor([n_neg / n_pos])
            criterion = nn.BCEWithLogitsLoss() 
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            for epoch in range(num_epoch):
                model.train()
                for batch_feature,batch_label in train_loader:
                    batch_feature, batch_label = batch_feature, batch_label
                    optimizer.zero_grad()
                    outputs = model(batch_feature)
                    loss = criterion(outputs,batch_label)
                    loss.backward()
                    optimizer.step()
            logits = model(X_test).view(-1)
            y_prob = torch.sigmoid(logits).tolist()
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = auc(fpr,tpr)
            print(train_fig_name,test_fig_name,auc_score)
            mlp_auc_df.loc[train_fig_name,test_fig_name] = auc_score
            precision,recall,_ = precision_recall_curve(y_test,y_prob)
            auc_prc = average_precision_score(y_test,y_prob)
            mlp_prc_df.loc[train_fig_name,test_fig_name] = auc_prc

    

mlp_auc_df.to_csv(PLOT_DIREC+'/Transfer/Transfer_AUROC_NN_ppi_only.csv')
mlp_prc_df.to_csv(PLOT_DIREC+'/Transfer/Transfer_AUPRC_NN_ppi_only.csv')

