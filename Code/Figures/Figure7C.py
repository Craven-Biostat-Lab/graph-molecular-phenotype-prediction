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
import os
import pickle
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
warnings.filterwarnings('ignore')

DATA_DIREC = '../../Data/'
PLOT_DIREC = '../../Plot/'


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
    
        
def train_rf(data,n_folds,target,rf_config_path = None):
    target_data = data.loc[target,:]
    target_data['label'] = 1
    data = data.drop(target)
    skf = StratifiedKFold(n_splits = n_folds, shuffle=True,random_state=123)
    all_pred,all_true = list(),list()
    for i, (train_index, test_index) in enumerate(skf.split(data.iloc[:,:-1], data.iloc[:,-1])):
        rf_config = rf_config_path+'fold_'+str(i)+'_ratio_1.0_5fold.json'
        rf_config = json.load(open(rf_config))
        all_train_data = data.iloc[train_index,:]
        train_index,val_index = split_val(all_train_data)
        train_index = data.loc[train_index,:].index.tolist()
        val_index = data.loc[val_index,:].index.tolist()
        test_index = data.iloc[test_index,:].index.tolist()
        fold_data = data.copy()
        target_fold = target_data.copy()
        fold_data = pd.concat([fold_data,target_fold])
        fold_data = move2last(fold_data,'label')
        test_data = fold_data.loc[test_index,:]
        train_data = fold_data.loc[train_index,]
        train_data = pd.concat([train_data,target_fold])
        X_train = np.array(train_data.iloc[:,:-1])
        y_train = np.array(train_data.iloc[:,-1])
        X_test = np.array(test_data.iloc[:,:-1])
        y_test = np.array(test_data.iloc[:,-1])
        if rf_config != None:
            seed_everything()
            model = RandomForestClassifier(**rf_config,class_weight='balanced')
        else:
            seed_everything()
            model = RandomForestClassifier(n_estimators=1000,class_weight='balanced')
        model.fit(X_train, y_train)
        test_score = model.predict_proba(X_test)[:, 1]
        all_true.extend(y_test)
        all_pred.extend(test_score)
    return all_true,all_pred
def main():
    phenotype = 'Influenza'
    data1 = pd.read_csv(DATA_DIREC+phenotype+'/train_node_attribute_'+phenotype+'_with_target_feature.csv',index_col = 0)
    target = list(np.load(DATA_DIREC+phenotype+'_target.npy'))
    true_1,pred_1 = train_rf(data1,n_folds=5,target=target,rf_config_path = DATA_DIREC+'config/'+phenotype+'/best_paras_'+phenotype+'_RandomForest_subset_False_')
    fpr_1, tpr_1, _ = roc_curve(true_1, pred_1)
    auc_1 = auc(fpr_1,tpr_1)
    data2 = pd.read_csv(DATA_DIREC+phenotype+'/train_node_attribute_'+phenotype+'_with_target_feature.csv',index_col = 0)
    pos = list(set(np.load(DATA_DIREC+phenotype+'/'+phenotype+'_pos_gene.npy')).intersection(set(data2.index.tolist())))
    all_ins = list()
    neg = list(set(np.load(DATA_DIREC+phenotype+'/'+phenotype+'_neg_gene.npy')).intersection(set(data2.index.tolist())))
    neg = list(set(neg)-set(pos))
    all_ins.extend(pos)
    all_ins.extend(neg)
    data2 = data2.loc[all_ins,]
    true_2,pred_2 = train_rf(data2,n_folds=5,target=target,rf_config_path = DATA_DIREC+'config/'+phenotype+'/best_paras_'+phenotype+'_RandomForest_subset_True_')
    fpr_2, tpr_2, _ = roc_curve(true_2, pred_2)
    auc_2 = auc(fpr_2,tpr_2)
    fig,ax = plt.subplots()
    ax.plot(fpr_1,tpr_1,label="%s (AUC = %0.2f)" % ('Original negatives',auc_1),lw=2,alpha=1)
    ax.plot(fpr_2,tpr_2,label="%s (AUC = %0.2f)" % ('Stringent negatives',auc_2),lw=2,alpha=1)
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.7)
    ax.legend( loc="lower right",fontsize = 'large')
    ax.set_xlabel("False positive rate",fontsize = 14)
    ax.set_ylabel("True positive rate",fontsize = 14)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.set_title('Influenza A virus replication',fontsize = 18)
    fig.savefig(PLOT_DIREC+''+phenotype+'/FigureS1_ROC_Curves.png',dpi = 800)

if __name__=='__main__':
    main()
