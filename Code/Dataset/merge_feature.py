import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import networkx as nx
import torch
from torch_geometric.utils.convert import from_networkx
import numpy as np

def merge(dict1,dict2):
    dict2.update(dict1)
    return dict2


def merge_all(directory_path,file_name,file_suffix,min_num,max_num):
    init_dict = dict()
    for n in range(min_num,max_num+1):
        file_path = directory_path+file_name+str(n)+file_suffix
        print(file_path)
        try:
            with open(file_path,'r') as f:
                new_dict = json.load(f)
                init_dict = merge(new_dict, init_dict)
                names = init_dict.keys()
        except FileNotFoundError:
            pass
    print(len(list(names)))
    return init_dict

def thermo_encoding(dataframe,col_tobe_encode):
    df = dataframe.copy()
    max_num = dataframe[col_tobe_encode].max()
    for i in range(max_num):
        new_name = 'thermo'+str(i)
        df[new_name]=" "
    print(df)
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

def binary_encode(dataframe,col_tobe_encode):
    df = dataframe.copy()
    new_col_name = 'binary_'+col_tobe_encode
    for i in range(len(df)):
        if df.loc[i,col_tobe_encode]!=0:
            df.loc[i,new_col_name] = 1
        else:
            df.loc[i,new_col_name]=0
    return df

def onehot_encode(dataframe,col_tobe_encode):
    df = dataframe.copy()
    df = df.fillna(0)
    orders = ['Not detected', 'Low', 'Medium', 'High']
    num_type = len(orders)
    col_name = list()
    for i in range(num_type):
        new_col_name = 'one_hot_'+str(i)
        col_name.append(new_col_name)
        df[new_col_name] = pd.Series()
    for i in range(len(df)):
        true_value = df.loc[i,col_tobe_encode]
        if true_value == 0:
            for names in col_name:
                df.loc[i,names] = 0.25
        else:
            idx = orders.index(true_value)
            for names in col_name:
                if col_name.index(names) == idx:
                    df.loc[i,names] = 1
                else:
                    df.loc[i,names] = 0
    df = df.drop(columns=col_tobe_encode)
    return df

def dict2df(dictionary):
    df = pd.DataFrame.from_dict(dictionary,orient='index')
    df['protein'] = df.index
    df = df.set_index(pd.Series(list(range(0,len(dictionary)))))
    return df

def concat_att_abun(df1,df2):
    # df1 node attritute extracted from graph
    # df2 abundance attribute
    all_proteins = df1['protein'].unique()
    df2 = df2.drop(['Enesembl ID', 'Gene'], axis=1)
    x = df2.drop_duplicates(subset='protein', keep='first')
    feature_df3 = pd.merge(df1,x,on='protein')
    exist_protein = feature_df3['protein'].unique()
    non_exist_protein = list(set(all_proteins)-set(exist_protein))
    print(len(non_exist_protein))
    comp_feature_df = pd.DataFrame()
    for protein in non_exist_protein:
        comp_feature_df = comp_feature_df.append(df1[df1['protein']==protein])
    #print(comp_feature_df)
    feature_df = pd.concat([feature_df3,comp_feature_df])
    feature_df = feature_df.fillna(0)
    return feature_df

def drop_same_features(feature_df):
    for column in feature_df.columns:
        if feature_df[column].nunique() == 1:
            print(column)
            feature_df.drop(column, axis=1, inplace=True)
    return feature_df

def normalize_features(feature_df,columns_not_norm):
    # columns_not_norm is a list containing column names of columns that you don't want to normalize
    cols = list(feature_df.columns)
    cols_to_normalize = list(set(cols)-set(columns_not_norm))
    scaler = MinMaxScaler()
    feature_df[cols_to_normalize] = scaler.fit_transform(feature_df[cols_to_normalize])
    return feature_df

def set_pruned_feature(pruned_list,original_dataframe):
    # label_list = original_dataframe[original_dataframe['protein'].isin(pruned_list)]['label']
    # train_mask_list = original_dataframe[original_dataframe['protein'].isin(pruned_list)]['train']
    shortest_path_len = [10000]* len(pruned_list)
    new_rows = pd.DataFrame({'protein':pruned_list,'shortest_path_len':shortest_path_len},index = range(len(original_dataframe),len(original_dataframe)+len(pruned_list)))

    original_dataframe = original_dataframe.append(new_rows)
    original_dataframe = original_dataframe.fillna(0)
    return original_dataframe

def df2graph_set(feature_df,graph_file_link):
    # feature_df = pd.read_csv('data/node_attribute_onehot_unremoved.csv', index_col=0)
    # print(feature_df.columns.tolist())
    # print(len(feature_df.columns.tolist()))
    protein_set = list(feature_df.index)
    # print(len(protein_set))
    df= pd.read_csv(graph_file_link, sep=',')
    node_features = feature_df.iloc[:, :-1]
    print(len(node_features))
    edge_features = ['experimental','database','textmining','combined_score',
                    'binary_experimental','binary_database','binary_textmining']
    #edge_features = ['combined_score']
    print('these are  the features!')
    #print(node_features.columns.tolist())
    #print('label' in features.columns.tolist())
    # Remain rows that the source and target proteins both show up
    ppi_pd = df[(df['protein1'].isin(protein_set)) & (df['protein2'].isin(protein_set))]
    #ppi_col_to_norm = ['combined_score']
    ppi_col_to_norm = ['experimental', 'database', 'textmining', 'combined_score']
    ppi_pd[ppi_col_to_norm] = ppi_pd[ppi_col_to_norm].apply(lambda x: x / 1000, axis=0)
    PPI_graph = nx.from_pandas_edgelist(ppi_pd, source='protein1', target='protein2', edge_attr=edge_features,create_using=nx.DiGraph)
    node_attributes = feature_df.to_dict('index')
    # print(node_attributes)
    nx.set_node_attributes(PPI_graph, node_attributes)
    graph_data = from_networkx(PPI_graph)
    node_features = node_features.drop(['train_mask','val_mask','test_mask'],axis = 1)
    feature_tensors = [graph_data[feature].reshape(-1, 1) for feature in node_features]
    graph_data['x'] = torch.cat(feature_tensors, dim=1)
    graph_data['y'] = graph_data['label'].long()
    edge_attr_tensors = [graph_data[feature].reshape(-1, 1) for feature in edge_features]
    #print(edge_attr_tensors)
    graph_data['edge_attr'] = torch.cat(edge_attr_tensors, dim=1)
    graph_data.num_classes = len(graph_data.y.unique())
    #print(graph_data)
    print(f'{graph_data.train_mask.sum()} training samples, {graph_data.y[graph_data.train_mask].sum()} positives '
            f'({graph_data.y[graph_data.train_mask].sum() / graph_data.train_mask.sum():.3f})')
    print(f'{graph_data.val_mask.sum()} validation samples, {graph_data.y[graph_data.val_mask].sum()} positives '
            f'({graph_data.y[graph_data.val_mask].sum() / graph_data.val_mask.sum():.3f})')
    print(f'{graph_data.test_mask.sum()} test samples, {graph_data.y[graph_data.test_mask].sum()} positives '
            f'({graph_data.y[graph_data.test_mask].sum() / graph_data.test_mask.sum():.3f})')
    return graph_data

def add_label(feature_df,pos_list):
    for i in range(len(feature_df)):
        if feature_df.loc[i, 'protein'] in pos_list:
            feature_df.loc[i, 'label'] = 1
        else:
            feature_df.loc[i, 'label'] = 0
    return feature_df

def add_mask(feature_df,pos_list):
    for i in range(len(feature_df)):
        if feature_df.loc[i, 'protein'] in pos_list:
            feature_df.loc[i, 'train_mask'] = True
            feature_df.loc[i, 'val_mask'] = False
            feature_df.loc[i, 'test_mask'] = False
        else:
            feature_df.loc[i, 'train_mask'] = False
            feature_df.loc[i, 'val_mask'] = False
            feature_df.loc[i, 'test_mask'] = False

    return feature_df

def move2last(feature_df,col_to_move):
    other_columns = [col for col in feature_df.columns if col != col_to_move]
    feature_df = feature_df.reindex(columns=[*other_columns, col_to_move])
    return feature_df


