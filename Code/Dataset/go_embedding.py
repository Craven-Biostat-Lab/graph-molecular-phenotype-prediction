import numpy as np
import pandas as pd
import torch
import networkx as nx
import argparse
from torch_geometric.utils import to_networkx, from_networkx
from torch_geometric.nn import Node2Vec
import pickle
import random
import os

def seed_everything(seed=42):
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_node2vec(model, loader, optimizer):
    """
    Training step for a node2vec model.  Must train differently because node2vec requires batched training.
    :param model: node2vec model
    :param loader: the data loader for batched training
    :param optimizer: loss function
    """
    if torch.backends.mps.is_built():
        device = 'mps'
    elif torch.cuda.is_available(): 
        device = "cuda:0" 
    else:
        device = 'cpu' 
    model.train()
    total_loss = 0
    # The data loader iterates over positive and negative random walks
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def main(args):
    if torch.backends.mps.is_built():
        device = 'mps'
    elif torch.cuda.is_available(): 
        device = "cuda:0" 
    else:
        device = 'cpu' 
    print(device)
    go_graph_file = pd.read_csv(args.graph,names = ['protein1','protein2','score']
                            ,keep_default_na=False)
    go_graph_file['protein1'] = go_graph_file['protein1'].str.strip()
    go_graph_file['protein2'] = go_graph_file['protein2'].str.strip()
    go_graph = nx.from_pandas_edgelist(go_graph_file,source='protein1',
                                target='protein2',
                                create_using=nx.DiGraph())
    node_list = np.load(args.nodelist)
    go_graph = go_graph.subgraph(node_list)
    node_to_index = {node: index for index, node in enumerate(go_graph.nodes())}
    index_to_node = {index: node for node, index in node_to_index.items()}
    data = from_networkx(go_graph)
    seed_everything()
    model = Node2Vec(data.edge_index, walk_length=20,
                    context_size=10, walks_per_node=10,
                    num_negative_samples=1, sparse=True,embedding_dim=args.dim,p=args.p,q=args.q).to(device)
    loader = model.loader(batch_size=128, shuffle=True, num_workers=1)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
    for epoch in range(1, args.epochs + 1):
        # node2vec also has a special training function that uses the loader
        loss = train_node2vec(model, loader, optimizer)
        if epoch % 10 == 0:
            print(f'epoch {epoch} loss {loss}', flush=True)
    model.eval()
    h = model.forward()
    # h are the node embeddings
    go_embedding = dict()
    for i in range(len(h)):
        node_name = index_to_node[i]
        go_embedding[node_name] = dict()
        features = h[i].tolist()
        for j in range(len(features)):
            go_embedding[node_name]['go_feature_'+str(j)] = features[j]
    go_embedding = pd.DataFrame.from_dict(go_embedding,orient='index')
    go_embedding.to_csv('go_embedding'+str(args.dim)+'.csv')
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph',
                        help='Path to the graph data file',
                        type=str)
    parser.add_argument('--nodelist',
                        help='Path to the node list file',
                        type=str)
    parser.add_argument('--dim',
                        default=32,
                        help='Dimension of the embedding',
                        type=int)
    parser.add_argument('--epochs',
                        default=50,
                        help='number of epochs to train (default 50)',
                        type=int)
    parser.add_argument('--p',default=1,
                        help = 'parameter p, default=1',
                        type = int)
    parser.add_argument('--q',default=1,
                         help = 'parameter q, default =1',
                         type = int)
    main(parser.parse_args())
    
            
        
    
