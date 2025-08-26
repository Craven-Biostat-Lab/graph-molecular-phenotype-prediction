import networkx as nx
import pandas as pd
from stringdb_alias import HGNCMapper
import numpy as np
import pickle
import random
import json



def init_protein_graph(protein_link_filename):
    G = nx.Graph()
    added_genes = list()
    with open(protein_link_filename, 'r') as f:
        temp = f.readlines()
        graphlist = list()
        for element in temp:
            element = element.replace('\n', '')
            element = element.split(' ')
            graphlist.append(element)
    #print(graphlist[0:10])
    print('linklist done')
    proteins = set()
    edge_count = 0
    print('Set Adding')
    for element in graphlist:
        # print(type(element[6]))
        proteins.add(element[0])
        proteins.add(element[1])
    print('There are '+str(len(proteins))+' proteins')
    print('Add Protein Nodes')
    for protein in proteins:
        G.add_node(protein,type = 'protein')
    print('There are '+str(len(G.nodes)) +' nodes added')
    print('Add Edges')
    #duplicate_edge = list()
    for element in graphlist:
        if int(element[2]) >0:
            G.add_edge(element[0],element[1],label = 'E',strength=int(element[2]),score=int(element[5]))
        elif int(element[3])>0:
            G.add_edge(element[0],element[1],label = 'D',strength=int(element[3]),score=int(element[5]))
        elif int(element[4]) >0:
            G.add_edge(element[0],element[1],label = 'T',strength=int(element[4]),score=int(element[5]))
        edge_count += 1

    
    return G,list(proteins)

def trim_edges(G,threshold):
    removed_edge = list()
    for edges in G.edges:
        #print(edges)
        source = edges[0]
        target = edges[1]
        if source.startswith('9606') and target.startswith('9606'):
            # This guarantee that only protein-protein edges are removed
            if int(G.get_edge_data(source,target)['score']) < threshold:
                G.remove_edge(source, target)
                removed_edge.append((source,target))
    print(str(len(removed_edge))+' edges have been removed.')
    return G

def change_edge_type(G,file,edge_type):
    ## The file should be in the following format
    ## Source, target, edge_type should be the first three columns
    edge_list = pd.read_csv(file)
    for i in range(len(edge_list)):
        Source = edge_list.iloc[i,0]
        Target = edge_list.iloc[i,1]
        G.add_edge(Source, Target, label = edge_type,score = 1)
    return G


def remove_nodes(G,node_list):
    G.remove_nodes_from(node_list)
    return G

if __name__ == '__main__':
    mapper = HGNCMapper('../Data/Graphs/9606.protein.info.v11.5.txt.gz', '../Data/Graphs/9606.protein.aliases.v11.5.txt.gz')
    ## Initiate PPI, core_pro are those genes that would connect to the phenotype
    ppi_file1 = '../Data/Graphs/protein_physical_link_no_header.txt'
    # core_pro = np.load('/Users/leojin/Desktop/IGVF/Mitomics_Data/mitomics_target.npy') 
    protein_graph = init_protein_graph(ppi_file1)
    protein_G = protein_graph[0]
    proteins = protein_graph[1]
    node_list = list(set(proteins))
    print(len(set(node_list)))
    print(len(protein_G.nodes()))
    print(len(protein_G.edges()))

    # Prune out some nodes based on the cell type.
    celltype = 'HepG2' ## Replace with your celltype
    non_exist_gene = pd.read_csv(f'../Data/Raw/cellular-localization/undetected_genes_{celltype}.tsv',sep='\t')['Gene'].tolist()
    pruned_gene = [mapper.get_string_ids(gene).item() for gene in non_exist_gene]
    pruned_node = list(set(pruned_gene)&set(proteins))
    print(len(pruned_node))
    protein_G = remove_nodes(protein_G,pruned_node)
    print(len(protein_G.nodes))

    # Save the graph as a .gpickle file

    with open(f'../Data/Graphs/Physical_graph_{celltype}.gpickle', 'wb') as f:
        pickle.dump(protein_G, f, pickle.HIGHEST_PROTOCOL)

    ## Genrate the instance list (.pickle) file used for feature extraction.

    with open('../Data/Graphs/instance_list.txt','w') as f:
        split_size = len(node_list) // 5000 ### decides how many instances there are in one .pickle file
        split_lists = [node_list[i:i+split_size] for i in range(0, len(node_list), split_size)]
        for i, split_list in enumerate(split_lists):
            pickle_file_name = f"file_{i}.pickle"
            f.write(pickle_file_name+'\n')
            with open('../Data/Graphs/instance_file/'+pickle_file_name,'wb') as pickle_file:
                pickle.dump(split_list,pickle_file)
    
