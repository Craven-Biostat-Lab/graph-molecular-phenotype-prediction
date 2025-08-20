import networkx as nx
import json
import sys
import pickle
import itertools
import pandas as pd
import numpy as np

def getFeaturesForSingleInstance(G,inst,is_phys=False, cutoff=4, get_enumerated_features=False):
    temp_dict = dict()  # Temp dict for this instance
   # gene = inst[1][:inst[1].index('-')]\
    #included_edge_types = ('E','T','D','F','N','U','M') ## This is for metabolism data
    included_edge_types = ('E','T','D')
    Source = inst
    prepend = ""
    if is_phys:
        prepend = "phys_"
    Target = 'Phenotype'
    try:
        count = 0
        max_mult_score = 0
        max_degree_score = 0
        short_path_len = 10000  # Some arbitrary large value
        has_path = False
        name = ['P'] 
        #name = target_list
        for i in range(cutoff):
            for perm in itertools.product(included_edge_types, repeat=i):
                #print(perm)
                temp_str = ''
                for val in perm:
                    temp_str += val
                for n in name:
                    temp_dict[temp_str + n]=0
        if nx.has_path(G, source=Source, target=Target):  # if path from gene to protein exists, do analysis
            has_path = True
            short_path = nx.shortest_path(G, source=Source, target=Target)
            short_path_len = len(short_path)-1
            #print(str(gene) + " " + str(protein))
            for path in nx.all_simple_edge_paths(G, source=Source, target=Target, cutoff=cutoff):
                count += 1
                temp_mult_score = 1
                temp_degree_score = 1
                first = True
                path_str = ""
                for edge in path:
                    if edge[0].startswith('9606') and edge[1].startswith('9606'):
                        temp_mult_score *= (G[edge[0]][edge[1]]['score'] / 1000)
                        temp_degree_score /= G.degree[edge[1]]
                        path_str += G[edge[0]][edge[1]]['label']
                    elif edge[0].startswith('9606') and edge[1] == Target:
                        end_protein = G[edge[0]][edge[1]]['label']
                        path_str += end_protein
                        temp_dict[path_str] += 1
                max_mult_score = temp_mult_score if max_mult_score < temp_mult_score else max_mult_score
                max_degree_score = temp_degree_score if max_degree_score < temp_degree_score else max_degree_score
        temp_dict[prepend + "has_path"] = int(has_path)
        temp_dict[prepend + "num_paths"] = count
        temp_dict[prepend + "shortest_path_len"] = short_path_len
        temp_dict[prepend + "max_path_score"] = max_mult_score
        temp_dict[prepend + "max_degree_score"] = max_degree_score
        return temp_dict, Source, Target  # return feature set
    except nx.NodeNotFound:  # If the gene is not in the graph, default to here
        for key in temp_dict.keys():
            temp_dict[key] = 0
        temp_dict[prepend + "has_path"] = 0
        temp_dict[prepend + "num_paths"] = 0
        temp_dict[prepend + "shortest_path_len"] = 10000
        temp_dict[prepend + "max_path_score"] = 0
        temp_dict[prepend + "max_degree_score"] = 0
        return temp_dict,Source,Target



if __name__ == "__main__":
    ## This script generate a .json file incuding the features extracted for all proteins in the .pickle file.
    ## Input files and arguments:
    ## 1. pickle file :  .pickle file, include StringIDs for gene $G$
    ## 2. graph file: .gpickle file,  the cell-type-specific PPI
    ## 3. target_list: .npy file, includes the target proteins selected for the phenotype
    pickle_file = str(sys.argv[1])
    print(pickle_file)
    n_prefix = pickle_file.index('le_')+3
    n_suffix = pickle_file.index('.p')
    number = pickle_file[n_prefix:n_suffix]
    obj = pd.read_pickle(pickle_file)
    # print(obj)
    graph_file = str(sys.argv[2])
    print(graph_file)
    print(np.load(sys.argv[3]))

    target_list = np.load(sys.argv[3]).tolist()
    if type(target_list) == type('a'):
        target_list = [target_list]
    print(target_list)
    phenotype_name = str(sys.argv[4])
    print(phenotype_name)
    if phenotype_name == 'None':
        phenotype_name = target_list[0]
    print(phenotype_name)

    with open(graph_file, 'rb') as f:
            ppi_graph = pickle.load(f)
            for target_protein in target_list:
                ppi_graph.add_edge(target_protein,'Phenotype',label = 'P')  # Add edges between all target protein and the phenotype
    print(nx.number_of_nodes(ppi_graph))
    instance_list = list()
    if type(obj) == type(list()):
        for instance in obj:
            Single_instance = getFeaturesForSingleInstance(ppi_graph,instance,get_enumerated_features=True)
            instance_list.append(Single_instance)
    elif type(obj) == type('a'):
        Single_instance = getFeaturesForSingleInstance(ppi_graph,obj,get_enumerated_features=True)
        instance_list.append(Single_instance)
    json_dump = dict()
    for instance in instance_list:
        feature_vector = instance[0]
        protein = instance[1]
        json_dump[protein] = feature_vector
    # print(json_dump)
    with open('node_features'+str(number)+'_'+phenotype_name+'.json', 'w') as f:
        json.dump(json_dump, f)
