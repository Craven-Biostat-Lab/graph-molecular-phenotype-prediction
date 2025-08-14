# GEARS-derived files

Files in this folder are derived from GEARS models;
[Paper link](https://www.nature.com/articles/s41587-023-01905-6);
[GitHub link](https://github.com/snap-stanford/GEARS).

## gene-go-embedding_2024-03-19.pt

Extracted on 2024-03-19 from the pretrained GEARS model used in the following notebook: https://colab.research.google.com/drive/11LlzGEUGoBk_Uj6DzlzizAeWse5_E9MK?usp=sharing#scrollTo=N91s3K5Kxaxm, using the dataloader from https://dataverse.harvard.edu/api/access/datafile/6979957 and the model from https://dataverse.harvard.edu/api/access/datafile/6979956.

Load using `torch.load`. The saved object is a dictionary with entries

 - `names`: A python list of perturbation (gene) names
 - `embedding_state_dict`: The `state_dict` of a `torch.nn.Embedding` layer.

The index of a gene in `names` is used to index into the embedding layer

## gene2go.pkl

A pickle file with the GO terms for genes from the GEARS files, a dict of sets.

## jaccard_go_graph

Jaccard similarity graph derived from gene2go.pkl (with code).

- `jaccard-top5-graph_2024-03-27.csv.gz`: The three columns are source, target, and Jaccard similarity of the go sets. In this graph I only keep the top 5 targets for each source (like they did in the paper). Some sources get more than 5 targets when more than 5 genes are at least as similar as the top 5th.
