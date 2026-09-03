from pathlib import Path
from argparse import ArgumentParser
import pickle
import pandas as pd

def make_parser():
    parser = ArgumentParser()
    parser.add_argument('gene2go', type=Path)
    parser.add_argument('first', type=int)
    parser.add_argument('last', type=int)
    parser.add_argument('out', type=Path)
    return parser

def main(args):
    gene2go = pickle.load(args.gene2go.open('rb'))
    targetlist = list(gene2go.items())
    sourcelist = targetlist[args.first:args.last]
    jaccards = [
        (src, tgt, len(a & b) / len(a | b))
        for src, a in sourcelist
        for tgt, b in targetlist
        if src != tgt and a & b
    ]
    pd.DataFrame(jaccards).to_csv(args.out, header=False, index=False)

if __name__ == '__main__':
    main(make_parser().parse_args())