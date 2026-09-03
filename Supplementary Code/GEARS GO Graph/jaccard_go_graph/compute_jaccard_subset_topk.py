from pathlib import Path
from argparse import ArgumentParser
import heapq
import pickle
import gzip

def make_parser():
    parser = ArgumentParser()
    parser.add_argument('gene2go', type=Path)
    parser.add_argument('first', type=int)
    parser.add_argument('last', type=int)
    parser.add_argument('k', type=int)
    parser.add_argument('out', type=Path)
    return parser

def main(args):
    gene2go = pickle.load(args.gene2go.open('rb'))
    targetlist = list(gene2go.items())
    sourcelist = targetlist[args.first:args.last]

    with gzip.open(args.out, 'wt') as out_stream:
        for src, a in sourcelist:
            target_jaccards = list(
                (- len(a & b) / len(a | b), tgt)
                for tgt, b in targetlist
                if src != tgt and a & b
            )
            heapq.heapify(target_jaccards)
            counter = args.k
            last = None
            while target_jaccards:
                neg_score, tgt = heapq.heappop(target_jaccards)
                if (counter > 0) or ((last is not None) and (neg_score <= last)):
                    out_stream.write(f'{src}, {tgt}, {-neg_score}\n')
                    last = neg_score
                    counter -= 1
                else:
                    break

if __name__ == '__main__':
    main(make_parser().parse_args())