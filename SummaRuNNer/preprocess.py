#!/usr/bin/env python3

import argparse
import json
import numpy as np
from collections import OrderedDict
from glob import glob
from time import time
from multiprocessing import Pool,cpu_count
from itertools import chain
from utils import iter_files
def build_vocab(args):
    print('start building vocab')

    PAD_IDX = 0
    UNK_IDX = 1
    PAD_TOKEN = 'PAD_TOKEN'
    UNK_TOKEN = 'UNK_TOKEN'
    
    f = open(args.embed)
    embed_dim = int(next(f).split()[1])

    word2id = OrderedDict()
    
    word2id[PAD_TOKEN] = PAD_IDX
    word2id[UNK_TOKEN] = UNK_IDX
    
    embed_list = []
    # fill PAD and UNK vector
    embed_list.append([0 for _ in range(embed_dim)])
    embed_list.append([0 for _ in range(embed_dim)])
    
    # build Vocab
    for line in f:
        tokens = line.split()
        word = tokens[:-1*embed_dim][0]
        vector = [float(num) for num in tokens[-1*embed_dim:]]
        embed_list.append(vector)
        word2id[word] = len(word2id)
    f.close()
    embed = np.array(embed_list,dtype=np.float32)
    np.savez_compressed(file=args.vocab, embedding=embed)
    with open(args.word2id,'w') as f:
        json.dump(word2id,f)

def worker(files):
    examples = []
    for f in files:
        paper= json.load(open(f))
        labels = ['0' for _ in range(len(paper['article']))]
        tmp = paper['extracted']
        for i in tmp:labels[i]='1'
        ex = {'doc':'\n'.join(paper['article']),'labels':'\n'.join(labels),'summaries':'\n'.join(paper['abstract'])}
        examples.append(ex)
    return examples

def build_dataset(args):
    t1 = time()
    
    print('start building dataset')
    if args.worker_num == 1 and cpu_count() > 1:
        print('[INFO] There are %d CPUs in your device, please increase -worker_num to speed up' % (cpu_count()))
        print("       It's a IO intensive application, so 2~10 may be a good choise")

    files = list(iter_files(args.source_dir))
    data_num = len(files)
    group_size = data_num // args.worker_num
    groups = []
    for i in range(args.worker_num):
        if i == args.worker_num - 1:
            groups.append(files[i*group_size : ])
        else:
            groups.append(files[i*group_size : (i+1)*group_size])
    p = Pool(processes=args.worker_num)
    multi_res = [p.apply_async(worker,(fs,)) for fs in groups]
    res = [res.get() for res in multi_res]
    
    with open(args.target_dir, 'w') as f:
        for row in chain(*res):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    t2 = time()
    print('Time Cost : %.1f seconds' % (t2 - t1))
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-build_vocab',default=False)
    parser.add_argument('-embed', type=str, default='/home/yhj/dataset/emnlp/word2vec.128d.72k.w2v')
    parser.add_argument('-vocab', type=str, default='data/embedding.npz')
    parser.add_argument('-word2id',type=str,default='data/word2id.json')

    parser.add_argument('-worker_num',type=int,default=8)
    parser.add_argument('-source_dir', type=str, default='/home/yhj/dataset/emnlp/val')
    parser.add_argument('-target_dir', type=str, default='data/val.json')

    args = parser.parse_args()
    
    if args.build_vocab:
        build_vocab(args)
    else:
        build_dataset(args)
