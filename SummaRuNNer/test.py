#!/usr/bin/env python3

import json
import models
import utils
import argparse,random,logging,numpy,os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from time import time
from tqdm import tqdm
from utils import eval_rouge
import time
import threading
import re
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')
parser = argparse.ArgumentParser(description='extractive summary')
# model
parser.add_argument('-save_dir',type=str,default='checkpoints/')
parser.add_argument('-embed_dim',type=int,default=128)
parser.add_argument('-embed_num',type=int,default=50)
parser.add_argument('-pos_dim',type=int,default=50)
parser.add_argument('-pos_num',type=int,default=70)
parser.add_argument('-seg_num',type=int,default=10)
parser.add_argument('-kernel_num',type=int,default=100)
parser.add_argument('-kernel_sizes',type=str,default='3,4,5')
parser.add_argument('-model',type=str,default='RNN_RNN')
parser.add_argument('-hidden_size',type=int,default=200)
# train
parser.add_argument('-lr',type=float,default=1e-3)
parser.add_argument('-batch_size',type=int,default=64)
parser.add_argument('-epochs',type=int,default=5)
parser.add_argument('-seed',type=int,default=1)
parser.add_argument('-train_dir',type=str,default='data/train.json')
parser.add_argument('-val_dir',type=str,default='data/val.json')
parser.add_argument('-embedding',type=str,default='data/embedding.npz')
parser.add_argument('-word2id',type=str,default='data/word2id.json')
parser.add_argument('-report_every',type=int,default=100)
parser.add_argument('-seq_trunc',type=int,default=50)
parser.add_argument('-max_norm',type=float,default=1.0)
# test
parser.add_argument('-load_dir',type=str,default='checkpoints/RNN_RNN_seed_1.pt')
parser.add_argument('-test_dir',type=str,default='data/int/test.json')
parser.add_argument('-ref',type=str,default='outputs/refs')
parser.add_argument('-dec',type=str,default='outputs/dec')
parser.add_argument('-topk',type=int,default=5)
# device
parser.add_argument('-device',type=int,default=4)
# option
parser.add_argument('-test',action='store_true')
parser.add_argument('-debug',default=True)
parser.add_argument('-predict',action='store_true')
args = parser.parse_args()
use_gpu = args.device is not None

if torch.cuda.is_available() and not use_gpu:
    print("WARNING: You have a CUDA device, should run with -device 0")

# set cuda device and seed
if use_gpu:
    torch.cuda.set_device(args.device)
torch.cuda.manual_seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
numpy.random.seed(args.seed) 

def eval(net,vocab,data_iter,criterion):
    net.eval()
    total_loss = 0
    batch_num = 0
    for batch in data_iter:
        features,targets,_,doc_lens = vocab.make_features(batch)
        features,targets = Variable(features), Variable(targets.float())
        if use_gpu:
            features = features.cuda()
            targets = targets.cuda()
        probs = net(features,doc_lens)
        loss = criterion(probs,targets)
        total_loss += loss.item()
        batch_num += 1
    loss = total_loss / batch_num
    net.train()
    return loss

def test():
     
    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    with open(args.word2id) as f:
        word2id = json.load(f)
    vocab = utils.Vocab(embed, word2id)

    if not os.path.exists(args.dec):
        os.makedirs(args.dec)
    if not os.path.exists(args.ref):
        os.makedirs(args.ref)

    with open(args.test_dir) as f:
        examples = [json.loads(line) for line in f]
    test_dataset = utils.Dataset(examples)

    test_iter = DataLoader(dataset=test_dataset,
                            batch_size=args.batch_size,
                            shuffle=False)
    if use_gpu:
        checkpoint = torch.load(args.load_dir)
    else:
        checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)

    # checkpoint['args']['device'] saves the device used as train time
    # if at test time, we are using a CPU, we must override device to None
    if not use_gpu:
        checkpoint['args'].device = None
    net = getattr(models,checkpoint['args'].model)(checkpoint['args'])
    net.load_state_dict(checkpoint['model'])
    if use_gpu:
        net.cuda()
    net.eval()
    
    doc_num = len(test_dataset)
    time_cost = 0
    file_id = 1
    for batch in tqdm(test_iter):
        features,_,summaries,doc_lens = vocab.make_features(batch)
        t1 = time.time()
        if use_gpu:
            probs = net(Variable(features).cuda(), doc_lens)
        else:
            probs = net(Variable(features), doc_lens)
        t2 = time.time()
        time_cost += t2 - t1
        start = 0
        for doc_id,doc_len in enumerate(doc_lens):
            stop = start + doc_len
            prob = probs[start:stop]
            tmp = summaries[doc_id].strip('\n')
            topk = len(tmp.split('\n'))
            topk = min(topk,doc_len)
            topk_indices = prob.topk(topk)[1].cpu().data.numpy()
            topk_indices.sort()
            doc = batch['doc'][doc_id].split('\n')[:doc_len]
            dec = [doc[index] for index in topk_indices]
            ref = summaries[doc_id]
            with open(os.path.join(args.ref,str(file_id)+'.ref'), 'w') as f:
                f.write(ref)
            with open(os.path.join(args.dec,str(file_id)+'.dec'), 'w') as f:
                f.write('\n'.join(dec))
            start = stop
            file_id = file_id + 1

    print('Speed: %.2f docs / s' % (doc_num / time_cost))
    print('')


def rouge():
    dec_pattern = r'(\d+).dec'
    ref_pattern = '#ID#.ref'
    output = eval_rouge(dec_pattern, args.dec, ref_pattern, args.ref)
    # print(output)
    with open('outputs/rouge_%d_%f.txt' % (args.topk,time.time()), 'w') as f:
        f.write(output)
    return output


if __name__=='__main__':
    last_time = None
    best_score = 0
    if not os.path.exists('./checkpoints/'):
        os.makedirs('./checkpoints/')

    if os.path.exists('./checkpoints/Best_RNN_RNN_seed_1.pt'):

        test()
        output = rouge()
        F = re.findall('Average_F:\s*(0\.\d{5})', output)
        f_score = 0
        for each in F: f_score += float(each)
        best_score = f_score/3
        print("Exist Best Model:")
        print(output)

    while True:
        if not os.path.exists(args.load_dir):
            time.sleep(30)
        else:
            cur_time = os.stat(args.load_dir).st_mtime
            if cur_time!=last_time:
                test()
                output = rouge()
                F = re.findall('Average_F:\s*(0\.\d{5})',output)
                f_score = 0
                for each in F:f_score += float(each)
                if f_score/3>best_score:
                    shutil.copy(args.load_dir,"./checkpoints/Best_RNN_RNN_seed_1.pt")
                    best_score = f_score/3
                    print(output)
                else:
                    print('Not the best!')
                last_time = cur_time

            else:
                time.sleep(30)


