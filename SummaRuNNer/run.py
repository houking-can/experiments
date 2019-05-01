#!/usr/bin/env python3

import argparse
import json
import logging
import numpy
import os
import random
import re
import shutil
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
import models
import utils
from utils import eval_rouge

logging.basicConfig(level=logging.INFO, format='%(asctime)s [INFO] %(message)s')
parser = argparse.ArgumentParser(description='extractive summary')
# model
parser.add_argument('-save_dir', type=str)
parser.add_argument('-embed_dim', type=int, default=128)
parser.add_argument('-embed_num', type=int, default=50)
parser.add_argument('-pos_dim', type=int, default=50)
parser.add_argument('-pos_num', type=int, default=70)
parser.add_argument('-seg_num', type=int, default=10)
parser.add_argument('-kernel_num', type=int, default=100)
parser.add_argument('-kernel_sizes', type=str, default='3,4,5')
parser.add_argument('-model', type=str, default='RNN_RNN')
parser.add_argument('-hidden_size', type=int, default=200)
# train
parser.add_argument('-lr', type=float, default=1e-3)
parser.add_argument('-batch_size', type=int, default=64)
parser.add_argument('-epochs', type=int, default=3)
parser.add_argument('-seed', type=int, default=1)
parser.add_argument('-train_dir', type=str)
parser.add_argument('-val_dir', type=str)
parser.add_argument('-embedding', type=str, default='data/embedding.npz')
parser.add_argument('-word2id', type=str, default='data/word2id.json')
parser.add_argument('-report_every', type=int, default=120)
parser.add_argument('-seq_trunc', type=int, default=50)
parser.add_argument('-max_norm', type=float, default=1.0)
# test
parser.add_argument('-load_dir', type=str)
parser.add_argument('-test_dir', type=str)
parser.add_argument('-ref', type=str)
parser.add_argument('-dec', type=str)
parser.add_argument('-topk', type=int, default=5)
# device
parser.add_argument('-device', type=int, required=True)
# option
parser.add_argument('-test', action='store_true')
parser.add_argument('-debug', default=True)
parser.add_argument('-predict', action='store_true')
parser.add_argument('-task', type=str, required=True)
parser.add_argument('-outputs_dir', type=str)
parser.add_argument('-mode', type=str)
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


def update_config():
    try:
        dataset, name, mode = args.task.split('_')
        if dataset not in ['km', 'he'] or mode not in ['train', 'test']:
            print("Please specify task name: he_baseline_train or km_embedding_test")
            exit()
    except:
        print("Please specify task name: he_baseline_train or km_embedding_test")

    args.outputs_dir = os.path.join('outputs', "%s_%s" % (dataset, name))
    args.save_dir = os.path.join('checkpoints', "%s_%s" % (dataset, name))
    args.train_dir = os.path.join('data', dataset, 'train.json')
    args.test_dir = os.path.join('data', dataset, 'test.json')
    args.val_dir = os.path.join('data', dataset, 'val.json')
    args.load_dir = os.path.join(args.save_dir, 'RNN_RNN_seed_1.pt')
    args.ref = os.path.join(args.outputs_dir, 'ref')
    args.dec = os.path.join(args.outputs_dir, 'dec')
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.mode = mode


def eval(net, vocab, data_iter, criterion):
    net.eval()
    total_loss = 0
    batch_num = 0
    for batch in data_iter:
        features, targets, _, doc_lens = vocab.make_features(batch)
        features, targets = Variable(features), Variable(targets.float())
        if use_gpu:
            features = features.cuda()
            targets = targets.cuda()
        probs = net(features, doc_lens)
        loss = criterion(probs, targets)
        total_loss += loss.item()
        batch_num += 1
    loss = total_loss / batch_num
    net.train()
    return loss


def train():
    logging.info('Loading vocab,train and val dataset.Wait a second,please')
    embed = torch.Tensor(np.load(args.embedding)['embedding'])
    with open(args.word2id) as f:
        word2id = json.load(f)
    vocab = utils.Vocab(embed, word2id)

    with open(args.train_dir) as f:
        examples = [json.loads(line) for line in f]
    train_dataset = utils.Dataset(examples)

    with open(args.val_dir) as f:
        examples = [json.loads(line) for line in f]
    val_dataset = utils.Dataset(examples)

    # update args
    args.embed_num = embed.size(0)
    args.embed_dim = embed.size(1)
    args.kernel_sizes = [int(ks) for ks in args.kernel_sizes.split(',')]
    # build model
    if os.path.exists(args.load_dir):
        if use_gpu:
            checkpoint = torch.load(args.load_dir)
        else:
            checkpoint = torch.load(args.load_dir, map_location=lambda storage, loc: storage)
        net = getattr(models, checkpoint['args'].model)(checkpoint['args'])
        net.load_state_dict(checkpoint['model'])
        print("=> loaded checkpoint '{}' ".format(args.load_dir))
    else:
        net = getattr(models, args.model)(args, embed)
    if use_gpu:
        net.cuda()
    # load dataset
    train_iter = DataLoader(dataset=train_dataset,
                            batch_size=args.batch_size,
                            shuffle=True)
    val_iter = DataLoader(dataset=val_dataset,
                          batch_size=args.batch_size,
                          shuffle=False)
    # loss function
    criterion = nn.BCELoss()
    # model info
    print(net)
    params = sum(p.numel() for p in list(net.parameters())) / 1e6
    print('#Params: %.1fM' % (params))

    min_loss = float('inf')
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    net.train()

    t1 = time.time()
    for epoch in range(1, args.epochs + 1):
        for i, batch in enumerate(train_iter):
            try:
                features, targets, _, doc_lens = vocab.make_features(batch)
                features, targets = Variable(features), Variable(targets.float())
                if use_gpu:
                    features = features.cuda()
                    targets = targets.cuda()
                probs = net(features, doc_lens)
                loss = criterion(probs, targets)
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(net.parameters(), args.max_norm)
                optimizer.step()
                if args.debug:
                    print('Batch ID:%d Loss:%f' % (i, loss.item()))
                    # continue
                if i % args.report_every == 0 :
                    cur_loss = eval(net, vocab, val_iter, criterion)
                    if cur_loss < min_loss:
                        min_loss = cur_loss
                        best_path = net.save()
                    logging.info('Epoch: %2d Min_Val_Loss: %f Cur_Val_Loss: %f'
                                 % (epoch, min_loss, cur_loss))
            except:
                print('pass a batch')
    t2 = time.time()
    logging.info('Total Cost:%f h' % ((t2 - t1) / 3600))


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
    net = getattr(models, checkpoint['args'].model)(checkpoint['args'])
    net.load_state_dict(checkpoint['model'])
    if use_gpu:
        net.cuda()
    net.eval()

    doc_num = len(test_dataset)
    time_cost = 0
    file_id = 1
    write_ref = True
    if len(os.listdir(args.ref))>0:
        write_ref = False
    for batch in tqdm(test_iter):
        features, _, summaries, doc_lens = vocab.make_features(batch)
        t1 = time.time()
        if use_gpu:
            probs = net(Variable(features).cuda(), doc_lens)
        else:
            probs = net(Variable(features), doc_lens)
        t2 = time.time()
        time_cost += t2 - t1
        start = 0
        for doc_id, doc_len in enumerate(doc_lens):
            stop = start + doc_len
            prob = probs[start:stop]
            # tmp = summaries[doc_id].strip('\n')
            # topk = len(tmp.split('\n'))
            # topk = min(topk, doc_len)
            topk = min(args.topk, doc_len)
            topk_indices = prob.topk(topk)[1].cpu().data.numpy()
            topk_indices.sort()
            doc = batch['doc'][doc_id].split('\n')[:doc_len]
            dec = [doc[index] for index in topk_indices]
            if write_ref:
                ref = summaries[doc_id]
                with open(os.path.join(args.ref, str(file_id) + '.ref'), 'w') as f:
                    f.write(ref)
            with open(os.path.join(args.dec, str(file_id) + '.dec'), 'w') as f:
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
    with open(os.path.join(args.outputs_dir,'rouge_%d_%f.txt' % (args.topk, time.time())), 'w') as f:
        f.write(output)
    return output


if __name__ == '__main__':
    update_config()
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        last_time = None
        best_score = 0

        best_path = os.path.join(args.save_dir, 'Best_RNN_RNN_seed_1.pt')
        if os.path.exists(best_path):
            test()
            output = rouge()
            F = re.findall('Average_F:\s*(0\.\d{5})', output)
            f_score = 0
            for each in F: f_score += float(each)
            best_score = f_score / 3
            print("Exist Best Model:")
            print(output)

        while True:
            if not os.path.exists(args.load_dir):
                time.sleep(30)
            else:
                cur_time = os.stat(args.load_dir).st_mtime
                if cur_time != last_time:
                    test()
                    output = rouge()
                    F = re.findall('Average_F:\s*(0\.\d{5})', output)
                    f_score = 0
                    for each in F: f_score += float(each)
                    if f_score / 3 > best_score:
                        shutil.copy(args.load_dir, best_path)
                        best_score = f_score / 3
                        print(output)
                    else:
                        print('Not the best!')
                    last_time = cur_time

                else:
                    time.sleep(30)
