import shutil
import random
import collections
import io
import pickle as pkl
import tarfile
import json
from tqdm import tqdm
import csv
import torch
import torch.utils.data as data
from torch.autograd import Variable
import numpy as np
import gensim
from torch import nn
import re
import os
import logging
import tempfile
import subprocess as sp
from cytoolz import curry
from pyrouge import Rouge155
from pyrouge.utils import log


class Vocab():
    def __init__(self, embed, word2id):
        self.embed = embed
        self.word2id = word2id
        self.id2word = {v: k for k, v in word2id.items()}
        assert len(self.word2id) == len(self.id2word)
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        self.PAD_TOKEN = 'PAD_TOKEN'
        self.UNK_TOKEN = 'UNK_TOKEN'

    def __len__(self):
        return len(self.word2id)

    def i2w(self, idx):
        return self.id2word[idx]

    def w2i(self, w):
        if w in self.word2id:
            return self.word2id[w]
        else:
            return self.UNK_IDX

    def make_features(self, batch, sent_trunc=50, doc_trunc=100, split_token='\n'):
        sents_list, targets, doc_lens = [], [], []
        # trunc document
        for doc, label in zip(batch['doc'], batch['labels']):
            sents = doc.split(split_token)
            labels = label.split(split_token)
            labels = [int(l) for l in labels]
            max_sent_num = min(doc_trunc, len(sents))
            sents = sents[:max_sent_num]
            labels = labels[:max_sent_num]
            sents_list += sents
            targets += labels
            doc_lens.append(len(sents))
        # trunc or pad sent
        max_sent_len = 0
        batch_sents = []
        for sent in sents_list:
            words = sent.split()
            if len(words) > sent_trunc:
                words = words[:sent_trunc]
            max_sent_len = len(words) if len(words) > max_sent_len else max_sent_len
            batch_sents.append(words)

        features = []
        for sent in batch_sents:
            feature = [self.w2i(w) for w in sent] + [self.PAD_IDX for _ in range(max_sent_len - len(sent))]
            features.append(feature)

        features = torch.LongTensor(features)
        targets = torch.LongTensor(targets)
        summaries = batch['summaries']

        return features, targets, summaries, doc_lens


class Dataset(data.Dataset):
    def __init__(self, examples):
        super(Dataset, self).__init__()
        # data: {'sents':xxxx,'labels':'xxxx', 'summaries':[1,0]}
        self.examples = examples
        self.training = False

    def train(self):
        self.training = True
        return self

    def test(self):
        self.training = False
        return self

    def shuffle(self, words):
        np.random.shuffle(words)
        return ' '.join(words)

    def dropout(self, words, p=0.3):
        l = len(words)
        drop_index = np.random.choice(l, int(l * p))
        keep_words = [words[i] for i in range(l) if i not in drop_index]
        return ' '.join(keep_words)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return ex
        # words = ex['sents'].split()
        # guess = np.random.random()

        # if self.training:
        #    if guess > 0.5:
        #        sents = self.dropout(words,p=0.3)
        #    else:
        #        sents = self.shuffle(words)
        # else:
        #    sents = ex['sents']
        # return {'id':ex['id'],'sents':sents,'labels':ex['labels']}

    def __len__(self):
        return len(self.examples)

def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)




def eval_rouge(dec_pattern, dec_dir, ref_pattern, ref_dir,
               cmd='-c 95 -r 1000 -n 2 -m', system_id=1):
    """ evaluate by original Perl implementation"""
    # silence pyrouge logging
    try:
        _ROUGE_PATH = '/home/yhj/ROUGE/RELEASE-1.5.5'
    except KeyError:
        print('Warning: ROUGE is not configured')
        _ROUGE_PATH = None

    assert _ROUGE_PATH is not None
    log.get_global_console_logger().setLevel(logging.WARNING)
    with tempfile.TemporaryDirectory() as tmp_dir:
        Rouge155.convert_summaries_to_rouge_format(
            dec_dir, os.path.join(tmp_dir, 'dec'))
        Rouge155.convert_summaries_to_rouge_format(
            ref_dir, os.path.join(tmp_dir, 'ref'))
        Rouge155.write_config_static(
            os.path.join(tmp_dir, 'dec'), dec_pattern,
            os.path.join(tmp_dir, 'ref'), ref_pattern,
            os.path.join(tmp_dir, 'settings.xml'), system_id
        )
        cmd = (os.path.join(_ROUGE_PATH, 'ROUGE-1.5.5.pl')
               + ' -e {} '.format(os.path.join(_ROUGE_PATH, 'data'))
               + cmd
               + ' -a {}'.format(os.path.join(tmp_dir, 'settings.xml')))
        output = sp.check_output(cmd.split(' '), universal_newlines=True)
    return output


def minus(src, des):
    """src: path of
       des: path of remove items
    """
    src_items = []
    des_items = []
    old = os.path.dirname(des) + '_OLD'

    for file in os.listdir(src):
        file, src_ext = os.path.splitext(os.path.basename(file))
        src_items.append(file)
    for file in os.listdir(des):
        file, des_ext = os.path.splitext(os.path.basename(file))
        des_items.append(file)

    src_items = set(src_items)
    des_items = set(des_items)

    move_items = des_items - src_items
    move_items = des_items - move_items
    print(len(move_items))
    if not os.path.exists(old):
        os.makedirs(old)

    for item in move_items:
        print(item)
        try:
            shutil.move(os.path.join(os.path.dirname(des), "%s%s" % (item, des_ext)), old)
        except Exception as e:
            print(e)

def split(src,des,ratio=0.94):
    files = list(iter_files(src))
    random.shuffle(files)
    len_train = int(len(files)*ratio)
    len_val =  int(len(files)*(1-ratio)/2)
    len_test = len(files)-len_train-len_val
    train = files[:len_train]
    val = files[len_train:len_train+len_val]
    test = files[-len_test:]

    train_dir = os.path.join(des,'train')
    test_dir = os.path.join(des, 'test')
    val_dir = os.path.join(des, 'val')
    if not os.path.exists(train_dir):os.makedirs(train_dir)
    if not os.path.exists(test_dir): os.makedirs(test_dir)
    if not os.path.exists(val_dir): os.makedirs(val_dir)
    for each in train:
        shutil.copy(each,train_dir)
    for each in test:
        shutil.copy(each,test_dir)
    for each in val:
        shutil.copy(each,val_dir)


def make_vocab(input, output):

    vocab_counter = collections.Counter()

    files = list(iter_files(input))

    for file in tqdm(files):
        paper = json.load(open(file))
        art_tokens = ' '.join(paper['article']).split()
        abs_tokens = ' '.join(paper['abstract']).split()
        con_tokens = ' '.join(paper['conclusion']).split()
        tokens = art_tokens + abs_tokens + con_tokens
        tokens = [t.strip() for t in tokens]  # strip
        tokens = [t for t in tokens if t != ""]  # remove empty
        vocab_counter.update(tokens)
    for each in ['<unk>','<pad>','<start>','<end>']:
        if each in vocab_counter:
            vocab_counter.pop(each)
    print("Writing vocab file...")
    with open(os.path.join(output, "vocab_cnt.pkl"),
              'wb') as vocab_file:
        pkl.dump(vocab_counter, vocab_file)
    print("Finished writing vocab file")


