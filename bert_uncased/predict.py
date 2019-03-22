import os
import json
import random
import tensorflow as tf
import tokenization

tokenizer = tokenization.FullTokenizer(
        vocab_file='./vocab.txt', do_lower_case=True)

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


# def shuffle_split(name, ratio=0.8):
#     with open(name) as f:
#         lines = f.readlines()
#         random.shuffle(lines)
#         len_train = int(len(lines) * ratio)
#         len_test = (len(lines) - len_train)//2
#
#     with open('./data/train.txt', 'w') as x:
#         x.write('\n'.join(lines[:len_train]))
#     with open('./data/dev.txt', 'w') as x:
#         x.write('\n'.join(lines[len_train:len_train+len_test]))
#     with open('./data/test.txt', 'w') as x:
#         x.write('\n'.join(lines[len_train+len_test:]))


def predict_sentences(name):
    category = ['background', 'problem', 'objective', 'method', 'result', 'conclusion', 'other']
    abstracts = []
    with open('./data/dev.txt') as f:
        lines = f.readlines()
        for line in lines:
            if len(line) > 1:
                tmp = json.loads(line)
                abstracts.extend(tmp['abstract_text'])

    # print(len(abstracts))
    with open(name) as f:
        results = f.readlines()
        # print(len(results))

        for i in range(len(results)):
            print(abstracts[i])
            tmp = results[i].split()
            x = [float(each) for each in tmp]
            print(category[x.index(max(x))])


def predict_template(name):
    abstracts = []

    with open('./data/test.txt') as f:
        lines = f.readlines()
        for line in lines:
            if len(line) > 1:
                tmp = json.loads(line)
                abstracts.extend(tmp['abstract_text'])
    with open(name) as f:
        results = f.readlines()
        for i in range(len(results)):

            print('')
            tmp = results[i].split()
            res = ''
            words = tokenizer.tokenize(abstracts[i])
            print(' '.join(words))
            for j in range(1,min(len(words),len(tmp))):
                if tmp[j] == '1':
                    res += words[j-1]+' '
                else:
                    res += '*'+' '
            print(res)
            print('')

if __name__ == '__main__':
    # shuffle_split('ijcai_110')
    # predict_sentences('./output/test_results.tsv')
    predict_template('./test_template_results.tsv')
    # pass
