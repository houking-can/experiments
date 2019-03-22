import os
import random
import json
import re
from nltk.tokenize import sent_tokenize, word_tokenize


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


def get_sentences(text):
    return [' '.join(each) for each in text]


def remove_samples(text):
    text = re.sub('[-|\\\\|{|}|(|)|\[|\]|\"|\'|/]', ' ', text)
    text = re.sub('e.g\s*.', ' e.g., ', text)
    text = re.sub('etc\s*.', ' etc.', text)
    text = re.sub('et\s*al\s*.\s*,', ' et al., ', text)
    text = re.sub('[;|:]', ' . \n ', text)
    text = sent_tokenize(text)
    text = [word_tokenize(each) for each in text]
    return text


def check_error(path):
    """check sentences and template is match in length"""
    files = iter_files(path)
    for file in files:
        paper = json.load(open(file))
        abstract = paper['abstract_text']
        template = paper['abstract_template']

        for i in range(len(abstract)):
            wrong = []
            if len(abstract[i]) != len(template[i]):
                wrong.append(i)
        if len(wrong) > 0:
            print("%s wrong label: %s !" % (os.path.basename(file), ' '.join(wrong)))
    print("check error done!")


def clean(path):
    """remove some samples"""
    # samples_remove = ['-','\"','\'','\\','/','[',']','{','}','(',')']
    # samples_repalce = [';']  #replace to .\n
    files = iter_files(path)
    for file in files:
        paper = json.load(open(file))
        abstract = '\n'.join(get_sentences(paper['abstract_text']))
        paper['abstract_text'] = remove_samples(abstract)
        introduction = '\n'.join(get_sentences(paper['introduction']))
        paper['introduction'] = remove_samples(introduction)
        json.dump(paper, open(file, 'w'))
    print('clean done!')


def write_dataset(files,filename):
    if not os.path.exists('./data/'):
        os.makedirs('./data/')
    with open('./data/'+filename,'w') as f:
        for file in files:
            paper = json.load(open(file))
            tmp = json.dumps(paper)
            f.write(tmp+'\n')
            f.flush()

def split_data(path, ratio=0.9):
    """generate dataset train eval test"""
    files = list(iter_files(path))
    random.shuffle(files)
    train_num = int(len(files) * ratio)
    eval_num = int(len(files) * (1 - ratio) / 2)
    test_num = len(files) - train_num - eval_num

    write_dataset(files[:train_num],'train.txt')
    write_dataset(files[train_num:train_num + eval_num],'dev.txt')
    write_dataset(files[-test_num:],'test.txt')

    print("split data done!")

def combine_data(path):
    files = list(iter_files(path))
    if not os.path.exists('./data/'):
        os.makedirs('./data/')
    with open('./data/sentences_%d' % len(files), 'w') as f:
        for file in files:
            paper = json.load(open(file))
            abstract = get_sentences(paper['abstract_text'])
            introduction = get_sentences(paper['introduction'])
            template = get_sentences(paper['abstract_template'])
            paper['abstract_text'] = abstract
            paper['introduction'] = introduction
            paper['abstract_template'] =template
            tmp = json.dumps(paper)
            f.write(tmp + '\n')
            f.flush()
    with open('./data/words_%d' % len(files),'w') as f:
        for file in files:
            paper = json.load(open(file))
            tmp = json.dumps(paper)
            f.write(tmp + '\n')
            f.flush()
    print('combine data done!')

if __name__ == "__main__":
    save_path = './save'
    clean_path = './v1'
    # clean(clean_path)
    check_error(save_path)
    split_data(save_path)
    combine_data(save_path)
