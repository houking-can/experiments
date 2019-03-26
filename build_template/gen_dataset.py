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
    text = re.sub('[-|\\\\|\{|\}|\(|\)|\[|\]|\"|\'|/]', ' ', text)
    text = re.sub('e\s*\.g\s*\.\s*,', ' e.g., ', text)
    text = re.sub('e\s*\.g\s*\.\s*', ' e.g., ', text)
    text = re.sub('etc\s*\.', ' etc. ', text)
    text = re.sub('et\s*al\s*\.\s*,', ' et al., ', text)
    text = re.sub('i\s*\.e\s*\.\s*,',' i.e., ',text)
    text = re.sub('[;|:]', ' . \n ', text)
    text = re.sub(',[\s|,]*,',' , ',text)

    text = text.split('\n')
    text = [word_tokenize(each) for each in text]
    return text


def check_error(path):
    """check sentences and template is match in length"""
    files = iter_files(path)
    for file in files:

        paper = json.load(open(file))
        abstract = paper['abstract_text']
        template = paper['abstract_template']
        # print(os.path.basename(file))
        # print(len(abstract),len(template))
        wrong = []
        for i in range(len(abstract)):

            if len(abstract[i]) != len(template[i]):
                wrong.append(i)
        if len(wrong) > 0:
            print("%s wrong label: %s !" % (os.path.basename(file), ' '.join([str(each) for each in wrong])))
        tmp = paper['label'].split()
        if len(abstract)!=len(tmp):
            print("%s skip label" % os.path.basename(file))
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


def write_dataset(files, filename):
    if not os.path.exists('./data/'):
        os.makedirs('./data/')
    with open('./data/' + filename, 'w') as f:
        for file in files:
            paper = json.load(open(file))
            abstract = get_sentences(paper['abstract_text'])
            introduction = get_sentences(paper['introduction'])
            template = get_sentences(paper['abstract_template'])
            paper['abstract_text'] = abstract
            paper['introduction'] = introduction
            paper['abstract_template'] = template
            tmp = json.dumps(paper)
            f.write(tmp + '\n')
            f.flush()


def split_data(path, ratio=0.9):
    """generate dataset train eval test"""
    files = list(iter_files(path))
    random.shuffle(files)
    train_num = int(len(files) * ratio)
    eval_num = int(len(files) * (1 - ratio) / 2)
    test_num = len(files) - train_num - eval_num

    write_dataset(files[:train_num], 'train.txt')
    write_dataset(files[train_num:train_num + eval_num], 'dev.txt')
    write_dataset(files[-test_num:], 'test.txt')

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
            paper['abstract_template'] = template
            tmp = json.dumps(paper)
            f.write(tmp + '\n')
            f.flush()
    with open('./data/words_%d' % len(files), 'w') as f:
        for file in files:
            paper = json.load(open(file))
            tmp = json.dumps(paper)
            f.write(tmp + '\n')
            f.flush()
    print('combine data done!')


def modify_save_template(path):
    files = list(iter_files(path))
    samples_remove = ['\"', '\'', '\\', '/', '[', ']', '{', '}', '(', ')']
    for file in files:
        paper = json.load(open(file))
        abstract = paper['abstract_text']
        template = paper['abstract_template']

        for i in range(len(abstract)):
            j = 0
            while j < len(abstract[i]):
                if abstract[i][j] in samples_remove:
                    abstract[i].pop(j)
                    template[i].pop(j)
                    continue
                j += 1

            j = 0
            while j < len(abstract[i]):
                if '-' in abstract[i][j]:
                    tmp = abstract[i][j].split('-')
                    tmp = tmp[::-1]
                    abstract[i].pop(j)
                    for each in tmp:
                        abstract[i].insert(j, each)
                    if template[i][j] == "XXX":
                        for _ in range(len(tmp) - 1):
                            template[i].insert(j, "XXX")
                    else:
                        template[i].pop(j)
                        for each in tmp:
                            template[i].insert(j, each)
                j += 1
            assert len(abstract[i]) == len(template[i])

        paper['abstract_text'] = abstract
        paper['abstract_template'] = template
        introduction = '\n'.join(get_sentences(paper['introduction']))
        paper['introduction'] = remove_samples(introduction)
        json.dump(paper,open(file,'w'))
    print('modify template done!')

def modify_save_introduction(path):
    files = list(iter_files(path))
    for file in files:
        paper = json.load(open(file))
        # introduction =
        name = os.path.basename(file)
        tmp = json.load(open('./v1/' + name))['introduction']
        paper['introduction'] = tmp
        json.dump(paper, open(file, 'w'))
    print("modify introduction done!")

if __name__ == "__main__":
    save_path = './save'
    clean_path = './v1'
    # clean(clean_path)
    # modify_save_introduction(save_path)
    # check_error(save_path)

    # check_error(save_path)
    # split_data(save_path)
    combine_data(r'C:\Users\Houking\Desktop\experiment_data\old\clean1')
