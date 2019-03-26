import json
from nltk.tag import StanfordPOSTagger
import os

def write_file(filename,string):
    if isinstance(string,str):
        with open(filename,'w',encoding='utf-8') as f:
            f.write(string)

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


def high_frequency_words(path):
    files = iter_files(path)
    JJ_extend = ['JJ', 'JJR', 'JJS']
    RB_extend = ['RB', 'RBR', 'RBS']
    NN_extend = ['NN', 'NNS', 'NNP', 'NNPS']
    VB_extend = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    JJ = dict()  # adj
    RB = dict()  # adv
    NN = dict()  # noun
    VB = dict()  # verb

    def add_to_dict(word, speech):
        if speech in JJ_extend:
            if word in JJ:
                JJ[word] += 1
            else:
                JJ[word] = 1
        elif speech in RB_extend:
            if word in RB:
                RB[word] += 1
            else:
                RB[word] = 1
        elif speech in NN_extend:
            if word in NN:
                NN[word] += 1
            else:
                NN[word] = 1
        elif speech in VB_extend:
            if word in VB:
                VB[word] += 1
            else:
                VB[word] = 1

    def process_paper(lines):
        for i, line in enumerate(lines):
            paper = json.loads(line)
            poses = paper['abstract_pos']
            abstracts = paper['abstract_text']
            for i in range(len(abstracts)):
                pos = poses[i].split()
                words = abstracts[i].split()
                for j, word in enumerate(words):
                    add_to_dict(word, pos[j])

    for file in files:
        name = os.path.basename(file)
        print(name)
        with open(file, encoding='utf-8') as fr:
            lines = fr.readlines()
            process_paper(lines)

    JJ = list(sorted(JJ.items(), key=lambda t: t[1], reverse=True))[:5000]
    RB = list(sorted(RB.items(), key=lambda t: t[1], reverse=True))[:5000]
    NN = list(sorted(NN.items(), key=lambda t: t[1], reverse=True))[:5000]
    VB = list(sorted(VB.items(), key=lambda t: t[1], reverse=True))[:5000]

    with open('./vocab/JJ.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join([tup[0] + '\t' + str(tup[1]) for tup in JJ]))
    with open('./vocab/RB.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join([tup[0] + '\t' + str(tup[1]) for tup in RB]))
    with open('./vocab/NN.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join([tup[0] + '\t' + str(tup[1]) for tup in NN]))
    with open('./vocab/VB.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join([tup[0] + '\t' + str(tup[1]) for tup in VB]))


def remove_words(lines):
    words = []
    for line in lines:
        line = line.split()
        if len(line) < 2: continue
        if len(line[0]) == 1 or '.' in line[0] or '/' in line[0] or '<' in line[0] or '\\' in line[0] \
                or '>' in line[0]: continue
        words.append((line[0], int(line[1])))
    return words


def gen_vocab(path):
    pass

    # def get_similar_words(word):
    #     if word in bert_words:
    #         index = bert_words.index(word)
    #         tmp = bert_words[max(0, index - 5):min(len(bert_words), index + 5)]
    #         for i in range(len(word), len(word) // 2, -1):
    #             cnt = 0
    #             res = []
    #             for each in tmp:
    #                 if each.startswith(word[:i]):
    #                     cnt += 1
    #                     res.append(each)
    #             if cnt > 3:
    #                 return res
    #     return []
    # with open('./vocab/JJ.txt', encoding='utf-8') as f:
    #     JJ = remove_words(f.readlines())
    # with open('./vocab/RB.txt', encoding='utf-8') as f:
    #     RB = remove_words(f.readlines())
    # with open('./vocab/NN.txt', encoding='utf-8') as f:
    #     NN = remove_words(f.readlines())
    # with open('./vocab/VB.txt', encoding='utf-8') as f:
    #     VB = remove_words(f.readlines())

    # files = iter_files(path)
    # tmp_vocab = set()
    # for file in files:
    #     paper = json.load(open(file))
    #     for sent in paper['abstract_template']:
    #         for word in sent:
    #             if word=='XXX':continue
    #             tmp_vocab.add(word.lower())

    # extend_words = open('./vocab/template_words.txt',encoding='utf-8').read()
    # extend_words = extend_words.split()
    # res = []
    # for word in extend_words:
    #     try:
    #         res.extend(get_similar_words(word))
    #     except Exception as e:
    #         print(e)
    #
    # vocab = set(res)
    # words = open('template_words.txt',encoding='utf-8').read()
    # words = words.split()
    # for word in JJ+RB+NN+VB:
    #     if word[0] in words:continue
    #     else:words.append(word[0])
    # with open('final.txt',encoding='utf-8') as f:
    #     words = f.read()
    #     words = words.split()
    #     print(len(words))
    # res = []
    # for word in words:
    #     if len(word)>3:
    #         res.append(word)
    #     else:
    #         print(word)

    # with open('final_2.txt',encoding='utf-8') as f:
    #     f.write('\n'.join(sorted(words)))


def get_vocab(path):
    words = open(path, encoding='utf-8').read()
    words =  words.split('\n')
    words.sort()
    # i = 0
    # while (i < len(bert_words)):
    #     if '#' in bert_words[i]:
    #         bert_words.pop(i)
    #     if all(ord(c) < 128 for c in bert_words[i]) and bert_words[i].isalpha():
    #         i += 1
    #     else:
    #         bert_words.pop(i)
    return words

def get_sentences(file):
    with open(file, encoding='utf-8') as f:
        for line in f.readlines():
            paper = json.loads(line)
            for i,each in enumerate(paper['abstract_text']):
                yield (each,paper["abstract_pos"][i])


def vocab_based(path):
    files = iter_files(path)
    res = ''
    for file in files:
        print(file)
        sentences = get_sentences(file)
        i = 0
        for sent in sentences:
            tmp = ''
            for word in sent[0].split():
                if word in vocab:
                    tmp+=word+' '
                else:tmp+=' XXX '
            res+=sent+'\n'+tmp+'\n\n'
            i+=1
            if i>1000:
                break
        break
    write_file('tmp.txt',res)

def standford_pos(text):
    eng_tagger = StanfordPOSTagger(
        model_filename=r'D:\Program Files\stanford-corenlp-full\stanford-postagger\models\english-bidirectional-distsim.tagger',
        path_to_jar=r'D:\Program Files\stanford-corenlp-full\stanford-postagger\stanford-postagger.jar')
    return eng_tagger.tag(text.split())


def pos_based(path):
    remain_speech = ["CC","DT","EX","MD","PRP","PRP$","RP","TO"]
    files = iter_files(path)
    res = ''
    for file in files:
        print(file)
        sentences = get_sentences(file)
        i = 0
        for sent in sentences:
            tmp = ''
            for id,word in enumerate(sent[0].split()):
                if word in vocab or sent[1][id] in remain_speech:
                    tmp += word + ' '
                else:
                    tmp += ' XXX '
            res += sent[0] + '\n' + tmp + '\n\n'
            # i += 1
            # if i > 1000:
            #     break
        break
    write_file('tmp.txt', res)

if __name__ == "__main__":

    vocab = get_vocab('./template.txt')
    # pos_based('./test.txt')
    sent = open('sent.txt',encoding='utf-8').readlines()
    pos = open('pos.txt').readlines()
    remain_speech = ["CC", "DT", "EX", "MD", "PRP", "PRP$", "RP", "TO"]
    res = ''
    for i,s in enumerate(sent):
        if len(s)<3:continue
        tmp = ''
        pos_tag = pos[i].split()
        for j,w in enumerate(s.split()):
            if w in vocab or pos_tag[j] in remain_speech:
                tmp += w + ' '
            else:
                tmp += 'XXX '
        res += s + tmp + '\n\n'
    write_file('predict.txt',res)
    # vocab_based('./dblp')
    # print(standford_pos('There have a cat .'))




