from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import  LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.sum_basic import SumBasicSummarizer
from sumy.parsers.plaintext import PlaintextParser
import shutil
from evaluate import eval_rouge
import os
import json
from tqdm import tqdm
from os.path import join,exists
from sumy.nlp.tokenizers import Tokenizer

def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dir_path, _, file_names in os.walk(path):
            for f in file_names:
                yield os.path.join(dir_path, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)

def summarize(test_path,decoder_path):
    summarizers = {
                   'lexrank':LexRankSummarizer(),
                   'lsa':LsaSummarizer(),
                   'sumbasic':SumBasicSummarizer(),
                   'textrank':TextRankSummarizer()
                   }
    for each in ['lexrank', 'lsa', 'sumbasic', 'textrank']:
        print("###################### %s #######################" % each)
        files = list(iter_files(test_path))
        dec_dir = join(decoder_path, each,'output')
        if not os.path.exists(dec_dir):
            os.makedirs(dec_dir)

        summarizer = summarizers[each]
        for file in tqdm(files):
            name = os.path.basename(file)
            name,_ = os.path.splitext(name)
            save_path = join(dec_dir,name+'.dec')
            article = ' '.join(json.load(open(file))['article'])
            article = PlaintextParser.from_string(article,Tokenizer('english'))
            output = summarizer(article.document,sentences_count=4)
            output = [each._text for each in output]
            with open(save_path,'w') as f:
                f.write('\n'.join(output))

def upper_bound():
    for split in ['val', 'test']:
        print(split)
        dec_rerank = '/home/yhj/emnlp/baseline/upperbound_%s/output_rerank' % split
        dec_order = '/home/yhj/emnlp/baseline/upperbound_%s/output_order' % split
        path = os.path.join('/home/yhj/dataset/emnlp/', split)
        if not os.path.exists(dec_rerank):
            os.makedirs(dec_rerank)

        if not os.path.exists(dec_order):
            os.makedirs(dec_order)

        for file in iter_files(path):
            paper = json.load(open(file))
            name = os.path.basename(file)
            name, _ = os.path.splitext(name)
            sents = [paper['article'][i] for i in paper['extracted']]
            with open(os.path.join(dec_rerank, name + '.dec'), 'w') as f:
                f.write('\n'.join(sents))
            order = sorted(paper['extracted'])
            sents = [paper['article'][i] for i in order]
            with open(os.path.join(dec_order, name + '.dec'), 'w') as f:
                f.write('\n'.join(sents))

        ref_dir = os.path.join('/home/yhj/dataset/emnlp/refs/', split)

        dec_pattern = r'(\d+).dec'
        ref_pattern = '#ID#.ref'

        output = eval_rouge(dec_pattern, dec_rerank, ref_pattern, ref_dir)
        print('%s rerank:' % split)
        print(output)
        with open('/home/yhj/emnlp/baseline/upperbound_%s/rouge_rerank.txt' % split, 'w') as f:
            f.write(output)

        output = eval_rouge(dec_pattern, dec_order, ref_pattern, ref_dir)
        print('%s order:' % split)
        print(output)
        with open('/home/yhj/emnlp/baseline/upperbound_%s/rouge_order.txt' % split, 'w') as f:
            f.write(output)

if __name__=="__main__":
    test_path = '/home/yhj/dataset/emnlp/test'
    decoder_path = '/home/yhj/emnlp/baseline'
    ref_dir = '/home/yhj/dataset/emnlp/refs/test'

    summarize(test_path,decoder_path)


    for each in ['lexrank','lsa','sumbasic','textrank']:
        print("###################### %s #######################" % each)

        dec_dir = join(decoder_path, each,'output')
        dec_pattern = r'(\d+).dec'
        ref_pattern = '#ID#.ref'
        output = eval_rouge(dec_pattern, dec_dir, ref_pattern, ref_dir)

        print(output)
        res_dir = join(decoder_path, each)
        with open(join(res_dir,'rouge.txt'), 'w') as f:
            f.write(output)


