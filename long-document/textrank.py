import json
import os
import shutil
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.parsers.plaintext import PlaintextParser
import tempfile
from pyrouge import Rouge155
import subprocess as sp

def readlines(path):
    """ iterate file per line """
    with open(path, 'r') as f:
        while True:
            line = f.readline()
            if line:
                yield line
            else:
                break



def write_summary(ref,sys):
    data_path = "/home/yhj/long-summarization/data/arxiv-release/test.txt"

    if os.path.exists(ref):
        shutil.rmtree(ref)
    os.makedirs(ref)
    if os.path.exists(sys):
        shutil.rmtree(sys)
    os.makedirs(sys)

    lines = readlines(data_path)
    summarizer = TextRankSummarizer()
    for id,line in enumerate(lines):
        print(id)
        article = json.loads(line)
        with open(os.path.join(ref,"%d_ref.txt" % id),'w') as f:
            for each in article['abstract_text']:
                each = each.replace('<S>','')
                each = each.replace('</S>', '')
                each.strip()
                f.write(each+'\n')

        text = ''
        for i,s in enumerate(article['section_names']):
            if 'intro' in s.lower():
                text +=' '.join(article['sections'][i])+' '
            if 'conclu' in s.lower():
                text += ' '.join(article['sections'][i]) + ' '
        text = PlaintextParser.from_string(text, Tokenizer('english'))

        output = summarizer(text.document, sentences_count=4)
        output = [each._text for each in output]
        with open(os.path.join(sys, "%d_dec.txt" % id), 'w') as f:
            f.write('\n'.join(output))


def eval_rouge(dec_pattern, dec_dir, ref_pattern, ref_dir,
               cmd='-c 95 -r 1000 -n 2 -m', system_id=1):
    """ evaluate by original Perl implementation"""
    # silence pyrouge logging

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
        cmd = (os.path.join('/home/yhj/ROUGE/RELEASE-1.5.5', 'ROUGE-1.5.5.pl')
               + ' -e {} '.format(os.path.join('/home/yhj/ROUGE/RELEASE-1.5.5', 'data'))
               + cmd
               + ' -a {}'.format(os.path.join(tmp_dir, 'settings.xml')))
        output = sp.check_output(cmd.split(' '), universal_newlines=True)
    return output


if __name__=="__main__":

    ref_dir = '/home/yhj/long-summarization/logroot/decode_textrank/ref'
    sys_dir = '/home/yhj/long-summarization/logroot/decode_textrank/sys'
    write_summary(ref_dir,sys_dir)

    dec_pattern = r'(\d+)_dec.txt'
    ref_pattern = '#ID#_ref.txt'

    res = eval_rouge(dec_pattern, sys_dir, ref_pattern, ref_dir)

    with open('/home/yhj/long-summarization/logroot/decode_textrank/rouge.txt','w') as f:
        f.write(res)