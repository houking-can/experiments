# from utils import eval_rouge, iter_files
import json
import os
import shutil
from utils import split
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
# for split in ['test','val']:
#     print(split)
#     # dec_rerank = '/home/yhj/emnlp/baseline/upperbound_%s/output_rerank' % split
#     dec_order = '/home/yhj/emnlp/baseline/upperbound_%s/output_order' % split
#     path = os.path.join('/home/yhj/dataset/emnlp/',split)
#     # if not os.path.exists(dec_rerank):
#     #     os.makedirs(dec_rerank)
#
#     if not os.path.exists(dec_order):
#         os.makedirs(dec_order)
#
#     for file in iter_files(path):
#         paper = json.load(open(file))
#         name = os.path.basename(file)
#         name,_ = os.path.splitext(name)
#         sents = [paper['article'][i] for i in paper['extracted']]
#         # with open(os.path.join(dec_rerank,name+'.dec'),'w') as f:
#         #     f.write('\n'.join(sents))
#         order = sorted(paper['extracted'])
#         sents = [paper['article'][i] for i in order]
#         with open(os.path.join(dec_order,name+'.dec'),'w') as f:
#             f.write('\n'.join(sents))
#
#     ref_dir = os.path.join('/home/yhj/dataset/emnlp/refs/',split)
#
#     dec_pattern = r'(\d+).dec'
#     ref_pattern = '#ID#.ref'
#
#     # output = eval_rouge(dec_pattern, dec_rerank, ref_pattern, ref_dir)
#     # print('%s rerank:' % split)
#     # print(output)
#     # with open('/home/yhj/emnlp/baseline/upperbound_%s/rouge_rerank.txt' % split, 'w') as f:
#     #     f.write(output)
#
#     output = eval_rouge(dec_pattern, dec_order, ref_pattern, ref_dir)
#     print('%s order:' % split)
#     print(output)
#     with open('/home/yhj/emnlp/baseline/upperbound_%s/rouge_order.txt' % split, 'w') as f:
#         f.write(output)
cnt=0
a=dict()
for file in iter_files(r'/home/yhj/dataset/emnlp_mix_int/train'):
    paper = json.load(open(file))
    # if len(paper['abstract'])<3:
    #     shutil.move(file,'/home/yhj/tmp/tmp_emnlp_mix')
    #     cnt+=1
    #     continue
    if str(len(paper['article'])) in a:
        a[str(len(paper['article']))]+=1
    else:
        a[str(len(paper['article']))]=1
b=sorted(a.items(),key=lambda d:d[1])
print(b)

# a=json.load(open('a.json'))
# b=json.load(open('b.json'))
# cnt_a = 0
# cnt_b = 0
# for k,v in a.items():
#     cnt_a+=v*int(k)
# for k,v in b.items():
#     cnt_b+=v*int(k)
# L   = len(os.listdir('/home/yhj/dataset/emnlp_mix/train'))
# print("a: %f" % (cnt_a/L))
# print("b: %f" % (cnt_b/L))



