from utils import eval_rouge,iter_files
import json
import os


for split in ['val_km']:
    print(split)
    dec_rerank = '/home/yhj/emnlp/baseline/upperbound_%s/output_rerank' % split
    dec_order = '/home/yhj/emnlp/baseline/upperbound_%s/output_order' % split
    path = os.path.join('/home/yhj/dataset/emnlp/',split)
    if not os.path.exists(dec_rerank):
        os.makedirs(dec_rerank)

    if not os.path.exists(dec_order):
        os.makedirs(dec_order)

    for file in iter_files(path):
        paper = json.load(open(file))
        name = os.path.basename(file)
        name,_ = os.path.splitext(name)
        sents = [paper['article'][i] for i in paper['extracted']]
        with open(os.path.join(dec_rerank,name+'.dec'),'w') as f:
            f.write('\n'.join(sents))
        order = sorted(paper['extracted'])
        sents = [paper['article'][i] for i in order]
        with open(os.path.join(dec_order,name+'.dec'),'w') as f:
            f.write('\n'.join(sents))

    ref_dir = os.path.join('/home/yhj/dataset/emnlp/refs/',split)

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