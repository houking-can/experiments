from utils import iter_files
from metric import compute_rouge_l, compute_rouge_n
from tqdm import tqdm
import time
import json
import os
import numpy as np
import multiprocessing as mp
from cytoolz import curry,compose
from scipy.optimize import linear_sum_assignment

def _split_words(texts):
    return map(lambda t: t.split(), texts)

@curry
def label(save_path,split,file):
    name = os.path.basename(file)
    name, _ = os.path.splitext(name)
    save_name = os.path.join(save_path,"%s.json" % name)
    paper = json.load(open(file))
    abstract = paper['abstract']
    article = paper['article']

    tokenize = compose(list, _split_words)
    article = tokenize(article)
    abstract = tokenize(abstract)
    km_matrix = []
    for i in range(len(abstract)):
        rouges = list(map(compute_rouge_l(reference=abstract[i], mode='r'),article))
        km_matrix.append([-i for i in rouges])
    km_matrix = np.array(km_matrix)

    row_ind, col_ind = linear_sum_assignment(km_matrix)
    paper['extracted'] = [int(i) for i in col_ind]
    paper['score'] = [-i for i in km_matrix[row_ind, col_ind]]
    json.dump(paper,open(save_name,'w'),indent=4)


if __name__ == "__main__":
    path = '/home/yhj/dataset/emnlp/'
    for split in ['train']:
        print("labeling %s..." % split)

        data_path = os.path.join(path, split)
        files = list(iter_files(data_path))

        t1 = time.time()
        save_path = os.path.join(path, "%s_km" % split)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with mp.Pool() as pool:
            list(pool.imap_unordered(label(save_path,split),files,chunksize=1024))
        # for file in tqdm(files):
        #
        #     p.apply_async(func=label, args=(file, save_name))

        t2 = time.time()
        print('%s time cost : %.1f seconds' % (split, (t2 - t1)))
