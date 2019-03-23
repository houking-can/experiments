from gen_dataset import get_sentences,iter_files
import json
import os
path = './save'
files = list(iter_files(path))
for file in files:
    paper = json.load(open(file))
    abstract = get_sentences(paper['abstract_text'])
    res = ''
    for i,each in enumerate(abstract):
        if ';' in each:
            res+=str(i)+' '
    if res!='':
        print(os.path.basename(file)+': '+res+'\n')
