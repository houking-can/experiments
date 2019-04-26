from evaluate import eval_rouge
import os


if __name__=="__main__":
    for each in ['decode_baseline','decode_introduction']:
        print(each)
        dec_dir = '/home/yhj/long-summarization/logroot/%s/decoded' % each
        ref_dir = '/home/yhj/long-summarization/logroot/%s/reference' % each
        dec_pattern = r'(\d+).decoded'
        ref_pattern = '#ID#.reference'
        output = eval_rouge(dec_pattern, dec_dir, ref_pattern, ref_dir)
        print(output)
        with open(os.path.join(os.path.dirname(dec_dir),'rouge.txt'),'w') as f:
            f.write(output)

