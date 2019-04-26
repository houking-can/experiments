import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
import json
from tensorflow.core.example import example_pb2
from tqdm import tqdm

dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote,
              ")"]  # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<S>'
SENTENCE_END = '</S>'

# To represent list of sections as string and retrieve it back
SECTION_SEPARATOR = ' <SCTN/> '
# to represent separator as string, end of item (ei)
LIST_SEPARATOR = ' <EI/> '

# To represent list as string and retrieve it back
SENTENCE_SEPARATOR = ' <SENT/> '



tokenized_dir = "tokenized"
bin_dir = "/home/yhj/long-summarization/data/bin_data"
chunks_dir = os.path.join(bin_dir, "chunked")

# These are the number of .story files we expect there to be in cnn_stories_dir and dm_stories_dir
num_expected_cnn_stories = 92579
num_expected_dm_stories = 219506

VOCAB_SIZE = 200000
CHUNK_SIZE = 1000  # num examples per chunk, for the chunked data


def chunk(data):
    # Make a dir to hold the chunks
    chunk_path = os.path.join(chunks_dir, data)
    if not os.path.isdir(chunk_path):
        os.makedirs(chunk_path)
    # Chunk the data
    # for set_name in ['train', 'val', 'test']:
    print("Splitting %s data into chunks..." % data)

    in_file = os.path.join(bin_dir, data + '.bin')
    reader = open(in_file, "rb")
    index = 0
    finished = False
    while not finished:
        chunk_fname = os.path.join(chunk_path, '%s_%03d.bin' % (data, index))  # new chunk
        with open(chunk_fname, 'wb') as writer:
            for _ in range(CHUNK_SIZE):
                len_bytes = reader.read(8)
                if not len_bytes:
                    finished = True
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, example_str))
            index += 1
    print("Saved chunked data in %s" % chunk_path)


def read_text_file(text_file):
    with open(text_file, "r") as f:
        for line in f:
            yield line


def write_to_bin(data_dir, bin_dir, makevocab=False,ischunk=False):
    """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""

    if makevocab:
        vocab_counter = collections.Counter()

    for data in os.listdir(data_dir):
        # make IO list file
        print("Writing %s bin file..." % data)
        papers = read_text_file(os.path.join(data_dir, data))
        with open(os.path.join(bin_dir, data[:-3]+'bin'), 'wb') as writer:
            for paper in tqdm(papers):
                paper = json.loads(paper)
                # Get the strings to write to .bin file

                article = SENTENCE_SEPARATOR.join(list(paper['article_text'])).encode()
                abstract = SENTENCE_SEPARATOR.join(list(paper['abstract_text'])).encode()
                article_id = paper['article_id'].encode()
                if paper['labels']:
                    labels = SENTENCE_SEPARATOR.join(list(paper['labels'])).encode()
                else:
                    labels = b''
                section_names = SENTENCE_SEPARATOR.join(list(paper['section_names'])).encode()
                list_section = [SENTENCE_SEPARATOR.join(section) for section in list(paper['sections'])]
                sections = SECTION_SEPARATOR.join(list_section).encode()

                # Write to tf.Example
                tf_example = example_pb2.Example()
                tf_example.features.feature['article_body'].bytes_list.value.extend([article])
                tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
                tf_example.features.feature['article_id'].bytes_list.value.extend([article_id])
                tf_example.features.feature['labels'].bytes_list.value.extend([labels])
                tf_example.features.feature['section_names'].bytes_list.value.extend([section_names])
                tf_example.features.feature['sections'].bytes_list.value.extend([sections])

                tf_example_str = tf_example.SerializeToString()
                str_len = len(tf_example_str)
                writer.write(struct.pack('q', str_len))
                writer.write(struct.pack('%ds' % str_len, tf_example_str))

                # Write the vocab to file, if applicable
                if makevocab:
                    art_tokens = article.split(' ')
                    abs_tokens = abstract.split(' ')
                    abs_tokens = [t for t in abs_tokens if
                                  t not in [SENTENCE_START, SENTENCE_END]]  # remove these tags from vocab
                    tokens = art_tokens + abs_tokens
                    tokens = [t.strip() for t in tokens]  # strip
                    tokens = [t for t in tokens if t != ""]  # remove empty
                    vocab_counter.update(tokens)

        # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
        if ischunk: chunk(data[:-4])

    # write vocab to file
    if makevocab:
        print("Writing vocab file...")
        with open(os.path.join(bin_dir, "vocab"), 'w') as writer:
            for word, count in vocab_counter.most_common(VOCAB_SIZE):
                writer.write(word + ' ' + str(count) + '\n')
        print("Finished writing vocab file")


if __name__ == '__main__':

    data_dir = 'data/arxiv-release'

    # Create some new directories

    if not os.path.exists(bin_dir): os.makedirs(bin_dir)

    # Run stanford tokenizer on both stories dirs, outputting to tokenized stories directories

    # Read the tokenized stories, do a little postprocessing then write to bin files

    write_to_bin(data_dir, bin_dir, ischunk=False)
    # write_to_bin(all_val_urls, os.path.join(bin_dir, "val.bin"))
    # write_to_bin(all_train_urls, os.path.join(bin_dir, "train.bin"), makevocab=True)


