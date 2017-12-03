import itertools
import numpy as np

class TaggingType(object):
    POS = 0,
    NER = 1

use_subword_sum = True
use_pretrained_embeddings = True
pretrained_unknown_word = 'UUUNKKK'
bos = 'bos' if use_pretrained_embeddings else 'BOS'
eos = 'eos' if use_pretrained_embeddings else 'EOS'
unknown_word = pretrained_unknown_word if use_pretrained_embeddings else 'Unknown123456'


def read_data_into_batches(fname):
    batches = []
    with open(fname) as f:
        lines = f.readlines()
    current_batch = []
    current_batch.append((bos, 'None'))
    current_batch.append((bos, 'None'))
    for line in lines:
        if line == "\n":
            current_batch.append((eos, 'None'))
            current_batch.append((eos, 'None'))
            batches.append(current_batch)
            current_batch = []
            current_batch.append((bos, 'None'))
            current_batch.append((bos, 'None'))
        else:
            text, label = line.strip().split()
            if use_pretrained_embeddings:
                text = text.lower()
            current_batch.append((text, label))
    return batches

POS_TRAIN_Batches = read_data_into_batches("data/pos/train")
POS_DEV_Batches    = read_data_into_batches("data/pos/dev")

NER_TRAIN_Batches = read_data_into_batches("data/ner/train")
NER_DEV_Batches    = read_data_into_batches("data/ner/dev")

word_vectors = np.loadtxt('data/wordVectors.txt')
with open('data/vocab.txt') as f:
    pretrained_vocab = [line.rstrip() for line in f]


def get_unique_words(tagging_type):
    all_tuples = get_all_tuples(tagging_type)
    all_words = set([word[0] for word in all_tuples])
    all_words.add('Unknown123456')
    return all_words

# def get_unique_words_vocab(words):
#     words = words[0]
#     unique_words = set([value[0] for value in words])
#     return unique_words


def get_unique_labels(tagging_type):
    all_tuples = get_all_tuples(tagging_type)
    labels = set([word[1] for word in all_tuples])
    labels.remove('None')
    return labels

def get_all_tuples(tagging_type):
    if tagging_type == TaggingType.POS:
        return list(itertools.chain.from_iterable(POS_TRAIN_Batches))
    return list(itertools.chain.from_iterable(NER_TRAIN_Batches))


# check how to read this...
# TEST  = [read_data("data/" + str(info_type) + "/test")]