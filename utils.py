import itertools
import numpy as np

class TaggingType(object):
    POS = 0,
    NER = 1

use_pretrained_embeddings = True
pretrained_unknown_word = 'UUUNKKK'
pretrained_unknown_prefix = 'UUUNKKK_PREFIX'
pretrained_unknown_suffix = 'UUUNKKK_SUFFIX'
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
            if not len(line.split()) > 1:
                line = line + 'None'
            text, label = line.strip().split()
            if use_pretrained_embeddings:
                text = text.lower()
            current_batch.append((text, label))

    return batches

POS_TRAIN_Batches = read_data_into_batches("data/pos/train")
POS_DEV_Batches    = read_data_into_batches("data/pos/dev")

NER_TRAIN_Batches = read_data_into_batches("data/ner/train")
NER_DEV_Batches    = read_data_into_batches("data/ner/dev")

POS_TEST_BATCHES = read_data_into_batches("data/pos/test")
NER_TEST_BATCHES = read_data_into_batches("data/ner/test")

word_vectors = np.loadtxt('data/wordVectors.txt')
with open('data/vocab.txt') as f:
    pretrained_vocab = [line.rstrip() for line in f]

dummy_word_vectors = np.loadtxt('data/dummyWordVectors.txt')

def get_unique_words(tagging_type):
    all_tuples = get_all_tuples(tagging_type)
    all_words = set([word[0] for word in all_tuples])
    all_words.add('Unknown123456')
    return all_words


def get_unique_labels(tagging_type):
    all_tuples = get_all_tuples(tagging_type)
    labels = set([word[1] for word in all_tuples])
    labels.remove('None')
    return labels


def get_all_tuples(tagging_type):
    if tagging_type == TaggingType.POS:
        return list(itertools.chain.from_iterable(POS_TRAIN_Batches))
    return list(itertools.chain.from_iterable(NER_TRAIN_Batches))


def get_suffix_prefix_regular_vocab(words):
    complete_vocab = []
    for word in words:
        complete_vocab.append(word)
        if len(word) > 3:
            complete_vocab.append(word[:3])
            complete_vocab.append(word[-3:])
    complete_vocab = set(complete_vocab)
    return complete_vocab


def get_suffix_prefix_pretrained_vocab_and_word_to_ix(pretrained_vocab, corpus_words):
    pretrained_words = []
    prefixes_suffixes = []
    for word in pretrained_vocab:
        pretrained_words.append(word)
    for word in corpus_words:
        if len(word) > 3:
            prefixes_suffixes.append(word[:3])
            prefixes_suffixes.append(word[-3:])
    prefixes_suffixes = set(prefixes_suffixes)
    prefixes_suffixes = [x for x in prefixes_suffixes if x not in pretrained_words]
    complete_vocab = pretrained_words + prefixes_suffixes + [pretrained_unknown_prefix, pretrained_unknown_suffix]
    word_to_ix = {word: i for i, word in enumerate(complete_vocab)}

    return complete_vocab, word_to_ix

