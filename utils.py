import itertools


def read_data_into_batches(fname):
    batches = []
    with open(fname) as f:
        lines = f.readlines()
    current_batch = []
    current_batch.append(('BOS', 'None'))
    current_batch.append(('BOS', 'None'))
    for line in lines:
        if line == "\n":
            current_batch.append(('EOS', 'None'))
            current_batch.append(('EOS', 'None'))
            batches.append(current_batch)
            current_batch = []
            current_batch.append(('BOS', 'None'))
            current_batch.append(('BOS', 'None'))
        else:
            text, label = line.strip().split()
            current_batch.append((text, label))
    return batches

POS_TRAIN_Batches = read_data_into_batches("data/pos/train")
POS_DEV_Batches    = read_data_into_batches("data/pos/dev")


def get_unique_words():
    all_tuples = list(itertools.chain.from_iterable(POS_TRAIN_Batches))
    all_words = set([word[0] for word in all_tuples])
    all_words.add('Unknown123456')
    return  all_words

# def get_unique_words_vocab(words):
#     words = words[0]
#     unique_words = set([value[0] for value in words])
#     return unique_words


def get_unique_labels():
    all_tuples = list(itertools.chain.from_iterable(POS_TRAIN_Batches))
    labels = set([word[1] for word in all_tuples])
    labels.remove('None')
    return labels




# check how to read this...
# TEST  = [read_data("data/" + str(info_type) + "/test")]