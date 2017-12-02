import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
import tagger1
import numpy as np
import random
import torch.backends.cudnn as cudnn
import datetime
from utils import TaggingType


EMBEDDING_DIM_SIZE = 50
CONTEXT_SIZE = 5
Epochs = 5
# LEARNING_RATE = 0.001
# LEARNING_RATE = 0.01
LEARNING_RATE = 0.02
HIDDEN_LAYER_SIZE = 100
START_INDEX = 2
tagging = TaggingType.NER

# def run_tagger1_training():
#     word_to_ix = {"hello": 0, "world": 1}
#     embeds = nn.Embedding(2, 5)  # 2 words in vocab, 5 dimensional embeddings
#     lookup_tensor = torch.LongTensor([word_to_ix["hello"]])
#     hello_embed = embeds(autograd.Variable(lookup_tensor))
#     print(hello_embed)
#
#     x = 8
#
# def prepare_data_for_training(training_words):
#     return [([training_words[0][i][0], training_words[0][i + 1][0]], training_words[0][i + 2][1]) for i in range(len(training_words[0]) - 2)]
#
# def get_one_hot_label(vocab, word, label_dim):
#     y = [0] * label_dim
#     y[vocab[word_to_ix[target]][1]] = 1
#     return y

def get_context_indexes(batch, i):
    indexes = []
    j = i - 2
    while j <= i + 2:
        # indexes.append(word_to_ix[batch[j][0]])
        indexes.append(word_to_ix[batch[j][0]])
        j += 1
    return indexes

def get_dev_context_indexes(batch, i):
    indexes = []
    j = i - 2
    while j <= i + 2:
        if batch[j][0] in word_to_ix:
            indexes.append(word_to_ix[batch[j][0]])
        else:
            indexes.append(word_to_ix['Unknown123456'])
        j += 1
    return indexes

def get_label_vec(label):
    y = np.zeros(label_dim)
    y[label_to_ix[label]] = 1
    return y

def execute_test(tagging_type):
    # test on dev
    total_loss = torch.FloatTensor([0]).cuda()
    # test_lose = torch.FloatTensor([0])
    total_tries = 0
    correct_preds = 0
    correct = 0
    total_bathces = str(len(dev_batches))
    for index, batch in enumerate(dev_batches):
        end_index = len(batch) - 3
        test_lose = torch.FloatTensor([0]).cuda()
        i = START_INDEX
        while i <= end_index:
            context_idxs = get_dev_context_indexes(batch, i)
            context_var = autograd.Variable(torch.LongTensor(context_idxs).cuda())
            log_probs = model(context_var)
            test_lose += loss_function(log_probs,
                                       autograd.Variable(torch.LongTensor([label_to_ix[batch[i][1]]]).cuda())).data

            if tagging_type == TaggingType.POS:
                total_tries += 1
                if label_to_ix[batch[i][1]] == log_probs.max(1)[1].data[0]:
                    correct_preds += 1
            else:
                if batch[i][1] == 'O' and ix_to_label[log_probs.max(1)[1].data[0]] == 'O':
                    i += 1
                    continue
                total_tries += 1
                if label_to_ix[batch[i][1]] == log_probs.max(1)[1].data[0]:
                    correct_preds += 1
            i += 1
        test_lose /= (len(batch) - 4)
        total_loss += test_lose
        if index % 100 == 0:
            print 'batch ' + str(index) + '/' + total_bathces  # ', current loss ' + str(test_lose)
    return total_loss / len(dev_batches), float(correct_preds) / total_tries


if __name__ == '__main__':
    vocab, labels = utils.get_unique_words(tagging), utils.get_unique_labels(tagging)
    train_batches = utils.POS_TRAIN_Batches if tagging == TaggingType.POS else utils.NER_TRAIN_Batches
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    label_to_ix = {label: i for i, label in enumerate(labels)}
    ix_to_label = {v: k for k, v in label_to_ix.iteritems()}
    label_dim = len(labels)
    # +1 for unknown words
    model = tagger1.SequenceTagger(len(vocab), EMBEDDING_DIM_SIZE, CONTEXT_SIZE, HIDDEN_LAYER_SIZE, label_dim)
    model.cuda()

    dev_batches = utils.POS_DEV_Batches if tagging == TaggingType.POS else utils.NER_DEV_Batches

    # pos_batches = read_into_barches('')
    # run_tagger1_training()
    # training_words = [word[0] for word in utils.POS_TRAIN[0]]
    # trigrams = prepare_data_for_training(utils.POS_TRAIN)
    # vocab = utils.get_unique_words_vocab(utils.POS_TRAIN)
    # word_to_ix = {word: i for i, word in enumerate(vocab)}
    # labels = utils.get_label_vector_size(utils.POS_TRAIN)
    # label_to_ix = {word: i for i, word in enumerate(labels)}
    # label_dim = len(labels)
    # word_to_ix = {word: i for i, word in enumerate(vocab)}
    losses = []
    accuracies = []
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    cudnn.benchmark = True
    cudnn.fastest = True


    for epoch in xrange(Epochs):
        print "start epoch " + str(epoch + 1)
        total_bathces = str(len(train_batches))
        random.shuffle(train_batches)
        pretrain_time = datetime.datetime.now()

        for index, batch in enumerate(train_batches):
            if index % 100 == 0:
                print 'batch ' + str(index) + '/' + total_bathces
            end_index = len(batch) - 3
            model.zero_grad()

            loss = 0
            i = START_INDEX

            while i <= end_index:
                context_idxs = get_context_indexes(batch, i)
                context_var = autograd.Variable(torch.LongTensor(context_idxs).cuda())
                log_probs = model(context_var)
                loss += loss_function(log_probs, autograd.Variable(torch.LongTensor([label_to_ix[batch[i][1]]]).cuda()))
                # context_idxs.append(get_context_indexes(batch, i))
                i += 1

            # context_var = context_var.view(len(batch) - 4, 5)
            # y = get_label_vec(batch[i][1])

            loss /= (len(batch) - 4)
            loss.backward()
            optimizer.step()
        print 'epoch took: ' + str(datetime.datetime.now() - pretrain_time)

        loss, accuracy = execute_test(tagging)

        losses.append(loss)
        accuracies.append(accuracy)
    print(losses)
    print(accuracies)
