import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils
import tagger3
import numpy as np
import random
import torch.backends.cudnn as cudnn
import datetime
from utils import TaggingType
import os
import matplotlib.pyplot as plt

EMBEDDING_DIM_SIZE = 50
CONTEXT_SIZE = 5
Epochs = 1
LEARNING_RATE = 0.01
HIDDEN_LAYER_SIZE = 200
START_INDEX = 2
tagging = TaggingType.POS


def ensure_directory_exists(self, directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def save_results(save_path, losses, accuracies):
    ensure_directory_exists(save_path)
    f = open(save_path + '/results.text', 'w')
    f.write('hidden layer size: ' + str(HIDDEN_LAYER_SIZE) + '\n')
    f.write('number of epochs: ' + str(Epochs) + '\n')
    f.write('learning rate: ' + str(LEARNING_RATE) + '\n')
    f.close()
    plt.title('Net Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.plot(losses, color='tab:red')
    plt.savefig(save_path + '/plot_loss.png', dpi=100)
    plt.cla()
    plt.title('Net Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.plot(losses, color='tab:blue')
    plt.savefig(save_path + '/plot_accuracy.png', dpi=100)

def get_suffix_prefix_inputs_offsets(batch, i):
    inputs = []
    offsets = []
    k = 0
    j = i - 2
    offsets.append(k)
    while j <= i + 2:
        current_word = batch[j][0]
        if current_word in word_to_ix and len(current_word) <= 3:
            inputs.append(word_to_ix[current_word])
            k += 1
            if j < i + 2:
                offsets.append(k)
        elif current_word in word_to_ix and len(current_word) > 3:
            inputs.append(word_to_ix[current_word])
            inputs.append(word_to_ix[current_word[:3]])
            inputs.append(word_to_ix[current_word[-3:]])
            k += 3
            if j < i + 2:
                offsets.append(k)
        else:
            inputs.append(word_to_ix[utils.unknown_word])
            k += 1
            if j < i + 2:
                offsets.append(k)
        j += 1
    return inputs, offsets
def get_context_indexes(batch, i):
    indexes = []
    j = i - 2
    while j <= i + 2:
        if batch[j][0] in word_to_ix:
            indexes.append(word_to_ix[batch[j][0]])
        else:
            indexes.append(word_to_ix[utils.unknown_word])
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
            # context_idxs = get_context_indexes(batch, i)
            # context_var = autograd.Variable(torch.LongTensor(context_idxs).cuda())
            inputs, offsets = get_suffix_prefix_inputs_offsets(batch, i)
            inputs = torch.autograd.Variable(torch.LongTensor(inputs).cuda())
            offsets = torch.autograd.Variable(torch.LongTensor(offsets).cuda())
            log_probs = model(inputs, offsets)
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
    if utils.use_pretrained_embeddings:
        vocab = utils.pretrained_vocab
    else:
        vocab = utils.get_suffix_prefix_vocab(utils.get_unique_words(tagging))
    labels = utils.get_unique_labels(tagging)
    train_batches = utils.POS_TRAIN_Batches if tagging == TaggingType.POS else utils.NER_TRAIN_Batches
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    label_to_ix = {label: i for i, label in enumerate(labels)}
    ix_to_label = {v: k for k, v in label_to_ix.iteritems()}
    label_dim = len(labels)
    model = tagger3.SequenceTagger(len(vocab), EMBEDDING_DIM_SIZE, CONTEXT_SIZE, HIDDEN_LAYER_SIZE, label_dim, utils.use_pretrained_embeddings)
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

    now = datetime.now()
    current_date = now.strftime("%d.%m.%Y")
    current_time = now.strftime("%H:%M:%S")
    for epoch in xrange(Epochs):
        print "start epoch " + str(epoch + 1)
        total_bathces = str(len(train_batches))
        random.shuffle(train_batches)
        pretrain_time = datetime.datetime.now()

        for index, batch in enumerate(train_batches):
            if index % 100 == 0:
                print 'epoch ' + str(epoch + 1) + ' batch ' + str(index) + '/' + total_bathces
            end_index = len(batch) - 3
            model.zero_grad()

            loss = 0
            i = START_INDEX

            while i <= end_index:
                inputs, offsets = get_suffix_prefix_inputs_offsets(batch, i)
                inputs = torch.autograd.Variable(torch.LongTensor(inputs).cuda())
                offsets = torch.autograd.Variable(torch.LongTensor(offsets).cuda())
                # embedding_sum = nn.EmbeddingBag(len(vocab), 50, mode='sum')
                # a batch of 2 samples of 4 indices each
                # w = embedding_sum(inputs, offsets)
                # context_idxs = get_context_indexes(batch, i)
                # context_var = autograd.Variable(torch.LongTensor(context_idxs).cuda())
                log_probs = model(inputs, offsets)
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
    save_results('results/' + current_date + '_' + current_time, losses, accuracies)
    print(losses)
    print(accuracies)
