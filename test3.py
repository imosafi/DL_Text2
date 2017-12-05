import torch
import torch.nn as nn
import utils
import os

print torch.__version__

embedding_sum = nn.EmbeddingBag(10, 2, mode='sum')
# embedding_sum.weight.data.copy_(torch.from_numpy(utils.dummy_word_vectors))
# a batch of 2 samples of 4 indices each
input = torch.autograd.Variable(torch.LongTensor([0, 1, 0, 1, 0, 1, 0,1,0,1,0]))
offsets = torch.autograd.Variable(torch.LongTensor([0,1,2,5,8]))
w = embedding_sum(input, offsets)

# embedding_sum = nn.EmbeddingBag(10, 3, mode='sum')
# x = torch.autograd.Variable(torch.LongTensor([[1,2,4,5],[4,3,2,9]]))
# embedding_sum(x)

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

ensure_directory_exists('results/test')
with open('results/test/pred.txt', 'w') as f:
    f.writelines(['a\n', 'b\n', 'c\n'])

d = 2