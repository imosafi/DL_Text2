import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils



class SequenceTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_layer_size, output_dim):
        super(SequenceTagger, self).__init__()
        self.output_dim = output_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        # self.linear1 = nn.Linear((two_way_context_size * 2 + 1) * embedding_dim, 200)
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size, output_dim)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.tanh(self.linear1(embeds))
        out = self.linear2(out)
        return F.log_softmax(out)
