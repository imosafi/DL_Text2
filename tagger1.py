import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import utils



class SequenceTagger(nn.Module):
    def __init__(self, vocab_size, embedding_dim, context_size, hidden_layer_size, output_dim, load_pretrained_embeddings = False):
        super(SequenceTagger, self).__init__()
        self.drop_out = nn.Dropout(0.5)
        self.output_dim = output_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        if load_pretrained_embeddings:
            self.embeddings.weight.data.copy_(torch.from_numpy(utils.word_vectors))
        self.linear1 = nn.Linear(context_size * embedding_dim, hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size, output_dim)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.tanh(self.linear1(embeds))
        out = F.dropout(out, training=self.training)
        out = self.linear2(out)
        return F.log_softmax(out)
