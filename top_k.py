import numpy as np
import scipy

words_to_search = ['dog', 'england', 'john', 'explode', 'office']

vecs = np.loadtxt('data/wordVectors.txt')
with open('data/vocab.txt') as f:
    vocab = [line.rstrip() for line in f]

word_vec_dict = dict(zip(vocab, vecs))


def get_cos_cdist(matrix, vector):
    v = vector.reshape(1, -1)
    return scipy.spatial.distance.cdist(matrix, v, 'cosine').reshape(-1)

def most_similar(word, k):
    distances = get_cos_cdist(vecs, word_vec_dict[word])

if __name__ == '__main__':
    most_similar_list = []

    for word in words_to_search:
        most_similar_list.append(most_similar(word, 5))
    c = 2