import numpy as np
import scipy.spatial

output_path = 'data/k_closest.txt'
K = 5

words_to_search = ['dog', 'england', 'john', 'explode', 'office']

vecs = np.loadtxt('data/wordVectors.txt')
with open('data/vocab.txt') as f:
    vocab = [line.rstrip() for line in f]

# words_to_search = ['cat', 'car']
#
# vecs = np.loadtxt('data/dummyWordVectors.txt')
# with open('data/dummyVocab.txt') as f:
#     vocab = [line.rstrip() for line in f]

word_to_vec_dict = dict(zip(vocab, vecs))
vec_to_word_dict = {tuple(v): k for k, v in word_to_vec_dict.iteritems()}


def get_cos_cdist(matrix, vector):
    v = vector.reshape(1, -1)
    return scipy.spatial.distance.cdist(matrix, v, 'cosine').reshape(-1)


def most_similar(word, k):
    k += 1
    distances = get_cos_cdist(vecs, word_to_vec_dict[word])
    distance_vector_pairs = dict(zip(distances, vecs))
    idx = np.argpartition(distances, k)
    k_smallest = distances[idx[:k]]
    k_closest_vectors = [distance_vector_pairs[dist] for dist in k_smallest]
    k_closest_words = [vec_to_word_dict[tuple(vec)] for vec in k_closest_vectors]
    return zip(k_closest_words, k_smallest)

if __name__ == '__main__':
    most_similar_list = []

    for word in words_to_search:
        tuples = most_similar(word, K)
        filtered_tuples = [i for i in tuples if i[0] != word]
        most_similar_list.append([word, filtered_tuples])
    with open('data/k_closest.txt', 'w') as f:
        f.write(str(most_similar_list))
