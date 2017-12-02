import numpy as np
import scipy
import torch

def cos_loop_spatial(matrix, vector):
    """
    Calculating pairwise cosine distance using a common for loop with the numpy cosine function.
    """
    neighbors = []
    for row in range(matrix.shape[0]):
        neighbors.append(scipy.spatial.distance.cosine(vector, matrix[row,:]))
    return neighbors

def cos_loop(matrix, vector):
    """
    Calculating pairwise cosine distance using a common for loop with manually calculated cosine value.
    """
    neighbors = []
    for row in range(matrix.shape[0]):
        vector_norm = np.linalg.norm(vector)
        row_norm = np.linalg.norm(matrix[row,:])
        cos_val = vector.dot(matrix[row,:]) / (vector_norm * row_norm)
        neighbors.append(cos_val)
    return neighbors

def cos_matrix_multiplication(matrix, vector):
    """
    Calculating pairwise cosine distance using matrix vector multiplication.
    """
    dotted = matrix.dot(vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    vector_norm = np.linalg.norm(vector)
    matrix_vector_norms = np.multiply(matrix_norms, vector_norm)
    neighbors = np.divide(dotted, matrix_vector_norms)
    return neighbors

cos_functions = [cos_loop_spatial, cos_loop, cos_matrix_multiplication]

# Test performance and plot the best results of each function
# mat = np.random.randn(1000,1000)
# vec = np.random.randn(1000)
# cos_performance = {}
# for func in cos_functions:
#     func_performance = %timeit -o func(mat, vec)
#     cos_performance[func.__name__] = func_performance.best

# pd.Series(cos_performance).plot(kind='bar')

print(torch.rand(2,3).cuda())
c = torch.cuda.current_device()
count = torch.cuda.device_count()
available = torch.cuda.is_available()



c = 2