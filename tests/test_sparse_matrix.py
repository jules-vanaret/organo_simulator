import numpy as np

from scipy.sparse import csr_matrix

row = np.array([0, 0, 1, 2, 2, 2])

col = np.array([0, 2, 2, 0, 1, 2])

data = np.array([1, 2, 3, 4, 5, 6])

mat = csr_matrix((data, (row, col)), shape=(3, 3))

foo ='bar'