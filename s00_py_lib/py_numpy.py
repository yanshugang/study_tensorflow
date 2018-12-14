"""
numpy, 挺重要。
"""

import numpy as np

vector = np.array([1, 2, 3])
matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(vector.size)
print(matrix.size)
print(matrix.shape)
print(matrix.dtype)


zeros = np.zeros((3, 4))
print(zeros)

ones = np.ones((3, 4))
print(ones)

eye = np.eye(4)
print(eye)

