from numba.decorators import jit
import numpy as np


@jit(nopython=True, fastmath=True)
def p_norm_distance(mat_A, mat_B, p):
    """
    This is going to calculate the pairwise p-norm distance from the samples
    :param mat_A:
    :param mat_B:
    :param p:
    :return:
    """

    return np.sum(
        np.abs((mat_A.reshape(mat_A.shape + (1,)) - mat_B.reshape(mat_B.shape + (1,)).T)) ** p,
        axis=1) ** (1.0/p)


@jit(nopython=True, fastmath=True)
def cosine_distance(mat_A, mat_B):
    """
    This is going to calculate the pairwise cosine distance from the samples
    :param mat_A:
    :param mat_B:
    :return:
    """

    n_A = mat_A.shape[0]
    n_B = mat_B.shape[0]

    row_norm_A = (np.array([np.linalg.norm(mat_A[i, :]) for i in range(n_A)]) + 1e-4) ** -1.0
    row_norm_B = (np.array([np.linalg.norm(mat_B[i, :]) for i in range(n_B)]) + 1e-4) ** -1.0

    return 1.0 - np.dot(np.dot(np.diag(row_norm_A), mat_A), np.dot(np.diag(row_norm_B), mat_B).T)


