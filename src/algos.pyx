import cython
from cython.parallel cimport prange, parallel
cimport numpy
import numpy
import torch


def adj2graph(adj):
    (n_rows, n_cols) = adj.shape
    assert n_rows == n_cols
    cdef unsigned int n = n_rows
    adj_copy = adj.astype(long, order='C', casting='safe', copy=True)
    assert adj_copy.flags['C_CONTIGUOUS']
    cdef numpy.ndarray[long, ndim=2, mode='c'] M = adj_copy
    cdef unsigned int i, j, k
    cdef list src_nodes = []
    cdef list dst_nodes = []

    for i in range(n):
        for j in range(n):
            if M[i][j] == 1:
                src_nodes.append(i), dst_nodes.append(j)
    connectivity = (torch.LongTensor(src_nodes), torch.LongTensor(dst_nodes))
    return connectivity