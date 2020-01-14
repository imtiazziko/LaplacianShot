"""
Created on Tue Dec  5 11:38:23 2017
@author: ziko
"""
import numpy as np
import multiprocessing
import itertools
import scipy.sparse as sps
# import os
import timeit
from utils.progressBar import printProgressBar
import math


def normalize(Q_in):
    maxcol = np.max(Q_in, axis=1)
    Q_in = Q_in - maxcol[:, np.newaxis]
    N = Q_in.shape[0]
    size_limit = 150000
    if N > size_limit:
        batch_size = 1280
        Q_out = []
        Q_out_2 = []
        num_batch = int(math.ceil(1.0 * N / batch_size))
        for batch_idx in range(num_batch):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, N)
            tmp = np.exp(Q_in[start:end, :])
            tmp = tmp / (np.sum(tmp, axis=1)[:, None])
            Q_out.append(tmp)
        del Q_in
        Q_out = np.vstack(Q_out)
    else:
        Q_out = np.exp(Q_in)
        Q_out = Q_out / (np.sum(Q_out, axis=1)[:, None])

    return Q_out

def normalize_2(S_in):
    S_in_sum = S_in.sum(1)[:,np.newaxis]
    S_in = np.divide(S_in,S_in_sum)
    # S_in = ne.evaluate('S_in/S_in_sum')
    return S_in

def mpassing(slices):
    i, k = slices
    Q_s, kernel_s_data, kernel_s_indices, kernel_s_indptr, kernel_s_shape = get_shared_arrays('Q_s', 'kernel_s_data',
                                                                                              'kernel_s_indices',
                                                                                              'kernel_s_indptr',
                                                                                              'kernel_s_shape')
    # kernel_s = sps.csc_matrix((SHARED_array['kernel_s_data'],SHARED_array['kernel_s_indices'],SHARED_array['kernel_s_indptr']), shape=SHARED_array['kernel_s_shape'], copy=False)
    kernel_s = sps.csc_matrix((kernel_s_data, kernel_s_indices, kernel_s_indptr), shape=kernel_s_shape, copy=False)
    Q_s[i, k] = kernel_s[i].dot(Q_s[:, k])


#    return Q_s

def entropy_energy(Q, unary, kernel, bound_lambda, batch=False):
    tot_size = Q.shape[0]
    pairwise = kernel.dot(Q)
    if batch == False:
        temp = (unary * Q) + (-bound_lambda * pairwise * Q)
        E = (Q * np.log(np.maximum(Q, 1e-20)) + temp).sum()
    else:
        batch_size = 1024
        num_batch = int(math.ceil(1.0 * tot_size / batch_size))
        E = 0
        for batch_idx in range(num_batch):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, tot_size)
            temp = (unary[start:end] * Q[start:end]) + (-bound_lambda * pairwise[start:end] * Q[start:end])
            E = E + (Q[start:end] * np.log(np.maximum(Q[start:end], 1e-20)) + temp).sum()

    return E


def bound_update(unary, X, kernel, bound_lambda, bound_iteration=20, batch=False, manual_parallel=False):
    """
    Here in this code, Q refers to Z in our paper.
    """
    start_time = timeit.default_timer()
    # print("Inside Bound Update . . .")
    N, K = unary.shape
    oldE = float('inf')

    # Initialize the unary and Normalize
        # print 'Parallel is FALSE'
    Q = normalize(-unary)
    # Q = np.exp((-unary))
    # Q = normalize_2(Q)
    for i in range(bound_iteration):
        # printProgressBar(i + 1, bound_iteration, length=12)
        additive = -unary
        mul_kernel = kernel.dot(Q)
        Q = -bound_lambda * mul_kernel
        additive = additive - Q
        Q = normalize(additive)
        E = entropy_energy(Q, unary, kernel, bound_lambda, batch)
        # print('entropy_energy is ' +repr(E) + ' at iteration ',i)
        report_E = E
        if (i > 1 and (abs(E - oldE) <= 1e-5 * abs(oldE))):
            # print('Converged')
            break

        else:
            oldE = E.copy();
            oldQ = Q.copy();
            report_E = E

    elapsed = timeit.default_timer() - start_time
    # print('\n Elapsed Time in bound_update', elapsed)
    l = np.argmax(Q, axis=1)
    ind = np.argmax(Q, axis=0)
    C = X[ind, :]
    return l, C, ind, Q, report_E