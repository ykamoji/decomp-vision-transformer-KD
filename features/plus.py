import numpy as np


def compute_plus(att_matrix):
    att_processed = np.empty_like(att_matrix[0])
    num_layers = att_matrix.shape[0]
    for layer in range(num_layers):
        att_processed += att_matrix[layer]
    return np.array([att_processed])


def compute_skip_plus(att_matrix, last=True):
    att_processed = np.empty_like(att_matrix[0])
    num_layers = att_matrix.shape[0]

    if last:
        start = num_layers // 2
        end = num_layers
    else:
        start = 0
        end = num_layers // 2

    for layer in range(start, end):
        att_processed += att_matrix[layer]

    return np.array([att_processed])



