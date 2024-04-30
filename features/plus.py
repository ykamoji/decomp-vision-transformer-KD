import numpy as np


def compute_plus(att_matrix):
    att_processed = np.empty_like(att_matrix[0])
    num_layers = att_matrix.shape[0]
    for layer in range(num_layers):
        att_processed += att_matrix[layer]
    return np.array([att_processed])


def compute_step_plus(att_matrix):
    att_processed = np.empty_like(att_matrix[0])
    num_layers = att_matrix.shape[0]
    for layer in range(num_layers // 2, num_layers):
        att_processed += att_matrix[layer]
    return np.array([att_processed])
