import numpy as np
import pandas as pd

__all__ = (
    'identify_continuous_features',
    'discretize_array'
)


def identify_continuous_features(data):
    n_feats = data.shape[1]
    cont_feats = []
    floats = [isinstance(data.dtypes.values[i], np.float64) for i in range(n_feats)]

    for i in range(n_feats):
        if floats[i]:
            cont_feats.append(i)
        elif len(np.unique(data.iloc[:, i])) / data.shape[0] > 0.2:
            cont_feats.append(i)

    discrete_feats = np.arange(n_feats)

    discrete_feats = np.delete(discrete_feats, cont_feats)

    return cont_feats, discrete_feats


def discretize_array(array, n_bins):

    bins = np.linspace(np.min(array), np.max(array) + 0.001,
                       (n_bins * 2))

    array_bins = np.digitize(array.values, bins[0::2])

    slot_labels = bins[1::2]

    return slot_labels[array_bins - 1], slot_labels


def safe_log(x):
    if x > 0:
        log = np.log(x)
    else:
        log = 0

    return log


def safe_divide(a, b):
    return np.array([a[i] / b[i] if np.logical_and(a[i] != 0, b[i] != 0) else -1
                     for i in range(len(a))])
