import numpy as np


def prod_non_zero_diag(x):
    """Compute product of nonzero elements from matrix diagonal.

    input:
    x -- 2-d numpy array
    output:
    product -- integer number


    Vectorized implementation.
    """

    diagonal = np.diag(x)
    return np.prod(diagonal[diagonal != 0])


def are_multisets_equal(x, y):
    """Return True if both vectors create equal multisets.

    input:
    x, y -- 1-d numpy arrays
    output:
    True if multisets are equal, False otherwise -- boolean

    Vectorized implementation.
    """

    return np.array_equal(np.sort(x), np.sort(y))


def max_after_zero(x):
    """Find max element after zero in array.

    input:
    x -- 1-d numpy array
    output:
    maximum element after zero -- integer number

    Vectorized implementation.
    """

    values_after_zero = x[1:][x[:-1] == 0]
    if values_after_zero.size == 0:
        raise ValueError("В массиве нет элементов, перед которыми стоит ноль")
    return np.max(values_after_zero)


def convert_image(img, coefs):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x 3)
    coefs -- 1-d numpy array (length 3)
    output:
    img -- 2-d numpy array

    Vectorized implementation.
    """

    return np.sum(img * coefs, axis=2)


def run_length_encoding(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Vectorized implementation.
    """

    if x.size == 0:
        return np.array([], dtype=x.dtype), np.array([], dtype=int)

    run_starts = np.r_[0, np.flatnonzero(x[1:] != x[:-1]) + 1]
    elements = x[run_starts]
    counters = np.diff(np.r_[run_starts, x.size])
    return elements, counters


def pairwise_distance(x, y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Vctorized implementation.
    """

    differences = x[:, np.newaxis, :] - y[np.newaxis, :, :]
    return np.sqrt(np.sum(differences ** 2, axis=2))
