def prod_non_zero_diag(x):
    """Compute product of nonzero elements from matrix diagonal.

    input:
    x -- 2-d numpy array
    output:
    product -- integer number


    Not vectorized implementation.
    """

    product = 1
    for index in range(min(len(x), len(x[0]) if x else 0)):
        if x[index][index] != 0:
            product *= x[index][index]
    return product


def are_multisets_equal(x, y):
    """Return True if both vectors create equal multisets.

    input:
    x, y -- 1-d numpy arrays
    output:
    True if multisets are equal, False otherwise -- boolean

    Not vectorized implementation.
    """

    if len(x) != len(y):
        return False

    unmatched = list(y)
    for value in x:
        if value not in unmatched:
            return False
        unmatched.remove(value)
    return True


def max_after_zero(x):
    """Find max element after zero in array.

    input:
    x -- 1-d numpy array
    output:
    maximum element after zero -- integer number

    Not vectorized implementation.
    """

    values_after_zero = []
    for index in range(1, len(x)):
        if x[index - 1] == 0:
            values_after_zero.append(x[index])

    if not values_after_zero:
        raise ValueError("В массиве нет элементов, перед которыми стоит ноль")
    return max(values_after_zero)


def convert_image(img, coefs):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x 3)
    coefs -- 1-d numpy array (length 3)
    output:
    img -- 2-d numpy array

    Not vectorized implementation.
    """

    result = []
    for row in img:
        result_row = []
        for pixel in row:
            weighted_sum = 0
            for channel, coef in zip(pixel, coefs):
                weighted_sum += channel * coef
            result_row.append(weighted_sum)
        result.append(result_row)
    return result


def run_length_encoding(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Not vectorized implementation.
    """

    if len(x) == 0:
        return [], []

    elements = [x[0]]
    counters = [1]
    for value in x[1:]:
        if value == elements[-1]:
            counters[-1] += 1
        else:
            elements.append(value)
            counters.append(1)
    return elements, counters


def pairwise_distance(x, y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Not vectorized implementation.
    """

    distances = []
    for x_object in x:
        row = []
        for y_object in y:
            squared_distance = 0
            for x_coordinate, y_coordinate in zip(x_object, y_object):
                squared_distance += (x_coordinate - y_coordinate) ** 2
            row.append(squared_distance ** 0.5)
        distances.append(row)
    return distances
