def prod_non_zero_diag(x):
    """Compute product of nonzero elements from matrix diagonal.

    input:
    x -- 2-d numpy array
    output:
    product -- integer number


    Not vectorized implementation.
    """

    product = 1
    diagonal_length = min(len(x), len(x[0]))

    for index in range(diagonal_length):
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

    return sorted(x) == sorted(y)


def max_after_zero(x):
    """Find max element after zero in array.

    input:
    x -- 1-d numpy array
    output:
    maximum element after zero -- integer number

    Not vectorized implementation.
    """

    result = []

    for index in range(1, len(x)):
        if x[index - 1] == 0:
            result.append(x[index])

    return max(result)


def convert_image(img, coefs):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x 3)
    coefs -- 1-d numpy array (length 3)
    output:
    img -- 2-d numpy array

    Not vectorized implementation.
    """

    new_image = []

    for row in img:
        new_row = []

        for pixel in row:
            new_pixel = 0

            for channel, coef in zip(pixel, coefs):
                new_pixel += channel * coef

            new_row.append(new_pixel)

        new_image.append(new_row)

    return new_image


def run_length_encoding(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Not vectorized implementation.
    """

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

    result = []

    for first_object in x:
        row = []

        for second_object in y:
            distance = 0

            for first, second in zip(first_object, second_object):
                distance += (first - second) ** 2

            row.append(distance ** 0.5)

        result.append(row)

    return result
