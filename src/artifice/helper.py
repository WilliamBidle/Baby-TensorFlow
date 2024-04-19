""" List of helper functions. """

import pickle
from typing import Tuple
import numpy as np

from artifice.network import NN


def apply_random_permutation(
    array_1: np.ndarray, array_2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform the same random permutation operation on two arrays.

    :params array_1: First array to permute.
    :params array_2: Second array to permute.
    :returns: Permuted arrays.
    """

    # Check that both arrays have same
    if len(array_1) != len(array_2):
        raise ValueError("Both arrays must be the same length.")

    # Get random permutation
    p = np.random.permutation(len(array_1))

    # Apply random permutation
    array_1 = array_1[p]
    array_2 = array_2[p]

    return array_1, array_2


def one_hot_encode(labels):
    """
    Performs One-Hot-Encoding of labels.

    :params filename:
    :returns encoded_labels:
    """

    num_unique_elements = np.unique(labels)

    # Create a dictionary mapping unique elements to their index
    dic = {val: idx for idx, val in enumerate(num_unique_elements)}

    encoded_labels = np.zeros((len(labels), len(num_unique_elements)))

    for index, label in enumerate(labels):
        encoded_labels[index][dic[label]] = 1.0

    return encoded_labels
