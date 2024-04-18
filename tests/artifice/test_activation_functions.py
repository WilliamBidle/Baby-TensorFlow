""" Tests for the artifice.activation_functions module. """

import pytest
import numpy as np
import tensorflow as tf

from artifice.activation_functions import (
    ReLU,
    Sigmoid,
    Tanh,
    Linear,
)

X = np.array([-1, 0, -1], dtype=float)


def test_relu():
    """Testing the ReLU activation function."""

    activation_func = ReLU()
    res = activation_func.evaluate(x=X)
    expected = tf.keras.activations.relu(X).numpy()
    assert np.allclose(res, expected)


def test_sigmoid():
    """Testing the Sigmoid activation function."""

    activation_func = Sigmoid()
    res = activation_func.evaluate(x=X)
    expected = tf.keras.activations.sigmoid(X).numpy()
    assert np.allclose(res, expected)


def test_tanh():
    """Testing the Tanh activation function."""

    activation_func = Tanh()
    res = activation_func.evaluate(x=X)
    expected = tf.keras.activations.tanh(X).numpy()
    assert np.allclose(res, expected)


def test_linear():
    """Testing the Linear activation function."""

    activation_func = Linear()
    res = activation_func.evaluate(x=X)
    expected = tf.keras.activations.linear(X)
    assert np.allclose(res, expected)
