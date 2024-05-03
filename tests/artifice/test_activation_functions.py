""" Tests for the artifice.activation_functions module. """

import pytest
import numpy as np
import tensorflow as tf

from artifice.activation_functions import (
    ActivationFunction,
    ReLU,
    Sigmoid,
    Tanh,
    Linear,
    ActivationFunctionConverter,
)

X = np.array([-1, 0, -1], dtype=float)


def test_activation_function():
    """Testing general properties of an activation function."""

    activation_func = ActivationFunction()

    # Check that initialization of expressions and expression derivative is None
    assert activation_func.expression is None
    assert activation_func.expression_diff is None

    # Check that trying to evaluate the base class unset loss function will result in an error.
    with pytest.raises(AssertionError, match="Activation function is not set!"):
        activation_func.evaluate(None, None)


def test_relu():
    """Testing the ReLU activation function."""

    activation_func = ReLU()
    res = activation_func.evaluate(x=X)
    expected = tf.keras.activations.relu(X).numpy()
    assert np.allclose(res, expected)

    # Evaluate derivative and compare with Tensorflow
    dydx = activation_func.evaluate(x=X, diff=True)
    x = tf.constant(X)
    with tf.GradientTape() as g:
        g.watch(x)
        y = tf.keras.activations.relu(x)
    expected_dydx = g.gradient(y, x).numpy()
    assert np.allclose(dydx, expected_dydx)

    # Simple test for __repr__
    repr_test = repr(activation_func)
    assert "Relu Activation Function:" in repr_test


def test_sigmoid():
    """Testing the Sigmoid activation function."""

    activation_func = Sigmoid()
    res = activation_func.evaluate(x=X)
    expected = tf.keras.activations.sigmoid(X).numpy()
    assert np.allclose(res, expected)

    # Evaluate derivative and compare with Tensorflow
    dydx = activation_func.evaluate(x=X, diff=True)
    x = tf.constant(X)
    with tf.GradientTape() as g:
        g.watch(x)
        y = tf.keras.activations.sigmoid(x)
    expected_dydx = g.gradient(y, x).numpy()
    assert np.allclose(dydx, expected_dydx)

    # Simple test for __repr__
    repr_test = repr(activation_func)
    assert "Sigmoid Activation Function:" in repr_test


def test_tanh():
    """Testing the Tanh activation function."""

    activation_func = Tanh()
    res = activation_func.evaluate(x=X)
    expected = tf.keras.activations.tanh(X).numpy()
    assert np.allclose(res, expected)

    # Evaluate derivative and compare with Tensorflow
    dydx = activation_func.evaluate(x=X, diff=True)
    x = tf.constant(X)
    with tf.GradientTape() as g:
        g.watch(x)
        y = tf.keras.activations.tanh(x)
    expected_dydx = g.gradient(y, x).numpy()
    assert np.allclose(dydx, expected_dydx)

    # Simple test for __repr__
    repr_test = repr(activation_func)
    assert "Tanh Activation Function:" in repr_test


def test_linear():
    """Testing the Linear activation function."""

    activation_func = Linear()
    res = activation_func.evaluate(x=X)
    expected = tf.keras.activations.linear(X)
    assert np.allclose(res, expected)

    # Evaluate derivative and compare with Tensorflow
    dydx = activation_func.evaluate(x=X, diff=True)
    x = tf.constant(X)
    with tf.GradientTape() as g:
        g.watch(x)
        y = tf.keras.activations.linear(x)
    expected_dydx = g.gradient(y, x).numpy()
    assert np.allclose(dydx, expected_dydx)

    # Simple test for __repr__
    repr_test = repr(activation_func)
    assert "Linear Activation Function:" in repr_test


class TestActivationFunctionConverter:
    """Tests the ActivationFunctionConverter class."""

    def test_from_str(self):
        """Testing the from_str function."""
        assert isinstance(ActivationFunctionConverter.from_str("relu"), ReLU)
        assert isinstance(ActivationFunctionConverter.from_str("sigmoid"), Sigmoid)
        assert isinstance(ActivationFunctionConverter.from_str("tanh"), Tanh)
        assert isinstance(ActivationFunctionConverter.from_str("linear"), Linear)

    def test_from_func(self):
        """Testing the from_func function."""
        assert ActivationFunctionConverter.from_func(ReLU) == "relu"
        assert ActivationFunctionConverter.from_func(Sigmoid) == "sigmoid"
        assert ActivationFunctionConverter.from_func(Tanh) == "tanh"
        assert ActivationFunctionConverter.from_func(Linear) == "linear"
