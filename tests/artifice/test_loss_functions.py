""" Tests for the artifice.loss_functions module that are compared to the corresponding Tensorflow loss function. """

import pytest
import numpy as np
import tensorflow as tf

from artifice.loss_functions import (
    LossFunction,
    MeanSquaredError,
    MeanAbsoluteError,
    MeanAbsolutePercentError,
    MeanLogSquaredError,
    PoissonError,
    BinaryCrossEntropy,
)


def test_loss_function():
    """Testing general properties of a loss function."""

    loss_func = LossFunction()

    # Check that initialization of expressions and expression derivative is None
    assert loss_func.expression is None
    assert loss_func.expression_diff is None

    # Check that trying to evaluate the base class unset loss function will result in an error.
    with pytest.raises(AssertionError, match="Loss function is not set!"):
        loss_func.evaluate(None, None)


Y_TRUE = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=float)
Y_PRED = np.array([[1.0, 2.0, 3.0], [0.1, 0.2, 0.3]], dtype=float)


@pytest.mark.parametrize(
    "y_pred, y_true",
    [(y_pred, y_true) for y_pred, y_true in zip(Y_PRED, Y_TRUE)],
)
def test_mean_squared_error(y_pred, y_true):
    """Testing the MeanSquaredError loss function."""

    loss_func = MeanSquaredError()
    res = loss_func.evaluate(y_pred=y_pred, y_true=y_true)
    expected = tf.keras.losses.MeanSquaredError()(y_true, y_pred).numpy()
    assert np.allclose(res, expected)

    # Evaluate derivative and compare with Tensorflow
    dydx = loss_func.evaluate(y_pred=y_pred, y_true=y_true, diff=True)
    y_true, y_pred = tf.constant(y_true), tf.constant(y_pred)
    with tf.GradientTape() as g:
        g.watch(y_pred)
        y = tf.keras.losses.MeanSquaredError()(y_true, y_pred)
    expected_dydx = g.gradient(y, y_pred).numpy()
    assert np.allclose(dydx, expected_dydx)


@pytest.mark.parametrize(
    "y_pred, y_true",
    [(y_pred, y_true) for y_pred, y_true in zip(Y_PRED, Y_TRUE)],
)
def test_mean_absolute_error(y_pred, y_true):
    """Testing the MeanAbsoluteError loss function."""

    loss_func = MeanAbsoluteError()
    res = loss_func.evaluate(y_pred=y_pred, y_true=y_true)
    expected = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred).numpy()
    assert np.allclose(res, expected)

    # Evaluate derivative and compare with Tensorflow
    dydx = loss_func.evaluate(y_pred=y_pred, y_true=y_true, diff=True)
    y_true, y_pred = tf.constant(y_true), tf.constant(y_pred)
    with tf.GradientTape() as g:
        g.watch(y_pred)
        y = tf.keras.losses.MeanAbsoluteError()(y_true, y_pred)
    expected_dydx = g.gradient(y, y_pred).numpy()
    assert np.allclose(dydx, expected_dydx)


@pytest.mark.parametrize(
    "y_pred, y_true",
    [
        (
            y_pred,
            y_true,
        )
        for y_pred, y_true in zip(Y_PRED, Y_TRUE)
    ],
)
def test_mean_absolute_percent_error(y_pred, y_true):
    """Testing the MeanAbsolutePercentError loss function."""

    loss_func = MeanAbsolutePercentError()
    res = loss_func.evaluate(y_pred=y_pred, y_true=y_true)
    expected = tf.keras.losses.MeanAbsolutePercentageError()(y_true, y_pred).numpy()
    assert np.allclose(res, expected)

    # Evaluate derivative and compare with Tensorflow
    dydx = loss_func.evaluate(y_pred=y_pred, y_true=y_true, diff=True)
    y_true, y_pred = tf.constant(y_true), tf.constant(y_pred)
    with tf.GradientTape() as g:
        g.watch(y_pred)
        y = tf.keras.losses.MeanAbsolutePercentageError()(y_true, y_pred)
    expected_dydx = g.gradient(y, y_pred).numpy()
    assert np.allclose(dydx, expected_dydx)


@pytest.mark.parametrize(
    "y_pred, y_true",
    [
        (
            y_pred,
            y_true,
        )
        for y_pred, y_true in zip(Y_PRED, Y_TRUE)
    ],
)
def test_mean_log_squared_error(y_pred, y_true):
    """Testing the MeanLogSquaredError loss function."""

    loss_func = MeanLogSquaredError()
    res = loss_func.evaluate(y_pred=y_pred, y_true=y_true)
    expected = tf.keras.losses.MeanSquaredLogarithmicError()(y_true, y_pred).numpy()
    assert np.allclose(res, expected)

    # Evaluate derivative and compare with Tensorflow
    dydx = loss_func.evaluate(y_pred=y_pred, y_true=y_true, diff=True)
    y_true, y_pred = tf.constant(y_true), tf.constant(y_pred)
    with tf.GradientTape() as g:
        g.watch(y_pred)
        y = tf.keras.losses.MeanSquaredLogarithmicError()(y_true, y_pred)
    expected_dydx = g.gradient(y, y_pred).numpy()
    assert np.allclose(dydx, expected_dydx)


@pytest.mark.parametrize(
    "y_pred, y_true",
    [(y_pred, y_true) for y_pred, y_true in zip(Y_PRED, Y_TRUE)],
)
def test_poisson_error(y_pred, y_true):
    """Testing the PoissonError loss function."""

    loss_func = PoissonError()
    res = loss_func.evaluate(y_pred=y_pred, y_true=y_true)
    expected = tf.keras.losses.Poisson()(y_true, y_pred).numpy()
    assert np.allclose(res, expected)

    # Evaluate derivative and compare with Tensorflow
    dydx = loss_func.evaluate(y_pred=y_pred, y_true=y_true, diff=True)
    y_true, y_pred = tf.constant(y_true), tf.constant(y_pred)
    with tf.GradientTape() as g:
        g.watch(y_pred)
        y = tf.keras.losses.Poisson()(y_true, y_pred)
    expected_dydx = g.gradient(y, y_pred).numpy()
    assert np.allclose(dydx, expected_dydx, atol=1e-5)


# Binary Crossentropy needs to deal with probability outputs
Y_TRUE = np.array([[0.5], [0.25]], dtype=float)
Y_PRED = np.array([[0.5], [0.75]], dtype=float)


@pytest.mark.parametrize(
    "y_pred, y_true",
    [(y_pred, y_true) for y_pred, y_true in zip(Y_PRED, Y_TRUE)],
)
def test_binary_cross_entropy_error(y_pred, y_true):
    """Testing the BinaryCrossEntropy loss function."""

    loss_func = BinaryCrossEntropy()
    res = loss_func.evaluate(y_pred=y_pred, y_true=y_true)
    expected = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred).numpy()
    assert np.allclose(res, expected)

    # Evaluate derivative and compare with Tensorflow
    dydx = loss_func.evaluate(y_pred=y_pred, y_true=y_true, diff=True)
    y_true, y_pred = tf.constant(y_true), tf.constant(y_pred)
    with tf.GradientTape() as g:
        g.watch(y_pred)
        y = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    expected_dydx = g.gradient(y, y_pred).numpy()
    assert np.allclose(dydx, expected_dydx)
