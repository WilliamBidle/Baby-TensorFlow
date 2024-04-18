""" Tests for the artifice.loss_functions module that are compared to the corresponding Tensorflow loss function. """

import pytest
import numpy as np
import tensorflow as tf

from artifice.loss_functions import (
    MeanSquaredError,
    MeanAbsoluteError,
    MeanAbsolutePercentError,
    MeanLogSquaredError,
    PoissonError,
    BinaryCrossEntropy,
)

Y_TRUE = np.array([[1, 2, 3], [1, 2, 3]], dtype=float)
Y_PRED = np.array([[1, 2, 3], [0, 0, 0]], dtype=float)


@pytest.mark.parametrize(
    "y_pred, y_true, expected",
    [
        (y_pred, y_true, tf.keras.losses.MeanSquaredError()(y_true, y_pred).numpy())
        for y_pred, y_true in zip(Y_PRED, Y_TRUE)
    ],
)
def test_mean_squared_error(y_pred, y_true, expected):
    """Testing the MeanSquaredError loss function."""

    loss_func = MeanSquaredError()
    res = loss_func.evaluate(y_pred=y_pred, y_true=y_true)
    assert np.allclose(res, expected)


@pytest.mark.parametrize(
    "y_pred, y_true, expected",
    [
        (y_pred, y_true, tf.keras.losses.MeanAbsoluteError()(y_true, y_pred).numpy())
        for y_pred, y_true in zip(Y_PRED, Y_TRUE)
    ],
)
def test_mean_absolute_error(y_pred, y_true, expected):
    """Testing the MeanAbsoluteError loss function."""

    loss_func = MeanAbsoluteError()
    res = loss_func.evaluate(y_pred=y_pred, y_true=y_true)
    assert np.allclose(res, expected)


@pytest.mark.parametrize(
    "y_pred, y_true, expected",
    [
        (
            y_pred,
            y_true,
            tf.keras.losses.MeanAbsolutePercentageError()(y_true, y_pred).numpy(),
        )
        for y_pred, y_true in zip(Y_PRED, Y_TRUE)
    ],
)
def test_mean_absolute_percent_error(y_pred, y_true, expected):
    """Testing the MeanAbsolutePercentError loss function."""

    loss_func = MeanAbsolutePercentError()
    res = loss_func.evaluate(y_pred=y_pred, y_true=y_true)
    assert np.allclose(res, expected)


@pytest.mark.parametrize(
    "y_pred, y_true, expected",
    [
        (
            y_pred,
            y_true,
            tf.keras.losses.MeanSquaredLogarithmicError()(y_true, y_pred).numpy(),
        )
        for y_pred, y_true in zip(Y_PRED, Y_TRUE)
    ],
)
def test_mean_log_squared_error(y_pred, y_true, expected):
    """Testing the MeanLogSquaredError loss function."""

    loss_func = MeanLogSquaredError()
    res = loss_func.evaluate(y_pred=y_pred, y_true=y_true)
    assert np.allclose(res, expected)


@pytest.mark.parametrize(
    "y_pred, y_true, expected",
    [
        (y_pred, y_true, tf.keras.losses.Poisson()(y_true, y_pred).numpy())
        for y_pred, y_true in zip(Y_PRED, Y_TRUE)
    ],
)
def test_poisson_error(y_pred, y_true, expected):
    """Testing the PoissonError loss function."""

    loss_func = PoissonError()
    res = loss_func.evaluate(y_pred=y_pred, y_true=y_true)
    assert np.allclose(res, expected)


# Binary Crossentropy needs to deal with probability outputs
Y_TRUE = np.array([[0.5], [0.25]], dtype=float)
Y_PRED = np.array([[0.5], [0.75]], dtype=float)


@pytest.mark.parametrize(
    "y_pred, y_true, expected",
    [
        (y_pred, y_true, tf.keras.losses.BinaryCrossentropy()(y_true, y_pred).numpy())
        for y_pred, y_true in zip(Y_PRED, Y_TRUE)
    ],
)
def test_binary_cross_entropy_error(y_pred, y_true, expected):
    """Testing the BinaryCrossEntropy loss function."""

    loss_func = BinaryCrossEntropy()
    res = loss_func.evaluate(y_pred=y_pred, y_true=y_true)
    assert np.allclose(res, expected)
