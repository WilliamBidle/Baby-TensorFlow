""" Tests for the artifice.network module. """

import numpy as np

from artifice.network import NN
from artifice.loss_functions import MeanSquaredError
from artifice.activation_functions import ReLU, Sigmoid


class TestNN:  # pylint: disable=too-few-public-methods
    """
    Tests for NN object.
    """

    def test_get_attributes(self):
        """
        Test get_attributes.
        """

        layer_sequence = [1, "relu", 2, "sigmoid", 3]
        loss_function = "MSE"

        nn = NN(layer_sequence, loss_function)

        # Check correct weight initialization
        assert nn.weights[0].shape == (2, 2)
        assert nn.weights[1].shape == (3, 3)

        # Check correct activation function initialization
        for activation_func in nn.activation_funcs:
            assert isinstance(activation_func, (ReLU, Sigmoid))

        # Check correct loss function initialization
        assert isinstance(nn.loss_func, MeanSquaredError)
        assert nn.loss_func_label == "MSE"

        # Check correct training error initialization
        assert nn.training_err is None

    def test_train(self):

        # Build a model
        layer_sequence = [10, "relu", 2, "sigmoid", 1]
        loss_function = "MSE"
        nn = NN(layer_sequence, loss_function)

        # Train the model on some custom data
        x_train, y_train = np.random.randn(100, layer_sequence[0]), np.random.randn(
            100, layer_sequence[-1]
        )
        nn.train(
            x_train,
            y_train,
            epochs=1,
            batch_size=2,
            epsilon=0.01,
            verbose=False,
            visualize=False,
        )

        # Check that training error updated
        assert nn.training_err is not None

    def test_evaluate(self):

        assert False

    def test_save_model(self):

        assert False

    def test_load_model(self):

        assert False
