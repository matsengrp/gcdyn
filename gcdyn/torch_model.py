
from functools import reduce
import numpy as onp

# CUT
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from gcdyn import poisson
from gcdyn.models import NeuralNetworkModel

class TorchModel(NeuralNetworkModel):
    def __init__(
        self,
        encoded_trees: list[onp.ndarray],
        responses: list[list[poisson.Response]],
        network_layers: list[callable] = None,
    ):
        """
        encoded_trees: list of encoded trees
        responses: list of response objects for each tree, i.e. list of lists of responses, with
                   first dimension the same length as encoded_trees, and second dimension with length the
                   number of parameters to predict for each tree. Each response (atm) should just be a constant response
                   with one parameter. (Responses that aren't being estimated need not be provided)
        network_layers: Ignored.
        """
        num_parameters = sum(len(response._param_dict) for response in responses[0])
        leaf_counts = set(
            [len(t[0]) for t in encoded_trees]
        )  # length of first row in encoded tree
        if len(leaf_counts) != 1:
            raise Exception(
                "encoded trees have different lengths: %s"
                % " ".join(str(c) for c in leaf_counts)
            )
        max_leaf_count = list(leaf_counts)[0]

        actfn = None #'elu'  # NOTE 'elu' was causing a bad lower threshold when scaling input variables
        network_layers = (
            # Rotate matrix from (4, leaf_count) to (leaf_count, 4)
            lambda x: tf.transpose(x, (0, 2, 1)),
            layers.Conv1D(filters=25, kernel_size=3, activation=actfn),
            layers.Conv1D(filters=25, kernel_size=8, activation=actfn),
            layers.MaxPooling1D(pool_size=10, strides=10),
            layers.Conv1D(filters=40, kernel_size=8, activation=actfn),
            layers.GlobalAveragePooling1D(),
            layers.Dense(32, activation=actfn),
            layers.Dense(16, activation=actfn),
            layers.Dense(8, activation=actfn),
            layers.Dense(num_parameters, activation=actfn),
        )

        inputs = keras.Input(shape=(4, max_leaf_count))
        outputs = reduce(lambda x, layer: layer(x), network_layers, inputs)
        self.network = keras.Model(inputs=inputs, outputs=outputs)
        self.network.summary(print_fn=lambda x: print("      %s" % x))

        # Note: the original deep learning model rescales trees, but we don't here
        # because we should always have the same root to tip height.
        self.max_leaf_count = max_leaf_count
        self.training_trees = encoded_trees

        self.responses = responses

    def fit(self, epochs: int = 30):
        """Trains neural network on given trees and response parameters."""
        response_parameters = self._encode_responses(self.responses)

        self.network.compile(loss="mean_squared_error")
        self.network.fit(
            onp.stack(self.training_trees), response_parameters, epochs=epochs, verbose=0,
        )

    def predict(
        self,
        encoded_trees: list[onp.ndarray],
    ) -> list[list[poisson.Response]]:
        """Returns the Response objects predicted for each tree."""

        response_parameters = self.network(onp.stack(encoded_trees))

        return self._decode_responses(
            response_parameters, example_responses=self.responses[0]
        )
