# %%
import numpy as np

# NOTE: sphinx is currently unable to present this in condensed form, using a string type hint
# of "array-like" in the docstring args for now, instead of ArrayLike hint in call signature
# from numpy.typing import ArrayLike

import tensorflow as tf
from functools import reduce, partial
from gcdyn import poisson, mutators, utils

# from tensorflow import keras
# from tensorflow.keras import layers
# Pylance can't recognize tf submodules imported the normal way
keras = tf.keras
layers = keras.layers


class NeuralNetworkModel:
    def __init__(self, trees, responses, max_leaf_count=200, ladderize_trees=True):
        """
        trees: sequence of `bdms.TreeNode`s
        responses: 2D sequence, with first dimension corresponding to trees
                   and second dimension to the Response objects to predict
                   (Responses that aren't being estimated need not be provided)
        max_leaf_count: will specify the size of the second dimension of encoded trees
        ladderize_trees: if trees are already ladderized, set this to `False` to save computing time
        """
        num_parameters = sum(len(response._param_dict) for response in responses[0])

        network_layers = (
            # Rotate matrix from (4, leaf_count) to (leaf_count, 4)
            lambda x: tf.transpose(x, (0, 2, 1)),
            layers.Conv1D(filters=50, kernel_size=3, activation="elu"),
            layers.Conv1D(filters=50, kernel_size=10, activation="elu"),
            layers.MaxPooling1D(pool_size=10, strides=10),
            layers.Conv1D(filters=80, kernel_size=10, activation="elu"),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation="elu"),
            layers.Dense(32, activation="elu"),
            layers.Dense(16, activation="elu"),
            layers.Dense(8, activation="elu"),
            layers.Dense(num_parameters, activation="elu"),
        )

        inputs = keras.Input(shape=(4, max_leaf_count))
        outputs = reduce(lambda x, layer: layer(x), network_layers, inputs)
        self.network = keras.Model(inputs=inputs, outputs=outputs)

        # TODO: should we rescale the trees too?
        self.max_leaf_count = max_leaf_count

        if ladderize_trees:
            for tree in trees:
                utils.ladderize_tree(tree)

        self.trees = trees
        self.encoded_trees = np.stack(
            [self._encode_tree(tree, self.max_leaf_count) for tree in trees]
        )
        self.responses = responses

    @classmethod
    def _encode_tree(cls, tree, max_leaf_count):
        """Returns the "Compact Bijective Ladderized Vector" form of the given
        ladderized tree."""

        def traverse_inorder(tree):
            """In-order traversal, generalizable to non-binary trees (by
            partitioning into left and right halves of children to visit)"""
            num_children = len(tree.children)

            for child in tree.children[: num_children // 2]:
                yield from traverse_inorder(child)

            yield tree

            for child in tree.children[num_children // 2 :]:
                yield from traverse_inorder(child)

        assert len(tree.get_leaves()) <= max_leaf_count
        matrix = np.zeros((4, max_leaf_count))

        leaf_index = 0
        ancestor_index = 0
        previous_ancestor = tree  # the root

        for node in traverse_inorder(tree):
            if node.is_leaf():
                matrix[0, leaf_index] = node.t - previous_ancestor.t
                matrix[2, leaf_index] = node.x
                leaf_index += 1
            else:
                matrix[1, ancestor_index] = node.t
                matrix[3, ancestor_index] = node.x
                ancestor_index += 1
                previous_ancestor = node

        return matrix

    @classmethod
    def _encode_responses(cls, responses):
        return np.array(
            [
                np.hstack([response._flatten()[2] for response in row])
                for row in responses
            ]
        )

    @classmethod
    def _decode_responses(cls, response_parameters, example_responses):
        result = []

        for row in response_parameters:
            responses = []
            i = 0

            response_structure = [response._flatten() for response in example_responses]

            for (
                response_cls,
                param_names,
                example_param_values,
            ) in response_structure:
                param_values = row[i : i + len(example_param_values)]
                responses.append(response_cls._from_flat(param_names, param_values))
                i += len(example_param_values)

            result.append(responses)

        return result

    def fit(self, epochs=30):
        """Trains neural network on given trees and response parameters."""

        response_parameters = self._encode_responses(self.responses)

        self.network.compile(loss="mean_squared_error")
        self.network.fit(self.encoded_trees, response_parameters, epochs=epochs)

    def predict(self, trees, ladderize_trees=True):
        """Returns the Response objects predicted for each tree."""

        if ladderize_trees:
            for tree in trees:
                utils.ladderize_tree(tree)

        encoded_trees = np.stack(
            [self._encode_tree(tree, self.max_leaf_count) for tree in trees]
        )

        response_parameters = self.network(encoded_trees)

        return self._decode_responses(
            response_parameters, example_responses=self.responses[0]
        )


# %%
if __name__ == "__main__":
    # A test to make sure we can correctly encode the tree
    # in Figure 2a of https://doi.org/10.1038/s41467-022-31511-0

    import ete3

    tree = ete3.Tree(newick="(a:3, ((b:2, c:1)C:2, (d:1, e:4)D:1)B:1)A:0;", format=3)

    # ete3.Tree doesn't have node.t but I rely on that
    tree.t = 0
    tree.x = 0
    for node in tree.iter_descendants("levelorder"):
        node.t = node.up.t + node.dist
        node.x = node.up.x + 1

    print(tree)

    utils.ladderize_tree(tree)
    print(tree)

    encoded_tree = NeuralNetworkModel._encode_tree(tree, max_leaf_count=10)
    print(encoded_tree)

    # Now try training a nnet

    N = 100

    params = [[poisson.ConstantResponse()] for _ in range(N)]

    sample_tree = partial(
        utils.sample_trees,
        n=1,
        t=2,
        birth_response=poisson.SigmoidResponse(),
        mutation_response=poisson.ConstantResponse(),
        mutator=mutators.GaussianMutator(-1, 1),
    )

    trees = [sample_tree(death_response=row[0])[0] for row in params]

    model = NeuralNetworkModel(trees, params)
    model.fit()

    print(model.predict(trees))

# %%
