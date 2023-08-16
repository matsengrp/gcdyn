from functools import reduce

import numpy as onp
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from gcdyn import poisson
from gcdyn.models import NeuralNetworkModel


class TorchModel(NeuralNetworkModel, nn.Module):
    @staticmethod
    def _partition_list(input_list, sublist_len):
        """
        Partitions a given list into sublists of size sublist_len.
        """
        if len(input_list) % sublist_len != 0:
            raise ValueError(
                f"The length of the input list ({len(input_list)}) is not divisible by {sublist_len}"
            )
        return [
            input_list[i : i + sublist_len]
            for i in range(0, len(input_list), sublist_len)
        ]

    @staticmethod
    def _collapse_identical_list(lst):
        if not lst:
            raise ValueError("List is empty")

        first_element = lst[0]

        for item in lst:
            if item != first_element:
                raise ValueError(
                    f"All items in the list are not identical: {first_element} vs {item}"
                )

        return first_element
    
    @staticmethod
    def _bundle_mean(tensor, bundle_size):
        """Computes the mean of a tensor along the first dimension, 
        after bundling the first dimension into groups of size bundle_size."""
        # Ensure that the tensor's first dimension is divisible by bundle_size
        assert tensor.size(0) % bundle_size == 0, "The tensor's size must be divisible by bundle_size"

        # Reshape the tensor so that the bundles are a separate dimension
        reshaped = tensor.view(-1, bundle_size, *tensor.shape[1:])
        
        # Compute the mean along the bundling dimension
        return reshaped.mean(dim=1)

    def __init__(
        self,
        encoded_trees: list[onp.ndarray],
        responses: list[list[poisson.Response]],
        bundle_size: int,
    ):
        """
        encoded_trees: list of encoded trees
        responses: list of response objects for each tree, i.e. list of lists of responses, with
                   first dimension the same length as encoded_trees, and second dimension with length the
                   number of parameters to predict for each tree. Each response (atm) should just be a constant response
                   with one parameter. (Responses that aren't being estimated need not be provided)
        network_layers: Ignored.
        """
        nn.Module.__init__(self)  # Initialize the PyTorch Module

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
        print("Leaf counts:", leaf_counts)
        self.max_leaf_count = max_leaf_count
        self.training_trees = encoded_trees
        self.responses = responses

        self.conv1 = nn.Conv1d(in_channels=4, out_channels=25, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=25, out_channels=25, kernel_size=8)
        self.maxpool = nn.MaxPool1d(kernel_size=10, stride=10)
        self.conv3 = nn.Conv1d(in_channels=25, out_channels=40, kernel_size=8)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(40, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, num_parameters)
        self.activation = nn.LeakyReLU(negative_slope=0.1)

        if torch.backends.cudnn.is_available():
            print("Using CUDA")
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("Using Metal Performance Shaders")
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.to(self.device)

        self.bundle_size = bundle_size

    def forward_per_bundle(self, x):
        """The x's are a bundle of trees."""
        # We are not doing any permuting because nn.Conv1d expects the input to be of shape (N, C, L)
        # here shape is [10, 4, 100]: 10 trees, 4 channels, 100 max leaf count
        x = self.activation(self.conv1(x)) # 25 convolutions with kernel size 3
        # here shape is [10, 25, 98]: 10 trees, 25 convolutions, and the convolutions have shrunk the length by 2
        x = self.activation(self.conv2(x)) # 25 convolutions with kernel size 8
        # here shape is [10, 25, 91]: 10 trees, 25 convolutions, and the convolutions have shrunk the length by 7
        x = self.maxpool(x) # kernel size 10, stride 10: take the max over 10 disjoint blocks of length 10
        # here shape is [10, 25, 9]: 10 trees, 25 convolutions, and after the max-ing step we have 9 blocks of length 1
        x = self.activation(self.conv3(x)) # 40 convolutions with kernel size 8
        # here shape is [10, 40, 2]: 10 trees, 40 convolutions, and the convolutions have shrunk the length by 7
        x = self.avgpool(x) # kernel size 1, stride 1: take the average over the sequence, now of length 2
        # here shape is [10, 40, 1]: 10 trees, 40 convolutions, and after the avg-ing step we have 1 block of length 1
        # The following line flattens the tensor from shape [10, 40, 1] to shape [10, 40]. 
        # The -1 in the view call means to put whatever is needed there so that the total number of elements is the same and we get to a 2D tensor.
        # In this case the last dimension is fake so we can just get rid of it.
        x = x.view(x.size(0), -1)  
        x = torch.mean(x, 0) # mean over the 10 trees
        return x
        
    def forward_slow(self, x):
        x = torch.stack([self.forward_per_bundle(bundle) for bundle in x])
        # With 400 training bundles, this is shape [400, 40]
        x = self.activation(self.fc1(x))
        # Then shape [400, 32]
        x = self.activation(self.fc2(x))
        # Then shape [400, 16]
        x = self.activation(self.fc3(x))
        # Then shape [400, 8]
        x = self.fc4(x)
        # Then shape [400, num_parameters]
        return x

    def forward(self, x):
        """The x's are all of the trees, ordered into bundles."""
        # We are not doing any permuting because nn.Conv1d expects the input to be of shape (N, C, L)
        # NOTE the following comments haven't been updated to reflect the fact that we are not bundling the trees.
        # here shape is [10, 4, 100]: 10 trees, 4 channels, 100 max leaf count
        x = self.activation(self.conv1(x)) # 25 convolutions with kernel size 3
        # here shape is [10, 25, 98]: 10 trees, 25 convolutions, and the convolutions have shrunk the length by 2
        x = self.activation(self.conv2(x)) # 25 convolutions with kernel size 8
        # here shape is [10, 25, 91]: 10 trees, 25 convolutions, and the convolutions have shrunk the length by 7
        x = self.maxpool(x) # kernel size 10, stride 10: take the max over 10 disjoint blocks of length 10
        # here shape is [10, 25, 9]: 10 trees, 25 convolutions, and after the max-ing step we have 9 blocks of length 1
        x = self.activation(self.conv3(x)) # 40 convolutions with kernel size 8
        # here shape is [10, 40, 2]: 10 trees, 40 convolutions, and the convolutions have shrunk the length by 7
        x = self.avgpool(x) # kernel size 1, stride 1: take the average over the sequence, now of length 2
        # here shape is [10, 40, 1]: 10 trees, 40 convolutions, and after the avg-ing step we have 1 block of length 1
        # The following line flattens the tensor from shape [10, 40, 1] to shape [10, 40]. 
        # The -1 in the view call means to put whatever is needed there so that the total number of elements is the same and we get to a 2D tensor.
        # In this case the last dimension is fake so we can just get rid of it.
        x = x.view(x.size(0), -1)
        # Take the mean over the 10 trees
        x = self._bundle_mean(x, self.bundle_size)
        # With 400 training bundles, this is shape [400, 40]
        x = self.activation(self.fc1(x))
        # Then shape [400, 32]
        x = self.activation(self.fc2(x))
        # Then shape [400, 16]
        x = self.activation(self.fc3(x))
        # Then shape [400, 8]
        x = self.fc4(x)
        # Then shape [400, num_parameters]
        return x

    def _make_tensor(self, array_list):
        return torch.tensor(onp.stack(array_list)).float().to(self.device)

    def _bundle_trees_into_tensors(self, encoded_trees):
        return [
            self._make_tensor(lst)
            for lst in self._partition_list(encoded_trees, self.bundle_size)
        ]

    def _mean_prediction_from_bundled_tensors(self, bundled_tensors):
        return torch.stack([torch.mean(self(bundle), 0) for bundle in bundled_tensors])

    def fit(self, epochs=30):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters())

        # original version (slow):
        # training_data = self._bundle_trees_into_tensors(self.training_trees)
        training_data = self._make_tensor(self.training_trees)
        response_parameters = self._make_tensor(
            self._encode_responses(
                [
                    self._collapse_identical_list(lst)
                    for lst in self._partition_list(self.responses, self.bundle_size)
                ]
            )
        )

        print(f"We have {len(training_data)} bundles of shape {training_data[0].shape}")
        print("Response shape:", response_parameters.shape)

        for _ in tqdm(range(epochs)):
            optimizer.zero_grad()
            outputs = self.forward(training_data)
            # outputs = self._mean_prediction_from_bundled_tensors(training_data)
            loss = criterion(outputs, response_parameters)
            loss.backward()
            optimizer.step()

    def predict(self, encoded_trees):
        with torch.no_grad():
            # Here's the version where we don't bundle the trees
            input_data = self._make_tensor(encoded_trees)
            predicted_responses = self(input_data)
            # Here's the version where we do bundle the trees:
            # input_data = self._bundle_trees_into_tensors(encoded_trees)
            # predicted_responses = self.forward(input_data)
            # TEMPORARY: Unbundle the responses
            predicted_responses = [
                item for item in predicted_responses for _ in range(self.bundle_size)
            ]

        return self._decode_responses(
            predicted_responses, example_responses=self.responses[0]
        )
