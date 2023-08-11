from functools import reduce
import numpy as onp

import torch
import torch.nn as nn
import torch.optim as optim

from gcdyn import poisson
from gcdyn.models import NeuralNetworkModel


class TorchModel(NeuralNetworkModel, nn.Module):

    @staticmethod
    def _partition_list(input_list, sublist_len):
        """
        Partitions a given list into sublists of size sublist_len.
        """
        if len(input_list) % sublist_len != 0:
            raise ValueError(f"The length of the input list ({len(input_list)}) is not divisible by {sublist_len}")
        return [input_list[i:i+sublist_len] for i in range(0, len(input_list), sublist_len)]

    @staticmethod
    def _collapse_identical_list(lst):
        if not lst:
            raise ValueError("List is empty")

        first_element = lst[0]

        for item in lst:
            if item != first_element:
                raise ValueError("All items in the list are not identical")

        return first_element


    def __init__(
        self,
        encoded_trees: list[onp.ndarray],
        responses: list[list[poisson.Response]],
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

    def forward(self, x):
        # We are not doing any permuting because nn.Conv1d expects the input to be of shape (N, C, L)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.maxpool(x)
        x = self.activation(self.conv3(x))
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x

    def _make_tensor(self, array_list):
        return torch.tensor(onp.stack(array_list)).float().to(self.device)

    def fit(self, epochs=30):
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters())

        training_data = self._make_tensor(self.training_trees)
        response_parameters = self._make_tensor(self._encode_responses(self.responses))

        print("Data shape:", training_data.shape)
        print("Response shape:", response_parameters.shape)

        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = self(training_data)
            loss = criterion(outputs, response_parameters)
            loss.backward()
            optimizer.step()

    def predict(self, encoded_trees):
        with torch.no_grad():
            input_data = torch.tensor(onp.stack(encoded_trees)).float().to(self.device)
            response_parameters = self(input_data)
        return self._decode_responses(
            response_parameters, example_responses=self.responses[0]
        )
