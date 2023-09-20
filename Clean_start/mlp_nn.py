# All functions regarding the Multi layer peceptron (basic neuronal network)
import numpy as np
import torch
from torch import nn

from Clean_start.helper_functions import build_sets


class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, hidden_size_2, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size_2)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size_2, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


def train_mlp(batches, test_size, window_size, epochs, learning_rate_value, truncate_value, truncate_value_front):

    x_train, x_test, y_train, y_test = build_sets(batches, test_size, truncate_value, truncate_value_front)

    # Convert data to PyTorch tensors
    x_train_tensor = torch.tensor(np.array(x_train), dtype=torch.float32)
    y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.float32)
    x_test_tensor = torch.tensor(np.array(x_test), dtype=torch.float32)
    y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.float32)

    # Hyperparameters
    input_size = ((truncate_value - truncate_value_front) * 2) + 1  # Size of each input vector (e.g., window size)
    hidden_size = 10  # Number of hidden units in the ANN
    hidden_size_2 = 5  # Number of hidden units in the ANN
    output_size = (truncate_value - truncate_value_front) * 2  # Size of the predicted output vector

    # Create an instance of the neural network
    neural_net = NeuralNetwork(input_size, hidden_size, hidden_size_2, output_size)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(neural_net.parameters(), lr=learning_rate_value)

    # Train the model
    for epoch in range(epochs):
        outputs = neural_net(x_train_tensor)
        optimizer.zero_grad()
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    with torch.no_grad():
        predictions = neural_net(x_test_tensor)

    return predictions, y_test
