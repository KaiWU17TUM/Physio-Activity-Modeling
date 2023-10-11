# All functions regarding the Multi layer peceptron (basic neuronal network)
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.utils.data import DataLoader

from Clean_start.helper_functions import build_sets, CustomDataset


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


log_dir = "logs/"  # Set your log directory
summary_writer = SummaryWriter(log_dir)

def train_mlp(batches, test_size, window_size, epochs, learning_rate_value, truncate_value, truncate_value_front, hidden_size, hidden_size2):

    x_train, x_test, y_train, y_test, y_test_untransformed = build_sets(batches, test_size, truncate_value, truncate_value_front)

    # # Convert data to PyTorch tensors
    # x_train_tensor = torch.tensor(np.array(x_train), dtype=torch.float32)
    # y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.float32)
    x_test_tensor = torch.tensor(np.array(x_test), dtype=torch.float32)
    #y_test_tensor_untransformed = torch.tensor(np.array(y_test_untransformed), dtype=torch.float32)

    # Create train and validation datasets
    train_dataset = CustomDataset(x_train, y_train)
    val_dataset = CustomDataset(x_test, y_test)

    # Step 3: Create DataLoader objects for training and validation datasets
    batch_size = 64  # Set your desired batch size

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) #TODO is true possible yes !
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Hyperparameters
    input_size = ((truncate_value - truncate_value_front) * 2) + 1  # Size of each input vector (e.g., window size)
    output_size = (truncate_value - truncate_value_front) * 2  # Size of the predicted output vector

    # Create an instance of the neural network
    neural_net = NeuralNetwork(input_size, hidden_size, hidden_size2, output_size)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(neural_net.parameters(), lr=learning_rate_value)

    # Define early stopping parameters
    patience = 10  # Number of epochs to wait for improvement
    min_val_loss = float('inf')  # Initialize the minimum validation loss
    early_stop_counter = 0  # Counter to track epochs without improvement

    # Train the model
    for epoch in range(epochs):
        # Training phase
        neural_net.train()
        for inputs, targets in train_loader:
            outputs = neural_net(inputs)
            optimizer.zero_grad()
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validation phase
        neural_net.eval()
        validation_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:  # Assuming you have a validation_loader for validation data
                val_outputs = neural_net(inputs)
                val_loss = criterion(val_outputs, targets)
                validation_loss += val_loss.item()

        # Calculate average validation loss for the epoch
        average_validation_loss = validation_loss / len(val_loader)

        # Check for early stopping
        if average_validation_loss < min_val_loss:
            min_val_loss = average_validation_loss
            early_stop_counter = 0
            # Save the model checkpoint here if needed
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f'Early stopping after {epoch + 1} epochs without improvement.')
                break

        # Log scalar values
        summary_writer.add_scalar("train_loss", loss.item(), epoch)
        summary_writer.add_scalar("validation_loss", average_validation_loss, epoch)

        # Log scalar values
        #summary_writer.add_scalar("loss", loss, epoch)

        # summary_writer.add_scalar("accuracy", accuracy, epoch)
        # print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    with torch.no_grad():
        predictions = neural_net(x_test_tensor)

    return predictions,y_test, y_test_untransformed
