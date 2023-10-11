# All functino regarding the LSTM
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader

from Clean_start.helper_functions import build_sets, CustomDataset


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

log_dir = "logs/"  # Set your log directory
summary_writer = SummaryWriter(log_dir)

def train_lstm(batches, input_size, hidden_size, num_layers, output_size, test_size, epochs, learning_rate,
               truncate_value, truncate_value_front):
    x_train, x_test, y_train, y_test, y_test_untransformed = build_sets(batches, test_size, truncate_value, truncate_value_front)

    # Convert data to PyTorch tensors
    x_train_tensor = torch.tensor(np.array(x_train), dtype=torch.float32)
    y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.float32)
    x_test_tensor = torch.tensor(np.array(x_test), dtype=torch.float32)
    y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.float32)

    # Step 3: Create DataLoader objects for training and validation datasets
    batch_size = 64  # Set your desired batch size

    batch_size, sequence_length, input_size = x_train_tensor.shape[0], 1, x_train_tensor.shape[1]
    x_train_tensor = x_train_tensor.view(batch_size, sequence_length, input_size)
    batch_size, sequence_length, input_size = x_test_tensor.shape[0], 1, x_test_tensor.shape[1]
    x_test_tensor = x_test_tensor.view(batch_size, sequence_length, input_size)

    # Create train and validation datasets
    train_dataset = CustomDataset(x_train_tensor, y_train_tensor)
    val_dataset = CustomDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False) #TODO is true possible yes !
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    #TODO fix input size !!

    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()  # Mean Squared Error loss for regression tasks
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define early stopping parameters
    patience = 5  # Number of epochs to wait for improvement
    min_val_loss = float('inf')  # Initialize the minimum validation loss
    early_stop_counter = 0  # Counter to track epochs without improvement

    # Step 5: Train the model
    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                val_outputs = model(inputs)
                val_loss += criterion(val_outputs, targets).item()

        # Calculate average validation loss for the epoch
        average_validation_loss = val_loss / len(val_loader)

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

        # print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    with torch.no_grad():
        predictions = model(x_test_tensor)

    return predictions,y_test, y_test_untransformed
