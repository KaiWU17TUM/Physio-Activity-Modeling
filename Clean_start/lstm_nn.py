# All functino regarding the LSTM
import numpy as np
import torch
from torch import nn, optim

from Clean_start.helper_functions import build_sets


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


def train_lstm(batches, input_size, hidden_size, num_layers, output_size, test_size, epochs, learning_rate,
               truncate_value):
    x_train, x_test, y_train, y_test = build_sets(batches, test_size, truncate_value)

    # Convert data to PyTorch tensors
    x_train_tensor = torch.tensor(np.array(x_train), dtype=torch.float32)
    y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.float32)
    x_test_tensor = torch.tensor(np.array(x_test), dtype=torch.float32)
    y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.float32)

    model = LSTMModel(input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()  # Mean Squared Error loss for regression tasks
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    batch_size, sequence_length, input_size = x_train_tensor.shape[0], 1, x_train_tensor.shape[1]
    x_train_tensor = x_train_tensor.view(batch_size, sequence_length, input_size)
    batch_size, sequence_length, input_size = x_test_tensor.shape[0], 1, x_test_tensor.shape[1]
    x_test_tensor = x_test_tensor.view(batch_size, sequence_length, input_size)

    # Step 5: Train the model
    for epoch in range(epochs):
        outputs = model(x_train_tensor)
        optimizer.zero_grad()
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    with torch.no_grad():
        predictions = model(x_test_tensor)

    return predictions, y_test
