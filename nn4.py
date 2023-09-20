import csv
import json

import torch
import wfdb
from matplotlib import pyplot as plt
from torch import nn, optim
from wfdb import processing
import numpy as np

from ecgdetectors import Detectors


##https://github.com/berndporr/py-ecg-detectors
# Authors
#
# Luis Howell, luisbhowell@gmail.com
#
# Bernd Porr, bernd.porr@glasgow.ac.uk
#
# Citation / DOI
#
# DOI: 10.5281/zenodo.3353396
#
# https://doi.org/10.5281/zenodo.3353396


def extract_qrs_complex(file, window_size):
    # Load ECG data from CSV
    data_as_tupel = []
    detectors = Detectors(1000)  # vorher 1000 300hz would be the correct one ?
    with open(file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            data_as_tupel.append([(row[0], row[1])])  # row[0] = signal row[1] = label

    # print([ecg[0][0] for ecg in data_as_tupel])

    r_peaks = detectors.hamilton_detector([float(ecg[0][0]) for ecg in data_as_tupel])

    # print(r_peaks)
    # plot_peaks([float(ecg[0][0]) for ecg in data_as_tupel], r_peaks)

    # build batches:
    result_batches = []
    for index in r_peaks:
        result_batches.append(data_as_tupel[int(index - window_size / 2):int(index + window_size / 2)])

    return result_batches


def plot_peaks(data, r_peaks):
    index_limit = 3000
    plt.plot(data[:index_limit])

    # Add vertical lines at specified indices
    for idx in r_peaks:
        if idx > index_limit:
            break
        plt.axvline(x=idx, color='red', linestyle='--', label=f'Value {idx}')

    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(f"Plot of First {index_limit} Elements")
    plt.show()


def plot_batches(batches):
    print(len(batches))
    batch = batches[2000]

    fft_batch = np.fft.fft([float(sig[0][0]) for sig in batch])

    ecg_signal = [float(sig[0][0]) for sig in batch]

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)

    # print(ecg_signal)
    plt.plot(ecg_signal)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(f"Plot batch 20")

    plt.subplot(2, 2, 2)
    plt.plot(np.real(fft_batch))
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(f"Plot batch 20")

    plt.subplot(2, 2, 3)
    plt.plot(np.imag(fft_batch))
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(f"Plot batch 20")
    plt.show()


# Step 3: Define the LSTM model
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layers, output_size):
#         super().__init__()
#         self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, )
#         #super(LSTMModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
#
#     def forward(self, x):
#         # Bestimmen Sie die Batch-Größe
#         batch_size = x.size(0)
#
#         # Initialisieren Sie hx und cx mit den richtigen Dimensionen
#         h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
#
#         # Geben Sie hx und cx an das LSTM weiter
#         out, _ = self.lstm(x, (h0, c0))
#
#         out = self.fc(out[:, -1, :])
#         return out

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


def build_test_array(batches, test_size):
    array_length = len(batches)
    percentage_of_ones = test_size
    # Calculate the number of ones and zeros
    num_ones = int(array_length * percentage_of_ones)
    num_zeros = array_length - num_ones
    # Create an array with ones and zeros
    array_index = np.concatenate((np.ones(num_ones), np.zeros(num_zeros)))
    # Shuffle the array randomly
    np.random.shuffle(array_index)
    return array_index


def split_img_values(value_array):
    """
    Splits complex numbers into their real and imaginary parts and creates a new array with alternating real and imaginary values.

    Args:
        value_array (np.ndarray): An array of complex numbers.

    Returns:
        np.ndarray: An array with twice the length, containing alternating real and imaginary parts of the complex numbers.
    """
    # Create a new array with twice the length
    transformed_array = np.zeros(len(value_array) * 2)

    # Fill the new array with real and imaginary parts alternately
    for i, num in enumerate(value_array):
        transformed_array[i * 2] = np.real(num)  # Real part
        transformed_array[i * 2 + 1] = np.imag(num)  # Imaginary part

    return transformed_array


def inverse_img_number(values_array):
    """
    Reconstructs complex numbers from their split real and imaginary parts.

    Args:
        values_array (np.ndarray): An array containing alternating real and imaginary values.

    Returns:
        np.ndarray: An array of complex numbers reconstructed from the split real and imaginary parts.
    """
    # Create an array of complex numbers
    complex_numbers = []
    for i in range(0, len(values_array), 2):
        real_part = values_array[i]
        imag_part = values_array[i + 1]
        complex_numbers.append(complex(real_part, imag_part))

    complex_numbers = np.array(complex_numbers)
    return complex_numbers


def train_lstm(batches, input_size, hidden_size, num_layers, output_size, test_size, epochs, learning_rate,
               truncate_value):
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    untransformed_test = []

    array_index = build_test_array(batches, test_size)

    for index in range(len(batches) - 1):
        ecg_signal = [float(sig[0][0]) for sig in batches[index]]
        fft_batch = split_img_values(np.fft.rfft(ecg_signal)[:truncate_value])  # take only the first 50 values

        ecg_signal_2 = [float(sig[0][0]) for sig in batches[index + 1]]
        label_2 = [float(sig[0][1]) for sig in batches[index + 1]]
        fft_batch_2 = split_img_values(np.fft.rfft(ecg_signal_2)[:truncate_value])  # take only the first 50 values
        score_next = np.mean(label_2)

        temp = np.concatenate((fft_batch, [score_next]))

        if array_index[index] == 0:
            x_train.append(temp)
            y_train.append(fft_batch_2)
        else:
            x_test.append(temp)
            y_test.append(fft_batch_2)
            untransformed_test.append(ecg_signal_2)

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
    num_epochs = epochs
    print(x_train_tensor.shape)
    print(y_train_tensor.shape)
    for epoch in range(num_epochs):
        outputs = model(x_train_tensor)
        optimizer.zero_grad()
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("start testing")
    with torch.no_grad():
        predictions = model(x_test_tensor)

    for pred, ground_truth, untrandsformed in zip(predictions, y_test, untransformed_test):
        # print(pred)
        # print(y_test)
        #FIXME dont use a golabe var here
        zero_array = np.zeros(window_size_setup - truncate_value)  # Create a zero array of the same length

        # Concatenate the arrays
        pred = np.concatenate((pred, zero_array))
        ground_truth = np.concatenate((ground_truth, zero_array))
        plot_prediction(pred, ground_truth, untrandsformed)


def plot_prediction(pred, ground, untransformed):
    # TODO do i need the irfft ? =?? RRRRR
    inverted_signal_pred = np.fft.ifft(inverse_img_number(pred))
    inverted_signal_ground = np.fft.ifft(inverse_img_number(ground))

    # Plot the original signal and its Fourier transform
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 4, 1)
    plt.plot(inverted_signal_ground)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.title('Original Signal ground_truth')

    # TODO set right
    # Add general information text
    general_info = "General information:\n"
    general_info += "Dataset: ECG Signal use lstm \n"
    general_info += "Analysis Type: Fourier Transform\n"
    general_info += "Window Size: 250, samples with a score not labels"

    plt.subplot(2, 4, 2)
    plt.text(0.5, 0.5, general_info, fontsize=10, ha='center', va='center')
    plt.axis('off')  # Turn off the axes for the text subplot

    # Plot the inverted signal
    plt.subplot(2, 4, 3)
    plt.plot(inverted_signal_pred)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.title('Inverted Signal prediction')

    # Plot the differences between original and inverted signals
    plt.subplot(2, 4, 4)
    plt.plot(inverted_signal_ground - inverted_signal_pred)
    plt.xlabel('Time (samples)')
    plt.ylabel('Difference')
    plt.title('Difference (Ground truth - predicted)')

    # Plot the groundtrouth fft
    plt.subplot(2, 4, 5)
    plt.plot(ground)
    plt.xlabel('freq')
    plt.ylabel('time')
    plt.title('Ground truth fft')

    # Plot the pred fft
    plt.subplot(2, 4, 6)
    plt.plot(pred)
    plt.xlabel('freq')
    plt.ylabel('time')
    plt.title('Pred fft')

    # Plot the pred fft without transform
    plt.subplot(2, 4, 7)
    plt.plot(untransformed)
    plt.xlabel('freq')
    plt.ylabel('time')
    plt.title('Ground without transforms fft')

    plt.tight_layout()
    plt.show()


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


def train_mlp(batches, test_size, window_size, epochs, learning_rate_value, truncate_value):
    array_index = build_test_array(batches, test_size)

    x_train, x_test, y_train, y_test = [], [], [], []

    for index in range(len(batches) - 1):
        ecg_signal = [float(sig[0][0]) for sig in batches[index]]
        fft_batch = split_img_values(np.fft.rfft(ecg_signal)[:truncate_value])  # take only the first 50 values

        ecg_signal_2 = [float(sig[0][0]) for sig in batches[index + 1]]
        label_2 = [float(sig[0][1]) for sig in batches[index + 1]]
        fft_batch_2 = split_img_values(np.fft.rfft(ecg_signal_2)[:truncate_value])  # take only the first 50 values
        score_next = np.mean(label_2)

        temp = np.concatenate((fft_batch, [score_next]))

        if array_index[index] == 0:
            x_train.append(temp)
            y_train.append(fft_batch_2)
        else:
            x_test.append(temp)
            y_test.append(fft_batch_2)
            # untransformed_test.append(ecg_signal_2)

    # Convert data to PyTorch tensors
    x_train_tensor = torch.tensor(np.array(x_train), dtype=torch.float32)
    y_train_tensor = torch.tensor(np.array(y_train), dtype=torch.float32)
    x_test_tensor = torch.tensor(np.array(x_test), dtype=torch.float32)
    y_test_tensor = torch.tensor(np.array(y_test), dtype=torch.float32)

    # Hyperparameters
    input_size = (truncate_value * 2) + 1  # Size of each input vector (e.g., window size)
    hidden_size = 300  # Number of hidden units in the ANN
    hidden_size_2 = 100  # Number of hidden units in the ANN
    output_size = truncate_value * 2  # Size of the predicted output vector

    # Create an instance of the neural network
    neural_net = NeuralNetwork(input_size, hidden_size, hidden_size_2, output_size)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(neural_net.parameters(), lr=learning_rate_value)

    # Step 5: Train the model
    num_epochs = epochs
    print(x_train_tensor.shape)
    print(y_train_tensor.shape)
    for epoch in range(num_epochs):
        outputs = neural_net(x_train_tensor)
        optimizer.zero_grad()
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("start testing")
    with torch.no_grad():
        predictions = neural_net(x_test_tensor)

    for pred, ground_truth in zip(predictions, y_test):
        # print(pred)
        # print(y_test)
        zero_array = np.zeros(window_size - truncate_value)  # Create a zero array of the same length

        # Concatenate the arrays
        pred = np.concatenate((pred, zero_array))
        ground_truth = np.concatenate((ground_truth, zero_array))
        plot_prediction(pred, ground_truth, "")  # TODO need to add dummy value for untranformed


if __name__ == "__main__":
    print("Starting ...")
    window_size_setup = 200

    batches_setup = extract_qrs_complex("data_shared/transformed_data3_andrei.csv", window_size_setup)
    # plot_batches(batches)
    truncate_value_setup = 50
    input_size_setup = truncate_value_setup * 2 + 1  # if not truncate use dont multiply with 2 and dont use plus 1 but plus 3
    hidden_size_setup = 200
    num_layers_setup = 2
    output_size_setup = input_size_setup - 1
    test_size_setup = 0.1
    epochs_setup = 100
    learning_rate_setup = 0.01
    train_lstm(batches_setup, input_size_setup, hidden_size_setup, num_layers_setup, output_size_setup, test_size_setup,
               epochs_setup, learning_rate_setup, truncate_value_setup)
    # train_mlp(batches_setup, test_size_setup, window_size_setup, epochs_setup, learning_rate_setup, truncate_value_setup)
    print("Complete")
