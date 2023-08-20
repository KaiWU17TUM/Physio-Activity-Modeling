import csv
import json

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

"""
Splits ECG data from a CSV file into batches based on a target length.

Args:
    ecg_file (str): Path to the CSV file containing ECG data and labels.
    target_length (int): The desired length for each batch.

Returns:
    list: A list of batches, where each batch is a list of tuples containing ECG data and its label.
"""


def split_to_batches(ecg_file, target_length):
    data_array_action = []  # List to hold batches of data
    # Split the data into a multi-dimensional array. Each list element is a tuple containing ECG data for an action and its label.
    with open(ecg_file, "r") as f:
        reader = csv.reader(f)
        data_array = []  # Temporary list to hold ECG data for a single action
        length_counter = 0  # Counter to track the current length of the data_array
        for row in reader:
            if length_counter <= target_length:
                data_array.append((row[0], row[1]))  # Append ECG data and label to the temporary array
                length_counter += 1
            else:
                data_array_action.append(data_array)  # Add the current batch of data to data_array_action
                data_array = [(row[0], row[1])]  # Start a new batch with the current data point
                length_counter = 1  # Reset the length counter for the new batch

    return data_array_action  # Return the list of batches


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


# Basic baseline
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


def plot_prediciton(pred, groud_truth):
    print("pred", inverse_img_number(pred))
    print("ground", inverse_img_number(groud_truth))

    inverted_signal_pred = np.fft.ifft(inverse_img_number(pred))
    inverted_signal_ground = np.fft.ifft(inverse_img_number(groud_truth))

    # Plot the original signal and its Fourier transform
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    plt.plot(inverted_signal_ground)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.title('Original Signal ground_truth')

    # TODO set right
    # Add general information text
    general_info = "General information:\n"
    general_info += "Dataset: ECG Signal\n"
    general_info += "Analysis Type: Fourier Transform\n"
    general_info += "Window Size: 100 samples"

    plt.subplot(2, 2, 2)
    plt.text(0.5, 0.5, general_info, fontsize=10, ha='center', va='center')
    plt.axis('off')  # Turn off the axes for the text subplot

    # Plot the inverted signal
    plt.subplot(2, 2, 3)
    plt.plot(inverted_signal_pred)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.title('Inverted Signal prediction')

    # Plot the differences between original and inverted signals
    plt.subplot(2, 2, 4)
    plt.plot(inverted_signal_ground - inverted_signal_pred)
    plt.xlabel('Time (samples)')
    plt.ylabel('Difference')
    plt.title('Difference (Ground truth - predicted)')

    plt.tight_layout()
    plt.show()


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


def train_ann(batches, test_size, window_size, epochs, learning_rate_value):
    array_index = build_test_array(batches, test_size)

    x_train, x_test, y_train, y_test = [], [], [], []

    for i in range(len(batches) - 1):

        signal_array_strings_curr = np.array([data for data, _ in data_array_action[i]])
        signal_array_strings_next = np.array([data for data, _ in data_array_action[i + 1]])
        label_curr = np.array([label for _, label in data_array_action[i]])
        label_next = np.array([label for _, label in data_array_action[i + 1]])

        signal_array_strings_curr = signal_array_strings_curr.astype(np.float32)
        signal_array_strings_next = signal_array_strings_next.astype(np.float32)
        label_curr = label_curr.astype(np.float32)
        label_next = label_next.astype(np.float32)

        fourier_transform = np.fft.fft(signal_array_strings_curr)
        temp = np.concatenate((split_img_values(fourier_transform), label_curr, label_next), axis=0)
        fourier_transform2 = np.fft.fft(signal_array_strings_next)

        if array_index[i] == 0:
            x_train.append(temp)
            y_train.append(split_img_values(fourier_transform2))
        else:
            x_test.append(temp)
            y_test.append(split_img_values(fourier_transform2))

    # Hyperparameters
    input_size = (window_size + 1) * 4  # Size of each input vector (e.g., window size)
    hidden_size = 300  # Number of hidden units in the ANN
    hidden_size_2 = 1000  # Number of hidden units in the ANN
    output_size = (window_size + 1) * 2  # Size of the predicted output vector

    # Convert data to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create an instance of the neural network
    neural_net = NeuralNetwork(input_size, hidden_size, hidden_size_2, output_size)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(neural_net.parameters(), lr=learning_rate_value)

    print("Start training!")
    # Training loop
    num_epochs = epochs
    for epoch in range(num_epochs):
        # TODO here change so that a fft inverse is performed ? to compare in ecg space
        # Forward pass
        outputs = neural_net(x_train_tensor)

        # Compute the loss
        loss = criterion(outputs, y_train_tensor)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training finished.")

    print("start testing")
    with torch.no_grad():
        predictions = neural_net(x_test_tensor)

    for wave, ground_truth in zip(predictions, y_test):
        plot_prediciton(wave, ground_truth)


def plot_pred_cnn(pred, groud_truth, test_size, window_size, epoch_count, learning_rate_value, kernal_size_value):
    # Plot the original signal and its Fourier transform
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    plt.plot(groud_truth)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.title('Ground_truth')

    # TODO set right
    # Add general information text
    general_info = "General information:\n"
    general_info += "Dataset: ECG Signal\n"
    general_info += f"Analysis Type: CNN raw\n"
    general_info += f"Test size: {test_size}\n"
    general_info += f"Window Size: {window_size} samples\n"
    general_info += f"Epochs: {epoch_count}\n"
    general_info += f"Learning rate: {learning_rate_value}\n"
    general_info += f"kernal Size: {kernal_size_value}\n"

    plt.subplot(2, 2, 2)
    plt.text(0.5, 0.5, general_info, fontsize=10, ha='center', va='center')
    plt.axis('off')  # Turn off the axes for the text subplot

    # Plot the inverted signal
    plt.subplot(2, 2, 3)
    plt.plot(pred)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.title('Signal prediction')

    # Plot the differences between original and inverted signals
    plt.subplot(2, 2, 4)
    plt.plot(groud_truth - pred)
    plt.xlabel('Time (samples)')
    plt.ylabel('Difference')
    plt.title('Difference (Ground truth - predicted)')

    plt.tight_layout()
    plt.show()


class TimeSeriesCNN(nn.Module):
    def __init__(self, input_size, output_size, kernal_size):
        super(TimeSeriesCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=41, out_channels=16, kernel_size=kernal_size, padding=20)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kernal_size, padding=20)
        self.fc1 = nn.Linear(34656, 512)  # nn.Linear(32 * input_size, 512)
        self.fc2 = nn.Linear(512, output_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def one_hot_encode(value, num_classes):
    encoding = [0] * (num_classes + 1)
    encoding[int(value)] = 1
    return encoding


def train_cnn(data_array_action, test_size, window_size, epoch_count, learning_rate_value, kernal_size_value):
    array_index = build_test_array(data_array_action, test_size)
    x_train, x_test, y_train, y_test = [], [], [], []

    for i in range(len(data_array_action) - 1):
        temp = data_array_action[i]
        temp_2 = data_array_action[i + 1]

        # Add the label as one hot encoding ??

        temp_pre = [(float(t1[0]), one_hot_encode(float(t1[1]), 19), one_hot_encode(float(t2[1]), 19)) for t1, t2 in
                    zip(temp, temp_2)]
        temp = []
        for x in temp_pre:
            temp.append(np.concatenate([np.array(item).flatten() for item in x]))

        temp_2 = [float(data) for data, label in temp_2]

        if array_index[i] == 0:
            x_train.append(temp)
            y_train.append(temp_2)
        else:
            x_test.append(temp)
            y_test.append(temp_2)

    input_data = torch.tensor(np.array(x_train), dtype=torch.float32)
    input_data = input_data.permute(0, 2, 1)
    labels = torch.tensor(np.array(y_train), dtype=torch.float32)

    test_data = torch.tensor(np.array(x_test), dtype=torch.float32)
    test_data = test_data.permute(0, 2, 1)
    labels_test = torch.tensor(np.array(y_test), dtype=torch.float32)

    # Initialize the TimeSeriesCNN model
    input_size = window_size + 1  # Length of input time series
    output_size = window_size + 1  # Length of output time series
    ts_cnn_model = TimeSeriesCNN(input_size, output_size, kernal_size_value)

    # Define loss function and optimizer
    criterion = nn.MSELoss()  # You can use other appropriate loss functions
    optimizer = torch.optim.Adam(ts_cnn_model.parameters(), lr=learning_rate_value)

    # Training loop
    num_epochs = epoch_count
    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Zero the gradients
        outputs = ts_cnn_model(input_data)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training finished.")

    print("start testing")
    with torch.no_grad():
        predictions = ts_cnn_model(test_data)

    for wave, ground_truth in zip(predictions, labels_test):
        plot_pred_cnn(wave, ground_truth, test_size, window_size, epoch_count, learning_rate_value, kernal_size_value)

def plot_pred_cnn_fft(pred, groud_truth, test_size, window_size, epoch_count, learning_rate_value, kernal_size_value):
    # Plot the original signal and its Fourier transform
    # complex_pred = [complex(item[0], item[1]) for item in pred.numpy()]
    pred_numpy = np.vectorize(complex)(pred.numpy()[0], pred.numpy()[1])
    # complex_ground = [complex(item[0], item[1]) for item in groud_truth.numpy()]
    ground_numpy = np.vectorize(complex)(groud_truth.numpy()[0], groud_truth.numpy()[1])
    inverted_signal_pred = np.fft.ifft(pred_numpy)
    inverted_signal_ground = np.fft.ifft(ground_numpy)

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    plt.plot(inverted_signal_ground)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.title('Ground_truth')

    # TODO set right
    # Add general information text
    general_info = "General information:\n"
    general_info += "Dataset: ECG Signal\n"
    general_info += f"Analysis Type: CNN FFT\n"
    general_info += f"Test size: {test_size}\n"
    general_info += f"Window Size: {window_size} samples\n"
    general_info += f"Epochs: {epoch_count}\n"
    general_info += f"Learning rate: {learning_rate_value}\n"
    general_info += f"kernal Size: {kernal_size_value}\n"

    plt.subplot(2, 2, 2)
    plt.text(0.5, 0.5, general_info, fontsize=10, ha='center', va='center')
    plt.axis('off')  # Turn off the axes for the text subplot

    # Plot the inverted signal
    plt.subplot(2, 2, 3)
    plt.plot(inverted_signal_pred)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.title('Signal prediction')

    # Plot the differences between original and inverted signals
    plt.subplot(2, 2, 4)
    plt.plot(inverted_signal_ground - inverted_signal_pred)
    plt.xlabel('Time (samples)')
    plt.ylabel('Difference')
    plt.title('Difference (Ground truth - predicted)')

    plt.tight_layout()
    plt.show()

class CNN_FFT(nn.Module):
    def __init__(self, input_size, output_size, kernal_size):
        super(CNN_FFT, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=42, out_channels=16, kernel_size=kernal_size, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=kernal_size, padding=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=2, kernel_size=15, padding=104)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)

        # Reshape the output tensor to match the desired shape (batch_size, 2, 1201)
        x = x.view(x.size(0), 2, -1)
        return x


def train_cnn_fft(data_array_action, test_size, window_size, epoch_count, learning_rate_value, kernal_size_value):
    array_index = build_test_array(data_array_action, test_size)

    x_train, x_test, y_train, y_test = [], [], [], []

    for i in range(len(data_array_action) - 1):

        signal_array_strings_curr = np.array([data for data, _ in data_array_action[i]])
        signal_array_strings_next = np.array([data for data, _ in data_array_action[i + 1]])
        label_curr = np.array([label for _, label in data_array_action[i]])
        label_next = np.array([label for _, label in data_array_action[i + 1]])

        signal_array_strings_curr = signal_array_strings_curr.astype(np.float32)
        signal_array_strings_next = signal_array_strings_next.astype(np.float32)
        label_curr = [one_hot_encode(item, 19) for item in label_curr.astype(np.float32)]
        label_next = [one_hot_encode(item, 19) for item in label_next.astype(np.float32)]

        fourier_transform = np.fft.fft(signal_array_strings_curr)
        temp_pre = list(zip(fourier_transform, label_curr, label_next))
        temp = []
        for x in temp_pre:
            # Get the real and imaginary parts of the complex number
            real_part = np.real(x[0])
            imag_part = np.imag(x[0])

            # Combine the lists and components
            array_pre = [real_part, imag_part] + x[1] + x[2]
            temp.append(np.array(array_pre))

        fourier_transform2 = np.fft.fft(signal_array_strings_next)

        if array_index[i] == 0:
            x_train.append(temp)
            y_train.append([[np.real(num), np.imag(num)] for num in fourier_transform2])
        else:
            x_test.append(temp)
            y_test.append([[np.real(num), np.imag(num)] for num in fourier_transform2])

    input_data = torch.tensor(np.array(x_train), dtype=torch.float32)
    input_data = input_data.permute(0, 2, 1)
    labels = torch.tensor(np.array(y_train), dtype=torch.float32)
    labels = labels.permute(0, 2, 1)

    test_data = torch.tensor(np.array(x_test), dtype=torch.float32)
    test_data = test_data.permute(0, 2, 1)
    labels_test = torch.tensor(np.array(y_test), dtype=torch.float32)
    labels_test = labels_test.permute(0, 2, 1)

    # Initialize the TimeSeriesCNN model
    input_size = window_size + 1  # Length of input time series
    output_size = window_size + 1  # Length of output time series
    cnn_fft_model = CNN_FFT(input_size, output_size, kernal_size_value)

    # Define loss function and optimizer
    criterion = nn.MSELoss()  # You can use other appropriate loss functions
    optimizer = torch.optim.Adam(cnn_fft_model.parameters(), lr=learning_rate_value)

    # Training loop
    num_epochs = epoch_count
    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Zero the gradients
        outputs = cnn_fft_model(input_data)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training finished.")

    print("start testing")
    with torch.no_grad():
        predictions = cnn_fft_model(test_data)

    for wave, ground_truth in zip(predictions, labels_test):
        plot_pred_cnn_fft(wave, ground_truth, test_size, window_size, epoch_count, learning_rate_value, kernal_size_value)


if __name__ == "__main__":
    print("Run Programm nn_3")
    # JSON-Datei öffnen und lesen
    json_filename = "config_nn.json"
    with open(json_filename, "r") as json_file:
        j_data = json.load(json_file)

    # Speichern der Daten in temporären Variablen
    type_value = j_data["type"]
    window_size_value = j_data["window_size"]
    method_value = j_data["method"]
    dataset_value = j_data["dataset"]
    plots_value = j_data["plots"]
    test_size_value = j_data["test_size"]
    epochs_value = j_data["epochs"]
    learning_rate_value = j_data["learning_rate"]
    kernal_size_value = j_data["kernal_size"]

    data_array_action = split_to_batches(dataset_value, window_size_value)

    # print overview
    print(
        f'Train an {type_value}\nwith {len(data_array_action)} batches\nwindow_length {window_size_value}\n'
        f'test size is {test_size_value}\nepochs {epochs_value}\nlearning_rate_value {learning_rate_value}'
        f'\nkernal_size {kernal_size_value}')

    if type_value == "ANN":
        train_ann(data_array_action, test_size_value, window_size_value, epochs_value, learning_rate_value)

    elif type_value == "CNN":
        train_cnn(data_array_action, test_size_value, window_size_value, epochs_value, learning_rate_value,
                  kernal_size_value)
    elif type_value == "CNN FFT":
        train_cnn_fft(data_array_action, test_size_value, window_size_value, epochs_value, learning_rate_value,
                      kernal_size_value)
    else:
        print("Not a valid type")

    print("Complete")
