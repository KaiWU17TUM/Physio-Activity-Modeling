# Was möchte ich tun: Ich möchte einen Array einlesen mit den letzen X ecg datenpunkten (ca 100) und zusätzlich eine Info
# pber die sportart dazu dann gebe ich die aktivität mit und predicte wie die nächsten n ecg datenpunkte aussehen.
# Wäre optimal vorzutrainieren was ein ecg ist und dann nur noch optimizen ?
import csv

# Look into stock market prediction, they try to do the same.
# Possible data sources for algo: https://towardsdatascience.com/the-complete-guide-to-time-series-analysis-and-forecasting-70d476bfe775
# Transformer for time series https://arxiv.org/pdf/2205.13504.pdf

# Ich möchte die abstände zwischen einzelnen beats predicten nicht die höhe oder sowas. Sondern die frequenz => deshlab
# benutz ich die Fourier transofmraion zum beispiel oder die wavelet decomposition ? will ich dannach zurück transfomieren ?

# Current approch: Transform data into Furnier => train a classifier, with transform and label predict next fourier
# inverse fournier and try to get the ecg signal in this way.

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn


# build batches without crossections between themself. Just cut the array in pieces of the same length.
def split_to_batches(ecg_file):
    data_array_action = []
    # split in a multi dim array. ech list element is a tuple a list of the ecg of a action and the lable
    with open(ecg_file, "r") as f:
        reader = csv.reader(f)
        data_array = []
        target_length = 1200  # ca 20 beats in theorie but it around 3-6
        length_counter = 0
        # score_cash = 0
        for row in reader:
            if length_counter <= target_length:
                data_array.append((row[0], row[1]))
                # score_cash += float(row[1])
                length_counter += 1
            else:
                data_array_action.append(data_array)
                data_array = [(row[0], row[1])]
                # score_cash = float(row[1])
                length_counter = 1

    return data_array_action


# read in batches make fft and fft back and compare the results ?
def plot_data_fft(data_array_action):
    # Prepare the data
    # Extract input data and labels
    signal_array_strings = np.array([data for data, _ in data_array_action[0]])
    # score = np.array([label for _, label in data_array_action[0]])

    signal_array = np.array(signal_array_strings, dtype=np.float32)

    # Perform Fourier Transform
    fourier_transform = np.fft.fft(signal_array)
    frequencies = np.fft.fftfreq(len(fourier_transform))

    # Plot the original signal and its Fourier transform
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    plt.plot(signal_array)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.title('Original Signal')

    plt.subplot(2, 2, 2)
    plt.plot(frequencies, np.abs(fourier_transform))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Fourier Transform')

    # Inverse Fourier Transform
    inverted_signal = np.fft.ifft(fourier_transform).real

    # Plot the inverted signal
    plt.subplot(2, 2, 3)
    plt.plot(inverted_signal)
    plt.xlabel('Time (samples)')
    plt.ylabel('Amplitude')
    plt.title('Inverted Signal')

    # Plot the differences between original and inverted signals
    plt.subplot(2, 2, 4)
    plt.plot(signal_array - inverted_signal)
    plt.xlabel('Time (samples)')
    plt.ylabel('Difference')
    plt.title('Difference (Original - Inverted)')

    plt.tight_layout()
    plt.show()


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

    # plt.subplot(2, 2, 2)
    # plt.plot(frequencies, np.abs(fourier_transform))
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Magnitude')
    # plt.title('Fourier Transform')

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


# Define the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        out, h = self.rnn(x, h)
        out = self.fc(out)
        return out, h


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


def split_img_values(value_array):
    # Create a new array with twice the length
    transformed_array = np.zeros(len(value_array) * 2)

    # Fill the new array with real and imaginary parts alternately
    for i, num in enumerate(value_array):
        transformed_array[i * 2] = np.real(num)  # Real part
        transformed_array[i * 2 + 1] = np.imag(num)  # Imaginary part

    return transformed_array


def inverse_img_number(values_array):
    # Create an array of complex numbers
    complex_numbers = []
    for i in range(0, len(values_array), 2):
        real_part = values_array[i]
        imag_part = values_array[i + 1]
        complex_numbers.append(complex(real_part, imag_part))

    complex_numbers = np.array(complex_numbers)
    return complex_numbers


def train_model_2(batches):
    # Data overview:
    print("batch size: 1201 1200 ecg datapoints + 1 score")
    print("Feature: FFT")
    print("batches: ", len(batches))
    print("test_size: 20% not implemented yet")

    # strategics, i need the score from the next batch to train because the score is the only thing that indicates
    # how it will shifts, the look into the furture.
    x_train = []
    x_test = []
    y_train = []
    y_test = []

    # TODO to get the test files, get 20 % random indices and if this index is chosen add the pair to test instead of train
    array_length = len(batches)
    percentage_of_ones = 0.2  # 20%
    # Calculate the number of ones and zeros
    num_ones = int(array_length * percentage_of_ones)
    num_zeros = array_length - num_ones
    # Create an array with ones and zeros
    array_index = np.concatenate((np.ones(num_ones), np.zeros(num_zeros)))
    # Shuffle the array randomly
    np.random.shuffle(array_index)

    for i in range(len(batches) - 1):

        signal_array_strings_curr = np.array([data for data, _ in data_array_action[i]])
        signal_array_strings_next = np.array([data for data, _ in data_array_action[i + 1]])
        label_curr = np.array([label for _, label in data_array_action[i]])
        label_next = np.array([label for _, label in data_array_action[i + 1]])

        signal_array_strings_curr = signal_array_strings_curr.astype(np.float32)
        signal_array_strings_next = signal_array_strings_next.astype(np.float32)
        label_curr = label_curr.astype(np.float32)
        label_next = label_next.astype(np.float32)

        # print(batches[i])
        # connect scor to signal davor brauchst du die Transformation

        # fourier_transform = np.fft.fft(signal_array_strings_curr).real
        fourier_transform = np.fft.fft(signal_array_strings_curr)
        frequencies = np.fft.fftfreq(len(fourier_transform))

        # Brauche den score form nächsten nciht vom jetzigen.
        temp = np.concatenate((split_img_values(fourier_transform), label_curr, label_next), axis=0)

        # fourier_transform2 = np.fft.fft(signal_array_strings_next).real
        fourier_transform2 = np.fft.fft(signal_array_strings_next)
        frequencies2 = np.fft.fftfreq(len(fourier_transform2))

        if array_index[i] == 0:
            x_train.append(temp)
            y_train.append(split_img_values(fourier_transform2))
        else:
            x_test.append(temp)
            y_test.append(split_img_values(fourier_transform2))

    # print(x_train, y_train)

    # Hyperparameters
    input_size = 4804  # Size of each input vector (e.g., window size)
    hidden_size = 300  # Number of hidden units in the RNN
    hidden_size_2 = 1000  # Number of hidden units in the RNN
    output_size = 2402  # Size of the predicted output vector

    # Convert data to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Create an instance of the neural network
    neural_net = NeuralNetwork(input_size, hidden_size, hidden_size_2, output_size)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(neural_net.parameters(), lr=0.001)

    print("Start training!")
    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
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

    """
    # Create an instance of the neural network
    rnn = RNN(input_size, hidden_size, output_size)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)

    # Convert data to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    # Training loop
    num_epochs = 100
    for epoch in range(num_epochs):
        # Initialize hidden state
        h = torch.zeros(1, x_train_tensor.size(0), hidden_size)

        # Forward pass
        outputs, h = rnn(x_train_tensor, h)

        # Compute the loss
        loss = criterion(outputs, y_train_tensor)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    print("Training finished.")
    """


if __name__ == "__main__":
    print("NN_model_2")
    # Output file name
    ecg_file = "../data/DHM 2/Andrei/transformed_data.csv"
    data_array_action = split_to_batches(ecg_file)
    # plot_data_fft(data_array_action)
    train_model_2(data_array_action)
    print("end")
