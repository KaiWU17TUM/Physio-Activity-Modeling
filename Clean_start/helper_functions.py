# This file contains all helping functings that were used in all files
import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


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


def build_test_array(batches, test_size):
    """
    Build a test array with a specified percentage of ones and zeros.

    This function takes an input 'batches' array and a 'test_size' that represents the
    desired percentage of ones in the test array. It calculates the number of ones
    and zeros needed to achieve the desired percentage, creates an array with those
    values, and shuffles it randomly to create a test array.

    Args:
        batches (array-like): The input array on which the test array is based.
        test_size (float): The desired percentage of ones in the test array.

    Returns:
        numpy.ndarray: A test array with the specified percentage of ones and zeros.

    Example:
        To build a test array with 30% ones from 'my_batches':
        test_array = build_test_array(my_batches, 0.30)
    """
    # Calculate the length of the input 'batches' array
    array_length = len(batches)

    # Calculate the percentage of ones in the test array
    percentage_of_ones = test_size

    # Calculate the number of ones needed based on the percentage
    num_ones = int(array_length * percentage_of_ones)

    # Calculate the number of zeros needed (complement to num_ones)
    num_zeros = array_length - num_ones

    # Create an array with ones (num_ones) followed by zeros (num_zeros)
    array_index = np.concatenate((np.ones(num_ones), np.zeros(num_zeros)))

    # Shuffle the array randomly to create a test array
    np.random.seed(42)
    np.random.shuffle(array_index)

    return array_index


def build_sets(batches, test_size, truncate_value, truncate_value_front):
    """
    Build training and testing datasets for a machine learning model.

    This function takes a list of batches, a test size, and a truncate value, and constructs
    training and testing datasets for a machine learning model. It extracts features from
    the input ECG signal data using Fourier transforms and organizes them into input-output pairs.

    Args:
        batches (list): A list of batches, where each batch contains ECG signal data.
        test_size (float): The desired percentage of data to be allocated for testing.
        truncate_value (int): The number of Fourier coefficients to keep in the feature extraction.

    Returns:
        tuple: A tuple containing the following four lists:
            - x_train (list): Training data for model input.
            - x_test (list): Testing data for model input.
            - y_train (list): Training data for model output.
            - y_test (list): Testing data for model output.

    Example:
        To build training and testing datasets with a test size of 20% and a truncate value of 50:
        x_train, x_test, y_train, y_test = build_sets(my_batches, 0.20, 50)
    """
    array_index = build_test_array(batches, test_size)

    # Initialize lists for training and testing data
    x_train, x_test, y_train, y_test, test_untransformed = [], [], [], [], []

    # Loop through batches to extract features and create input-output pairs
    for index in range(len(batches) - 1):
        ecg_signal = [float(sig[0][0]) for sig in batches[index]]
        fft_batch = split_img_values(np.fft.rfft(ecg_signal)[truncate_value_front:truncate_value])

        ecg_signal_2 = [float(sig[0][0]) for sig in batches[index + 1]]
        label_2 = [float(sig[0][1]) for sig in batches[index + 1]]
        fft_batch_2 = split_img_values(np.fft.rfft(ecg_signal_2)[truncate_value_front:truncate_value])
        score_next = np.mean(label_2)

        temp = np.concatenate((fft_batch, [score_next]))

        if array_index[index] == 0:
            x_train.append(temp)
            y_train.append(fft_batch_2)
        else:
            x_test.append(temp)
            y_test.append(fft_batch_2)
            test_untransformed.append(ecg_signal_2)

    return x_train, x_test, y_train, y_test, test_untransformed


def plot_prediction(preds, y_test, ground_truths, type_of_network, window_size, truncate_value, truncate_value_front, plot_y_n, test_set):
    """
    Plot predictions and ground truths for a time series forecasting task.

    Args:
        preds (numpy.ndarray): Predicted values (spectrum) from the model.
        ground_truths (numpy.ndarray): Ground truth values (spectrum).
        type_of_network (str): Type or name of the neural network used.
        window_size (int): Size of the data window.
        truncate_value (int): Number of values to truncate from the spectrum.

    Returns:
        None: This function displays the plot but does not return any values.
    """
    pearson_cc = []
    spearman_cc = []
    mse = []
    pearson_cc_test = []
    spearman_cc_test = []
    mse_test = []
    for pred, y_test, ground_truth in zip(preds, y_test, ground_truths):# TODO welches test set soll ich verwenden ? trans or not ?
        zero_array = np.zeros(int(window_size / 2) + 1 - truncate_value)

        array_to_add_front = np.zeros(truncate_value_front)

        # Concatenate the arrays
        pred = inverse_img_number(pred)
        inverse_array = np.append(array_to_add_front, pred)
        inverted_signal_pred = np.fft.irfft(np.append(inverse_array, zero_array))

        y_test = inverse_img_number(y_test)
        inverse_array_test = np.append(array_to_add_front, y_test)
        inverted_signal_test = np.fft.irfft(np.append(inverse_array_test, zero_array))

        ##### Untransformed Test
        # Calculate Pearson correlation coefficient
        ground_truth = np.array(ground_truth)
        correlation_coefficient, p_value = pearsonr(inverted_signal_pred, ground_truth)
        # Calculate Spearman correlation coefficient
        rho, _ = spearmanr(inverted_signal_pred, ground_truth)
        pearson_cc.append(correlation_coefficient)
        spearman_cc.append(rho)

        # Calculate the squared differences
        squared_diff = (ground_truth - inverted_signal_pred) ** 2 #TODO mit oder ohne s
        mse_local = np.mean(squared_diff)
        mse.append(mse_local)

        ### Transfomred Test
        # Calculate Pearson correlation coefficient
        correlation_coefficient_test, p_value_test = pearsonr(inverted_signal_pred, inverted_signal_test)
        # Calculate Spearman correlation coefficient
        rho_test, _ = spearmanr(inverted_signal_pred, inverted_signal_test)
        pearson_cc_test.append(correlation_coefficient_test)
        spearman_cc_test.append(rho_test)

        # Calculate the squared differences
        squared_diff_test = (inverted_signal_test - inverted_signal_pred) ** 2
        mse_local_test = np.mean(squared_diff_test)
        mse_test.append(mse_local_test)
        if test_set:
            if plot_y_n:
                # Add a text annotation or note
                plt.text(120, 0.1, f'Pearson CC {round(correlation_coefficient_test, 4)}', fontsize=12, color='blue')
                plt.text(120, 0.2, f'Spearman CC {round(rho_test, 4)}', fontsize=12, color='blue')
                plt.text(120, 0.3, f'MSE {round(mse_local_test, 4)}', fontsize=12, color='blue')

                plt.plot(inverted_signal_pred, linestyle='dotted', label='prediction')
                plt.plot(inverted_signal_test, linestyle='-', label='inverted test')
                plt.legend()
                plt.title(f"Prediction vs Test in the {type_of_network}")
                plt.ylabel("Amplitude")
                plt.xlabel("Time")

                plt.tight_layout()
                plt.show()
        else:
            if plot_y_n:
                # Add a text annotation or note
                plt.text(120, 0.1, f'Pearson CC {round(correlation_coefficient, 4)}', fontsize=12, color='blue')
                plt.text(120, 0.2, f'Spearman CC {round(rho, 4)}', fontsize=12, color='blue')
                plt.text(120, 0.3, f'MSE {round(mse_local, 4)}', fontsize=12, color='blue')

                plt.plot(inverted_signal_pred, linestyle='dotted', label='prediction')
                plt.plot(ground_truth, linestyle='-', label='ground truth')
                plt.legend()
                plt.title(f"Prediction vs Ground truth in the {type_of_network}")
                plt.ylabel("Amplitude")
                plt.xlabel("Time")

                plt.tight_layout()
                plt.show()
    if test_set:
        return np.mean(pearson_cc_test), np.mean(spearman_cc_test), np.mean(mse_test)
    else:
        return np.mean(pearson_cc), np.mean(spearman_cc), np.mean(mse)


def up_sample(array, freq, target_freq):
    t_original = np.arange(0, len(array) / freq, 1 / freq)

    # Create a new time array for the target frequency
    t_target = np.arange(0, len(array) / freq, 1 / target_freq)

    interpolator = interp1d(t_original, array, kind='linear', fill_value="extrapolate")
    signal_resampled = interpolator(t_target)

    return signal_resampled


def norm_sig(data):

    # Calculate the minimum and maximum values in the data
    min_val = np.min(data)
    max_val = np.max(data)

    # Normalize the data between 0 and 1
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

def norm_sig_mhealth(data):

    # Calculate the minimum and maximum values in the data
    min_val = np.min(-1)
    max_val = np.max(1)

    # Normalize the data between 0 and 1
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data


class CustomDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y
