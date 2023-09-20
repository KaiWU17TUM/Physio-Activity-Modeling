# This file reads in DHM data, and feeds it to the different NN algos
import csv
from ecgdetectors import Detectors
from matplotlib import pyplot as plt

from Clean_start.evaluate_fft import evaluate_fft, evaluate_rfft, evaluate_rfft_trunc
from Clean_start.helper_functions import plot_prediction, norm_sig
from Clean_start.lstm_nn import train_lstm
from Clean_start.mlp_nn import train_mlp


# Read in data, split into batches for each peak.
def extract_qrs_complex(file, window_size):
    """
    Extract QRS complexes from ECG data in a CSV file.

    This function takes an input CSV file containing ECG data and extracts QRS complexes
    using the Hamilton detector from the 'ecgdetectors' package. It loads the ECG data,
    detects R-peaks, and creates batches of data centered around each R-peak. The batches
    contain the data as tuples, datapoint (signal) and label.

    Args:
        file (str): The path to the CSV file containing ECG data.
        window_size (int): The size of the window for creating batches centered around R-peaks.

    Returns:
        list: A list of batches, each containing ECG data centered around an R-peak.

    Example:
        To extract QRS complexes from 'ecg_data.csv' with a window size of 100:
        qrs_batches = extract_qrs_complex('ecg_data.csv', 100)
    """
    # Load ECG data from CSV
    data_as_tupel = []
    data_signal = []
    data_lable = []
    detectors = Detectors(1000)  # You may need to adjust the sampling rate here
    with open(file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            # data_as_tupel.append([(row[0], row[1])])  # row[0] = signal, row[1] = label
            data_signal.append(float(row[0]))
            data_lable.append(int(row[1]))
    norm_signal = norm_sig(data_signal)

    for sig, lable in zip(norm_signal, data_lable):
        data_as_tupel.append([(sig, lable)])

    # Detect R-peaks using the Hamilton detector
    r_peaks = detectors.hamilton_detector([float(ecg[0][0]) for ecg in data_as_tupel])

    # Build batches centered around R-peaks
    result_batches = []
    for index in r_peaks:
        result_batches.append(data_as_tupel[int(index - window_size / 2):int(index + window_size / 2)])

    print(len(result_batches))
    return result_batches

def inspect_dhm(file):
    data_as_tupel = []
    detectors = Detectors(1000)  # You may need to adjust the sampling rate here
    with open(file, "r") as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            data_as_tupel.append([(row[0], row[1])])  # 3 is safe an ECG but 4 ??
    ecg_data = [float(ecg[0][0]) for ecg in data_as_tupel][1020:1400]

    # Create a time axis (assuming a sample rate of 1)
    time = range(len(ecg_data))
    print(ecg_data)

    # Create a plot
    plt.figure(figsize=(8, 4))  # Optional: Set the figure size
    plt.plot(time, ecg_data, linestyle='-')
    plt.title('Signal Plot')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)

    # Show the plot
    plt.show()

# Run the code:
if __name__ == "__main__":
    print("start DHM")
    window_size = 200
    test_size = 0.1
    epochs = 200
    learning_rate_value = 0.01
    truncate_value = 25
    truncate_value_front = 2
    input_size = truncate_value * 2 + 1
    hidden_size = 200
    num_layers = 2
    output_size = input_size - 1

    #inspect_dhm("/Users/vet/Documents/Uni Allgemein/Master/SS23/Guided Research/Physio-Activity-Modeling/data_shared/transformed_data3_andrei.csv")
    batches = extract_qrs_complex(
        "/Users/vet/Documents/Uni Allgemein/Master/SS23/Guided Research/Physio-Activity-Modeling/data_shared/transformed_data3_andrei.csv",
        window_size)
    # Now optimized to Mhealth
    # evaluate_fft(batches[0])
    # evaluate_rfft(batches[0])
    # evaluate_rfft_trunc(batches[0], truncate_value, truncate_value_front)


    pred, ground = train_mlp(batches, test_size, window_size, epochs, learning_rate_value, truncate_value, truncate_value_front)
    pearson_mean, spearman_mean, mse = plot_prediction(pred, ground, "MLP", window_size, truncate_value, truncate_value_front, True)
    print(f"Pearson_cc: {pearson_mean}")
    print(f"Spearman_cc: {spearman_mean}")
    print(f"MSE: {mse}")
    #pred, ground = train_lstm(batches, input_size, hidden_size, num_layers, output_size, test_size, epochs,
    #                          learning_rate_value,
    #                          truncate_value, False)
    #pearson_mean, spearman_mean, mse = plot_prediction(pred, ground, "LSTM", window_size, truncate_value, False)

