#prepare the mhealth data like the dhm data.
import csv

import numpy as np
from ecgdetectors import Detectors
from matplotlib import pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d

from Clean_start.evaluate_fft import evaluate_fft, evaluate_rfft, evaluate_rfft_trunc
from Clean_start.helper_functions import plot_prediction, up_sample, norm_sig
from Clean_start.mlp_nn import train_mlp


def extract_qrs_complex(files, window_size):
    # Load ECG data from CSV
    data_as_tupel = []
    data_signal = []
    data_lable = []
    detectors = Detectors(300)  # mhealth uses 50hz up sampled to 300
    for file in files:
        with open(file, "r") as f:
            reader = csv.reader(f, delimiter='\t')
            for row in reader:
                #data_as_tupel.append([(row[3], row[23])])  # row[3] = signal, row[23] = label
                data_signal.append(float(row[3]))
                data_lable.append(int(row[23]))

    print(len(data_signal))
    # Oversample the signal and the labels
    up_sample_signal = norm_sig(up_sample(data_signal, 50, 300)) # signal.resample(data_signal, len(data_signal) * 6) # up_sample(data_signal, 50, 300)
    up_sample_lable = up_sample(data_lable, 50, 300) # signal.resample(data_lable, len(data_lable) * 6) # up_sample(data_lable, 50, 300)
    print(len(up_sample_signal))

    for sig, lable in zip(up_sample_signal, up_sample_lable):
        data_as_tupel.append([(sig, lable)])

    # Detect R-peaks using the Hamilton detector
    r_peaks = detectors.hamilton_detector([float(ecg[0][0]) for ecg in data_as_tupel])

    # Build batches centered around R-peaks
    result_batches = []
    for index in r_peaks:
        result_batches.append(data_as_tupel[int(index - window_size / 2):int(index + window_size / 2)])

    # time = range(len(result_batches[0]))
    # plot_data = np.array([ecg[0][0] for ecg in result_batches[0]]).astype(float)
    #
    # # Create a plot
    # plt.figure(figsize=(8, 4))  # Optional: Set the figure size
    # plt.plot([float(ecg[0][0]) for ecg in data_as_tupel][:600], linestyle=':')
    # for position in r_peaks:
    #     if position > 600:
    #         break
    #     plt.axvline(x=position, color='red', linestyle='--')
    # #plt.plot(time, plot_data, linestyle='-')
    # #plt.axvline(r_peaks[0])
    # plt.title('Signal Plot')
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    #
    # # Show the plot
    # plt.show()

    #TODO where are the peaks gone it were 35 K previously
    print(len(result_batches))
    return result_batches

def inspect_mhealth(file):
    data_as_tupel = []
    detectors = Detectors(1000)  # You may need to adjust the sampling rate here
    with open(file, "r") as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            data_as_tupel.append([(row[3], row[23])])  # 3 is safe an ECG but 4 ??
    ecg_data = [float(ecg[0][0]) for ecg in data_as_tupel][10000:10050]

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
    print("start Mhealth")
    window_size = 200
    test_size = 0.1
    epochs = 200
    learning_rate_value = 0.01
    truncate_value = 25
    truncate_value_front = 2
    input_size = (truncate_value - truncate_value_front) * 2 + 1
    hidden_size = 200
    num_layers = 2
    output_size = input_size - 1

    #inspect_mhealth("/Users/vet/Documents/Uni Allgemein/Master/SS23/Guided Research/Physio-Activity-Modeling/data/MHEALTHDATASET/mHealth_subject1.log")

    files = [
        "/Users/vet/Documents/Uni Allgemein/Master/SS23/Guided Research/Physio-Activity-Modeling/data/MHEALTHDATASET/mHealth_subject1.log",
        "/Users/vet/Documents/Uni Allgemein/Master/SS23/Guided Research/Physio-Activity-Modeling/data/MHEALTHDATASET/mHealth_subject2.log",
        "/Users/vet/Documents/Uni Allgemein/Master/SS23/Guided Research/Physio-Activity-Modeling/data/MHEALTHDATASET/mHealth_subject3.log",
        "/Users/vet/Documents/Uni Allgemein/Master/SS23/Guided Research/Physio-Activity-Modeling/data/MHEALTHDATASET/mHealth_subject4.log",
        "/Users/vet/Documents/Uni Allgemein/Master/SS23/Guided Research/Physio-Activity-Modeling/data/MHEALTHDATASET/mHealth_subject5.log",
        "/Users/vet/Documents/Uni Allgemein/Master/SS23/Guided Research/Physio-Activity-Modeling/data/MHEALTHDATASET/mHealth_subject6.log",
        "/Users/vet/Documents/Uni Allgemein/Master/SS23/Guided Research/Physio-Activity-Modeling/data/MHEALTHDATASET/mHealth_subject7.log",
        "/Users/vet/Documents/Uni Allgemein/Master/SS23/Guided Research/Physio-Activity-Modeling/data/MHEALTHDATASET/mHealth_subject8.log",
        "/Users/vet/Documents/Uni Allgemein/Master/SS23/Guided Research/Physio-Activity-Modeling/data/MHEALTHDATASET/mHealth_subject9.log"
    ]

    batches = extract_qrs_complex(files, window_size)

    # evaluate_fft(batches[0])
    # evaluate_rfft(batches[0])
    #evaluate_rfft_trunc(batches[0], truncate_value, truncate_value_front)
    pred, ground = train_mlp(batches, test_size, window_size, epochs, learning_rate_value, truncate_value, truncate_value_front)
    pearson_mean, spearman_mean, mse = plot_prediction(pred, ground, "MLP_Mhealth", window_size, truncate_value, truncate_value_front,  True)
    print(f"Pearson_cc: {pearson_mean}")
    print(f"Spearman_cc: {spearman_mean}")
    print(f"MSE: {mse}")
    #pred, ground = train_lstm(batches, input_size, hidden_size, num_layers, output_size, test_size, epochs,
    #                          learning_rate_value,
    #                          truncate_value, False)
    #pearson_mean, spearman_mean, mse = plot_prediction(pred, ground, "LSTM", window_size, truncate_value, False)
