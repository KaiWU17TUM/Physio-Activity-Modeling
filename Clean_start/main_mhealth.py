# prepare the mhealth data like the dhm data.
import csv

import numpy as np
from ecgdetectors import Detectors
from matplotlib import pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d

from Clean_start.evaluate_fft import evaluate_fft, evaluate_rfft, evaluate_rfft_trunc
from Clean_start.helper_functions import plot_prediction, up_sample, norm_sig, norm_sig_mhealth
from Clean_start.lstm_nn import train_lstm
from Clean_start.mlp_nn import train_mlp


def clense_batch(batches):
    result_batches = []
    for element in batches:
        signal_data = [float(sig[0][0]) for sig in element]
        if not (np.max(signal_data) > 1 or np.min(signal_data) < -1):
            signal_data = norm_sig_mhealth(signal_data)
            label_data = [float(sig[0][1]) for sig in element]
            res = []
            for sig, lable in zip(signal_data, signal_data):
                res.append([(sig, lable)])

            result_batches.append(res)

    return result_batches

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
                # data_as_tupel.append([(row[3], row[23])])  # row[3] = signal, row[23] = label
                data_signal.append(float(row[3]))
                data_lable.append(int(row[23]))

    # Oversample the signal and the labels
    up_sample_signal = up_sample(data_signal, 50,
                                 300)  # signal.resample(data_signal, len(data_signal) * 6) # up_sample(data_signal, 50, 300)
    up_sample_signal = norm_sig(
       up_sample_signal)  # signal.resample(data_signal, len(data_signal) * 6) # up_sample(data_signal, 50, 300)
    up_sample_lable = up_sample(data_lable, 50,
                                300)  # signal.resample(data_lable, len(data_lable) * 6) # up_sample(data_lable, 50, 300)


    for sig, lable in zip(up_sample_signal, up_sample_lable):
        data_as_tupel.append([(sig, lable)])

    # Detect R-peaks using the Hamilton detector
    r_peaks = detectors.hamilton_detector([float(ecg[0][0]) for ecg in data_as_tupel])


    # Build batches centered around R-peaks
    result_batches = []
    for index in r_peaks:
        new_batch = data_as_tupel[int(index - window_size / 2):int(index + window_size / 2)]
        if len(new_batch) < 200:
            break
        result_batches.append(new_batch)

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
    #print(len(result_batches))
    #result_batches = clense_batch(result_batches)
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
    hidden_size = 10
    hidden_size2 = 5
    num_layers = 2
    output_size = input_size - 1
    plot = True
    # To use the  uuntransfomred set this to = False
    test_set = False

    # inspect_mhealth("/Users/vet/Documents/Uni Allgemein/Master/SS23/Guided Research/Physio-Activity-Modeling/data/MHEALTHDATASET/mHealth_subject1.log")

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
    print(len(batches))

    print(f"person \t spearman \t mse")
    # for x in range(90, 100):
    #     test_size = x * 0.01
    # evaluate_fft(batches[0])
    # evaluate_rfft(batches[0])
    # evaluate_rfft_trunc(batches[0], truncate_value, truncate_value_front)
    print(test_size)
    # pred, y_test, ground = train_mlp(batches, test_size, window_size, epochs, learning_rate_value, truncate_value, truncate_value_front, hidden_size, hidden_size2)
    # pearson_mean, spearman_mean, mse = plot_prediction(pred[:20], y_test[:20], ground[:20], "MLP_Mhealth", window_size, truncate_value, truncate_value_front, plot, test_set)
    #
    # print(f"{pearson_mean}\t{spearman_mean}\t{mse}")
    pred, y_test, ground = train_lstm(batches, input_size, hidden_size, num_layers, output_size, test_size, epochs,
                                      learning_rate_value,
                                      truncate_value, truncate_value_front)
    pearson_mean, spearman_mean, mse = plot_prediction(pred[:20], y_test[:20], ground[:20], "LSTM", window_size, truncate_value,
                                                       truncate_value_front, plot, test_set)

    # print(test_size)
    print(f"{pearson_mean}\t{spearman_mean}\t{mse}")
