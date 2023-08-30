import csv
import json
import wfdb
from matplotlib import pyplot as plt
from wfdb import processing
import numpy as np

from ecgdetectors import Detectors

##https://github.com/berndporr/py-ecg-detectors
#Authors
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
    detectors = Detectors(1000)
    with open(file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            data_as_tupel.append([(row[0], row[1])])  # row[0] = signal row[1] = label

    # print([ecg[0][0] for ecg in data_as_tupel])

    r_peaks = detectors.hamilton_detector([float(ecg[0][0]) for ecg in data_as_tupel])

    #print(r_peaks)
    #plot_peaks([float(ecg[0][0]) for ecg in data_as_tupel], r_peaks)

    #build batches:
    result_batches = []
    for index in r_peaks:
        result_batches.append(data_as_tupel[int(index-window_size/2):int(index+window_size/2)])


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
    batch = batches[2000]

    ecg_signal = [float(sig[0][0]) for sig in batch]

    print(ecg_signal)
    plt.plot(ecg_signal)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title(f"Plot batch 20")
    plt.show()

if __name__ == "__main__":
    print("Starting ...")

    batches = extract_qrs_complex("data_shared/transformed_data3_andrei.csv", 200)
    plot_batches(batches)

    print("Complete")
