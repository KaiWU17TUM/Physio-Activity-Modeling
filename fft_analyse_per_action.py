import csv
import numpy as np
import pywt
from matplotlib import pyplot as plt

def split_ecg_to_action_lists(ecg_file):
    data_array_action = []
    # split in a multi dim array. ech list element is a tuple a list of the ecg of a action and the lable
    with open(ecg_file, "r") as f:
        reader = csv.reader(f)
        data_array = []
        row_curr_label = '0'
        for row in reader:
            if row[1] == row_curr_label:
                data_array.append(row[0])
            else:
                data_array_action.append((data_array, row_curr_label))
                data_array = []
                data_array.append(row[0])
                row_curr_label = row[1]

    return data_array_action


def perform_fft(list_of_tupel_action):
    action_number = 0
    for ecg_list, label in list_of_tupel_action:

        # Load ecg data into an array
        ecg_data = np.loadtxt(ecg_list)

        # Perform Fourier transform
        transformed_data = np.fft.fft(ecg_data)

        # Compute the frequencies corresponding to the transformed data
        frequencies = np.fft.fftfreq(len(ecg_data))

        # Plot the transformed data
        plt.plot(frequencies, np.abs(transformed_data))
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        plt.title(f'Fourier Transform of Ecg Data, action_number {action_number} action {label}')
        plt.savefig(f'Fourier Transform of Ecg Data, action_number {action_number} action {label} .png')
        plt.show()
        action_number += 1

def perform_wavelet(list_of_tupel_action):
    action_number = 0
    for ecg_list, label in list_of_tupel_action:

        # Load ecg data into an array
        ecg_data = np.loadtxt(ecg_list)

        coeffs = pywt.dwt(ecg_data, 'db4')
        # Plot the wavelet coefficients
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, c in enumerate(coeffs):
            ax.plot(c, label='Level {}'.format(i))
        ax.set_title(f'Wavelet Coefficients action {label} number {action_number}')
        plt.savefig(f'Wavelet Coefficients action {label} action_number {action_number}.png')
        action_number += 1
        ax.legend()
        plt.show()


if __name__ == "__main__":
    print("Hello")
    # Output file name
    ecg_file = "data/DHM 2/Andrei/transformed_data.csv"
    data_array_action = split_ecg_to_action_lists(ecg_file)
    perform_fft(data_array_action)
    print("end")
