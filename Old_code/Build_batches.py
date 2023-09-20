#Split the data into batches of the same size (elements) and a label for the base data and that for fft and wavelet
import csv

import numpy as np
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

        #TODO das ist flasch glaube ich kürzt die batches unnötig stark
        # Add the last batch if it's not empty
        if data_array:
            data_array_action.append((data_array, row_curr_label))

        # Determine the length of the shortest array
        shortest_length = min(len(data) for data, _ in data_array_action)

        # Split arrays to match the length of the shortest array
        #TODO change so that its like a sliding window over the array of a fixed length.
        split_data_array_action = []
        for data, label in data_array_action:
            num_batches = len(data) // shortest_length
            for i in range(num_batches):
                split_data = data[i * shortest_length: (i + 1) * shortest_length]
                split_data_array_action.append((split_data, label))
            remaining_data = data[num_batches * shortest_length:]
            #TODO remove this because i dont want a half fill array at the end
            #if remaining_data:
            #    split_data_array_action.append((remaining_data, label))

    return split_data_array_action

def split_ecg_to_action_lists_sliding_window(ecg_file):
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

        # Add the last batch if it's not empty
        if data_array:
            data_array_action.append((data_array, row_curr_label))

        # Determine the length of the shortest array
        shortest_length = min(len(data) for data, _ in data_array_action)

        # Split arrays to match the length of the shortest array
        # TODO change so that its like a sliding window over the array of a fixed length.
        split_data_array_action = []
        for data, label in data_array_action:
            num_batches = len(data) // shortest_length
            for i in range(num_batches):
                split_data = data[i * shortest_length: (i + 1) * shortest_length]
                split_data_array_action.append((split_data, label))
            remaining_data = data[num_batches * shortest_length:]

    return split_data_array_action


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

if __name__ == "__main__":
    print("Hello")
    # Output file name
    ecg_file = "../data/DHM 2/Andrei/transformed_data.csv"
    data_array_action = split_ecg_to_action_lists(ecg_file)
    #perform_fft(data_array_action)
    print("end")


