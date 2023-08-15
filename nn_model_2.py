#Was möchte ich tun: Ich möchte einen Array einlesen mit den letzen X ecg datenpunkten (ca 100) und zusätzlich eine Info
# pber die sportart dazu dann gebe ich die aktivität mit und predicte wie die nächsten n ecg datenpunkte aussehen.
# Wäre optimal vorzutrainieren was ein ecg ist und dann nur noch optimizen ?
import csv

# Look into stock market prediction, they try to do the same.
#Possible data sources for algo: https://towardsdatascience.com/the-complete-guide-to-time-series-analysis-and-forecasting-70d476bfe775
# Transformer for time series https://arxiv.org/pdf/2205.13504.pdf

# Ich möchte die abstände zwischen einzelnen beats predicten nicht die höhe oder sowas. Sondern die frequenz => deshlab
# benutz ich die Fourier transofmraion zum beispiel oder die wavelet decomposition ? will ich dannach zurück transfomieren ?

# Current approch: Transform data into Furnier => train a classifier, with transform and label predict next fourier
# inverse fournier and try to get the ecg signal in this way.

import numpy as np
from matplotlib import pyplot as plt

from Build_batches import split_ecg_to_action_lists

#build batches without crossections between themself. Just cut the array in pieces of the same length.
def split_to_batches(ecg_file):
    data_array_action = []
    # split in a multi dim array. ech list element is a tuple a list of the ecg of a action and the lable
    with open(ecg_file, "r") as f:
        reader = csv.reader(f)
        data_array = []
        target_length = 1200  # ca 20 beats in theorie but it around 3-6
        length_counter = 0
        score_cash = 0
        for row in reader:
            if length_counter <= target_length:
                data_array.append(row[0])
                score_cash += float(row[1])
                length_counter += 1
            else:
                data_array_action.append((data_array, score_cash/target_length))
                data_array = [row[0]]
                score_cash = float(row[1])
                length_counter = 1

    return data_array_action


#read in batches make fft and fft back and compare the results ?
def plot_data_fft(data_array_action):
    # Prepare the data
    # Extract input data and labels
    signal_array_strings = np.array([data for data, _ in data_array_action])[0]
    score = np.array([label for _, label in data_array_action])

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

def train_model_2(batches):
    #Data overview:
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
    for i in range(len(batches) - 1):
        # print(batches[i])
        #connect scor to signal davor brauchst du die Transformation
        #TODO how ???
        fourier_transform = np.fft.fft(batches[i][0])
        #TODO do i need this ?
        frequencies = np.fft.fftfreq(len(fourier_transform))

        #Brauche den score form nächsten nciht vom jetzigen.
        temp = list(fourier_transform + np.array([batches[i+1][1]]))
        x_train.append(temp)

        fourier_transform2 = np.fft.fft(batches[i+1][0])
        frequencies = np.fft.fftfreq(len(fourier_transform))

        y_train.append(fourier_transform2)

    #print(x_train, y_train)



if __name__ == "__main__":
    print("NN_model_2")
    # Output file name
    ecg_file = "data/DHM 2/Andrei/transformed_data.csv"
    data_array_action = split_to_batches(ecg_file)
    # plot_data_fft(data_array_action)
    train_model_2(data_array_action)
    print("end")
