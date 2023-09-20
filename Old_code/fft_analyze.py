import csv
import numpy as np
from matplotlib import pyplot as plt

def analyze_with_fft(ecg_file):
    data_array = []
    with open(ecg_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            data_array.append(row[0])

    # Load egg data into an array
    egg_data = np.loadtxt(data_array)

    # Perform Fourier transform
    transformed_data = np.fft.fft(egg_data)

    # Compute the frequencies corresponding to the transformed data
    frequencies = np.fft.fftfreq(len(egg_data))

    # Plot the transformed data
    plt.plot(frequencies, np.abs(transformed_data))
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.title('Fourier Transform of Egg Data')
    plt.show()


def analyze_with_fft_old(ecg_file):

    #TODO make a plot for each label
    label_list = ['0', '1', '2', '3', '9', '14', '15', '16', '17', '18', '19']

    data_array_0 = []
    data_array_1 = []
    data_array_2 = []
    data_array_3 = []
    data_array_9 = []
    data_array_14 = []
    data_array_15 = []
    data_array_16 = []
    data_array_17 = []
    data_array_18 = []
    data_array_19 = []
    with open(ecg_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[1] == '0':
                data_array_0.append(row[0])
            elif row[1] == '1':
                data_array_1.append(row[0])
            elif row[1] == '2':
                data_array_2.append(row[0])
            elif row[1] == '3':
                data_array_3.append(row[0])
            elif row[1] == '9':
                data_array_9.append(row[0])
            elif row[1] == '14':
                data_array_14.append(row[0])
            elif row[1] == '15':
                data_array_15.append(row[0])
            elif row[1] == '16':
                data_array_16.append(row[0])
            elif row[1] == '17':
                data_array_17.append(row[0])
            elif row[1] == '18':
                data_array_18.append(row[0])
            elif row[1] == '19':
                data_array_19.append(row[0])
    list_arrays = [data_array_0, data_array_1, data_array_2, data_array_3, data_array_9, data_array_14, data_array_15, data_array_16, data_array_17, data_array_18, data_array_19]
    count = 0
    for x in list_arrays:
        plt.figure()
        # Load egg data into an array
        egg_data = np.loadtxt(x)

        # Perform Fourier transform
        transformed_data = np.fft.fft(egg_data)

        # Compute the frequencies corresponding to the transformed data
        frequencies = np.fft.fftfreq(len(egg_data))

        # Plot the transformed data
        plt.plot(frequencies, np.abs(transformed_data))
        plt.xlabel('Frequency')
        plt.ylabel('Amplitude')
        plt.title(f'Fourier Transform of Egg Data {label_list[count]}')
        plt.savefig(f'Fourier Transform of Egg Data {label_list[count]}.png')
        plt.show()

        count += 1
        #print(x)


if __name__ == "__main__":
    print("Hello")
    # Output file name
    ecg_file = "../data/DHM 2/Andrei/transformed_data.csv"

    analyze_with_fft_old(ecg_file)
    print("end")
