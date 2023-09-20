import csv

import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from nn_3 import split_to_batches

if __name__ == "__main__":
    ecg_file = "../data/DHM 2/Andrei/transformed_data.csv"

    data_array = []
    # split in a multi dim array. ech list element is a tuple a list of the ecg of a action and the lable
    with open(ecg_file, "r") as f:
        reader = csv.reader(f)
        count = 0
        for row in reader:
            if count == 1200:
                break
            data_array.append(row[0])
            count += 1

    noise = np.random.normal(0, 100, size=(1200, 2)).view(np.complex128).flatten()
    fourier = np.fft.fft(data_array)

    four_1 = noise + fourier
    print(four_1)

    # Inverse Fourier transform
    inv_fourier = np.fft.ifft(four_1)
    inv_fourier_orig = np.fft.ifft(fourier)
    # Plot the original and inverse Fourier-transformed data
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(np.abs(fourier))
    plt.title("Fourier Transformed Data")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")

    plt.subplot(2, 1, 2)
    plt.plot(np.real(inv_fourier))
    plt.plot(np.real(inv_fourier_orig))
    plt.title("Inverted Fourier Transformed Data")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()
