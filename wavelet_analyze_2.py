import csv
import numpy as np
import pywt
import matplotlib.pyplot as plt

def analyze_with_wavelet(ecg_file):
    ecg_signal = []

    with open(ecg_file, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            ecg_signal.append(float(row[0]))

    # Apply wavelet transform
    #coeffs = pywt.wavedec(ecg_signal, 'db4', level=5)  # Use 'db4' wavelet and 5 decomposition levels
    coeffs = pywt.dwt(ecg_signal, 'db4')
    # Plot the wavelet coefficients
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, c in enumerate(coeffs):
        ax.plot(c, label='Level {}'.format(i))
    ax.set_title('Wavelet Coefficients')
    ax.legend()
    plt.show()

    # Reconstruct the signal
    reconstructed_data = pywt.waverec(coeffs, 'db4')

    # Plot the original and reconstructed signals
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(reconstructed_data, label='Reconstructed')
    ax.plot(ecg_signal, label='Original')
    ax.set_title('Original vs. Reconstructed Signal')
    ax.legend()
    plt.show()


if __name__ == "__main__":
    print("Hello Wavelet")
    # Output file name
    ecg_file = "data/DHM 2/Andrei/transformed_data.csv"

    analyze_with_wavelet(ecg_file)
    print("end")