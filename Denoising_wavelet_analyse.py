import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt  # for wavelet denoising

def analyze_with_wavelet(file_name):
    noisy_ecg = []

    with open(file_name, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            noisy_ecg.append(float(row[0]))

    # Generate sample ECG data
    # Replace this with your actual ECG data
    time = np.arange(0, len(noisy_ecg), 1)

    # Method 1: Moving average
    window_size = 100
    smooth_ecg_ma = pd.Series(noisy_ecg).rolling(window=window_size, center=True).mean()

    # Method 2: Wavelet denoising
    wavelet = 'db4'  # Choose a wavelet function
    level = 3  # Choose the level of decomposition (adjust as needed)

    # Decompose the signal using wavelet transform
    coeffs = pywt.wavedec(noisy_ecg, wavelet, level=level)

    # Set the threshold for noise reduction
    threshold = np.std(coeffs[-level]) * np.sqrt(2 * np.log(len(noisy_ecg)))

    # Apply thresholding to the wavelet coefficients
    denoised_coeffs = [pywt.threshold(c, threshold) for c in coeffs]

    # Reconstruct the signal from the denoised coefficients
    smooth_ecg_wd = pywt.waverec(denoised_coeffs, wavelet)

    plt.subplot(3, 1, 2)
    plt.plot(time, noisy_ecg, 'r', label='Noisy ECG')
    plt.plot(time, smooth_ecg_ma, 'g', label='Moving Average')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('ECG Data with Moving Average Smoothing')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time, noisy_ecg, 'r', label='Noisy ECG')
    plt.plot(time, smooth_ecg_wd, 'm', label='Wavelet Denoising')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('ECG Data with Wavelet Denoising')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Hello Wavelet")
    # Output file name
    ecg_file = "data/DHM 2/Andrei/transformed_data.csv"

    analyze_with_wavelet(ecg_file)
    print("end")