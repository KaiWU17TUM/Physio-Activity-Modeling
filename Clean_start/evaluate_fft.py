import numpy as np
from matplotlib import pyplot as plt


def evaluate_fft(batch):
    """

    It will plot the different FFTs and after transformation.
    3 plots: FFT, vs rFFT vs rFFT truncated original and inverse
    4 auf 4 plots one original, one explainitation, one tih noise one without ??

    :param batch: input signal from a ecg
    :return: None just plot
    """

    #TODO the noise needs to be applied to the signal not the frequency
    ecg_signal = [float(sig[0][0]) for sig in batch]

    #Classic fourier
    #here show that noise reduction ? but noise in the signal not later
    fourier = np.fft.fft(ecg_signal)

    noise_classic = np.random.normal(0, 1, size=(len(batch), 2)).view(np.complex128).flatten()
    # noise_signal = np.random.normal(0, 1, size=(len(batch), 2)).view(np.complex128).flatten()
    # noise_signal = ecg_signal + noise_signal
    #fourier_noise_sig = np.fft.fft(noise_signal)
    four_1 = noise_classic + fourier
    # Inverse Fourier transform
    inv_fourier_orig = np.fft.ifft(fourier)
    inv_fourier = np.fft.ifft(four_1)
    #inv_fourier_noise_sig = np.fft.ifft(fourier_noise_sig)

    # Plot the original and inverse Fourier-transformed data
    plt.figure(figsize=(6, 6))

    plt.subplot(2, 1, 1)
    plt.plot(np.abs(fourier))
    plt.plot(np.abs(four_1))
    #plt.plot(np.abs(fourier_noise_sig))
    plt.title("Fourier Transformed Data")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")

    plt.subplot(2, 1, 2)
    plt.plot(np.real(inv_fourier), label='inverse fourier noise in fft')
    plt.plot(np.real(inv_fourier_orig), linestyle='dashed', label='inverse fourier')
    plt.plot(ecg_signal, linestyle='dotted', label='original signal')
    #plt.plot(np.real(inv_fourier_orig), linestyle='dashed', label='inverse fourier')
    #plt.plot(ecg_signal, linestyle='dotted', label='original signal')
    #plt.plot(inv_fourier_noise_sig, linestyle='dotted', label='fft noise in signal')
    #plt.plot(noise_signal, linestyle='--', label='fft noise in signal')
    plt.legend()
    plt.title("Inverted Fourier Transformed Data")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

def evaluate_rfft(batch):
    #rfft

    ecg_signal = [float(sig[0][0]) for sig in batch]
    fourier = np.fft.rfft(ecg_signal)

    noise_rfft = np.random.normal(0, 1, size=(int(len(batch)/2)+1, 2)).view(np.complex128).flatten()
    four_1 = noise_rfft + fourier

    # Inverse Fourier transform
    inv_fourier_orig = np.fft.irfft(fourier)
    inv_fourier = np.fft.irfft(four_1)

    # Plot the original and inverse Fourier-transformed data
    plt.figure(figsize=(6, 6))

    plt.subplot(2, 1, 1)
    plt.plot(np.abs(fourier))
    plt.plot(np.abs(four_1))
    plt.title("r Fourier Transformed Data")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")

    plt.subplot(2, 1, 2)
    plt.plot(np.real(inv_fourier), label='inverse fourier with noise')
    plt.plot(np.real(inv_fourier_orig), linestyle='dashed', label='inverse fourier')
    plt.plot(ecg_signal, linestyle='dotted', label='original signal')
    plt.legend()
    plt.title("Inverted r Fourier Transformed Data")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

def evaluate_rfft_trunc(batch, truncate_value, truncate_value_front):
    np.random.seed(42)
    #here show if noise reduction i mean noise to signal
    # make two plots 1 to show that no info is lost
    # one that shows noise reduction

    ecg_signal = [float(sig[0][0]) for sig in batch]

    #rfft truncated
    fourier = np.fft.rfft(ecg_signal)[truncate_value_front:truncate_value]
    # noise_rfft_trunc = np.random.normal(0, 10, size=(truncated_value, 2)).view(np.complex128).flatten()
    # four_1 = noise_rfft_trunc + fourier
    # Inverse Fourier transform
    #Add zeros als much as trucated:
    zeros_to_add = int(len(batch)/2+1) - truncate_value
    array_to_add = np.zeros(zeros_to_add)
    array_to_add_front = np.zeros(truncate_value_front)
    inverse_array = np.append(array_to_add_front, fourier)
    inv_fourier_orig = np.fft.irfft(np.append(inverse_array, array_to_add))
    # inv_fourier = np.fft.irfft(np.append(four_1, array_to_add))

    # Plot the original and inverse Fourier-transformed data
    plt.figure(figsize=(6, 6))

    plt.subplot(2, 1, 1)
    plt.plot(np.abs(fourier))
    plt.title(f"r FFT trunc {len(batch)} to {truncate_value} front cut {truncate_value_front}")
    plt.xlabel("Frequency")
    plt.ylabel("Magnitude")

    plt.subplot(2, 1, 2)
    # plt.plot(np.real(inv_fourier), label='inverse fourier with noise')
    plt.plot(np.real(inv_fourier_orig), linestyle='dashed', label='inverse fourier')
    plt.plot(ecg_signal, linestyle='dotted', label='original signal')
    plt.legend()
    plt.title(f"Irfft truncated {len(batch)} to {truncate_value} front cut {truncate_value_front}")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

    #noise reduction:

    ecg_signal = [float(sig[0][0]) for sig in batch]

    noise_signal = np.random.normal(0, 0.01, size=(len(batch), 2)).view(np.complex128).flatten()
    noise_signal = ecg_signal + noise_signal

    #rfft truncated
    fourier = np.fft.rfft(noise_signal)[truncate_value_front:truncate_value]

    # Inverse Fourier transform
    #Add zeros als much as trucated:
    zeros_to_add = int(len(batch)/2+1) - truncate_value
    array_to_add = np.zeros(zeros_to_add)
    array_to_add_front = np.zeros(truncate_value_front)
    inverse_array = np.append(array_to_add_front, fourier)
    inv_fourier_orig = np.fft.irfft(np.append(inverse_array, array_to_add))
    # inv_fourier = np.fft.irfft(np.append(four_1, array_to_add))

    # Plot the original and inverse Fourier-transformed data
    plt.figure(figsize=(6, 6))

    plt.subplot(2, 1, 1)
    # plt.plot(np.real(inv_fourier), label='inverse fourier with noise')
    #plt.plot(np.real(inv_fourier_orig), linestyle='dashed', label='inverse fourier')
    plt.plot(ecg_signal, linestyle='dotted', label='original signal')
    plt.plot(noise_signal, linestyle='-', label='noise signal')
    plt.legend()
    plt.title(f"Irfft truncated without noise reduction")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    # plt.plot(np.real(inv_fourier), label='inverse fourier with noise')
    plt.plot(np.real(inv_fourier_orig), linestyle='dashed', label='inverse fourier')
    plt.plot(ecg_signal, linestyle='dotted', label='original signal')
    #plt.plot(noise_signal, linestyle='-', label='noise signal')
    plt.legend()
    plt.title(f"Irfft  truncated noise reduction")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

