import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

def highpass_filter_difference_eq(ecg_signal, cutoff_freq, fs):
    # Calculate coefficients
    nyq = 0.5 * fs
    fc = cutoff_freq / nyq
    k = np.tan(np.pi * fc)
    k2 = k * k
    sqrt2 = np.sqrt(2)
    k_over_sqrt2 = k / sqrt2
    a = 1 + sqrt2 * k + k2
    b0 = 1
    b1 = -2
    b2 = 1
    a0 = 1
    a1 = 2 * (k2 - 1)
    a2 = 1 - sqrt2 * k + k2

    # Initialize filtered signal
    filtered_signal = np.zeros_like(ecg_signal)

    # Apply high-pass filter difference equation
    for n in range(2, len(ecg_signal)):
        filtered_signal[n] = (1 / a) * (b0 * ecg_signal[n] + b1 * ecg_signal[n - 1] + b2 * ecg_signal[n - 2]
                                        - a1 * filtered_signal[n - 1] - a2 * filtered_signal[n - 2])

    return filtered_signal

# Example usage
fs = 1000  # Sampling frequency in Hz
cutoff_freq = 5  # Cutoff frequency in Hz

# Generate example ECG signal (replace this with your actual ECG signal)
t = np.linspace(0, 10, 10000, endpoint=False)
ecg_signal = 0.5 * np.sin(2*np.pi*0.01*t)

# Apply high-pass filter using difference equation
filtered_ecg_signal = highpass_filter_difference_eq(ecg_signal, cutoff_freq, fs)

# Plot the original and filtered signals
plt.figure(figsize=(10, 6))
plt.plot(t, ecg_signal, label='Original ECG Signal')
plt.plot(t, filtered_ecg_signal, label='Filtered ECG Signal (HPF)')
plt.title('High-Pass Filtered ECG Signal (Difference Equation)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()
