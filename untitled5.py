import numpy as np
import matplotlib.pyplot as plt
def downsample_signal(signal, original_hz, target_hz):
    # Calculate the downsampling factor
    downsample_factor = original_hz / target_hz
    
    # Determine the new length of the downsampled signal
    new_length = int(len(signal) / downsample_factor)
    
    # Resample the signal
    downsampled_signal = np.zeros(new_length)
    for i in range(new_length):
        downsampled_signal[i] = signal[int(i * downsample_factor)]
    
    return downsampled_signal

# Example ECG signal array sampled at 500 Hz
t = np.linspace(0, 10, 5000)
waveform = np.sin(2 * np.pi * t)
ecg_signal_500hz = waveform  # Replace with your actual ECG signal array

# Downsample the signal to 200 Hz
downsampled_signal_200hz = downsample_signal(ecg_signal_500hz, original_hz=500, target_hz=200)

# Check the length of the downsampled signal
print("Length of downsampled signal (200 Hz):", len(downsampled_signal_200hz))


plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(ecg_signal_500hz, label='1', color='b')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('1 & 2')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(downsampled_signal_200hz, label='3', color='g')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('3')
plt.legend()
plt.grid(True)