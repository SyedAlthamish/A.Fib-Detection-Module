#import random
import pandas as pd
#loading the support dataframe
df = pd.read_csv('Data\coorteeqsrafva.csv', sep=';', header=0, index_col=0)

#loading ecg data
import numpy as np
data= np.load('Data\ecgeq-500hzsrfava.npy')

# #taking single person's data from lead 2 ecg
# singlepersondata=data[0,:,:]
# print(singlepersondata)
# lead2singleperson=singlepersondata[:,1]
# print(lead2singleperson)

#plotting the secluded data
import matplotlib.pyplot as plt
# plt.plot(lead2singleperson[0:1000])
# plt.show()



# #taking random person's data who's normal
afib_df=df.copy()
# normal_case = random.choice(list(afib_df[afib_df['ritmi']=='SR'].index))
# print(normal_case)
random_normal_indexes=[4799,310,4780,4653,4943,2024,449]
random_afibrillation_indexes=[167,2547,819,3868,2376,4925,2320]
allrelevindexes=np.concatenate([random_normal_indexes, random_afibrillation_indexes])

# Plot original and smoothed ECG signals
def plot_ecg_signals(ecg_signal, peaks, derivative_signal):
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(ecg_signal, label='1', color='b')
    plt.scatter(peaks, ecg_signal[peaks], label='2', color='r')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('1 & 2')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(derivative_signal, label='3', color='g')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('3')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

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

def low_pass_filter(ecg_signal, T):
    y = [0, 0]  # Initial conditions for y(nT - T) and y(nT - 2T)
    filtered_signal = []
    
    for i in range(0,len(ecg_signal)):
        # Ensure we have enough previous samples for calculation
        # Apply the low-pass filter formula
        #if i > 6:
        y_nT = 2 * y[-1] - y[-2] + ecg_signal[i] - 2 * ecg_signal[i-6] + ecg_signal[i-12]
        filtered_signal.append(y_nT)
        # else:
        #     y_nT=0
        #     filtered_signal.append(0)  # Fill initial samples with zeros
    
        # Update the y array for the next iteration
        y.append(y_nT)

    return filtered_signal

def hpf(ecg_signal, T):
    y = [0]  # Initial conditions for y(nT - T) and y(nT - 2T)
    filtered_signal = []
    
    for i in range(1,len(ecg_signal)):
        # Ensure we have enough previous samples for calculation
        # Apply the low-pass filter formula
        #if i > 6:
        y_nT = 32 * ecg_signal[i - 16] - ( y[i - 1] + ecg_signal[i] - ecg_signal[ i - 32 ])
        filtered_signal.append(y_nT)
        # else:
        #     y_nT=0
        #     filtered_signal.append(0)  # Fill initial samples with zeros
    
        # Update the y array for the next iteration
        y.append(y_nT)

    return filtered_signal
       
def differentiate_signal(ecg_signal, T):
    diff_signal = np.zeros_like(ecg_signal)
    for n in range(2, len(ecg_signal) - 2):
        diff_signal[n] = (1/(8*T)) * (-ecg_signal[n - 2] - 2*ecg_signal[n - 1] + 2*ecg_signal[n + 1] + ecg_signal[n + 2])
    return diff_signal

def square(ecg_signal):
    squaredsig=ecg_signal[:]*ecg_signal[:]
    return squaredsig

def moving_window_integrator(signal, N):
    integrator_output = np.zeros_like(signal)
    for n in range(N-1, len(signal)):
        integrator_output[n] = (1/N) * np.sum(signal[n-(N-1):n+1])
    return integrator_output

def detect_peaks(signal):
    peaks = []
    for i in range(15, len(signal) - 15):
        if signal[i] > signal[i-15] and signal[i] > signal[i+15]:
            peaks.append(i)
            
    return peaks

def initializethreshold(ecg_signal):
    SigLevel = 0.7*max(ecg_signal[:1000])
    NoiseLevel = np.mean(ecg_signal[:1000])
    Th1 =  NoiseLevel + 0.25 * (SigLevel-NoiseLevel)
    
    return SigLevel, NoiseLevel, Th1

def findqrs(ecg_signal,peaks,Th1,SigLevel,NoiseLevel):
    qrspeaks=[]
    for i in peaks:
        peakvalue=ecg_signal[i]
        if peakvalue > Th1:
            qrspeaks.append(i)
            SigLevel = 0.125 * peakvalue + 0.875 * SigLevel
        else:
            NoiseLevel = 0.125 * peakvalue + 0.875 * NoiseLevel
        Th1 =  NoiseLevel + 0.25 * (SigLevel-NoiseLevel)

    return qrspeaks

def removeqrsduplicates(peaks):
    polishedqrs=[]
    polishedqrs.append(peaks[0])
    for i in range(1,len(peaks)):
        if peaks[i]-peaks[i-1] > 50:
            polishedqrs.append(peaks[i])
    return polishedqrs

def remove_qrs_duplicates(peaks):                      # Define a function to remove duplicate QRS peaks
    polished_qrs = []                                  # Initialize a list to store the filtered QRS peaks
    polished_qrs.append(peaks[0])                      # Always add the first peak to the polished list
    
    for i in range(1, len(peaks)):                     # Iterate through the peaks starting from the second element
        if peaks[i] - peaks[i - 1] > 50:               # Check if the current peak is more than 50 units away from the previous peak
            polished_qrs.append(peaks[i])              # If so, add it to the polished list
            
    return polished_qrs                                # Return the list of polished QRS peaks

def qrspeaks2time(peaks):
    timeofpeaks= []
    for i in peaks:
        timeofpeaks.append(i*(10/5000))
    return timeofpeaks

def findintervals(timeofpeaks):
    beatintervals=[]
    for i in range(1,len(timeofpeaks)):
        beatintervals.append(timeofpeaks[i]-timeofpeaks[i-1])
    return beatintervals

def absintervalsdifference(intervals):
    intervalsdifference=[]
    for i in range(1,len(intervals)):
        intervalsdifference.append(abs(intervals[i]-intervals[i-1]))
    return intervalsdifference

T=1/100
for i in allrelevindexes: 
    testdata500= data[i,:,1]
    testdata=downsample_signal(testdata500, 500, 200)
    lpftestdata=low_pass_filter(testdata, T)
    hpftestdata=hpf(lpftestdata, T)
    difftestdata=differentiate_signal(hpftestdata, T)
    squaretestdata=square(difftestdata)
    N=30
    integratordata=moving_window_integrator(squaretestdata, N)
    #plot_ecg_signals([0],testdata,integratordata)
    
    
    
    #################################################################################
    
    # def compute_thresholds(signal, peaks, is_signal=True):
    #     PEAK = max(signal[peaks])
    #     if is_signal:
    #         SPK = PEAK
    #         NPK = np.mean(signal)
    #     else:
    #         SPK = np.mean(signal)
    #         NPK = PEAK
    #     THRESHOLD1 = NPK + 0.25 * (SPK - NPK)
    #     THRESHOLD2 = 0.5 * THRESHOLD1
    #     return THRESHOLD1, THRESHOLD2
    
    # def adjust_thresholds_for_irregular_rates(THRESHOLD1, THRESHOLD2):
    #     THRESHOLD1 *= 0.5
    #     THRESHOLD2 *= 0.5
    #     return THRESHOLD1, THRESHOLD2
    
    # # Example usage
    # # Assuming ecg_signal is your ECG signal array
    # ecg_signal = integratordata # Your ECG signal array
    
    # # Detect peaks in the ECG signal
    # peaks,peakedsignal = detect_peaks(ecg_signal)
    
    # # Compute thresholds for the integration waveform
    # THRESHOLD1_int, THRESHOLD2_int = compute_thresholds(ecg_signal, peaks, is_signal=True)
    
    # # Compute thresholds for the filtered ECG signal
    # THRESHOLD1_filt, THRESHOLD2_filt = compute_thresholds(ecg_signal, peaks, is_signal=False)
    
    # # Adjust thresholds for irregular heart rates
    # THRESHOLD1_int, THRESHOLD2_int = adjust_thresholds_for_irregular_rates(THRESHOLD1_int, THRESHOLD2_int)
    # THRESHOLD1_filt, THRESHOLD2_filt = adjust_thresholds_for_irregular_rates(THRESHOLD1_filt, THRESHOLD2_filt)
    
    peaks=detect_peaks(integratordata)
    Th1, SigLevel, NoiseLevel = initializethreshold(integratordata)
    qrspeaks = findqrs(integratordata, peaks, Th1, SigLevel, NoiseLevel)
    polishedqrs=removeqrsduplicates(qrspeaks)
    
    #plot_ecg_signals(integratordata,peaks,testdata)
    #plot_ecg_signals(integratordata,qrspeaks,testdata)
    #plot_ecg_signals(integratordata,polishedqrs,testdata)
    # plt.figure(figsize=(12,8))
    # #plt.plot(t, waveform, label='Waveform')
    # plt.scatter(peaks, integratordata[peaks], color='red', label='Peaks')
    # plt.title('Identifying Peaks in a Waveform')
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude')
    # plt.legend()
    # plt.grid(True)
    # plt.show()
    timeofpeaks=qrspeaks2time(polishedqrs)
    #print(timeofpeaks)
    intervals=findintervals(timeofpeaks)
    #print(intervals)
    mean=np.mean(intervals)
    #print(mean)
    intervaldifference=absintervalsdifference(intervals)
    #print(intervaldifference)
    meanintervaldifference=np.mean(intervaldifference)
    MSBID=meanintervaldifference/mean
    if (MSBID>0.11): #threshold selected based on paper: "langley2012.pdf"
        print("AF")
    else:
        print("SA")