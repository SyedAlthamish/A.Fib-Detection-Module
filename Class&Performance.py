'''{
    File Description:
        This file contains the implementation of the classification
        algorithm and performance estimation. The base paper for classification
        is the "langley2012.pdf" given under papers subfolder i.e. Mean 
        successive beat interval difference. The basis of performance 
        estimation is accuracy. The data used is taken from Data subfolder.
    }'''

import random
import pandas as pd
import matplotlib.pyplot as plt
#loading the support dataframe
df = pd.read_csv('Data\coorteeqsrafva.csv', sep=';', header=0, index_col=0)

#loading ecg data
import numpy as np
data= np.load('Data\ecgeq-500hzsrfava.npy')

afib_df=df.copy()

#taking a master_indices which holds noof_epochs lists of randomly selected sr and af samples
sr_indices = afib_df[afib_df['ritmi'] == 'SR'].index.tolist()
af_indices = afib_df[afib_df['ritmi'] == 'AF'].index.tolist()
master_indices=[]
noof_epochs=5
num_samples = 7 # Change this to the number of indices you want
for _ in range(noof_epochs):
    random_SR_indices = random.sample(sr_indices, num_samples)
    random_AF_indices = random.sample(af_indices, num_samples)
    random_both_indices= random_AF_indices + random_SR_indices
    master_indices.append(random_both_indices)


#Pan-Tomkins Algorithm
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

def prediction_accuracy(pred_list,relev_index):
    labeled_list= afib_df['ritmi'].iloc[relev_index]
    totaltrue=sum(labeled_list==pred_list)
    total=len(labeled_list)
    accuracy=totaltrue/total
    return accuracy

predicted_list=[]
T=1/100
accuracy_list=[]
for allrelevindexes in master_indices:
    for i in allrelevindexes: 
        testdata500= data[i,:,1]
        testdata=downsample_signal(testdata500, 500, 200)
        lpftestdata=low_pass_filter(testdata, T)
        hpftestdata=hpf(lpftestdata, T)
        difftestdata=differentiate_signal(hpftestdata, T)
        squaretestdata=square(difftestdata)
        N=30
        integratordata=moving_window_integrator(squaretestdata, N)
        
        peaks=detect_peaks(integratordata)
        Th1, SigLevel, NoiseLevel = initializethreshold(integratordata)
        qrspeaks = findqrs(integratordata, peaks, Th1, SigLevel, NoiseLevel)
        polishedqrs=removeqrsduplicates(qrspeaks)
        

        timeofpeaks=qrspeaks2time(polishedqrs)
   
        intervals=findintervals(timeofpeaks)
      
        mean=np.mean(intervals)
   
        intervaldifference=absintervalsdifference(intervals)
    
        meanintervaldifference=np.mean(intervaldifference)
        MSBID=meanintervaldifference/mean
        if (MSBID>0.11): #threshold selected based on paper: "langley2012.pdf"
            predicted_list.append('AF')
        else:
            predicted_list.append('SR')
        
    accuracy = prediction_accuracy(predicted_list,allrelevindexes)
    accuracy_list.append(accuracy)
    predicted_list=[]

mean_accuracy= sum(accuracy_list)/noof_epochs
print("accuracy of class algo:",mean_accuracy)

    