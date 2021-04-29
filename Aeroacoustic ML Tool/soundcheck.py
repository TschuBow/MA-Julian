from numpy.lib.function_base import hanning
import streamlit as st
import pyaudio
import librosa
import librosa.display
import json
import own_functions
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import shutil
import os
import numpy as np
from scipy.fftpack import fft


testpath = Path("audio-files")
testpathfile = testpath / "500Hz.wav" 

#sample_rate, samples = wavfile.read(testpathfile)                        # opening the temporary audio file that was just recorded before
sample_amplitudes, sample_rate = librosa.load(testpathfile, sr=44100, mono=True)

frequencies, times, spectrogram = signal.spectrogram(sample_amplitudes, 
sample_rate, 
window='hanning',      #str or tuple or array_like, optional. Desired window to use. If window is a string or tuple, it is passed to get_window to generate the window values, which are DFT-even by default. See get_window for a list of windows and required parameters. If window is array_like it will be used directly as the window and its length must be nperseg. Defaults to a Hann window.
nperseg=4096,   #Length of each segment. Defaults to None, but if window is str or tuple, is set to 256, and if window is array_like, is set to the length of the window.
noverlap=2048)   #Number of points to overlap between segments. If None, noverlap = nperseg // 2. Defaults to None.
                                                                                                # window = ('hanning')
st.write("Spectrogram")
temporary_spectrogram_fig=plt.figure()
plt.pcolormesh(times, frequencies, spectrogram)                                 # creating the spectogram from the arrays of times, frequencies and spectrogram where amplitudes are expressed as colours
plt.ylim(1, 10000)                                                              # setting the scale of the frequencies axis from 1 to 20000Hz
plt.yscale("log")                                                               # logarithmizing the frequencies axis of the spectrogram
plt.ylabel('Frequency [Hz]')                                                    # labeling the axes
plt.xlabel('Time [sec]')
#plt.z
st.pyplot(temporary_spectrogram_fig)   

st.write("Welchdiagram")
frequency, Pxx_den = signal.welch(
sample_amplitudes, 
fs=sample_rate, 
window='hanning', 
nperseg=1024, 
noverlap=512, 
return_onesided=True)



sample_amplitudes = sample_amplitudes[sample_rate:(sample_rate+round(0.02*sample_rate))] #Kürzen auf 0,02 Sekunden

st.write(sample_rate)

plt.figure
plt.plot()
plt.figure(figsize=(10,4))


librosa.display.waveplot(sample_amplitudes, sr=sample_rate)
plt.ylim((-3,3))
plt.xlabel('Zeit in s')
plt.ylabel('Amplitude')
st.pyplot()







# st.write("Spectrogram")
# temporary_spectrogram_fig=plt.figure()
# plt.pcolormesh(times, frequencies, spectrogram)                                 # creating the spectogram from the arrays of times, frequencies and spectrogram where amplitudes are expressed as colours
# plt.ylim(1, 20000)                                                              # setting the scale of the frequencies axis from 1 to 20000Hz
# plt.yscale("log")                                                               # logarithmizing the frequencies axis of the spectrogram
# plt.ylabel('Frequency [Hz]')                                                    # labeling the axes
# plt.xlabel('Time [sec]')
# st.pyplot(temporary_spectrogram_fig)     







# frequencies, times, spectrogram = signal.spectrogram(sample_amplitudes, sample_rate)      # extracting frequencies, times and amplitdues from the wav file
        
# st.write("Spectrogram")
# temporary_spectrogram_fig=plt.figure()
# plt.pcolormesh(times, frequencies, spectrogram)                                 # creating the spectogram from the arrays of times, frequencies and spectrogram where amplitudes are expressed as colours
# plt.ylim(1, 20000)                                                              # setting the scale of the frequencies axis from 1 to 20000Hz
# plt.yscale("log")                                                               # logarithmizing the frequencies axis of the spectrogram
# plt.ylabel('Frequency [Hz]')                                                    # labeling the axes
# plt.xlabel('Time [sec]')
# st.pyplot(temporary_spectrogram_fig)                                            # visualization of the spectrogram on Streamlit
        

# st.write("Welch-Diagram")
# sample_fr, powerspectraldensity = signal.welch(sample_amplitudes, sample_rate)
# temporary_welch=plt.figure()
# plt.semilogy(sample_fr, np.sqrt(powerspectraldensity))
# plt.xlim(1,20000)
# plt.xscale("log")
# plt.xlabel('frequency [Hz]')
# plt.ylabel('Linear Spectrum')
# plt.grid()
# st.pyplot(temporary_welch)
 

# sampleplot=plt.figure()
# print(sample_amplitudes.size)
# plt.plot(np.arange(100), sample_amplitudes[:100])
# st.pyplot(sampleplot)

# st.write(sample_amplitudes)

# fs = 10e3
# N = 1e5
# amp = 2*np.sqrt(2)
# freq = 1234.0
# noise_power = 0.001 * fs / 2
# time = np.arange(N) / fs
# x = amp*np.sin(2*np.pi*freq*time)
# x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)

# st.write(x)


# st.write("Welch-Diagram")
# sample_fr, powerspectraldensity = signal.welch(x, fs)
# temporary_welch=plt.figure()
# plt.semilogy(sample_fr, powerspectraldensity)
# plt.xlim(1,20000)
# plt.xscale("log")
# plt.xlabel('frequency [Hz]')
# plt.ylabel('Linear Spectrum')
# plt.grid()
# st.pyplot(temporary_welch)'