import streamlit as st
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
import librosa
import librosa.display
import scipy, pylab



def main():

    st.header("Audio File Plotter")
    st.write("Upload your audiofiles and create spectrograms and Welch diagrams.")
    audio_files = st.file_uploader(
        "Upload Audios", type=["wav", "mp3"], accept_multiple_files=True
    )
    # st.button("Clear Cache")
    st.write("number of uploaded audio files:", len(audio_files))
    plot = st.radio(
        "Do you want to plot the functions for each uploaded audio file?", ("Yes", "No")
    )
    loaded_npersegs = ("512", "1024", "2048", "4096")
    # npersegsspectro = int(
    #     st.selectbox(
    #         "Length of each segment (nperseg) for spectrogram", loaded_npersegs
    #     )
    # )
    #npersegswelch = int(
    #    st.selectbox(
    #         "Length of each segment (nperseg) for Welch diagram", loaded_npersegs
    #     )
    # )

    if audio_files:  # gibt True an, sobald Einträge in audio_files enthalten sind

        # Create plot.
        common_welch = plt.figure(figsize=(15, 8))
        plt.grid(True)
        plt.xscale("log")
        plt.xlabel("frequency [Hz]")
        plt.ylabel("PSD [V**2/Hz]")
        common_welch_container = (
            st.empty()
        )

        common_rms = plt.figure(figsize=(10, 5))
        
        plt.xlabel('Window / Time')
        plt.ylabel('RMS Energy')
        plt.yscale('linear')
        common_rms_container = (
            st.empty()
        )
        
        
        
        
          # sichert hier den Platz für das Diagram, welches erst mit den im Folgenden gelesenen Daten befüllt wird.

        # while i>=0 and i<number_of_elements:
        for (
            audio_file
        ) in (
            audio_files
        ):  # iteriert audio_files durch und setzt für jeden Wert von audio_files einen Wert in audio_file +1.

            expander = st.beta_expander(
                f"Click here to show plots for {audio_file.name}"
            )

            with expander:

                st.audio(audio_file, format="audio/wav")
                y, sr = librosa.load(audio_file, sr=None)
                st.write("native sample rate of the file [Hz]: ", sr)
                st.write(f"length of the audio file [s]: {(len(y)-1)/sr}")
                t = np.array(range(0, len(y))) / sr

                if plot == "Yes":
                    st.write("Sound pressure level:")
                    soundpressure = plt.figure(figsize=(15, 5))
                    plt.title(audio_file.name)
                    plt.plot(t, y)
                    plt.xlabel("time in seconds")
                    plt.ylabel("amplitude")
                    st.pyplot(soundpressure)

                    st.write("RMS")
                    S, phase = librosa.magphase(librosa.stft(y))
                    rms_x = librosa.feature.rms(y=y, frame_length=256, hop_length=512)
                    rms = plt.figure(figsize=(10, 5))
                    plt.semilogy(rms_x.T, label= f'{audio_file.name}')
                    plt.xlabel('Window / Time')
                    plt.ylabel('RMS Energy')
                    plt.yscale('linear')
                    st.pyplot(rms)

                    plt.figure(common_rms.number)
                    plt.semilogy(rms_x.T, label= f'{audio_file.name}')


                    st.write("Spectrogram:")
                    # width=st.slider('spectogram width', 5, 25, 15)
                    # height[i]=st.slider('spectogran height', 5, 25, 12)
                    spectrogram = plt.figure(figsize=(10, 10))
                    #noverlaps = npersegsspectro / 2
                    noverlaps = 256
                    plt.specgram(
                        y, Fs=sr, scale="dB", NFFT=512, noverlap=noverlaps
                    )
                    plt.ylim(1, 20000)
                    plt.yscale("log")
                    #plt.colorbar(format="%+2.0f dB")
                    plt.title("Spectrogram 500Hz")
                    plt.xlabel("Time")
                    plt.ylabel("Frequency")
                    st.pyplot(spectrogram)
                   

                    #
                    st.write("Welch Diagram:")
                    welch = plt.figure(figsize=(15, 5))
                    f, Pxx_den = signal.welch(
                        y,
                        sr,
                        nperseg=2048,
                        return_onesided=True,
                        scaling="density",
                    )
                    plt.semilogy(f, Pxx_den)
                    plt.grid(True)
                    plt.xscale("log")
                    plt.xlabel("frequency [Hz]")
                    plt.ylabel("PSD [V**2/Hz]")
                    st.pyplot(welch)

                    # Plot into common plot.
                    plt.figure(common_welch.number)
                    plt.semilogy(f, Pxx_den, label=audio_file.name)

        plt.figure(common_welch.number)
        plt.legend()
        common_welch_container.pyplot(common_welch)

        plt.figure(common_rms.number)
        plt.legend()
        common_rms_container.pyplot(common_rms)
