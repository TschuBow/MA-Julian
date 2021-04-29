from PIL import Image
import streamlit as st
import json
import own_functions
import pyaudio
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import shutil
import os
import numpy as np
import librosa
import librosa.display

import serial #pip install pyserial, NOT serial!!!
import time
import multiprocessing


# function main() that is run by Main.py:
def main():
    # selectable metadata regarding the testobject and testing hardware:
    loaded_objects = ("Sandvik CNMG 12 04 08-MM 2025", "other")
    loaded_machine_tools = ("DOOSAN DNM 500", "other")
    loaded_nozzles = ("Schlick Modellreihe 629", "Schlick Modellreihe 650", "FOS-45°-Flachstrahldüse 25mm Länge", "FOS-Flachstrahldüse 35mm Länge", "FOS-Flachstrahldüse 60mm Länge",
                      "Jet-Düse MV6, Innendurchm. 6mm", "Jet-Düse MV12, Innendurchm. 6mm", "Düse 1200SSF 1/8 verstellbar, Edelstahl", "Silvent MJ4", "Silvent MJ5", "")
    loaded_microphones = ("Renkforce UM-80 USB-Mikrofon", "other")
    loaded_recording_modes = ("Recording by time", "Recording on Karl I")

    # selectable audio recording settings:
    loaded_sampling_rates = ("44100", "48000", "88200", "96000")
    loaded_sample_formats = ("16 bit", "24 bit")
    loaded_chunks = ("1024", "2048")
    #loaded_channels = ("1", "2")

    # header and subheader of the page:
    st.header("Metadata Capture of Audio Files")
    st.write(
        "Name your audio file you want to record and select or type in the right parameters")
    # select metadata regarding the testobject and testing hardware:
    audio_file_name = st.text_input("File name", "example")
    test_object = st.selectbox("Test object", loaded_objects)
    test_object_number = str(st.text_input("Testobjekt-Nummer", "1"))
    test_machine_tool = st.selectbox("Machine tool", loaded_machine_tools)
    test_nozzle = st.selectbox("Nozzle", loaded_nozzles)
    test_microphone = st.selectbox("Microphone", loaded_microphones)
    test_deviation = st.number_input(
        'Insert the measured deviation of the test object [µm]', step=0.01)
    #test_recording_mode = st.selectbox("Recording mode", loaded_recording_modes)
    test_recording_mode = st.radio("Recording mode", loaded_recording_modes)

    if test_recording_mode == "Recording by time":
        st.header("Manual setting of the recording time")
        test_required_recording_time = st.number_input(
            'Insert the recording time [s] (between 0 and 240 seconds)', step=0.01, min_value=0.01, max_value=240.00)
        mode_path_name = "0_recording_by_time"
    elif test_recording_mode == "Recording on Karl I":
        st.header("Recording with Karl I")
        st.write('the test rig for determining the wear of indexable inserts')
        Infoimage = Image.open('images\info_cutting_parameters.png')
        st.image(Infoimage, caption=None, width=600, height=600)
        test_feed_speed = st.number_input(
            'Feed speed [mm/s]', step=0.01, min_value=0.01, max_value=10000.00)
        test_feed_acceleration = st.number_input(
            'Acceleration [mm/s²]', step=0.01, min_value=0.01, max_value=2000.00)
        test_measuring_path = st.number_input(
            'Measuring path (a) [mm]', step=0.01, min_value=0.01, max_value=1000.00)
        test_offset = st.number_input(
            'Offset (b) [mm]', min_value=0.01, max_value=1000.00)
        test_number_of_measuring_paths = st.slider(
            'Number of measuring paths (n)', 0, 50, 0, 1)
        st.write(test_number_of_measuring_paths,
                 'Measuring paths per audiofile')
        n = test_number_of_measuring_paths
        ss_p = test_measuring_path  # Messstrecke
        ss_o = test_offset  # Versatz/Abstand zwischen den Messstrecken
        vs = test_feed_speed  # eingestellte Vorschubgeschwindigkeit
        acs = test_feed_acceleration  # eingestellte Beschleunigung
        ta = vs/acs  # benötigte Zeit, bis mit gegebener Beschleunigung die Vorschubgeschwindigkeit erreicht ist
        # Strecke, die während Beschleunigungsphase auf Vorschubgeschwindigkeit zurückgelegt wird.
        sa = 0.5*acs*(ta**2)
        # Bei den Versatzwegen wird die Sollvorschubgeschwindigkeit in der Regel nicht erreicht, daher dieser Ausnahmefall:
        if sa <= (0.5*ss_o):
            tv_o = 0  # Keine Zeit beim Versatz wird dann mit der Sollvorschubgeschwindigkeit zurückgelegt
            # Die Zeit, die für das Zurücklegen des halben Versatzes mit der Beschleunigung zurückgelegt wird
            ta_o = (ss_o/acs)**0.5
            ts_o = 2*ta_o  # 2x, jeweils einmal für Beschleunigung und einmal für Bremsen
        else:
            sv_o = ss_o-(2*sa)
            tv_o = sv_o/vs
            ts_o = tv_o+(2*ta)

        sv_p = ss_p-(2*sa)
        tv_p = sv_p/vs
        ts_p = tv_p+(2*ta)

        ttotal = n*ts_p+(n-1)*ts_o
        if ttotal <= 0:
            ttotal = 0
        ttolerance_fix = st.number_input(
            'Specify a one-time tolerance time value. [s]')
        ttolerance_per_path = st.number_input(
            'Specify a tolerance value per path. [s]', step=0.01)
        ttotal_with_tolerance = ttotal + ttolerance_fix + n*ttolerance_per_path

        st.write('The calculated recording time including tolerances counts',
                 ttotal_with_tolerance, 'seconds')
        test_required_recording_time = ttotal_with_tolerance
        mode_path_name = "1_recording_on_karl_I"

    # select audio recording settings:

    audio_file_name = f"{audio_file_name}_objectnr_{test_object_number}.wav"
    test_sampling_rate = int(st.selectbox(
        'Sampling rate of the recording in kHz', loaded_sampling_rates))
    selected_test_sample_format = st.selectbox(
        'Sample format of the recording', loaded_sample_formats)
    test_chunk = int(st.selectbox('Buffer in Bytes', loaded_chunks))
    #test_channels = int(st.selectbox('Number of channels', loaded_channels))

    # translating the input of the selectbox for the sample format into they right variable for the recording algorithm:
    if selected_test_sample_format == "8 bit":
        test_sample_format = pyaudio.paInt8
    elif selected_test_sample_format == "16 bit":
        test_sample_format = pyaudio.paInt16
    elif selected_test_sample_format == "24 bit":
        test_sample_format = pyaudio.paInt24
    elif selected_test_sample_format == "32 bit":
        test_sample_format = pyaudio.paInt32
    else:
        test_sample_format = None

    # reminder to check if all settings are done correctly and button to start the recording:

    #loaded_npersegs = ("512", "1024", "2048", "4096")
    #npersegs=int(st.selectbox('Length of each segment (nperseg)', loaded_npersegs))
    npersegs = 4096
    noverlaps = npersegs/2

    st.write('Check if everything is filled in correctly and save your metadata by pressing the "Save" button below.')

    start_recording = st.button("Start recording")

    #
    if start_recording == True:
        if test_recording_mode == "Recording by time":
            # determining the path for temporary files
            temp = Path(f"audio-files/{mode_path_name}/0_temporary_files")
            # creating the path if it doesn't exist yet
            temp.mkdir(parents=True, exist_ok=True)
            # path and name for the temporary recording file
            temporary_file_path = temp / "temporary_file.wav"
        elif test_recording_mode == "Recording on Karl I":
            # determining the path for temporary files
            temp = Path(f"audio-files/{mode_path_name}/0_temporary_files")
            # creating the path if it doesn't exist yet
            temp.mkdir(parents=True, exist_ok=True)
            temporary_file_path = temp / "temporary_file.wav"
            triggerpath = Path("x")
            triggerpath.mkdir(parents=True, exist_ok=True)
            # Trigger file for starting G-Code on the testing rig
            triggerpathfile = triggerpath / "trigger"
            with open(f'{str(triggerpathfile)}.json', 'w') as json_file:
                json.dump("trigger", json_file)

        own_functions.record(test_sample_format, 1, test_sampling_rate, test_chunk, test_required_recording_time, str(
            temporary_file_path))  # recording function that is defined in Aufnahme.py

        # sample_rate, samples = wavfile.read(temporary_file_path)                        # opening the temporary audio file that was just recorded before
        samples, sample_rate = librosa.load(
            temporary_file_path, sr=test_sampling_rate)

        st.write('sample rate:', sample_rate)

        # extracting frequencies, times and amplitdues from the wav file
        frequencies, times, spectrogram = signal.spectrogram(
            samples, sample_rate, nperseg=npersegs, noverlap=noverlaps)

        st.write("Spectrogram logarithmic scaling")
        temporary_spectrogram_fig = plt.figure()
        # creating the spectogram from the arrays of times, frequencies and spectrogram where amplitudes are expressed as colours
        plt.pcolormesh(times, frequencies, spectrogram)
        # setting the scale of the frequencies axis from 1 to 20000Hz
        plt.ylim(1, (sample_rate/2))
        # logarithmizing the frequencies axis of the spectrogram
        plt.yscale("log")
        # labeling the axes
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        # plt.z
        # visualization of the spectrogram on Streamlit
        st.pyplot(temporary_spectrogram_fig)

        if test_recording_mode == "Recording by time":
            # opening the last recorded wav file in a player on Streamlit
            st.audio(
                f"audio-files/{mode_path_name}/0_temporary_files/temporary_file.wav", format="audio/wav")
        elif test_recording_mode == "Recording on Karl I":
            st.audio(
                f"audio-files/{mode_path_name}/0_temporary_files/temporary_file.wav", format="audio/wav")

    savefile = st.button("Save")
    #temp = Path(f"audio-files/{mode_path_name}/0_temporary_files")
    #temporary_file_path = temp / "temporary_file.wav"
    if savefile == True:

        # 0_recording_by_time
        old_temp_path = Path(f"audio-files/{mode_path_name}/0_temporary_files")
        old_temp_path_file = old_temp_path / "temporary_file.wav"
        if os.path.exists(old_temp_path_file):

            new_path = Path(
                f"audio-files/{mode_path_name}/{test_object}/{test_nozzle}")
            new_path.mkdir(parents=True, exist_ok=True)
            new_path_file = new_path/audio_file_name
            new_path_file = own_functions.uniquify(new_path_file)

            shutil.copyfile(old_temp_path_file, new_path_file)

            os.remove(old_temp_path_file)

            if test_recording_mode == "Recording by time":
                audio_file_meta = {"file": audio_file_name,
                                   "microphone": test_microphone,
                                   "sampling rate": test_sampling_rate,
                                   "recording mode": test_recording_mode,
                                   "object": test_object,
                                   "nozzle": test_nozzle,
                                   "machine tool": test_machine_tool,
                                   "optically measured deviation": test_deviation,
                                   "recording time": test_required_recording_time
                                   }
            elif test_recording_mode == "Recording on Karl I":
                audio_file_meta = {"file": audio_file_name,
                                   "microphone": test_microphone,
                                   "sampling rate": test_sampling_rate,
                                   "recording mode": test_recording_mode,
                                   "paths": test_number_of_measuring_paths,
                                   "measuring path": test_number_of_measuring_paths,
                                   "offset": test_offset,
                                   "feed speed": test_feed_speed,
                                   "feed acceleration": test_feed_acceleration,
                                   "object": test_object,
                                   "nozzle": test_nozzle,
                                   "machine tool": test_machine_tool,
                                   "optically measured deviation": test_deviation,
                                   "recording time": test_required_recording_time
                                   }
            with open(f'{str(new_path_file)}.json', 'w') as json_file:
                json.dump(audio_file_meta, json_file)
            st.write("Audio file and metadata file for " +
                     audio_file_name + "  were saved.")
        else:
            st.write("Record a new audio file first.")

    deletefile = st.button("Delete")

    if deletefile == True:

        old_temp_path = Path(f"audio-files/{mode_path_name}/0_temporary_files")
        old_temp_path_file = old_temp_path / "temporary_file.wav"
        if os.path.exists(old_temp_path_file):
            os.remove(old_temp_path_file)
            st.write("Your recording was deleted")
        else:
            st.write("There is no recording yet that could be deleted.")
