from numpy.lib.mixins import _reflected_binary_method
import streamlit as st
import librosa  # importiert die Bibliothek librosa für Audio-Anwendung
import os
import soundfile
import librosa.display
import json
from pathlib import Path
import soundfile
import math


def main():

    # loaded_cropping_modes = (
    #     "cropping files that were made by Karl I",
    #     "cropping by reference sound",
    # )   
    #chosen_cropping_mode = st.radio(
    #    "Cropping Mode", loaded_cropping_modes
    # )  # Setting the recording mode decides which cropping mode will be used.

    # if (
    #     chosen_cropping_mode == "cropping files that were made by Karl I"
    # ):  # First case: the files were recorded by Karl I
    st.header("Cropping audio files with timestamps and metadata from Karl I")

    col1, col2, = st.beta_columns((2, 1,))

    metadata_files = col1.file_uploader(
        "upload the metadata json files of your audio recordings.",
        type=["json"],
        accept_multiple_files=True,
    )  # Implementing a drag-and-drop tool in which the metadata Json file have to be uploaded
    col1.write(
        f"Number of uploaded metadata files for audios that need to be cropped: {len(metadata_files)}"
    )

    recording_delay = col1.number_input(
        "Set the delay of the timestamps/recording (positive value when the timestamps are before the right position, negative if the timestamps are after the right position) [ms]",
        -100000,
        100000,
        0,
        1,
    )

    open_metadata_files = []

    for (
        metadata_file
    ) in (
        metadata_files
    ):  # iterates metadata_files through and sets for each value of metadata_files a value in metadata_file +1.
        open_metadata_file = json.load(
            metadata_file
        )  # opens each metadata_file to read the Json
        open_metadata_files.append(open_metadata_file)
        audio_file_path = Path(
            open_metadata_file["audio_file_path"]
        )  # extracting the path of the belonging audio file that shall be cropped
        timestamp_file_path = Path(
            open_metadata_file["timestamp_file_path"]
        )  # extracting the path of the belonging timestamp file that gives information about the cutting points
        st.write("audio file path:", audio_file_path)
        st.write(
            "audio file exists:", os.path.isfile(audio_file_path)
        )  # checks if the audio file exists
        st.write("timestamp file path:", timestamp_file_path)
        st.write(
            "timestamp file exists:", os.path.isfile(timestamp_file_path)
        )  # checks if the timestamp file exists
        st.write("recorded time:", open_metadata_file["recording_time"])

        if os.path.isfile(audio_file_path) == True:  # if the audio file exists
            col1, col2, = st.beta_columns((2, 1,))
            audio_open = open(audio_file_path, "rb")
            audio_bytes = audio_open.read()
            col1.audio(audio_bytes)  # it will be shown on a player in the browser
        else:
            None
        # if (
        #     os.path.isfile(audio_file_path) == True
        #     and os.path.isfile(timestamp_file_path) == True
        # ):  # if both, audio and timestamp file exist, all needs for cutting are given.
        #     all_prepared_for_cropping = True
        # else:
        #     all_prepared_for_cropping = False

        
    cropping_activate = st.button("Crop audio files")
  
    if cropping_activate == True:
        for (
            open_metadata_file
        ) in (
            open_metadata_files
        ):  # iteriert metadata_files durch und setzt für jeden Wert von metadata_files einen Wert in metadata_file +1.

            audio_file_path = open_metadata_file["audio_file_path"].replace("\\", "/")
                
            timestamp_file_path = open_metadata_file["timestamp_file_path"].replace("\\", "/")
            sampling_rate = open_metadata_file["sampling_rate"]
            with open(timestamp_file_path) as f:
                open_timestamp_file = json.load(f)

            n = open_timestamp_file["turns"]  # number of measuring paths
            recording_time_per_path = open_metadata_file["time_per_path"]
            # if cropping_length_mode == "manually":
            #     cropping_length = manual_cropping_length
            # else:
            cropping_length = math.ceil(recording_time_per_path*1000)

            if (
                os.path.isfile(audio_file_path) == True
                and os.path.isfile(timestamp_file_path) == True
            ):

                y, sr = librosa.load(audio_file_path, sampling_rate, mono=True)

                for a in range(
                    0, n
                ):  # iterates a through with values until including n-1
                    tsl_start_second = (open_timestamp_file[f"TSL{a+1}"] + recording_delay) / 1000 
                    tsr_start_second = (open_timestamp_file[f"TSR{a+1}"] + recording_delay) / 1000
                    tsl_start_sample = tsl_start_second * sampling_rate
                    tsr_start_sample = tsr_start_second * sampling_rate
                    tsl_end_sample = (
                        tsl_start_sample + (cropping_length/1000) * sampling_rate
                    )
                    tsr_end_sample = (
                        tsr_start_sample + (cropping_length/1000) * sampling_rate
                    )

                        # export_path_tsl = f"{open_metadata_file['path']}/{open_metadata_file['file']}_tsl{a+1}.wav"
                        # export_path_tsr = f"{open_metadata_file['path']}/{open_metadata_file['file']}_tsr{a+1}.wav"

                    # export_path_tsl = f"{open_metadata_file['test_series_path']}/{os.path.basename(open_metadata_file['audio_file_path'])}_tsl{a+1}.wav"
                    # export_path_tsr = f"{open_metadata_file['test_series_path']}/{os.path.basename(open_metadata_file['audio_file_path'])}_tsr{a+1}.wav"

                    export_path_tsl = f"{open_metadata_file['test_series_path']}/{open_metadata_file['pure_base_name']}_tsl{a+1}.wav"
                    export_path_tsr = f"{open_metadata_file['test_series_path']}/{open_metadata_file['pure_base_name']}_tsr{a+1}.wav"


                    audio_tsl = y[
                        int(round(tsl_start_sample)) : int(round(tsl_end_sample))
                    ]
                    audio_tsr = y[
                        int(round(tsr_start_sample)) : int(round(tsr_end_sample))
                    ]

                    soundfile.write(export_path_tsl, audio_tsl, sr, format="WAV")
                    soundfile.write(export_path_tsr, audio_tsr, sr, format="WAV")

                    st.write(
                        f"Cropping file {open_metadata_file['pure_base_name']} was successful."
                    )

                cropped = {
                    "cropped": True
                }  # additional meta data that will be written in the already existing metadata file. It informs, that the raw audio has successfully been cropped

                with open(
                    open_metadata_file["meta_file_path"], "r+"
                ) as file:  # https://www.kite.com/python/answers/how-to-append-to-a-json-file-in-python#:~:text=Use%20open(filename%2C%20mode),to%20append%20a_dict%20to%20dict%20.
                    data = json.load(file)
                    data.update(cropped)
                    file.seek(0)
                    json.dump(data, file)

            else:
                st.write(
                    f"Cropping file {open_metadata_file['file']} was NOT successful."
                )

 