###audio_feature_generator
import pandas as pd
from pandas import DataFrame
import streamlit as st
import json
import librosa
import librosa.display
import numpy as np
from scipy import signal
from scipy.io import wavfile
import own_functions
from pathlib import Path
import codecs, json

def main():
    st.header("Feature Generator")
    st.write(
        "With the Feature Generator you can extract the values of features of your audio croppings and save them for following analysis."
    )

    metadata_files = st.file_uploader(
        "upload the metadata json files of your audio recordings.",
        type=["json"],
        accept_multiple_files=True,
    )

    sample_rates_from_meta = []
    croppeds_from_meta = []

    open_metadata_files = []

    for metadata_file in metadata_files:
        open_metadata_file = json.load(metadata_file)
        open_metadata_files.append(open_metadata_file)
        sample_rate_from_meta = open_metadata_file["sampling_rate"]
        sample_rates_from_meta.append(sample_rate_from_meta)
        a = "cropped"
        if a in open_metadata_file:
            cropped_from_meta = True
        else:
            cropped_from_meta = False
        croppeds_from_meta.append(cropped_from_meta)
    same_sample_rate = False
    cropped = False

    if len(sample_rates_from_meta) > 0:
        same_sample_rate = all(
            elem == sample_rates_from_meta[0] for elem in sample_rates_from_meta
        )
        cropped = all(elem == croppeds_from_meta[0] for elem in croppeds_from_meta)
    if same_sample_rate and cropped:
        st.write("All files are already cropped and have the same sampling rate.")
    elif same_sample_rate == True and cropped == False:
        st.write("All files have the same sampling rate but havn't been cropped yet.")
    elif same_sample_rate == False and cropped == True:
        st.write(
            "All files have already been cropped but have different sampling rates."
        )
    else:
        st.write(
            "The uploaded files dont have the same sampling rate nor are they cropped yet."
        )
    identify_features = st.button("Extract Features")

    if identify_features:
        features_for_different_testing_objects = {}
        for open_metadata_file in open_metadata_files:
                       
            test_series_path = open_metadata_file["test_series_path"]
            optically_measured_deviation = open_metadata_file["optically_measured_deviation"]
            sampling_rate = open_metadata_file["sampling_rate"]
            n = open_metadata_file["number_measuring_paths"]
            #nperseg = 1024
            nperseg = sampling_rate
            test_series = open_metadata_file["test_series"]
            audio_file_name = open_metadata_file["audio_file"]

            features_in_measuring_paths = {}
            for a in range(0, n):  # starts to iterate through all paths
                features_in_tsx = {}
                for b in range(
                    0, 2
                ):  # using a for loop to differ between the left -> right and right -> left measuring path
                    if b == 0:
                        tsx = "tsl"
                    else:
                        tsx = "tsr"

                    tsxt = f"{tsx}{a+1}"  # combining the information of the measuring path direction with the current iteration
                    current_file_path = Path(f"{open_metadata_file['test_series_path']}/{open_metadata_file['pure_base_name']}_{tsxt}.wav")              
                    y, sr = librosa.load(
                        current_file_path,
                        sampling_rate,
                        mono=True,
                    )
                    hoplength = 256
                    length = (round(len(y)/hoplength+0.49))
                    width_feature = np.zeros ((6, length))
                    width_feature[0] = librosa.feature.rms(y=y, hop_length=hoplength)
                    width_feature[1] = librosa.feature.spectral_centroid(y=y, hop_length=hoplength) 
                    width_feature[2] = librosa.feature.spectral_flatness(y=y, hop_length=hoplength)
                    width_feature[3] = librosa.feature.spectral_rolloff(y=y, hop_length=hoplength)
                    width_feature[4] = librosa.feature.spectral_bandwidth(y=y, hop_length=hoplength)
                    width_feature[5] = librosa.feature.zero_crossing_rate(y=y, hop_length=hoplength)
                    
                  
                    width_features = {
                        "rms_mean" : np.mean(width_feature[0]),
                        "rms_std" : np.std(width_feature[0]),
                    
                        "spectral_centroid_mean" : np.mean(width_feature[1]),
                        "spectral_centroid_std" : np.std(width_feature[1]),
                    
                        "spectral_flatness_mean" : np.mean(width_feature[2]),
                        "spectral_flatness_std" : np.std(width_feature[2]),
                    
                        "spectral_rolloff_mean" : np.mean(width_feature[3]),
                        "spectral_rolloff_std" : np.std(width_feature[3]),
                    
                        "spectral_bandwidth_mean" : np.mean(width_feature[4]),
                        "spectral_bandwidth_std" : np.std(width_feature[4]),

                        "zero_crossing_rate_mean" : np.mean(width_feature[5]),
                        "zero_crossing_rate_std" : np.std(width_feature[5]),
                    }
                    

                    (
                        nr_frequency_ranges,
                        start_frequency_ranges,
                        end_frequency_ranges,
                    ) = own_functions.get_frequency_ranges(
                        sr, 2000
                    )  # opening own function that returns the number of different frequency ranges and two arrays, containing the first and last frequency of each frequency range
                    f_welch, Pxx_den = signal.welch(
                        y,
                        sr,
                        window="hanning",
                        nperseg=nperseg,
                        noverlap=nperseg / 2,
                        return_onesided=True,
                        scaling="density",
                    )
                    # print(f_welch)
                    # print(len(f_welch))
                    total_mean = np.mean(Pxx_den)
                    total_standard_deviation = np.std(Pxx_den)

                    features_in_frequency_range = {}

                    for c in range(0, nr_frequency_ranges):
                        # print(start_frequency_ranges[c])
                        # print(end_frequency_ranges[c])
                        # print(Pxx_den)
                        current_welch_range = Pxx_den[
                            start_frequency_ranges[c] : end_frequency_ranges[c]
                        ]  
                        # print(current_welch_range)
                        # print(len(Pxx_den)) 

                        current_mean = np.mean(current_welch_range)
                        current_standard_deviation = np.std(current_welch_range)
                        current_max = max(current_welch_range)
                        current_min = min(current_welch_range)

                        current_features = {
                            "mean": current_mean.item(),
                            "std": current_standard_deviation.item(),
                            "max": current_max.item(),
                            "min": current_min.item(),
                        }
                        features_in_frequency_range[
                            f"{start_frequency_ranges[c]}:{end_frequency_ranges[c]}"
                        ] = current_features
                    
                    welch_features = {
                        "features_in_frequency_range": features_in_frequency_range,
                        "total_mean" : total_mean.item(),
                        "total_std" : total_standard_deviation.item(),
                    }

                    features_in_tsx[f"{tsx}"] = {
                        "welch_features": welch_features,
                        "width_features": width_features
                    }   
                features_in_measuring_paths[f"{a+1}"] = features_in_tsx
            features_for_different_testing_objects[f"{audio_file_name}"] = {
                "optically_measured_deviation": optically_measured_deviation,
                "features_in_measuring_paths": features_in_measuring_paths
            }

        
            
        test_series_data_file_path= Path(f"{test_series_path}/{test_series}_features")
        # print(features_for_different_testing_objects)   
        # print(type(features_for_different_testing_objects["example_objectnr_1.wav"]["features_in_measuring_paths"]["1"]["tsl"]["width_features"]["rms_mean"])) 
     

        with open (f"{test_series_data_file_path}.json", "w") as json_file:
            json.dump(features_for_different_testing_objects, json_file)
        st.write(
            "Test series data file for " + test_series + "  was saved."
        )
