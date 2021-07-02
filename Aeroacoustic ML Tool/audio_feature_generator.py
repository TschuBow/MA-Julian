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
from scipy import stats

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

    frequency_ranges_for_welch = st.radio("Features of Frequency Range Windows of Welch required?", ["yes", "no"])
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
                       
            test_series_path = open_metadata_file["test_series_path"].replace("\\", "/")
            optically_measured_deviation = open_metadata_file["optically_measured_deviation"]
            sampling_rate = open_metadata_file["sampling_rate"]
            n = open_metadata_file["number_measuring_paths"]
            #nperseg = 1024
            nperseg = sampling_rate
            test_series = open_metadata_file["test_series"].replace("\\", "/")
            audio_file_name = open_metadata_file["audio_file"].replace("\\", "/")

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




                    test_series_path_apple_correct = open_metadata_file["test_series_path"].replace("\\", "/")

                    pure_base_name_apple_correct = open_metadata_file["pure_base_name"].replace("\\", "/")



                    current_file_path = Path(f'{ test_series_path_apple_correct}/{pure_base_name_apple_correct}_{tsxt}.wav')            

                    #current_file_path_apple_corrected = current_file_path.replace("\\", "/")



                    y, sr = librosa.load(

                        current_file_path,

                        sampling_rate,
        
                        mono=True,
                    )
                    hoplength = 256
                    length = (round(len(y)/hoplength+0.49))

                    width_feature = list()
                    #width_feature = np.zeros(1)
                     
                    fr, Pxx = signal.welch(x=y, nperseg=hoplength)
                    width_feature.append(Pxx)

                    S1, phase = librosa.magphase(librosa.stft(y))
                    rms1 = librosa.feature.rms(S=S1, frame_length=2048, hop_length=hoplength)
                    
                    savgol_rms= signal.savgol_filter(rms1, 51, 3)
                    width_feature.append(savgol_rms)
                    

                    S1, phase = librosa.magphase(librosa.stft(y))
                    rms1 = librosa.feature.rms(S=S1, frame_length=2048, hop_length=hoplength)
                    yhat = signal.savgol_filter(rms1, 51, 3)
                    dif= np.diff(yhat)
                    
                    savgol_diff= signal.savgol_filter(dif, 51, 3)
                    width_feature.append(savgol_diff)

                    mfcc = librosa.feature.mfcc(y)
                    width_feature.append(mfcc)

                    poly = librosa.feature.poly_features(y)
                    width_feature.append(poly)

                    centroid = librosa.feature.spectral_centroid(y=y, hop_length=hoplength)
                    width_feature.append(centroid)

                    bandwidth = librosa.feature.spectral_bandwidth(y=y, hop_length=hoplength)
                    width_feature.append(bandwidth)

                    contrast = librosa.feature.spectral_contrast(y=y, hop_length=hoplength)
                    width_feature.append(contrast)

                    flatness = librosa.feature.spectral_flatness(y=y, hop_length=hoplength)
                    width_feature.append(flatness)

                    rolloff = librosa.feature.spectral_rolloff(y=y, hop_length=hoplength)
                    width_feature.append(rolloff)


                    

            

                    
                  
                    width_features = {

                        "freq_mean" : float(np.mean(width_feature[0])), #Arithmetisches Mittel
                        "freq_std" : float(np.std(width_feature[0])), #Standardabweichung
                        "freq_maxv" : float(np.amax(width_feature[0])), #Maximum Wert
                        "freq_maxs" : float(np.argmax(width_feature[0])), #Maximum Stelle
                        "freq_minv" : float(np.amin(width_feature[0])), #Minimum Wert
                        "freq_mins" : float(np.argmin(width_feature[0])), #Minimum Stelle
                        "freq_median" : float(np.median(width_feature[0])), #Median
                        "freq_skew" : float(np.mean(stats.skew(width_feature[0]))), #Schiefe
                        "freq_kurt" : float(np.mean(stats.kurtosis(width_feature[0]))), #Wölbung
                        "freq_q1" : float(np.quantile(width_feature[0], 0.25)), #Quantile1
                        "freq_q3" : float(np.quantile(width_feature[0], 0.75)), #Quantile2
                        "freq_mode" : float(np.mean(stats.mode(width_feature[0])[0][0])), #Modal
                        "freq_iqr" : float(stats.iqr(width_feature[0])), #Interquartilsabstand
                        "freq_zerocross" : float(np.sum(librosa.zero_crossings(width_feature[0], pad=False))), #Zero-Crossings

                        "rms_mean" : float(np.mean(width_feature[1])), #Arithmetisches Mittel
                        "rms_std" : float(np.std(width_feature[1])), #Standardabweichung
                        "rms_maxv" : float(np.amax(width_feature[1])), #Maximum Wert
                        "rms_maxs" : float(np.argmax(width_feature[1])), #Maximum Stelle
                        "rms_minv" : float(np.amin(width_feature[1])), #Minimum Wert
                        "rms_mins" : float(np.argmin(width_feature[1])), #Minimum Stelle
                        "rms_median" : float(np.median(width_feature[1])), #Median
                        "rms_skew" : float(np.mean(stats.skew(width_feature[1]))), #Schiefe
                        "rms_kurt" : float(np.mean(stats.kurtosis(width_feature[1]))), #Wölbung
                        "rms_q1" : float(np.quantile(width_feature[1], 0.25)), #Quantile1
                        "rms_q3" : float(np.quantile(width_feature[1], 0.75)), #Quantile2
                        "rms_mode" : float(np.mean(stats.mode(width_feature[1])[0][0])), #Modal
                        "rms_iqr" : float(stats.iqr(width_feature[1])), #Interquartilsabstand
                        "rms_zerocross" : float(np.sum(librosa.zero_crossings(width_feature[1], pad=False))), #Zero-Crossings

                        "ableitung_std" : float(np.std(width_feature[2])), #Standardabweichung
                        "ableitung_maxv" : float(np.amax(width_feature[2])), #Maximum Wert
                        "ableitung_maxs" : float(np.argmax(width_feature[2])), #Maximum Stelle
                        "ableitung_minv" : float(np.amin(width_feature[2])), #Minimum Wert
                        "ableitung_mins" : float(np.argmin(width_feature[2])), #Minimum Stelle
                        "ableitung_median" : float(np.median(width_feature[2])), #Median
                        "ableitung_skew" : float(np.mean(stats.skew(width_feature[2]))), #Schiefe
                        "ableitung_kurt" : float(np.mean(stats.kurtosis(width_feature[2]))), #Wölbung
                        "ableitung_q1" : float(np.quantile(width_feature[2], 0.25)), #Quantile1
                        "ableitung_q3" : float(np.quantile(width_feature[2], 0.75)), #Quantile2
                        "ableitung_mode" : float(np.mean(stats.mode(width_feature[2])[0][0])), #Modal
                        "ableitung_iqr" : float(stats.iqr(width_feature[2])), #Interquartilsabstand
                        "ableitung_zerocross" : float(np.sum(librosa.zero_crossings(width_feature[2], pad=False))), #Zero-Crossings

                        "mfcc_mean" : float(np.mean(width_feature[3])), #Arithmetisches Mittel
                        "mfcc_std" : float(np.std(width_feature[3])), #Standardabweichung
                        "mfcc_maxv" : float(np.amax(width_feature[3])), #Maximum Wert
                        "mfcc_maxs" : float(np.argmax(width_feature[3])), #Maximum Stelle
                        "mfcc_minv" : float(np.amin(width_feature[3])), #Minimum Wert
                        "mfcc_mins" : float(np.argmin(width_feature[3])), #Minimum Stelle
                        "mfcc_median" : float(np.median(width_feature[3])), #Median
                        "mfcc_skew" :float(np.mean(stats.skew(width_feature[3]))), #Schiefe
                        "mfcc_kurt" : float(np.mean(stats.kurtosis(width_feature[3]))), #Wölbung
                        "mfcc_q1" : float(np.quantile(width_feature[3], 0.25)), #Quantile1
                        "mfcc_q3" : float(np.quantile(width_feature[3], 0.75)), #Quantile2
                        "mfcc_mode" : float(np.mean(stats.mode(width_feature[3])[0][0])), #Modal
                        "mfcc_iqr" : float(stats.iqr(width_feature[3])), #Interquartilsabstand
                        "mfcc_zerocross" : float(np.sum(librosa.zero_crossings(width_feature[3], pad=False))), #Zero-Crossings

                        "poly_mean" : float(np.mean(width_feature[4])), #Arithmetisches Mittel
                        "poly_std" : float(np.std(width_feature[4])), #Standardabweichung
                        "poly_maxv" : float(np.amax(width_feature[4])), #Maximum Wert
                        "poly_maxs" : float(np.argmax(width_feature[4])), #Maximum Stelle
                        "poly_minv" : float(np.amin(width_feature[4])), #Minimum Wert
                        "poly_mins" : float(np.argmin(width_feature[4])), #Minimum Stelle
                        "poly_median" : float(np.median(width_feature[4])), #Median
                        "poly_skew" : float(np.mean(stats.skew(width_feature[4]))), #Schiefe
                        "poly_kurt" : float(np.mean(stats.kurtosis(width_feature[4]))), #Wölbung
                        "poly_q1" : float(np.quantile(width_feature[4], 0.25)), #Quantile1
                        "poly_q3" : float(np.quantile(width_feature[4], 0.75)), #Quantile2
                        "poly_mode" : float(np.mean(stats.mode(width_feature[4])[0][0])), #Modal
                        "poly_iqr" : float(stats.iqr(width_feature[4])), #Interquartilsabstand
                        "poly_zerocross" : float(np.sum(librosa.zero_crossings(width_feature[4], pad=False))), #Zero-Crossings

                        "spec_centroid_mean" : float(np.mean(width_feature[5])), #Arithmetisches Mittel
                        "spec_centroid_std" : float(np.std(width_feature[5])), #Standardabweichung
                        "spec_centroid_maxv" : float(np.amax(width_feature[5])), #Maximum Wert
                        "spec_centroid_maxs" : float(np.argmax(width_feature[5])), #Maximum Stelle
                        "spec_centroid_minv" : float(np.amin(width_feature[5])), #Minimum Wert
                        "spec_centroid_mins" : float(np.argmin(width_feature[5])), #Minimum Stelle
                        "spec_centroid_median" : float(np.median(width_feature[5])), #Median
                        "spec_centroid_skew" : float(np.mean(stats.skew(width_feature[5]))), #Schiefe
                        "spec_centroid_kurt" : float(np.mean(stats.kurtosis(width_feature[5]))), #Wölbung
                        "spec_centroid_q1" : float(np.quantile(width_feature[5], 0.25)), #Quantile1
                        "spec_centroid_q3" : float(np.quantile(width_feature[5], 0.75)), #Quantile2
                        "spec_centroid_mode" : float(np.mean(stats.mode(width_feature[5])[0][0])), #Modal
                        "spec_centroid_iqr" : float(stats.iqr(width_feature[5])), #Interquartilsabstand
                        "spec_centroid_zerocross" : float(np.sum(librosa.zero_crossings(width_feature[5], pad=False))), #Zero-Crossings

                        "spec_bandwidth_mean" : float(np.mean(width_feature[6])), #Arithmetisches Mittel
                        "spec_bandwidth_std" : float(np.std(width_feature[6])), #Standardabweichung
                        "spec_bandwidth_maxv" : float(np.amax(width_feature[6])), #Maximum Wert
                        "spec_bandwidth_maxs" : float(np.argmax(width_feature[6])), #Maximum Stelle
                        "spec_bandwidth_minv" : float(np.amin(width_feature[6])), #Minimum Wert
                        "spec_bandwidth_mins" : float(np.argmin(width_feature[6])), #Minimum Stelle
                        "spec_bandwidth_median" : float(np.median(width_feature[6])), #Median
                        "spec_bandwidth_skew" : float(np.mean(stats.skew(width_feature[6]))), #Schiefe
                        "spec_bandwidth_kurt" : float(np.mean(stats.kurtosis(width_feature[6]))), #Wölbung
                        "spec_bandwidth_q1" : float(np.quantile(width_feature[6], 0.25)), #Quantile1
                        "spec_bandwidth_q3" : float(np.quantile(width_feature[6], 0.75)), #Quantile2
                        "spec_bandwidth_mode" : float(np.mean(stats.mode(width_feature[6])[0][0])), #Modal
                        "spec_bandwidth_iqr" : float(stats.iqr(width_feature[6])), #Interquartilsabstand
                        "spec_bandwidth_zerocross" : float(np.sum(librosa.zero_crossings(width_feature[6], pad=False))), #Zero-Crossings

                        "spec_contrast_mean" : float(np.mean(width_feature[7])), #Arithmetisches Mittel
                        "spec_contrast_std" : float(np.std(width_feature[7])), #Standardabweichung
                        "spec_contrast_maxv" : float(np.amax(width_feature[7])), #Maximum Wert
                        "spec_contrast_maxs" : float(np.argmax(width_feature[7])), #Maximum Stelle
                        "spec_contrast_minv" : float(np.amin(width_feature[7])), #Minimum Wert
                        "spec_contrast_mins" : float(np.argmin(width_feature[7])), #Minimum Stelle
                        "spec_contrast_median" : float(np.median(width_feature[7])), #Median
                        "spec_contrast_skew" : float(np.mean(stats.skew(width_feature[7]))), #Schiefe
                        "spec_contrast_kurt" : float(np.mean(stats.kurtosis(width_feature[7]))), #Wölbung
                        "spec_contrast_q1" : float(np.quantile(width_feature[7], 0.25)), #Quantile1
                        "spec_contrast_q3" : float(np.quantile(width_feature[7], 0.75)), #Quantile2
                        "spec_contrast_mode" : float(np.mean(stats.mode(width_feature[7])[0][0])), #Modal
                        "spec_contrast_iqr" : float(stats.iqr(width_feature[7])), #Interquartilsabstand
                        "spec_contrast_zerocross" : float(np.sum(librosa.zero_crossings(width_feature[7], pad=False))), #Zero-Crossings

                        "spec_flatness_mean" : float(np.mean(width_feature[8])), #Arithmetisches Mittel
                        "spec_flatness_std" : float(np.std(width_feature[8])), #Standardabweichung
                        "spec_flatness_maxv" : float(np.amax(width_feature[8])), #Maximum Wert
                        "spec_flatness_maxs" : float(np.argmax(width_feature[8])), #Maximum Stelle
                        "spec_flatness_minv" : float(np.amin(width_feature[8])), #Minimum Wert
                        "spec_flatness_mins" : float(np.argmin(width_feature[8])), #Minimum Stelle
                        "spec_flatness_median" : float(np.median(width_feature[8])), #Median
                        "spec_flatness_skew" : float(np.mean(stats.skew(width_feature[8]))), #Schiefe
                        "spec_flatness_kurt" : float(np.mean(stats.kurtosis(width_feature[8]))), #Wölbung
                        "spec_flatness_q1" : float(np.quantile(width_feature[8], 0.25)), #Quantile1
                        "spec_flatness_q3" : float(np.quantile(width_feature[8], 0.75)), #Quantile2
                        "spec_flatness_mode" : float(np.mean(stats.mode(width_feature[8])[0][0])), #Modal
                        "spec_flatness_iqr" : float(stats.iqr(width_feature[8])), #Interquartilsabstand
                        "spec_flatness_zerocross" : float(np.sum(librosa.zero_crossings(width_feature[8], pad=False))), #Zero-Crossings

                        "spec_rolloff_mean" : float(np.mean(width_feature[9])), #Arithmetisches Mittel
                        "spec_rolloff_std" : float(np.std(width_feature[9])), #Standardabweichung
                        "spec_rolloff_maxv" : float(np.amax(width_feature[9])), #Maximum Wert
                        "spec_rolloff_maxs" : float(np.argmax(width_feature[9])), #Maximum Stelle
                        "spec_rolloff_minv" : float(np.amin(width_feature[9])), #Minimum Wert
                        "spec_rolloff_mins" : float(np.argmin(width_feature[9])), #Minimum Stelle
                        "spec_rolloff_median" : float(np.median(width_feature[9])), #Median
                        "spec_rolloff_skew" : float(np.mean(stats.skew(width_feature[9]))), #Schiefe
                        "spec_rolloff_kurt" : float(np.mean(stats.kurtosis(width_feature[9]))), #Wölbung
                        "spec_rolloff_q1" : float(np.quantile(width_feature[9], 0.25)), #Quantile1
                        "spec_rolloff_q3" : float(np.quantile(width_feature[9], 0.75)), #Quantile2
                        "spec_rolloff_mode" : float(np.mean(stats.mode(width_feature[9])[0][0])), #Modal
                        "spec_rolloff_iqr" : float(stats.iqr(width_feature[9])), #Interquartilsabstand
                        "spec_rolloff_zerocross" : float(np.sum(librosa.zero_crossings(width_feature[9], pad=False))), #Zero-Crossings

                    }
                    
                   
                    # (
                    #     nr_frequency_ranges,
                    #     start_frequency_ranges,
                    #     end_frequency_ranges,
                    # ) = own_functions.get_frequency_ranges(
                    #     sr, 2000
                    # )  # opening own function that returns the number of different frequency ranges and two arrays, containing the first and last frequency of each frequency range
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

                    if frequency_ranges_for_welch == "yes":
                        (
                            nr_frequency_ranges,
                            start_frequency_ranges,
                            end_frequency_ranges,
                        ) = own_functions.get_frequency_ranges(
                            sr, 2000
                        )  # opening own function that returns the number of different frequency ranges and two arrays, containing the first and last frequency of each frequency range
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
                    else:
                        welch_features = {
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
            "Test series data file for " + test_series + "_features.json  was saved."
        )
