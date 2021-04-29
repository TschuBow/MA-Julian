import pyaudio
import wave
import streamlit as st
import os
import math
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression




# FUNCTIONS FOR RECORDER

def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path

def record(FORMAT, CHANNELS, RATE, CHUNK, RECORD_SECONDS, WAVE_OUTPUT_FILENAME):
    # FORMAT = pyaudio.paInt16
    # CHANNELS = 2
    # RATE = 44100
    # CHUNK = 1024
    # RECORD_SECONDS = 5
    # WAVE_OUTPUT_FILENAME = "file.wav"
    with st.spinner("recording..."):
        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
        )
        
        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        st.write("finished recording")

        # stop Recording
        stream.stop_stream()
        stream.close()
        audio.terminate()

        waveFile = wave.open(WAVE_OUTPUT_FILENAME, "wb")
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b"".join(frames))
        waveFile.close()



# FUNCTIONS FOR FEATURE GENERATOR

def get_frequency_ranges(sr, frequency_range):
    number_of_frequency_ranges = 1 + math.ceil(
        (math.ceil(sr / 2) - 1000) / frequency_range
    )
    start_frequency_range = []
    end_frequency_range = []

    for i in range(0, number_of_frequency_ranges):

        if i == 0:
            start_frequency_range.append(0)
            end_frequency_range.append(1000)
        elif i < number_of_frequency_ranges - 1:
            start_frequency_range.append((i * frequency_range) - 999)
            end_frequency_range.append(((i + 1) * frequency_range) - 1000)
        else:
            start_frequency_range.append((i * frequency_range) - 999)
            end_frequency_range.append(math.floor(sr / 2))

    return number_of_frequency_ranges, start_frequency_range, end_frequency_range



# FUNCTIONS FOR FEATURE REGRESSOR

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "Linear Regression":
        None
    elif clf_name == "Polynomial Regression":
        degree=st.number_input("Set Polynomial Degree", 1, 10, 2, 1)
        params["degree"] = degree
    elif clf_name == "KNN":
        K = st.slider("K", 1 , 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.slider("C", 0.01, 10.00)
        params["C"] = C
    elif clf_name == "Random Forest":
        max_depth = st.slider("Max Depths", 2 , 15)
        n_estimators = st.slider("Number of Estimators", 1 , 15)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params

def get_classifier(clf_name, params):
    if clf_name == "Linear Regression":
        clf = LinearRegression()
    elif clf_name == "Polynomial Regression":
        clf = make_pipeline(PolynomialFeatures(degree= params["degree"]), Ridge())
    elif clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    elif clf_name == "Random Forrest":
        clf = RandomForestClassifier(n_estimators=params["n_estimators"],
                                    max_depth=params["max_depth"], random_state=1234)
    return clf

