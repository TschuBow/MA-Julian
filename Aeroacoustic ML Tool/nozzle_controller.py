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

import serial  # pip install pyserial, NOT serial!!!
import time
import multiprocessing

def main():
    st.header("Nozzle Controller")

    nozzle_on = st.button("switch nozzle on")
    

    
    if nozzle_on == True:
        nozzle_off = False
        port1 = "COM4"
        ser1 = serial.Serial(port1, 115200, timeout = 10)
        time.sleep(2)
        print("opened Serial to Arduino.")
        ser1.write(bytes("<0>".encode('ascii'))) #Ventil aus und LED an
        ser1.close() #close Serial Port
        nozzle_off=st.button("switch nozzle off")  
        
    if nozzle_off == True:
        nozzle_on = False
        port1 = "COM4"
        ser1 = serial.Serial(port1, 115200, timeout = 10)
        time.sleep(2)
        print("opened Serial to Arduino.")
        ser1.write(bytes("<1>".encode('ascii'))) #Ventil aus und LED an
        ser1.close() #close Serial Port
        nozzle_on = st.button("switch nozzle on")
        


        