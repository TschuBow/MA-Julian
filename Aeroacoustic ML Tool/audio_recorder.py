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

def G1_withprint(x_pos, y_pos, z_pos, speed):
    x_pos = round(x_pos,1)
    y_pos = round(y_pos,1)
    z_pos = round(z_pos,1)
    tmp_GCode = "G1"+" "+"X"+str(x_pos)+" "+"Y"+str(y_pos)+" "+"Z"+str(z_pos)+" "+"F"+str(speed)+ "\n" #Position anfahren
    tmp_GCode = tmp_GCode + "M400" + "\n" #Warten mit Ausführung des nächsten G-Codes bis Beendigung des vorigen
    tmp_GCode = tmp_GCode + "M118"+" TX: "+"X"+str(x_pos)+" "+"Y"+str(y_pos)+" "+"Z"+str(z_pos)+" "+ "\n" #M118 für Json
    return tmp_GCode

def G1_withoutprint(x_pos, y_pos, z_pos, speed):
    x_pos = round(x_pos,1)
    y_pos = round(y_pos,1)
    z_pos = round(z_pos,1)
    tmp_GCode = "G1"+" "+"X"+str(x_pos)+" "+"Y"+str(y_pos)+" "+"Z"+str(z_pos)+" "+"F"+str(speed)+ "\n" #Position anfahren
    tmp_GCode = tmp_GCode + "M400" + "\n" #Warten mit Ausführung des nächsten G-Codes bis Beendigung des vorigen
    return tmp_GCode


#####Worker=Timestampgenerator & Karl I Ansteuerung
def worker(temppath):
    temp = temppath #temporären Übergabepfad nutzen
    #temp = r"C:\Users\heiko\Downloads\WPy64-3680\notebooks\gcode_projektarbeit"
    json_array = {} #json array to save important information
    startms = int(round(time.time() * 1000)) #start time of operation in ms #####MIGHT BE BETTER IN GCODE GENERATION
    lastms = startms

    #read G-Code from temporary file
    gcode_array_position = 0 #flag to store where we are in the gcode_array right now
    gcode_array = [] #file to array
    tempsave = os.path.join(temp, "temporary_file_gcode.g") 
    with open(tempsave) as my_file:
        gcode_array = my_file.readlines()

    #read G-Code waiter from temporary file
    #quasi prepare for "ok"s to expect from printer to stream gcode without error
    gcode_waiter_array_positon = 0 #flag to store where we are in the gcode_waiter_array right now
    gcode_waiter_array = [] #file to array
    temp_gcode_waiter_file = []
    tempsave = os.path.join(temp, "temporary_file_gcode_waiter.g") 
    with open(tempsave) as my_file:
        temp_gcode_waiter_file = my_file.readlines()
    temp_gcode_waiter_file = temp_gcode_waiter_file[0]
    temp_gcode_waiter_file = temp_gcode_waiter_file.strip("[")
    temp_gcode_waiter_file = temp_gcode_waiter_file.strip("]")
    gcode_waiter_array = temp_gcode_waiter_file.split(",")

    #Open Serial
    port = "COM3" #Hardcoded port, change if other port used #COM4 X230 #COM3 Unirechner linker USB-Port
    ser = serial.Serial(port, 250000, timeout = 10) #start serial connection
    print("start of operation.") #debug print
    print("serial port opened.") #debug print
    time.sleep(2) #sleept to give time to initialize #do not delete or expect trouble

    ok_counter = 0 #count "ok" from demonstrator to not overflow serial buffer and check if action is executed
    new_gcode_flag = 1 #flag to signal new gcode should be transmitted
    timestampart = "TSL" #Konvention zur Benennung links: TSL rechts: TSR
    timstampcount = 1 #Counting timestamps starting with 1
    leseserial = 1 #if 1 keep reading

    while leseserial == 1:
        textreceived = ser.readline() #read serial output of demonstrator line by line
        print("demonstrator output: " + str(textreceived)) #print the output to console
        with open("SerialData.txt", "a+") as myfile: #open logging file
            myfile.write(str(textreceived)+"\n") #write to logging file

        if b'SETTINGS' in textreceived: #receive the settings output
            splitters = textreceived.split() #split the string based on spaces
            json_array['feedrate'] = int(splitters[1][1:]) #save to json
            json_array['acceleration'] = int(splitters[2][1:]) #save to json
            json_array['turns'] = int(splitters[3][1:]) #save to json
            port1 = "COM4"
            ser1 = serial.Serial(port1, 115200, timeout = 10)
            time.sleep(2)
            print("opened Serial to Arduino.")
            ser1.write(bytes("<1>".encode('ascii'))) #Ventil auf und LED an
            ser1.close() #close Serial Port

        if b'ok' in textreceived: #okcounter to check if code was successfully transmitted and executed
            ok_counter = ok_counter + 1 #add 1 to okcounter
            if(int(gcode_waiter_array[int(gcode_waiter_array_positon)]) == ok_counter): #compare the "ok" received to the expected "ok"
                ok_counter = 0 #reset ok_counter for the next block
                gcode_waiter_array_positon = gcode_waiter_array_positon + 1 #next position in gcode_waiter_array_positon
                new_gcode_flag = 1 #set flag to start new gcode transmission
            else:
                continue #do nothing here if we have not reached the expected "ok" amount

        if(new_gcode_flag == 1):
            new_gcode_flag = 0
            with open("SerialData.txt", "a") as myfile: #to make serial data more readable
                    myfile.write("BLOCK NEU"+"\n")
            for a in range(int(gcode_waiter_array[int(gcode_waiter_array_positon)])): #send commands based on amount previously specified
                try:
                    ser.write(bytes(gcode_array[gcode_array_position], 'utf-8'))
                    gcode_array_position = gcode_array_position + 1 #next gcode
                except:
                    print("A Error occured while trying to send data to demonstrator.")
                    continue
                
        if b'TX' in textreceived: #Timestamp generation
            now = int(round(time.time() * 1000)) #current time in ms
            tookms = now - lastms #calculate used ms between M118 with "TX"
            lastms = now #set lastms to current time
            print('time between last M118 with "TX": ' + str(tookms)) #print for comparision and debug to console
            if(timestampart == "TSL"): #timestamp on left side
                timestampname = timestampart + str(timstampcount) #create jsoneintrag name
                json_array[timestampname] = now -startms #create timestamp with time since start 
                timestampart = "TSR" #change for next timestamp
            else: #timestamp on right side
                timestampname = timestampart + str(timstampcount) #create jsoneintrag name
                json_array[timestampname] = now - startms #create timestamp with time since start
                timstampcount = timstampcount + 1 #always one timestamp left then one right then next turn so we add +1
                timestampart = "TSL" #change for next timestamp


        if b'START' in textreceived: #START GCODE BAUSTEIN
            continue

        if b'END' in textreceived: #END GCODE M118 Baustein
            leseserial = 0 #end the while loop
            ser.close() #close the serial port
            print("end of operation.")
            print("serial port closed.")

            port1 = "COM4"
            ser1 = serial.Serial(port1, 115200, timeout = 10)
            time.sleep(2)
            print("opened Serial to Arduino.")
            ser1.write(bytes("<0>".encode('ascii'))) #Ventil aus und LED an
            ser1.close() #close Serial Port

    tempsave = os.path.join(temp, "temporary_file_timestamp.json") #create temporary path
    with open(tempsave, "w+") as outfile: #create file
        outfile.write(json.dumps(json_array)) #dump json file
    return #end this part of code
#################

# function main() that is run by Main.py:
def main():
    # selectable metadata regarding the testobject and testing hardware:
    loaded_objects = ("Sandvik CNMG 12 04 08-MM 2025", "Mischkoerper", "other")
    loaded_testing_rigs = ("KARL 1",)
    loaded_nozzles = (
        "Schlick Modellreihe 629",
        "Schlick Modellreihe 650",
        "FOS-45°-Flachstrahldüse 25mm Länge",
        "FOS-Flachstrahldüse 35mm Länge",
        "FOS-Flachstrahldüse 60mm Länge",
        "Jet-Düse MV6, Innendurchm. 6mm",
        "Jet-Düse MV12, Innendurchm. 6mm",
        "Düse 1200SSF 1/8 verstellbar, Edelstahl",
        "Silvent MJ4",
        "Silvent MJ5",
        "",
    )
    loaded_microphones = ("Renkforce UM-80 USB-Mikrofon", "other")
    loaded_recording_modes = ("Recording by time", "Recording on Karl I")

    # selectable audio recording settings:
    loaded_sampling_rates = ("44100", "48000", "88200", "96000")
    loaded_sample_formats = ("16", "24")
    loaded_chunks = ("1024", "2048")

    # header and subheader of the page:
    st.header("Metadata Capture of Audio Files")
    st.write(
        "Name your recording and select or type in the right parameters"
    )
    col_meta_1, col_meta_2, col_meta_3, = st.beta_columns((2, 2, 2,))
     
    # select metadata regarding the testobject and testing hardware:
    test_series_name = str(col_meta_1.text_input("Test Series Name", "TestserienNr", help='The test series name designates a series of tests on comparable bodies. For example "width_Schlick_629" for the test of several inserts of the same model. Consider carefully how to name the test series.Subsequent changes are not possible or lead to incorrect processing of the metadata. Avoid letters like "ä, ö, ü, ß" and special characters.'))
    audio_file_name_raw = col_meta_1.text_input("File Name", "example", help='The file name of the recording is based on the name to be entered here and the number of the test object in the line below.')
    test_object_number = str(col_meta_1.text_input("Test Object Number", "1", help='The number of the test object is specified here. The numbering helps to distinguish between different objects of the same type or model and is included in the file name of the recording.'))
    test_deviation = col_meta_1.number_input(
        "Target Value [µm]", step=0.01, help='Insert the measured target value of the test object [µm]. For example, the optically measured deviation of the test object'
    )
    

    #col_meta_1, col_meta_2, col_meta_3 = st.beta_columns((2, 2, 2))

   
    test_nozzle = col_meta_2.selectbox("Nozzle", loaded_nozzles, help='Select the nozzle used for compressed air blasting. If the desired model is not selectable, it must be added to the "loaded_nozzles" list in the source code of the "audio_recorder.py" file.')
    test_object = col_meta_2.selectbox("Test Object", loaded_objects, help='Select the test tested object / model that is to be tested. If the desired model is not selectable, it must be added to the "loaded_objects" list in the source code of the "audio_recorder.py" file.')
    test_testing_rig = col_meta_2.selectbox("Test Rig", loaded_testing_rigs, help='Select the test rig on which the recording will be taken. If the desired model is not selectable, it must be added to the "loaded_testing_rigs" list in the source code of the "audio_recorder.py" file.')
    
    
    
    # test_recording_mode = st.selectbox("Recording mode", loaded_recording_modes)
    
    st.header("G-Code Settings and Recording Time")
    
    test_recording_mode = st.radio("Recording mode", loaded_recording_modes)
    
    if test_recording_mode == "Recording by time":
        #col_record_settings.write("Manual setting of the recording time")
        col_mode_1, col_mode_2, = st.beta_columns((2, 1))
        test_required_recording_time = col_mode_1.number_input(
            "Recording Time [s]", help='Enter the desired recording length here. This can be between 0 and 240 seconds.',
            step=0.01,
            min_value=0.01,
            max_value=240.00,
            value=2.00
        )
        mode_path_name = "0_recording_by_time"

    elif test_recording_mode == "Recording on Karl I":
        #col_record_settings.header("Recording with Karl I")
        col_mode_1, col_mode_2, col_mode_3, = st.beta_columns((1, 1, 1,))

        Infoimage = Image.open("images\info_cutting_parameters.png")
        col_mode_3.image(Infoimage, caption=None, use_column_width=True)#, width=600, height=600)

        x_start = col_mode_1.number_input("start position of x axis [mm]", min_value = 0.00, max_value = 120.00, value = 12.00, step = 0.1) 
        y_start = col_mode_1.number_input("start position of y axis [mm]", min_value = 0.00, max_value = 130.00, value = 0.00, step = 0.1) 
        z_start = col_mode_1.number_input("start position of z axis [mm]", min_value = 0.00, max_value = 100.00, value = 10.00, step = 0.1) 

        test_feed_speed = col_mode_2.number_input(
            "Feed Speed [mm/s]",
            step=0.01,
            min_value=0.01,
            max_value=10000.00,
            value=8.00,
        )
        test_feed_acceleration = col_mode_2.number_input(
            "Acceleration [mm/s²]",
            step=0.01,
            min_value=0.01,
            max_value=2000.00,
            value=25.00,
        )
        test_measuring_path = col_mode_2.number_input(
            "Measuring Path (a) [mm]",
            step=0.01,
            min_value=0.01,
            max_value=1000.00,
            value=16.00,
        )
        test_offset = col_mode_2.number_input(
            "Offset (b) [mm]", min_value=0.01, max_value=1000.00, value=1.00
        )
        test_number_of_measuring_paths = col_mode_1.slider(
            "Number of Measuring Paths (n)", 0, 50, 1, 1
        )
        #st.write(test_number_of_measuring_paths, "Measuring paths per audiofile")
        n = test_number_of_measuring_paths
        ss_p = test_measuring_path  # Messstrecke
        ss_o = test_offset  # Versatz/Abstand zwischen den Messstrecken
        vs = test_feed_speed  # eingestellte Vorschubgeschwindigkeit
        acs = test_feed_acceleration  # eingestellte Beschleunigung
        ta = (
            vs / acs
        )  # benötigte Zeit, bis mit gegebener Beschleunigung die Vorschubgeschwindigkeit erreicht ist
        # Strecke, die während Beschleunigungsphase auf Vorschubgeschwindigkeit zurückgelegt wird.
        sa = 0.5 * acs * (ta ** 2)
        # Bei den Versatzwegen wird die Sollvorschubgeschwindigkeit in der Regel nicht erreicht, daher dieser Ausnahmefall:
        if sa <= (0.5 * ss_o):
            tv_o = 0  # Keine Zeit beim Versatz wird dann mit der Sollvorschubgeschwindigkeit zurückgelegt
            # Die Zeit, die für das Zurücklegen des halben Versatzes mit der Beschleunigung zurückgelegt wird
            ta_o = (ss_o / acs) ** 0.5
            ts_o = (
                2 * ta_o
            )  # 2x, jeweils einmal für Beschleunigung und einmal für Bremsen
        else:
            sv_o = ss_o - (2 * sa)
            tv_o = sv_o / vs
            ts_o = tv_o + (2 * ta)

        sv_p = ss_p - (2 * sa)  # path length while having the set feed speed
        tv_p = sv_p / vs  # measuring time while having the set feed speed
        ts_p = tv_p + (2 * ta)  # total measuring time per path

        ttotal = (
            2 * n * ts_p + (n - 1) * ts_o
        )  # 2 times the measuring path length times its iterations plus the offsets
        if ttotal <= 0:
            ttotal = 0
        
        st.write("Adjust recording time manually:")
        col_mode_1, col_mode_2, col_mode_3,  = st.beta_columns((1, 1, 1,))
        
        ttolerance_fix = col_mode_1.number_input(
            "One-Time Time Tolerance [s]", step=0.01, value=25.00
        )
        ttolerance_per_path = col_mode_2.number_input(
            "Time Tolerance per Path [s]", step=0.01, value=1.25
        )
        ttotal_with_tolerance = ttotal + ttolerance_fix + n * ttolerance_per_path

        st.write("")
        col_mode_1, col_mode_2, = st.beta_columns((2, 1,))

        col_mode_1.write(
            f"The calculated recording time including tolerances counts {ttotal_with_tolerance} seconds"
        )
        test_required_recording_time = ttotal_with_tolerance
        mode_path_name = "1_recording_on_karl_I"
        demonstrator_active = col_mode_1.radio("Is the demonstrator connected?", ("yes", "no"))
    # select audio recording settings:

    audio_file_name = f"{audio_file_name_raw}_objectnr_{test_object_number}.wav"

    st.header("Microphone and Recording Settings")
    
    col_rec_set_1, col_rec_set_2, col_rec_set_3, col_rec_set_4, col_rec_set_5 = st.beta_columns((2, 2, 2, 2, 4))

    test_microphone = col_rec_set_1.selectbox("Microphone", loaded_microphones)
    
    test_sampling_rate = int(
        col_rec_set_2.selectbox("Sampling rate [kHz]", loaded_sampling_rates, help='The sample rate is the number of samples played in each second. Sample rates are measured in "Hertz" (abbreviated "Hz"), which means "per second," or in "kilohertz" (abbreviated "kHz"), which means "per second, times one thousand." The sample rate used on audio CDs can be written as 44 100 Hz, or 44.1 kHz, which both have the same meaning. Common sample rates are 44.1 kHz, 48 kHz, and 96 kHz. Other possible sample rates include 22 kHz, 88.2 kHz, and 192 kHz.')
    )
    selected_test_sample_format = col_rec_set_3.selectbox(
        "Sample format [bit]", loaded_sample_formats, help="The sample format is the number of bits used to describe each sample. The greater the number of bits, the more data will be stored in each sample. Common sample formats are 16 bits and 24 bits. 8-bit samples are low-quality, and not used often. 20-bit samples are not commonly used on computers. 32-bit samples are possible, but not supported by most audio interfaces."
    )
    test_chunk = int(col_rec_set_4.selectbox("Buffer [byte]", loaded_chunks, help="The chunk is like a buffer, so therefore each buffer will contain 1024 samples, which you can then either keep or throw away. We use chunks of data, instead of a continuous amount of audio because of processing power."))
    # test_channels = int(st.selectbox('Number of channels', loaded_channels))

    # translating the input of the selectbox for the sample format into they right variable for the recording algorithm:
    if selected_test_sample_format == "8":
        test_sample_format = pyaudio.paInt8
    elif selected_test_sample_format == "16":
        test_sample_format = pyaudio.paInt16
    elif selected_test_sample_format == "24":
        test_sample_format = pyaudio.paInt24
    elif selected_test_sample_format == "32":
        test_sample_format = pyaudio.paInt32
    else:
        test_sample_format = None

    # reminder to check if all settings are done correctly and button to start the recording:

    # loaded_npersegs = ("512", "1024", "2048", "4096")
    # npersegs=int(st.selectbox('Length of each segment (nperseg)', loaded_npersegs))
    npersegs = 4096
    noverlaps = npersegs / 2

    st.write(
        'Check if everything is filled in correctly and save your metadata by pressing the "Save" button below.'
    )

    col_rec_1, col_rec_2, col_rec_3, col_rec_4 = st.beta_columns((1.3, 1, 1, 12))

    start_recording = col_rec_1.button("Record")

    
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
            ################################################################################################    
            ################################################################################################
            #Parameter aus Streamlit übernehmen
            x_length = float(round(ss_p,1))
            y_length = float(round(ss_o,1))
            speed = int(round(vs*60,0)) #Umrechnung Geschwindigkeit in mm/min
            accel = int(round(acs,0)) #in mm/s2
            turns = int(n) # nur gerade Zahlen, damit auch zurückkommt
            ##################################################    
            #Startpunkt definieren
            # x_start = 12.00
            # y_start = 0.00
            # z_start = 10.00
            #Safety Z definieren
            Z_safe = 5
            ##################################################
            #maximale Werte festlegen
            x_max = 120 #not used at this time
            y_max = 130 #not used at this time
            z_max = 100 #not used at this time
            ##################################################        
            #Datei
            G_codefile = ""
            G_codewaiter = []

            #Start G-Code - 4 Operationen
            tmp_GCode = "M118 START" + "\n" #Start M118 für Json Generation
            tmp_GCode = tmp_GCode + "G28" + "\n" #Home Operation in X,Y,Z
            tmp_GCode = tmp_GCode + "M204 T" + str(accel) + "\n" #Beschleunigung festlegen
            tmp_GCode = tmp_GCode + "M118 SETTINGS F"+ str(int(round(speed/60,0))) + " A" + str(accel) + " T" + str(turns) + "\n" #M118 für Json
            G_codefile = G_codefile + tmp_GCode #zu G_codefile hinzufügen
            G_codewaiter.append(4)

            #Startpositon G-Code - 3 Operationen
            x_pos, y_pos, z_pos = x_start, y_start, z_start #Ausgangswerte für Operationen definieren
            G_codefile = G_codefile + G1_withprint(x_pos, y_pos, z_pos, speed) #zu G_codefile hinzufügen
            G_codewaiter.append(3)

            ####Start Operations G-Code
            for n in range(turns):
                #+X Verfahrweg + M118 - 3 Operationen
                x_pos = x_pos + x_length #Ausgangswerte für Operationen definieren
                G_codefile = G_codefile + G1_withprint(x_pos, y_pos, z_pos, speed) #zu G_codefile hinzufügen
                G_codewaiter.append(3)
                #-X Verfahrweg - 2 Operationen
                x_pos = x_pos - x_length #Ausgangswerte für Operationen definieren
                G_codefile = G_codefile + G1_withoutprint(x_pos, y_pos, z_pos, speed) #zu G_codefile hinzufügen
                G_codewaiter.append(2)  
                if (n<turns-1): #um letzten Verfahrweg in Y nicht durchzuführen
                    #+Y Verfahrweg + M118 - 3 Operationen
                    y_pos = y_pos + y_length #Ausgangswerte für Operationen definieren
                    G_codefile = G_codefile + G1_withprint(x_pos, y_pos, z_pos, speed) #zu G_codefile hinzufügen
                    G_codewaiter.append(3)    
            ####Ende Operations G-Code

            #Z-Achse positiv Verfahren um sicheres Zurückfahren zu ermöglichen G-Code - 3 Operationen
            z_pos = z_pos + Z_safe #Verfahren in Z für sichere Endoperation
            G_codefile = G_codefile + G1_withoutprint(x_pos, y_pos, z_pos, speed) #zu G_codefile hinzufügen
            G_codewaiter.append(2)

            #Endpositon G-Code - 3 Operationen
            x_pos, y_pos, z_pos = 0,0,0 #Ausgangspunkt definieren
            G_codefile = G_codefile + G1_withoutprint(x_pos, y_pos, z_pos, speed) #zu G_codefile hinzufügen
            G_codewaiter.append(2)

            #End G-Code - 3 Operationen
            tmp_GCode = "M84" + "\n" #Stepper Motoren ausschalten
            tmp_GCode = tmp_GCode + "M118 END" + "\n" #End
            G_codefile = G_codefile + tmp_GCode #zu G_codefile hinzufügen
            G_codewaiter.append(2)

            tempsave = os.path.join(temp, "temporary_file_gcode.g") #temporären Pfad generieren
            f = open(tempsave, "w+") #Datei neu erstellen oder überschreiben
            f.write(G_codefile) #in Datei schreiben
            f.close() #Datei schließen

            tempsave = os.path.join(temp, "temporary_file_gcode_waiter.g") #temporären Pfad generieren
            f = open(tempsave, "w+") #Datei neu erstellen oder überschreiben
            f.write(str(G_codewaiter)) #in Datei schreiben
            f.close() #Datei schließen

            #Errorcheck
            if (sum(G_codewaiter) != G_codefile.count("\n")): #compare gcodewaiter to gcode
                print("You might have a Error in your G-Code. Expect it not to work")
            ##################################################
            ##################################################
            
            #START COMMUNICATION
            if demonstrator_active == 'yes':
                jobs = []
                p = multiprocessing.Process(target=worker, args=(temp,))
                jobs.append(p)
                p.start()
            else:
                None
            ##################################################
            ##################################################
            
            temporary_file_path = temp / "temporary_file.wav"

        own_functions.record(
            test_sample_format,
            1,
            test_sampling_rate,
            test_chunk,
            test_required_recording_time,
            str(temporary_file_path),
        )  # recording function that is defined in Aufnahme.py

        # sample_rate, samples = wavfile.read(temporary_file_path)                        # opening the temporary audio file that was just recorded before
        samples, sample_rate = librosa.load(temporary_file_path, sr=test_sampling_rate)

        st.write("sample rate:", sample_rate)

        # extracting frequencies, times and amplitdues from the wav file
        frequencies, times, spectrogram = signal.spectrogram(
            samples, sample_rate, nperseg=npersegs, noverlap=noverlaps
        )

        st.write("Spectrogram logarithmic scaling")
        temporary_spectrogram_fig = plt.figure()
        # creating the spectogram from the arrays of times, frequencies and spectrogram where amplitudes are expressed as colours
        plt.pcolormesh(times, frequencies, spectrogram)
        # setting the scale of the frequencies axis from 1 to 20000Hz
        plt.ylim(1, (sample_rate / 2))
        # logarithmizing the frequencies axis of the spectrogram
        plt.yscale("log")
        # labeling the axes
        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Time [sec]")
        # plt.z
        # visualization of the spectrogram on Streamlit
        st.pyplot(temporary_spectrogram_fig)

        if test_recording_mode == "Recording by time":
            # opening the last recorded wav file in a player on Streamlit
            st.audio(
                f"audio-files/{mode_path_name}/0_temporary_files/temporary_file.wav",
                format="audio/wav",
            )
        elif test_recording_mode == "Recording on Karl I":
            st.audio(
                f"audio-files/{mode_path_name}/0_temporary_files/temporary_file.wav",
                format="audio/wav",
            )

    savefile = col_rec_2.button("Save")
    # temp = Path(f"audio-files/{mode_path_name}/0_temporary_files")
    # temporary_file_path = temp / "temporary_file.wav"
    if savefile == True:

        # 0_recording_by_time
        old_temp_path = Path(f"audio-files/{mode_path_name}/0_temporary_files")
        old_temp_path_file = old_temp_path / "temporary_file.wav"
        if os.path.exists(old_temp_path_file):

            new_path = Path(f"audio-files/{mode_path_name}/{test_series_name}")
            new_path.mkdir(parents=True, exist_ok=True)
            new_path_file = new_path / audio_file_name
            new_path_file = own_functions.uniquify(new_path_file)

            shutil.copyfile(old_temp_path_file, new_path_file)

            os.remove(old_temp_path_file)
            
            pure_file_name = os.path.basename(new_path_file)

            pure_base_name = pure_file_name.replace(".wav", "")
            meta_file_name = Path(f"{pure_base_name}_meta.json")
            time_stamp_file_name = Path(f"{pure_base_name}_ts.json")

            meta_file_path = new_path / meta_file_name
            time_stamp_file_path = new_path / time_stamp_file_name

            if test_recording_mode == "Recording by time":
                audio_file_meta = {
                    "optically_measured_deviation": test_deviation,
                    "test_series": test_series_name,
                    "test_series_path": str(new_path),
                    "pure_base_name": str(pure_base_name),
                    "audio_file": str(pure_file_name),
                    "audio_file_path": str(new_path_file),
                    "meta_file": str(meta_file_name),
                    "meta_file_path": str(meta_file_path),
                    "microphone": test_microphone,
                    "samplingrate": test_sampling_rate,
                    "recording_mode": test_recording_mode,             
                    "object": test_object,
                    "nozzle": test_nozzle,
                    "machine_tool": test_testing_rig,
                    "recording_time": test_required_recording_time,
                    "test_object_number": test_object_number,
                }
            elif test_recording_mode == "Recording on Karl I":
                if demonstrator_active == "yes":
                    old_time_stamp_path_file = (
                        old_temp_path / "temporary_file_timestamp.json"
                    )
                    shutil.copyfile(old_time_stamp_path_file, time_stamp_file_path)
                else:
                    time_stamp_file_path = (
                        "test recording: no stamp file generated."
                    )

                audio_file_meta = {
                    "optically_measured_deviation": test_deviation,
                    "test_series": test_series_name,
                    "test_series_path": str(new_path),
                    "pure_base_name": str(pure_base_name),
                    "audio_file": str(pure_file_name),
                    "audio_file_path": str(new_path_file),
                    "meta_file": str(meta_file_name),
                    "meta_file_path": str(meta_file_path),
                    "timestamp_file": str(time_stamp_file_name),
                    "timestamp_file_path":  str(time_stamp_file_path),                
                    "microphone": test_microphone,
                    "sampling_rate": test_sampling_rate,
                    "recording_mode": test_recording_mode,
                    "number_measuring_paths": test_number_of_measuring_paths,
                    "measuring_path_length": test_measuring_path,
                    "offset": test_offset,
                    "feed_speed": test_feed_speed,
                    "feed_acceleration": test_feed_acceleration,
                    "object": test_object,
                    "nozzle": test_nozzle,
                    "recording_time": test_required_recording_time,
                    "time_per_path": ts_p,
                    "test_object_number": test_object_number,
                }

            with open(meta_file_path, "w") as json_file:
                json.dump(audio_file_meta, json_file)
            st.write(
                "Audio file and metadata file for " + str(audio_file_name) + "  were saved in " + str(meta_file_name) + " ."
            )

        else:
            st.write("Record a new audio file first.")

    deletefile = col_rec_3.button("Delete")

    if deletefile == True:

        old_temp_path = Path(f"audio-files/{mode_path_name}/0_temporary_files")
        old_temp_path_file = old_temp_path / "temporary_file.wav"
        if os.path.exists(old_temp_path_file):
            os.remove(old_temp_path_file)
            st.write("Your recording was deleted")
        else:
            st.write("There is no recording yet that could be deleted.")
