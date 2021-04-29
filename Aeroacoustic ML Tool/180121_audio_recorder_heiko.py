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

#########################################################################################################################################
import serial # pip3 install pyserial; NICHT serial
import time
import multiprocessing

temp = ""

def worker():
    global temp
    """worker function"""
    testsite_array = [] #file to array
    x = 0 # position im testsite_array
    
    time_array = []
    turn = 1 #messweg oder verfahrweg 1 = messweg 0 = verfahrweg
    turns = 1 #anzahl der messwege
    
    startms = 0 #ms zu start der Bewegungsfolge
    lastms = 0 #um zeit zwischen abläufen zu checken
    
    my_dict = {}

    startms = int(round(time.time() * 1000))
    lastms = startms

    time_array.append(0) #start ist bei 0 ms quasi
    tempsave = temp / "temporary_file_gcode.txt"

    with open(tempsave) as my_file:
        testsite_array = my_file.readlines()
    
    Port = "COM4"
    ser = serial.Serial(Port, 250000, timeout = 10)
    print("opened Serial.")
    time.sleep(2)
    
    for a in range(4):
        ser.write(bytes(testsite_array[x], 'utf-8'))
        x = x + 1
    start = 1 # um am anfang 4 ok zu lesen
    
    leseserial = 1
    okcounter = 0
    
    while leseserial == 1:
        
        textreceived = ser.readline()
        with open("SerialData.txt", "a+") as myfile:
            myfile.write(str(textreceived)+"\n") #schreibe serial TimestampOutput in file für debug
        
        if b'SETTINGS'  in textreceived: #START GCODE BAUSTEIN
            print(textreceived)
            splitters = textreceived.split() #put speed and accel in variables and later file
            Speed = int(splitters[1][1:])
            Accel = int(splitters[2][1:])
            Turns = int(splitters[3][1:])

            my_dict['feedrate'] = Speed
            my_dict['acceleration'] = Accel
            my_dict['turns'] = Turns
            
        if b'START'  in textreceived: #START GCODE BAUSTEIN
            print(textreceived)
            
        if b'ok'  in textreceived: # OK = Ausgeführt signal von Controller
            okcounter = okcounter + 1
            
        if b'TX'  in textreceived: #Codewort für M118 Serial Data
            tookms = int(round(time.time() * 1000)) - lastms #MS berechnen für Timestamp
            lastms = int(round(time.time() * 1000))
            
            if(turn == 1 and turns<=my_dict['turns']): #wenn gerade strecke abgefahren wird
                dicttext = "turn"+str(turns)
                my_dict[dicttext] = lastms-startms
                turns = turns + 1
                turn = 0
            else:
                turn = 1
            #time_array.append(lastms-startms) #add timestamp to array
            
            print(textreceived)
            print(tookms)
            
        if b'END'  in textreceived: #END GCODE M118 Baustein
                print(textreceived)
                leseserial = 0
                ser.close()
                print("finished")
        
        if(start == 1 and okcounter>=4 and x<len(testsite_array)): # beim ersten start auf 4 ok warten
            try:
                with open("SerialData.txt", "a") as myfile:
                    myfile.write("BLOCK NEU"+"\n")
                for a in range(3): #immer 3 lines ausführen #quasi 1 block mit G1 M400 M118
                    ser.write(bytes(testsite_array[x], 'utf-8'))
                    x = x + 1
                okcounter = 0
                start = 0
            except:
                print("Error")
                continue
            
        if(start != 1 and okcounter>=3 and x<len(testsite_array)):
            try:
                with open("SerialData.txt", "a") as myfile:
                    myfile.write("BLOCK NEU"+"\n")
                for a in range(3): #immer 3 lines ausführen #quasi 1 block mit G1 M400 M118
                    ser.write(bytes(testsite_array[x], 'utf-8'))
                    x = x + 1
                okcounter = 0
            except:
                print("Error")
                continue
                
    tempsave = os.path.join(temp, "temporary_file_json.txt")       
    with open(tempsave, "w+") as outfile: #create the TimestampOutput file
        outfile.write(json.dumps(my_dict)) # use `json.loads` to do the reverse
    return
#########################################################################################################################################

# function main() that is run by Main.py:
def main():
    # selectable metadata regarding the testobject and testing hardware:
    loaded_objects = ("Sandvik CNMG 12 04 08-MM 2025", "other")
    loaded_machine_tools = ("DOOSAN DNM 500", "other")
    loaded_nozzles = ("Schlick Modellreihe 629", "Schlick Modellreihe 650", "FOS-45°-Flachstrahldüse 25mm Länge", "FOS-Flachstrahldüse 35mm Länge", "FOS-Flachstrahldüse 60mm Länge", "Jet-Düse MV6, Innendurchm. 6mm", "Jet-Düse MV12, Innendurchm. 6mm", "Düse 1200SSF 1/8 verstellbar, Edelstahl", "Silvent MJ4", "Silvent MJ5", "")
    loaded_microphones = ("Renkforce UM-80 USB-Mikrofon", "other")
    loaded_recording_modes = ("Recording by time", "Recording on Karl I")
    
    # selectable audio recording settings:
    loaded_sampling_rates = ("44100", "48000", "88200", "96000")
    loaded_sample_formats = ("16 bit", "24 bit")
    loaded_chunks = ("1024","2048")
    #loaded_channels = ("1", "2")
        
    # header and subheader of the page:
    st.header("Metadata Capture of Audio Files")
    st.write("Name your audio file you want to record and select or type in the right parameters")
    # select metadata regarding the testobject and testing hardware:
    audio_file_name = st.text_input("File name", "example")
    test_object = st.selectbox("Test object", loaded_objects)
    test_object_number = str(st.text_input("Testobjekt-Nummer", "1"))
    test_machine_tool = st.selectbox("Machine tool", loaded_machine_tools)
    test_nozzle = st.selectbox("Nozzle", loaded_nozzles)
    test_microphone = st.selectbox("Microphone", loaded_microphones)
    test_deviation = st.number_input('Insert the measured deviation of the test object [µm]', step=0.01)
    #test_recording_mode = st.selectbox("Recording mode", loaded_recording_modes)
    test_recording_mode = st.radio("Recording mode", loaded_recording_modes)
    
    
            

    if test_recording_mode == "Recording by time":
        st.header("Manual setting of the recording time")
        test_required_recording_time = st.number_input('Insert the recording time [s] (between 0 and 240 seconds)', step=0.01, min_value=0.01, max_value=240.00)   
        mode_path_name="0_recording_by_time" 
    elif test_recording_mode == "Recording on Karl I":
        st.header("Recording with Karl I")
        st.write('the test rig for determining the wear of indexable inserts')
        Infoimage = Image.open('images\info_cutting_parameters.png')
        st.image(Infoimage, caption=None, width=600, height=600)
        test_feed_speed = st.number_input('Feed speed [mm/s]', step=0.01, min_value=0.01, max_value=10000.00)
        test_feed_acceleration = st.number_input('Acceleration [mm/s²]', step=0.01, min_value=0.01, max_value=2000.00)
        test_measuring_path = st.number_input('Measuring path (a) [mm]', step=0.01, min_value=0.01, max_value=1000.00)
        test_offset = st.number_input('Offset (b) [mm]', min_value=0.01, max_value=1000.00)
        test_number_of_measuring_paths = st.slider('Number of measuring paths (n)', 0, 50, 0, 1)
        st.write(test_number_of_measuring_paths, 'Measuring paths per audiofile')
        n = test_number_of_measuring_paths
        ss_p = test_measuring_path    
        ss_o = test_offset
        vs = test_feed_speed 
        acs = test_feed_acceleration
        ta = vs/acs
        sa = 0.5*acs*(ta**2)
        sv_p = ss_p-(2*sa)
        sv_o = ss_o-(2*sa)
        tv_p = sv_p/vs
        tv_o = sv_o/vs
        ts_p = tv_p+(2*ta)
        ts_o = tv_o+(2*ta)    
        ttotal = n*ts_p+(n-1)*ts_o
        if ttotal <= 0:
            ttotal = 0
        ttolerance_fix = st.number_input('Specify a one-time tolerance time value. [s]')
        ttolerance_per_path = st.number_input('Specify a tolerance value per path. [s]', step=0.01)
        ttotal_with_tolerance = ttotal + ttolerance_fix + n*ttolerance_per_path
        
        st.write('The calculated recording time including tolerances counts', ttotal_with_tolerance, 'seconds')
        test_required_recording_time = ttotal_with_tolerance
        mode_path_name="1_recording_on_karl_I"

    # select audio recording settings:
    
    audio_file_name = f"{audio_file_name}_objectnr_{test_object_number}.wav"    
    test_sampling_rate = int(st.selectbox('Sampling rate of the recording in kHz', loaded_sampling_rates))
    selected_test_sample_format = st.selectbox('Sample format of the recording', loaded_sample_formats)
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
    npersegs=4096
    noverlaps = npersegs/2
    
    st.write('Check if everything is filled in correctly and save your metadata by pressing the "Save" button below.')

    start_recording = st.button("Start recording")
   
    
    # 
    if start_recording == True:  
        if test_recording_mode == "Recording by time":
            temp = Path(f"audio-files/{mode_path_name}/0_temporary_files")                                    # determining the path for temporary files
            temp.mkdir(parents=True, exist_ok=True)                                         # creating the path if it doesn't exist yet
            temporary_file_path = temp / "temporary_file.wav"                               # path and name for the temporary recording file
        elif test_recording_mode == "Recording on Karl I":
            temp = Path(f"audio-files/{mode_path_name}/0_temporary_files")                                    # determining the path for temporary files
            temp.mkdir(parents=True, exist_ok=True)                                         # creating the path if it doesn't exist yet
#########################################################################################################################################
            #HEIKO#################################################
            #EINSTELLPARAMETER################################
            #Parameter
            x_length = ss_p
            zwischenabstand = ss_o
            speed = int(vs) #in mm/s und ganze zahlen
            accel = int(acs) #in mm/s2
            turns = int(n) # nur gerade Zahlen, damit auch zurückkommt
            ##################################################    
            #Startpunkt
            x_start = 5
            y_start = 5
            z_start = 5
            ##################################################
            #maximale Werte
            speed_max = 500 #in mm/min
            x_max = 120
            y_max = 130
            z_max = 100
            ##################################################        
            #Arbeitswerte
            posX = x_start
            posY = y_start
            posZ = z_start
            #Datei
            G_codefile = ""    
            #Start G-Code
            G_codefile = G_codefile + "G28" + "\n"
            G_codefile = G_codefile + "M204 T" + str(accel) + "\n"
            G_codefile = G_codefile + "M118 SETTINGS F"+ str(speed) + " A" + str(accel) + " T" + str(turns) + "\n"
            G_codefile = G_codefile + "M118 START" + "\n"
            text = "G1"+" "+"X"+str(posX)+" "+"Y"+str(posY)+" "+"Z"+str(posZ)+" "+"F"+str(speed*60)+ "\n"
            text = text + "M400" + "\n"
            text = text + "M118"+" TX: "+"X"+str(posX)+" "+"Y"+str(posY)+" "+"Z"+str(posZ)+" "+ "\n"
            G_codefile = G_codefile + text
            #RestG-Code erstellen
            i = 0
            posneg = "+" #wenn zuvor positiv verfahren muss nun negativ
            while i <= turns-1:
                i = i+1
                if(posneg == "+"):
                    posX = posX + x_length #X+ Verfahren
                    text = "G1"+" "+"X"+str(posX)+" "+"Y"+str(posY)+" "+"Z"+str(posZ)+" "+"F"+str(speed*60)+ "\n"
                    text = text + "M400" + "\n"
                    text = text + "M118"+" TX: "+"X"+str(posX)+" "+"Y"+str(posY)+" "+"Z"+str(posZ)+" "+ "\n"
                    G_codefile = G_codefile + text
                    #print("plus")
                    if(i<=turns-1): #nur zwischenabstand fahren wenn 2 weniger als turns, damit unnötiger zwischenabstand nicht verfahren wird
                        posY = posY + zwischenabstand #Y Verfahren
                        text = "G1"+" "+"X"+str(posX)+" "+"Y"+str(posY)+" "+"Z"+str(posZ)+" "+"F"+str(speed*60)+ "\n"
                        text = text + "M400" + "\n"
                        text = text + "M118"+" TX: "+"X"+str(posX)+" "+"Y"+str(posY)+" "+"Z"+str(posZ)+" "+ "\n"
                        G_codefile = G_codefile + text
                        #print("zwischen")
                    posneg = "-"
                
                elif(posneg == "-"):
                    posX = posX - x_length #X- Verfahren
                    text = "G1"+" "+"X"+str(posX)+" "+"Y"+str(posY)+" "+"Z"+str(posZ)+" "+"F"+str(speed*60)+ "\n"
                    text = text + "M400" + "\n"
                    text = text + "M118"+" TX: "+"X"+str(posX)+" "+"Y"+str(posY)+" "+"Z"+str(posZ)+" "+ "\n"
                    G_codefile = G_codefile + text
                    #print("minus")
                    if(i<=turns-1):
                        posY = posY + zwischenabstand #Y Verfahren
                        text = "G1"+" "+"X"+str(posX)+" "+"Y"+str(posY)+" "+"Z"+str(posZ)+" "+"F"+str(speed*60)+ "\n"
                        text = text + "M400" + "\n"
                        text = text + "M118"+" TX: "+"X"+str(posX)+" "+"Y"+str(posY)+" "+"Z"+str(posZ)+" "+ "\n"
                        G_codefile = G_codefile + text
                        #print("zwischen")
                    posneg = "+"
                else:
                    continue
            #auf Startpunkt zurückfahren
            posX = x_start
            posY = y_start
            posZ = z_start
            text = "G1"+" "+"X"+str(posX)+" "+"Y"+str(posY)+" "+"Z"+str(posZ)+" "+"F"+str(speed*60)+ "\n"
            text = text + "M400" + "\n"
            text = text + "M118"+" TX: "+"X"+str(posX)+" "+"Y"+str(posY)+" "+"Z"+str(posZ)+" "+ "\n"
            G_codefile = G_codefile + text
            
            #End G-Code
            G_codefile = G_codefile + "M118 END" + "\n"
            G_codefile = G_codefile + "M84" + "\n"
            
            #write to file
            tempsave = os.path.join(temp, "temporary_file_gcode.g")
            f = open(tempsave, "w")
            f.write(G_codefile)
            f.close()
            ##################################################
            ##################################################
            
            #START COMMUNICATION
            jobs = []
            p = multiprocessing.Process(target=worker)
            jobs.append(p)
            p.start()
            ##################################################
            ##################################################
#########################################################################################################################################  
        
            temporary_file_path = temp / "temporary_file.wav" 

        own_functions.record(test_sample_format, 1, test_sampling_rate, test_chunk, test_required_recording_time, str(temporary_file_path))  # recording function that is defined in Aufnahme.py
        
        #sample_rate, samples = wavfile.read(temporary_file_path)                        # opening the temporary audio file that was just recorded before
        samples, sample_rate = librosa.load(temporary_file_path, sr=test_sampling_rate)


        
        
        st.write('sample rate:', sample_rate)
        
        

        frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, nperseg=npersegs, noverlap=noverlaps)       # extracting frequencies, times and amplitdues from the wav file
        
        
    
        st.write("Spectrogram logarithmic scaling")
        temporary_spectrogram_fig=plt.figure()
        plt.pcolormesh(times, frequencies, spectrogram)                                 # creating the spectogram from the arrays of times, frequencies and spectrogram where amplitudes are expressed as colours
        plt.ylim(1, (sample_rate/2))                                                              # setting the scale of the frequencies axis from 1 to 20000Hz
        plt.yscale("log")                                                               # logarithmizing the frequencies axis of the spectrogram
        plt.ylabel('Frequency [Hz]')                                                    # labeling the axes
        plt.xlabel('Time [sec]')
        #plt.z
        st.pyplot(temporary_spectrogram_fig)                                            # visualization of the spectrogram on Streamlit
        

        # st.write("Spectrogram linear scaling")
        # temporary_spectrogram_fig=plt.figure()
        # plt.pcolormesh(times, frequencies, spectrogram)                                 # creating the spectogram from the arrays of times, frequencies and spectrogram where amplitudes are expressed as colours
        # plt.ylim(1, (sample_rate/2))                                                              # setting the scale of the frequencies axis from 1 to 20000Hz                                                             
        # plt.ylabel('Frequency [Hz]')                                                    # labeling the axes
        # plt.xlabel('Time [sec]')
        # #plt.z
        # st.pyplot(temporary_spectrogram_fig)  

        # st.write("Welch-Diagram")
        # sample_fr, powerspectraldensity = signal.welch(samples, sample_rate)
        # temporary_welch=plt.figure()
        # plt.semilogy(sample_fr, np.sqrt(powerspectraldensity))
        # plt.xlim(1,20000)
        # plt.xscale("log")
        # plt.xlabel('frequency [Hz]')
        # plt.ylabel('Linear Spectrum')
        # st.pyplot(temporary_welch)
       
        # spectrogramsimon=plt.figure(figsize=(14,5))                      #Code von Simon
        # #Develop 't' array to match x
        # t = np.array(range(0, len(samples))) / sample_rate 
        # #plt.plot(t[0:1000], x[0:1000])
        # plt.title('Spectrogram of DL 7bar 02mm')
        # plt.plot(t, samples)
        # plt.xlabel('time in seconds')
        # plt.ylabel('Pressure')
        # st.pyplot(spectrogramsimon)

        if test_recording_mode == "Recording by time":
            st.audio(f"audio-files/{mode_path_name}/0_temporary_files    emporary_file.wav", format="audio/wav") # opening the last recorded wav file in a player on Streamlit
        elif test_recording_mode == "Recording on Karl I":
            st.audio(f"audio-files/{mode_path_name}/0_temporary_files    emporary_file.wav", format="audio/wav")   
        

    savefile = st.button("Save")
    #temp = Path(f"audio-files/{mode_path_name}/0_temporary_files") 
    #temporary_file_path = temp / "temporary_file.wav" 
    if savefile == True:  
        
        old_temp_path = Path(f"audio-files/{mode_path_name}/0_temporary_files")         #0_recording_by_time
        old_temp_path_file = old_temp_path / "temporary_file.wav"
        old_temp_path_gcode_file = old_temp_path / "temporary_file_gcode.g"
        old_temp_path_json_file = old_temp_path / "temporary_file_json.txt"

        if os.path.exists(old_temp_path_file):

            new_path = Path(f"audio-files/{mode_path_name}/{test_object}/{test_nozzle}")
            new_path.mkdir(parents=True, exist_ok=True)
            new_path_file = new_path/audio_file_name
            new_path_file = own_functions.uniquify(new_path_file)
            
            new_path_gcode_file = new_path / "temporary_file_gcode.g"
            new_path_json_file = new_path / "temporary_file_json.txt"
            
            shutil.copyfile(old_temp_path_file, new_path_file)
            
            #forhieko
            shutil.copyfile(old_temp_path_gcode_file, new_path_gcode_file)
            shutil.copyfile(old_temp_path_json_file, new_path_json_file)

            os.remove(old_temp_path_file)

            audio_file_meta = {"file": audio_file_name,
            #"optically measured deviation": test_deviation,
            "microphone": test_microphone,
            "sampling rate": test_sampling_rate,
            "object": test_object,
            "nozzle": test_nozzle,
            #"outlet pressure": test_outlet_pressure,
            "machine tool": test_machine_tool
            #"feed speed": test_feed_speed
            }

            with open(f'{str(new_path_file)}.json', 'w') as json_file:
                json.dump(audio_file_meta, json_file)
            st.write("Audio file and metadata file for " + audio_file_name + "  were saved.")
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
        
