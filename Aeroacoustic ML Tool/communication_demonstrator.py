# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 09:14:23 2020

@author: Heiko
"""

import serial
import time
import json
    
testsite_array = [] #file to array
x = 0 # position im testsite_array

time_array = []
turn = 1 #messweg oder verfahrweg 1 = messweg 0 = verfahrweg
turns = 1 #anzahl der messwege

startms = 0 #ms zu start der Bewegungsfolge
lastms = 0 #um zeit zwischen abläufen zu checken

my_dict = {}
##for calculation of abweichung soll-ist
#Xnow = 0
#Xlast = 0
#Ynow = 0
#Ylast = 0
#Znow = 0
#Zlast = 0

if __name__ == '__main__':
    startms = int(round(time.time() * 1000))
    lastms = startms

    time_array.append(0) #start ist bei 0 ms quasi
    
    with open('audio-files/1_recording_on_karl_I/abfahren.g') as my_file:
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
            
#            splitters = textreceived.split()
#            Xnow = int(splitters[1][1:])
#            Ynow = int(splitters[2][1:])
#            Znow = int(splitters[3][1:])
#            
#            Xweg = Xnow - Xlast
#            Yweg = Ynow - Ylast
#            Zweg = Znow - Zlast
#            
#            Xlast = Xnow
#            Ylast = Ynow
#            Zlast = Znow
            
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
            
    with open("outfile.txt", "w+") as outfile: #create the TimestampOutput file
        #outfile.write("\n".join(str(item) for item in time_array))
        outfile.write(json.dumps(my_dict)) # use `json.loads` to do the reverse
        
        
        
        
        
        
        
        
        
        
        