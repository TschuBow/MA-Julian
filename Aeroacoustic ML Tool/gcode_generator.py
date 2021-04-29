##################################################
#EINSTELLPARAMETER################################

#Parameter
x_length = 0
print('Enter xlength')
try:
  x_length = int(input())
except:
  print("An exception occurred")
  x_length = 50

zwischenabstand = 0
print('Enter zwischenabstand')
try:
  zwischenabstand = int(input())
except:
  print("An exception occurred")
  zwischenabstand = 5

speed = 0 #in mm/s und ganze zahlen
print('Enter speed')
try:
  speed = int(input())
except:
  print("An exception occurred")
  speed = 8

accel = 0 #in mm/s2
print('Enter accel')
try:
  accel = int(input())
except:
  print("An exception occurred")
  accel = 25

turns = 0 # nur gerade Zahlen, damit auch zurückkommt
print('Enter turns')
try:
  turns = int(input())
except:
  print("An exception occurred")
  turns = 6
  
##################################################

#Startpunkt
x_start = 5
y_start = 5
z_start = 5

##################################################
##################################################

#maximale Werte
speed_max = 500 #in mm/min
x_max = 120
y_max = 130
z_max = 100

#Arbeitswerte
posX = x_start
posY = y_start
posZ = z_start

#Datei
G_codefile = ""

#-----------------
#++++++++++++++++-
#-----------------
#-++++++++++++++++
#-----------------
#++++++++++++++++-
#-----------------

print("Verfahrweg in Y: ",turns*zwischenabstand+y_start)

import math

#Fehlerabfrage
if((x_length + x_start) > x_max):
    print("Error: Weg in X ist zu groß.")
if(z_start > z_max):
    print("Error: Z ist zu groß.")
if(speed > speed_max/60):
    print("Error: Speed ist zu hoch. Speed nur zwischen 0 und ", speed_max/60,"mm/s")
if((y_start+turns*zwischenabstand)>y_max):
    print("Error! Zu viele turns.")
	
def G1_command():
    global G_codefile
    text = "G1"+" "+"X"+str(posX)+" "+"Y"+str(posY)+" "+"Z"+str(posZ)+" "+"F"+str(speed*60)+ "\n"
    text = text + "M400" + "\n"
    text = text + "M118"+" TX: "+"X"+str(posX)+" "+"Y"+str(posY)+" "+"Z"+str(posZ)+" "+ "\n"
    G_codefile = G_codefile + text

#Start G-Code
G_codefile = G_codefile + "G28" + "\n"
G_codefile = G_codefile + "M204 T" + str(accel) + "\n"
G_codefile = G_codefile + "M118 SETTINGS F"+ str(speed) + " A" + str(accel) + " T" + str(turns) + "\n"
G_codefile = G_codefile + "M118 START" + "\n"

G1_command()

i = 0
posneg = "+" #wenn zuvor positiv verfahren muss nun negativ
while i <= turns-1:
    i = i+1
    if(posneg == "+"):
        posX = posX + x_length #X+ Verfahren
        G1_command()
        #print("plus")
        if(i<=turns-1): #nur zwischenabstand fahren wenn 2 weniger als turns, damit unnötiger zwischenabstand nicht verfahren wird
            posY = posY + zwischenabstand #Y Verfahren
            G1_command() 
            #print("zwischen")
        posneg = "-"
    
    elif(posneg == "-"):
        posX = posX - x_length #X- Verfahren
        G1_command()
        #print("minus")
        if(i<=turns-1):
            posY = posY + zwischenabstand #Y Verfahren
            G1_command()
            #print("zwischen")
        posneg = "+"
    else:
        continue
#auf Startpunkt zurückfahren
posX = x_start
posY = y_start
posZ = z_start
G1_command()

#End G-Code
G_codefile = G_codefile + "M118 END" + "\n"
G_codefile = G_codefile + "M84" + "\n"

#write to file
f = open("abfahren.g", "w")
f.write(G_codefile)
f.close()

#tell me I am ready
print("code finished")