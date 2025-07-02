
"""
Created on Tue May 16 14:27:31 2023
    


*****************
NB: BEFORE RUNNING!!! 
* MAKE SURE YOU HAVE ALL THE LIBRARIES ON YOUR ENVIRONMENT 
* TURN ON PYTHON CODE BEFORE JavaScript CODE
*****************


@author: yannaelB
"""

import os
import time 
import asyncio as asy
import websockets 
import serial
from datetime import datetime



# Chose your eye tracker device's (well in fact you don't really have choice for now)
if False: #IS_EYELINK ? Put True if is it
    from eyelink_functions import init_eyelink, end_eyelink, end_recording, start_recording
    from pylink import getEYELINK
elif True: #IS_TOBII ? Put True if is it
    import tobii_research as tobii #Be carefull with this library, it requires python=3.10 version. Otherwise it probably won't work. If you have an other version and don't want to lose it, you can create an anaconda environnement (cf https://www.pythoniste.fr/anaconda/les-environnements-virtuels-en-python-avec-anaconda/)
    import time



def synchro(session_index):   

    if True: #IS_TOBII ? Put True if is it
        alltobii = tobii.find_all_eyetrackers()
        mytobii = alltobii[0] # Make sure that you have only one eyetracker connected to your computer, otherwise it will be ambiguous
        
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")

        eyetracking_outputfilename = os.path.join('Tobii_data', ("TXTFile_tobii_subject_" + str(subject) + "_run_" + str(session_index) + "_" + formatted_time + ".txt"))
        eyetracking_outputfile = open(eyetracking_outputfilename, "w")
        eyetracking_outputfile.write("time flow" + " \t " + "RE(x,y)" + " \t " + "LE(x,y)" + "\t diam (RE,LE)" 
                +"\t " + "\n") 
        

    def what_tobii_records(flux_oculo,):
        global timeflow
        timeflow        = flux_oculo['device_time_stamp']
        coord_RE        = flux_oculo['right_gaze_point_on_display_area']
        coord_LE        = flux_oculo['left_gaze_point_on_display_area']
        diam_RE_pupil   = flux_oculo['right_pupil_diameter']
        diam_LE_pupil   = flux_oculo['left_pupil_diameter']            
        
        eyetracking_outputfile.write(str(timeflow) +" \t "+ str(coord_RE) +" \t "+ str(coord_LE) +" \t "
                            + str(diam_RE_pupil) +" \t "+ str(diam_LE_pupil)+ "\t "+ "\n")

    mytobii.subscribe_to(tobii.EYETRACKER_GAZE_DATA, what_tobii_records, as_dictionary=True) #start data capturing ,

    number_itera = 0
    return mytobii,what_tobii_records,eyetracking_outputfile, number_itera
    
    

    # Writting information from the experience on the webpage into the txt.file
def Write_information_from_expe(eyetracking_outputfile,message,timeflow):
    eyetracking_outputfile.write(str(timeflow)+ " MSG: Received : " + str(message) + "\n")


    # Stop the eye-tracker and save data
def Stop_recording(mytobii,eyetracking_outputfile,what_tobii_records,timeflow):
    if False: #IS_EYELINK ? Put True if is it
        end_recording()
        end_eyelink(exp, edfFileName)
    elif True: #IS_TOBII ? Put True if is it
        eyetracking_outputfile.write(str(timeflow)+" MSG: End\n")
        time.sleep(0.1) 
        mytobii.unsubscribe_from(tobii.EYETRACKER_GAZE_DATA, what_tobii_records)
        print("Tobii's job done")


# =============================================================================
# MAIN FUNCTION : COMMUNICATION BETWEEN PYTHON and WEBSITE
# =============================================================================


#This code allow to create a WebSocket (kind of a private local servor for making possible the communication between Web (Cedric's experiences) and Python (EyeTracker)))
#It's important to create the websocket at first, before every other task. Otherwise the communication is impossible. (here is the sources from where i got inspired : https://www.youtube.com/watch?v=lv0oEnQY1pM)

def start_websocket_server():
    global event_loop
    event_loop = asy.get_event_loop() # Get the asyncio event loop
    start_server = websockets.serve(server,"localhost",5000) # Start the WebSocket server using the websockets.serve function
    asy.get_event_loop().run_until_complete(start_server) # Asynchronously run the WebSocket server creation
    asy.get_event_loop().run_forever() # Run the asyncio event loop indefinitely

if __name__ == '__main__':
    subject = int(input("Number of the subject : ")) #inter the number of the subject. needed for saving files correctly
    async def server(websocket, path):
        global session_index
        session_index = 0
        async for message in websocket:
            try:
                await websocket.send(f'test message: {message}')
                # messages are coming from the webpage Ada Prob

                # Turn on  Tobii 
                if message == 'synchronisation': 
                    session_index += 1
                    mytobii,what_tobii_records ,eyetracking_outputfile,number_itera = synchro(session_index)
                    time.sleep(1)
                    eyetracking_outputfile.write(str(timeflow)+" MSG: Session starts" + "\n")
                    print(f"trig for sessions {session_index} starts")

                # Add a line in the txt file informing the beginning of the experience
                elif message == "session starts":
                    eyetracking_outputfile.write(str(timeflow)+" MSG: Displaying Introduction" + "\n")

                # Add a line in the txt file informing the end of the experience and stop recording
                elif message == 'session ends':
                    eyetracking_outputfile.write(str(timeflow)+" MSG: Session ends" + "\n")                    
                    time.sleep(1)
                    Stop_recording(mytobii,eyetracking_outputfile,what_tobii_records,timeflow)

                # At the end of a session, stop recording with Tobii and close/save the txt file
                elif message == "end session and stop recording":
                    Stop_recording(mytobii,eyetracking_outputfile,what_tobii_records,timeflow)
                    # => stop recording and start recording again when clicking                 
                    
                # When the subject click on "i'm okay", start recording and open new txt file 
                elif message == "new session : start recording and open new txt file":
                    session_index += 1 
                    mytobii,what_tobii_records ,eyetracking_outputfile,number_itera = synchro(session_index)
                    time.sleep(1)
                    eyetracking_outputfile.write(str(timeflow)+" MSG: Displaying Introduction" + "\n")

                # stop recording with Tobii
                elif message == "stop_recording":
                    Stop_recording(mytobii,eyetracking_outputfile,what_tobii_records,timeflow)
                    
                # Close the websocket so the communication between python and the webpage
                elif message == "close_websocket":
                    print(" here in close_websocket : CLOSE WEBSOCKET")
                    await websocket.close()
                    print("close the websocket")
                    event_loop.stop()
                    break
                
                elif message == "stim_1":
                    Write_information_from_expe(eyetracking_outputfile,message,timeflow)

                elif message == "stim_0":
                    Write_information_from_expe(eyetracking_outputfile,message,timeflow)

                # Add information in the txt file about Ada prob experiment's states
                else:
                    print("message reçu = ", message )
                    Write_information_from_expe(eyetracking_outputfile,message,timeflow)
                    print({message})



            except websockets.exceptions.ConnectionClosedOK:
                print("Connexion WebSocket fermée")

    start_websocket_server()

            
