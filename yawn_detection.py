
# %% libraries
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
import winsound

# %% main code

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance



ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
args = vars(ap.parse_args())


EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 24
alarm_status = False
alarm_status2 = False
saying = False
EAR_COUNTER = 0
YAWN_COUNTER = 0
YAWN_TIME = 5
TIME = 20 # total time
warning_duration = 3
time_1 = 0
time_2 = 0
time_3 = 0

print("-> Loading the predictor and detector...")
#detector = dlib.get_frontal_face_detector()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    #Faster but less accurate

# This object is a tool that takes in an image region containing some object and outputs a set of point locations that define the pose of the object.
# The classic example of this is human face pose prediction, where you take an image of a human face as input and are expected 
# to identify the locations of important facial landmarks such as the corners of the mouth and eyes, tip of the nose, and so forth.
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)


while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, #Detects objects of different sizes in the input image.
		minNeighbors=5, minSize=(30, 30), #The detected objects are returned as a list of rectangles.
		flags=cv2.CASCADE_SCALE_IMAGE)
    
    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
        
    
        #rects = detector(gray, 0)
        rects = detector.detectMultiScale(gray, scaleFactor=1.1, #Detects objects of different sizes in the input image.
		minNeighbors=5, minSize=(30, 30), #The detected objects are returned as a list of rectangles.
		flags=cv2.CASCADE_SCALE_IMAGE)
    
    
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
    
        distance = lip_distance(shape)
    
        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (255, 255, 255), 1)
    
        if (distance > YAWN_THRESH):
            
            if YAWN_COUNTER <= 0 :
                YAWN_COUNTER += 1
                time_1 = time.clock()
                
                
            elif YAWN_COUNTER <= 1 and (time.clock() - time_1) >= YAWN_TIME:
                if time.clock() - time_1 > TIME:
                    YAWN_COUNTER = 1
                    time_1 = time.clock()
                    
                else:
                    YAWN_COUNTER += 1
                    time_2 = time.clock()
                    
                    
            elif YAWN_COUNTER >= 2:
                if (time.clock() - time_2) >= YAWN_TIME:
                    if time.clock() - time_1 < TIME:
                        
                        YAWN_COUNTER = 3
                        
                        time_3 = time.clock()
                        if alarm_status2 == False and saying == False:
                            alarm_status2 = True
                            
                            
        
                    
                    elif time.clock() - time_1 >= TIME:
                        time_3 = time.clock()
                        if time.clock()- time_2 >= TIME:
                            YAWN_COUNTER = 1
                            time_1 = time_3
                        
                        else :
                            YAWN_COUNTER = 2
                            time_1 = time_2
                            time_2 = time_3 
               
                
            
        print("yawn counter :",YAWN_COUNTER)
        
        if YAWN_COUNTER == 1:
                cv2.putText(frame, "Yawn Alert1", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if time.clock() - time_1 >= TIME:
                    YAWN_COUNTER = 0
                    
                
                
        elif YAWN_COUNTER == 2:
                cv2.putText(frame, "Yawn Alert2", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if (time.clock() - time_2) >= TIME:
                    YAWN_COUNTER = 1
                    time_1 = time.clock()
                
        elif YAWN_COUNTER == 3 :
                cv2.putText(frame, "Third Yawn. I guess You're SLEEPY", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
                
                if (time.clock() - time_3) <= warning_duration :
                    duration = 800
                    frequence = 440
                    winsound.Beep(frequence , duration)
                
                if time.clock() - time_3 >= YAWN_TIME:
                    YAWN_COUNTER = 1
                    time_1 = time.clock()
            
            
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        


    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()












