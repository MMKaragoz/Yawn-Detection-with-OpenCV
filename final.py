
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



def alarm(msg):
    global alarm_status
    global alarm_status2
    global saying

    while alarm_status:
        print('call')
        s = 'espeak "' + msg + '"'
        os.system(s)

    if alarm_status2:
        print('call')
        saying = True
        s = 'espeak "' + msg + '"'
        os.system(s)
        saying = False

# is used to compute the ratio of distances between the horizontal eye landmarks and the distances between the vertical eye landmarks.
# from Real-Time Eye Blink Detection using Facial Landmarks(article)
# in order for more details in article , http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf

def eye_aspect_ratio(eye): 
    A = dist.euclidean(eye[1], eye[5])  # 2 - 6 
    B = dist.euclidean(eye[2], eye[4])  # 3 - 5

    C = dist.euclidean(eye[0], eye[3])  # 1 - 4

    ear = (A + B) / (2.0 * C)

    return ear


def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

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

# initializing
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

BLINK_COUNTER = 0
WITHOUT_BLINK_COUNTER = 0
TOTAL_BLINK = 0
ALARM_ON = False
tot = 0
oldx = 0
oldy = 0
count_head = 0
EYE_AR_CONSEC_FRAMES_BLINK = 2


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

    #rects = detector(gray, 0)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, #Detects objects of different sizes in the input image.
		minNeighbors=5, minSize=(30, 30), #The detected objects are returned as a list of rectangles.
		flags=cv2.CASCADE_SCALE_IMAGE)

    if len(rects) == 0:
        if not ALARM_ON:
            ALARM_ON = True
        
        
               
        cv2.putText(frame, "NO FACES DETECTED!", (10, 200),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    #for rect in rects:
    # https://www.pyimagesearch.com/2017/05/08/drowsiness-detection-opencv/
    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))
        
        x1 = int((x+w)/2)
        y1 = int((y+h)/2)
        tot = abs(oldx - x1) + abs(oldy - y1)
        tot += tot
        xdif = abs(oldx - x1)
        ydif = abs(oldy - y1)

        if (xdif < 2 and ydif < 2):
                count_head = count_head + 1
                print("count_head: ", count_head)


        if (count_head > 100):
            rint("x dif", abs(oldx - x1), "y dif", abs(oldy - y1), "total :", tot)
            oldx = x1
            oldy = y1    
            
            if  ((xdif > 2) or (ydif > 2)):


                    print("DROWSY ALERTTTTTT!!!!!!!!!!!!")
                    print("DROWSY ALERTTTTTT!!!!!!!!!!!!")
                    print("DROWSY ALERTTTTTT!!!!!!!!!!!!")
                    print("DROWSY ALERTTTTTT!!!!!!!!!!!!")
                    print("DROWSY ALERTTTTTT!!!!!!!!!!!!")

                    count_head = 0
                    i = 0
                    while (i<5000):
                        cv2.putText(frame, "BE CAREFUL!!!!", (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        i = i+1      
                        
                        
                        
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        #is used function finar_ear
        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye [1]
        rightEye = eye[2]

        distance = lip_distance(shape)

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (255, 0, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (255, 0, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (255, 255, 255), 1)

        if ear < EYE_AR_THRESH:
            EAR_COUNTER += 1
            BLINK_COUNTER += 1
            if EAR_COUNTER >= EYE_AR_CONSEC_FRAMES:
                if alarm_status == False:
                     alarm_status = True
                     time_4 = time.clock()
                    
                

                     cv2.putText(frame, "DROWSINESS ALERT!", (10, 180),
                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            EAR_COUNTER = 0
            alarm_status = False
            
            WITHOUT_BLINK_COUNTER += 1
            if BLINK_COUNTER >= EYE_AR_CONSEC_FRAMES_BLINK:
                TOTAL_BLINK += 1
                WITHOUT_BLINK_COUNTER = 0
                if TOTAL_BLINK >=5:
                    if not ALARM_ON:
                        ALARM_ON = True

            
                        # draw an alarm on the frame
                        cv2.putText(frame, "DROWSINESS ALERT FOR BLINK!", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if TOTAL_BLINK >=10:
                        TOTAL_BLINK = 0
            
            if WITHOUT_BLINK_COUNTER >= 15:
                TOTAL_BLINK = 0
                
            BLINK_COUNTER = 0
            ALARM_ON = False
            
            # draw the computed eye aspect ratio on the frame to help
        # with debugging and setting the correct eye aspect ratio
        # thresholds and frame counters
        cv2.putText(frame, "Blinks: {}".format(TOTAL_BLINK), (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "BLINK_COUNTER: {}".format(BLINK_COUNTER), (25, 90),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "WITHOUT_BLINK_COUNTER: {}".format(WITHOUT_BLINK_COUNTER), (40, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        
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
    
        if alarm_status == True and (time.clock() - time_4) <= 2:
            duration = 1000
            frequence = 440
            winsound.Beep(frequence , duration)                  
        
        
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "YAWN: {:.2f}".format(distance), (300, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
