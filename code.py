#!/usr/bin/python3
import numpy as np
import cv2
import time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

global number_of_times
global number_flag
global global_time
global timer_count
cap = cv2.VideoCapture(0)
number_of_times = 0
number_flag = 0
global_time = 0
timer_count = 0

while 1:

    #global_time = time.time()
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #HERE FACE WILL BE DETECTED

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2) #255 FOR BLUE RECT AND 2 FOR LINE WIDTH
        roi_gray = gray[y:y+h, x:x+w] #REEASON OF IMAGE starting point and ending point
        roi_color = img[y:y+h, x:x+w] #its coloured image not grey

        eyes = eye_cascade.detectMultiScale(roi_gray,scaleFactor=1.3, minNeighbors=4, minSize=(70, 70))
        for (ex,ey,ew,eh) in eyes:
            global_time = time.time()
            #print global_time
            eye_img = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_img_c = roi_color[ey:ey+eh, ex:ex+ew]
            cv2.imshow('face',eye_img)
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) #drawing eye in roi color , #color is green
            circles = cv2.HoughCircles(eye_img,cv2.HOUGH_GRADIENT,1,5)
            #timer_count = timer_count+1
            time_elapsed = time.time() - global_time
            timer_count = timer_count + time_elapsed
            print(timer_count*100) #*time_elapsed
            if circles is None:
                continue
            for i in circles[0,:]:
                   # draw the outer circle
                   cv2.circle(eye_img_c,(i[0],i[1]),i[2],(0,255,0),2)
                   # draw the center of the circle
                   cv2.circle(eye_img_c,(i[0],i[1]),2,(0,0,255),3)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
