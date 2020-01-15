import cv2
import os
import numpy as np

def colorExtract():
    cap = cv2.VideoCapture(0)
    
    while True:
        _,frame= cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        lowerBlue=np.array([110,50,50])
        upperBlue=np.array([130,255,255])
        
        lowerRed=np.array([0,50,50])
        upperRed=np.array([10,255,255])
        
        lowerGreen=np.array([50,50,50])
        upperGreen=np.array([70,255,255])
        
        # Threshold the HSV image to get only blue colors
        maskBlue = cv2.inRange(hsv, lowerBlue, upperBlue)
        maskRed = cv2.inRange(hsv, lowerRed, upperRed)
        maskGreen = cv2.inRange(hsv, lowerGreen, upperGreen)
        
        # Bitwise-AND mask and original image
        resBlue = cv2.bitwise_and(frame,frame, mask= maskBlue)
        resRed = cv2.bitwise_and(frame,frame, mask= maskRed)
        resGreen = cv2.bitwise_and(frame,frame, mask= maskGreen)
        cv2.imshow('frame',frame)
        cv2.imshow('resBlue',resBlue)
        cv2.imshow('resRed',resRed)
        cv2.imshow('resGreen',resGreen)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    
    cv2.destroyAllWindows()
    
if __name__=="__main__":
    colorExtract()