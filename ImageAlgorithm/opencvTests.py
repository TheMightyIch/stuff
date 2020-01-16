import cv2
import os
from matplotlib import pyplot as plt
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
    
def filterImg():
    img=cv2.imread("1.jpg")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    edge= cv2.Canny(gray,50,150,apertureSize = 3)
    kernel = np.ones((5,5),np.uint8)
    edge=cv2.morphologyEx(edge, cv2.MORPH_DILATE,kernel)
    ret,thresh = cv2.threshold(edge,127,255,0)
    contours,hierarchy = cv2.findContours(thresh, 2, 2)
    cnt = contours[0]
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    img = cv2.drawContours(img,[box],0,(0,0,255),2)
    # minLineLength = 50
    # maxLineGap = 20
    # lines = cv2.HoughLinesP(edge,1,np.pi/180,100,minLineLength,maxLineGap)
    # for x1,y1,x2,y2 in lines[0]:
    #     cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

    
    noiseFilter=cv2.morphologyEx(edge,cv2.MORPH_OPEN,kernel)    
    plt.subplot(121),plt.imshow(img)
    plt.subplot(122),plt.imshow(edge)
    #plt.subplot(122),plt.imshow(thresh, cmap='gray')
    
    plt.show()
    
def ORB_detector():
    img = cv2.imread('handyCam.jpg',0)

    # Initiate STAR detector
    orb = cv2.ORB_create()
    orb.getDefaultName()
    # find the keypoints with ORB
    kp = orb.detect(img,None)
    
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    print(type(kp[0]),des)
    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(img,kp,None,color=(0,255,0), flags=0)
    plt.subplot(121), plt.imshow(img2)
    plt.show()
if __name__=="__main__":
    ORB_detector()