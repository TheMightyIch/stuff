import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
import ImageWarping as iw

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
    img=cv2.imread("Simetic.jpg")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    row,col = gray.shape
    M= cv2.getRotationMatrix2D((col/2, row/2),0,1)
    gray=img=cv2.warpAffine(gray,M,(col,row))
    
    edge= cv2.Canny(gray,50,150,apertureSize = 3)
    kernel = np.ones((5,5),np.uint8)
    edge=cv2.morphologyEx(edge, cv2.MORPH_DILATE,kernel)
    ret,thresh = cv2.threshold(edge,127,255,0)
    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    our_cnt=None
    for idx in contours:
        rect = cv2.minAreaRect(idx)
        box = cv2.boxPoints(rect)
        
        box = np.int0(box)
        [cv2.circle(img, (p[0][0], p[0][1]), 10,(0,255,0)) for p in idx]

        peri = cv2.arcLength(idx, True)
        approx = cv2.approxPolyDP(idx, 0.028 * peri, True)
        if len(approx) == 4:
            our_cnt = approx
            break
    our_cnt=[x[:][0] for x in our_cnt]
    our_cnt=np.int0(our_cnt)
    img = cv2.drawContours(img,[our_cnt],0,(255,0,0),2)
    imgWarped=iw.four_point_transform(img, our_cnt)
    # minLineLength = 50
    # maxLineGap = 20
    # lines = cv2.HoughLinesP(edge,1,np.pi/180,100,minLineLength,maxLineGap)
    # for x1,y1,x2,y2 in lines[0]:
    #     cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)


    # noiseFilter=cv2.morphologyEx(edge,cv2.MORPH_OPEN,kernel)    
    plt.subplot(221),plt.imshow(img)
    plt.subplot(222),plt.imshow(imgWarped)
    plt.subplot(223),plt.imshow(edge)
    plt.subplot(224),plt.imshow(thresh, cmap='gray')
    
    plt.show()
    cv2.imshow("img", imgWarped)

    cv2.waitKey()
    cv2.destroyAllWindows()
    return imgWarped
    
def ORB_detector(img):
    img = img

    # Initiate STAR detector
    orb = cv2.ORB_create()
    orb.getDefaultName()
    # find the keypoints with ORB
    kp = orb.detect(img,None)
    
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(img,kp,None,color=(0,255,0), flags=0)
    return kp,des,img2
    
def ORB_matcher():
    orb= cv2.ORB_create()
    #image to detect from
    img1 = cv2.imread('simetic7.jpg',0)
    #image to be detected
    img2 =cv2.imread('SchaltschrankS7.jpg', 0)
    
    kp1,des1,img1 = ORB_detector(img1)
    kp2,des2,img2 = ORB_detector(img2)
    
    bf=cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    matches = sorted(matches, key = lambda x:x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], None,flags=2)
    
    plt.subplot(221),plt.imshow(img1)
    plt.subplot(222),plt.imshow(img2)
    plt.subplot(223),plt.imshow(img3)
   
    plt.show()

if __name__=="__main__":
    ORB_matcher()
    filterImg()