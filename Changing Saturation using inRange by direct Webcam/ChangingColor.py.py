'''
Program :- Changing color from Virtual camera using cv.inRange() Function.

Author:- AmRiyaz

Last MOdified:- September 2020
'''
#importing packages
import cv2
import numpy as np
 
def nothing(x):
    pass
 
#Opening Machine's Web Cam
webcam = cv2.VideoCapture(0) 
 
#Creating a new Window for Thresholding
cv2.namedWindow('Shield')
 
#Creating SlideBars(Trackbar) for changing color
cv2.createTrackbar('lowH','Shield',0,179,nothing)
cv2.createTrackbar('highH','Shield',179,179,nothing)
cv2.createTrackbar('lowS','Shield',0,255,nothing)
cv2.createTrackbar('highS','Shield',255,255,nothing)
cv2.createTrackbar('lowV','Shield',0,255,nothing)
cv2.createTrackbar('highV','Shield',255,255,nothing)
 
while(True):
    ret, frame = webcam.read()
 
    #Getting the Current Position of Braces in Trackbars
    ilowH = cv2.getTrackbarPos('lowH', 'Shield')
    ihighH = cv2.getTrackbarPos('highH', 'Shield')
    ilowS = cv2.getTrackbarPos('lowS', 'Shield')
    ihighS = cv2.getTrackbarPos('highS', 'Shield')
    ilowV = cv2.getTrackbarPos('lowV', 'Shield')
    ihighV = cv2.getTrackbarPos('highV', 'Shield')
    
    #Converting the Color to HSV (Hie Saturation Value Model)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([ilowH, ilowS, ilowV])
    higher_hsv = np.array([ihighH, ihighS, ihighV])

    #Appliying and exhibiting the cv2.inRange() masktion
    mask = cv2.inRange(hsv, lower_hsv, higher_hsv)

    #Extracting the original color by Applying the mask on the Shield 
    frame = cv2.bitwise_and(frame, frame, mask=mask)
    cv2.imshow('Shield', frame)
    
    
    #To Exit the Shield press Z in Keyboard
    if cv2.waitKey(1) & 0xFF == ord('z'):
        break
    
#Destroying Sessions
webcam.release()
cv2.destroyAllWindows()