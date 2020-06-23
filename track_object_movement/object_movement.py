from collections import deque
from imutils.video import VideoStream
import numpy as np
import time

import imutils
import cv2
import argparse

ap=argparse.ArgumentParser()
ap.add_argument("-v","--video",help="path to the video file")
ap.add_argument("-b","--buffer",path="buffer size ")
args=vars(ap.parse_args())
#define the lower and upper bound in the HSV color space
greenlower=(29,86,6)
greenupper=(64,255,255)
#initialise the list of tracked points,the frame counter ,
#and the co-ordinate deltas
pts=deque(maxlen=args["buffer"])
counter=0
(dX, dY)=(0,0)
direction=""
if not args.get("video",False):
    print("starting the Video SStream")
    vs=VideoStream(src=0).start() # initialise the points and the deque with max size buffer
else:
    vs=cv2.VideoCapture(args["video"])
time.sleep(2)
while True:
    # grab the current frame
    frame = vs.read()

    # handle the frame from video file or videostream
    frame = frame[1] if args.get("video", False) else frame

    # if we are viewing the video
    # #and we are grabbing no frame then we have reached the end of the video
    if frame is None:
        break

    # resizee the frame ,blur it,and convert it to the HSV
    frame = imutils.resize(frame, width=600)
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    # construct a mask for the color green ,then perform
    # a series of erosion and dilation to remove any small
    # blob left in the mask
    mask = cv2.inRange(hsv, greenlower, greenupper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None
    if len(cnts)>0:
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        centre = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.circle(frame, centre, 5, (0, 0, 255), -1)
            pts.appendleft(centre)

        for i in range(1, len(pts)):
            # if either of the tracked points are none, ignore
            # them
            if pts[i - 1] is None or pts[i] is None:
                # if the current frame or previous frame is not present then ignore the current idexcontinue looping over the points
                # this means the not successfully detected in the given frame
                continue
            #check to see if enough points
            # have been accumalated in the buffer
            if counter>=10 and i==1 and pts[-10] is not None:
                #compute the difference between X and Y co-ordinate
                #and reinitialise the directions
                #text variables
                dX=pts[-10][0] - pts[i][0]
                dY=pts[-10][1] - pts[i][0]
                (dirX, dirY)= ("","")
                #ensure there is
                # significant movement in xdirection
                if np.abs(dX) > 20:
                    dirX = "East" if np.sign(dX)==1 else "West"
                #ensure that there is significant movement in Ydirection
                if np.abs(dY) > 20:
                    dirY= "North" if np.sign(dY)==1 else "South"
                #handle when both direction not empty
                if dirX!="" and dirY !="":
                    direction="{}-{}".format(dirY,dirX)
                #otherwise only one direction is non empty
                else:
                    direction=dirX if dirX!="" else dirY

            # otherwise compute the thickness of the line
            # and draw the connecting lines
            # compute the thickness

            thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)  # draw the line
            gray = cv2.cvtColor(frame, cv2.BGR2GRAY)
        cv2.putText(frame,direction,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,255), 3)
        cv2.putText(frame,"dX: {}, dY: {}".format(dX,dY),(10, frame.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.35,(0,0,255), 1)
        # draw the frames to our screen
        cv2.imshow("Frame", frame)
        cv2.imshow("Gray",gray)
        key = cv2.waitKey(2) & 0xFF
        counter+=1
        # if the q key is pressed
        if key == ord('q'):
            break
# if we are not using the video file ,
# stop  the camera video stream
if not args.get("video", False):
    vs.stop()
# last step or imp step is to draw the contrail of the ball, or simply the past N(x,y)cd ..co-ordinates detected
# close all windows
cv2.destroyAllWindows()



