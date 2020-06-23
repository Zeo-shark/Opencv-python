from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2

ap=argparse.ArgumentParser()
ap.add_argument("-p","--shape-predictor",required=True,help="path to the facial landmark detector")
ap.add_argument("-r","--picamera",type=int, default=-1, help="Whether or not picamera should be used or not")
args=vars(ap.parse_args())
print("Loading the face detector model")
detector= dlib.get_frontal_face_detector()
predictor= dlib.shape_predictor(args["shape_predictor"])

vs=VideoStream(usePiCamera=args["picamera"] >0).start()
time.sleep(2.0)
while True:
    frame=vs.read()
    frame=imutils.resize(frame,width=400)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    rects=detector(gray ,0)
    for rect in rects:
        #determine for facial landmark for the face region,then
        #convert the x,y coor into numpy array
        shape=predictor(gray , rect)
        shape= face_utils.shape_to_np(shape)

        for (x,y) in shape:
            cv2.circle(frame, (x,y), 2, (0,0,255), -1)

        cv2.imshow("frame",frame)
        key= cv2.waitKey(1) & 0xFF
        if key== ord('q'):
            break

        cv2.destroyAllWindows()
        vs.stop()


