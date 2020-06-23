import imutils
import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS
import time
import argparse
import cv2
from collections import deque

ap=argparse.ArgumentParser()
ap.add_argument("-v","--video",type=str,help="Path to the video file")#optional otherwise  usewebcam
ap.add_argument("-t","--tracker",type=str, default="kcf",help="Opencv object Tracker type")
ap.add_argument("-b","--buffer", type=int, default= 64,help= "max buffer size")
args=vars(ap.parse_args())
(major,minor) = cv2.__version__.split(".")[:2]
if int(major)== 3 and int(minor) <3:#spl factory function
     tracker = cv2.Tracker_create(args["tracker"].upper())
    #explicit call to the const
else:
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf":  cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create,
        }
    pts = deque(maxlen=args["buffer"])
    #grab the appt object tracker
    tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
# bounding box coordinate
initBB=None
if not args.get("video",False):
    print("starting the Video SStream")
    vs=VideoStream(src=0).start()
    time.sleep(2.0)

else:
    vs = cv2.VideoCapture(args["video"])
fps=None
#otherwise --video ini the video stream from a video file
# keep looping
while True:

    # grab the current frame
    frame = vs.read()
    centre=None
    # handle the frame from VideoCapture or VideoStream
    frame = frame[1] if args.get("video", False) else frame

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        break
    frame = imutils.resize(frame, width=500)
    (H,W) = frame.shape[:2]
    # check to see if we are currently tracking an object
    if initBB is not None:
        # grab the new bounding box coordinate
        (success, box) = tracker.update(frame)
        # check to see if tracking was successful
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            center = (int(x+w/2), int(y+h/2))
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
        pts.appendleft(center)
        fps.update()
        fps.stop()
        # update the points queue
        pts.appendleft(center)

        # loop over the set of tracked points
        for i in range(1, len(pts)):
            # if either of the tracked points are None, ignore
            # them
            if pts[i - 1] is None or pts[i] is None:
                continue

            # otherwise, compute the thickness of the line and
            # draw the connecting lines
            thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

        # show the output frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(0) & 0xFF
        if key == ord('s'):
            # select the bounding box of the object of the object we want to track ( make sure you press Enter or space before implementing them
            initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)
            # start Opencv object tracker using the supplied bounding box coordinate then Start FPS through the estimator as well
            tracker.init(frame, initBB)
            fps = FPS().start()
            # select s to manually select an ROI
            # while the video stream is Frozen
            # esp ini our FPS counter on the subsequent 121
        elif key == ord("q"):
            break
        # if we are using the webcam release the pointer vs
    if not args.get("video", False):
        vs.stop()
        # otherwise release the file pointer
    else:
        vs.release()
    cv2.destroyAllWindows()
