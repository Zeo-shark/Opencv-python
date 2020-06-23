from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
import argparse

def midpoint(ptA,ptB):
    return((ptA[0]+ ptB[0])*0.5,(ptA[1]+ptB[1])*0.5)

ap=argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True,help="path to the input image")
ap.add_argument("-w","--width",type=float,required=True,help="width of the left most object in the image [in inches]")
args=vars(ap.parse_args())
image=cv2.imread(args["image"])
gray=cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
gray=cv2.GaussianBlur(gray, (7, 7), 0)
edged= cv2.Canny(gray, 50, 100)
edged=cv2.dilate(edged,None, iterations=1)
edged=cv2.erode(edged, None,iterations=2)
cnts=cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts=imutils.grab_contours(cnts)
(cnts, _)=contours.sort_contours(cnts)
pixelsPerMetric=None
for c in cnts:
    if cv2.contourArea(c)<100:
        continue
    #compute the rotated bounding box of the contour
    orig=image.copy()
    box=cv2.minAreaRect(c)
    box=cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box= np.array(box,dtype="int")

    #order the points such that they appear in the top left ,top-right,top