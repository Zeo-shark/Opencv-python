from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker():
    def __init__(self, maxDisappeared=50 ):
        #ini the next unique id along with two ordered dict track of mapping a given object
        #ID to its centroid and number of consecutive frames it has been marked as "disappeared" respectively
        self.nextObjectID=0
        self.objects= OrderedDict()#dict
        self.disappeared=OrderedDict()#Dict
        self.maxDisappeared= maxDisappeared#Dict

    def register(self,centroid):
        #when registering an object we use the next available object
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID]= 0
        self.nextObjectID += 1#numvber array index

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
    #heart of the centroid tracker
    def update(self, rects,objectID):
        #check if input bounding box rectangle is empty
        if len(rects)== 0:
            # loop over an exist tracked object and mark them disappear
            for objectsID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                #if we reach maximum number of frames of object declared as dereg
                if self.disappeared[objectID]> self.maxDisappeared:
                    self.deregister(objectID)
            #return early as there are no centroids for tracking info to update
            return self.objects
        #inii array of input centroids for the current frame
        inputCentroids= np.zeros((len(rects),2), dtype="int")#numpy 2D array for each ret
        #lop over the bpund box rectangle
        for(i, (startX,startY, endX,endY)) in enumerate(rects):
            cx=int((startX+endX)/2.0)
            cy=int((startY+endY)/2.0)
            inputCentroids[i] = (cx , cy)
        #if we trck no object ini take the input centroid and register each of them
        if len(self.objects)== 0:
            for i in range(0 , len(inputCentroids)):
                self.register(inputCentroids[i])
        # Core step
        #otherwise the tracking centroids should be matched with the new centroids based on les euclidean dist
        else:
            #grab the set of objectID
            objectIDs=list(self.objects.keys())
            objectCentroids= list(self.objects.values())
            D= dist.cdist(np.array(objectCentroids), inputCentroids)# EUCLIDEAN DISTANCE cdist((objectCentroids),inputCentroids)
            # in order to matching
            #sort to smallest value in each row
            #indexes based on their min values row with the smalllest values infront of the list
            rows = D.min(axis = 1).argsort()
            # find the min column
            cols= D.argmin(axis = 1)[rows]
            #output numpya array shape  of uor distance map
            # (object centroids, input centroids)
            #which rows or cols already examined
            usedRows= set()
            usedCols= set()
            for (row, col) in zip(rows, cols):
                if row  in usedRows or col in usedCols:
                    continue
                #grab the objectid for the current row, set its new centroid reset disappeared counter
                objectID=objectIDs[row]
                self.objects[objectID]= inputCentroids[col]
                self.disappeared[objectID]= 0
                usedRows.add(row)#add ing row and col to the respective Usedrows and Usedcols ie for new centroid
                usedCols.add(col)
                # compute rows & cols we havent examined yet
                unusedRows= set(range(0, D.shape[0])).difference(usedRows)
                unusedCols = set(range(0, D.shape[0])).difference(usedCols)
                #in case the num of object centroids greater or equal to thenew centroids we need to check if some have disappeard
                if D.shape[0] >= D.shape[1]:
                    for row in unusedRows:
                        objectID = objectIDs[row]
                        self.disappeared[objectID]+=1#disappeared count in dict

                        # check the num of objects mark disappear to mark disregister by using unused rows if any
                        if self.disappeared[objectID] >= self.maxDisappeared:
                            self.deregister(objectID)
                        else:
                            for col in unusedCols:
                                self.register(inputCentroids[col])

                    # finally we return the set of all trackable objects
                    return self.objects














