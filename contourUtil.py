import numpy as np
import cv2
import math

def getContoursIndexesAndAreaOrderedByArea(contours):

    contoursNum = np.size(contours)

    contoursIndexesWithArea = np.empty(contoursNum*3,dtype='i').reshape(contoursNum,3)
 
    for i in range(contoursNum):
        contoursIndexesWithArea[i,1] = i
        contoursIndexesWithArea[i,2] = cv2.contourArea(contours[i])

    # Order the contoursIndexesWithArea array by it's second column
    contoursIndexesWithAreaOrder = contoursIndexesWithArea[contoursIndexesWithArea[:,2].argsort()]

    contoursIndexesWithAreaOrder[:,0] = range(contoursNum)

    return contoursIndexesWithAreaOrder


def getContoursIndexesAndSizeOrderedBySize(contours):

    contoursNum = np.size(contours)

    contoursIndexesWithSize = np.empty(contoursNum*3,dtype='i').reshape(contoursNum,3)
 
    for i in range(contoursNum):
        contoursIndexesWithSize[i,1] = i
        contoursIndexesWithSize[i,2] = np.size(contours[i])

    # Order the contoursIndexesWithArea array by it's second column
    contoursIndexesWithSizeOrder = contoursIndexesWithSize[contoursIndexesWithSize[:,2].argsort()]

    contoursIndexesWithSizeOrder[:,0] = range(contoursNum)

    return contoursIndexesWithSizeOrder


def removeImagePerimeterContour(img, contours):

    height,width = img.shape

    contourNum = np.size(contours)
    contourFound = False
    imageArea = height * width

    for i in range(contourNum):
        if ((cv2.contourArea(contours[i]) / imageArea) > 0.80):
            contourFound = True
            break

    if contourFound:
        return np.delete(contours,i)
    else:
        return contours
