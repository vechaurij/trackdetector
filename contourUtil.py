import numpy as np
import cv2
import math
from enum import Enum

class ContourOrderFeature(Enum):
    AREA = 1
    SIZE = 2

def getContoursIndexesOrderedByFeature(contours, contourFeature = ContourOrderFeature.AREA):
    """Given a set of contours found by cv2.findContours function, 
        returns a ndarray containing the contours indexes ordered by a contour feature 
        (Contour Area or Contour Size) selected by the parameter contourFeature.
        The returned array will be of shape nx3 of integers, given that there are n contours.
        
        For each tuple on the returned array, each column represent the following
        0: Tuple number of the returned array, from 0 to n, where n is the number of contours.
        1: Contour area or contour size of one of the contours
        2: Index of the contours with area or size indicated on column 1 on the input contours array

        Parameters
        ----------
        contours : ndarray
            Array returned by the function cv2.findContours containing all contours found on an image
        contourFeature: Enum
            Indicates the contour feature to be used to order the returned array, two possibilities
            ContourOrderFeature.AREA (Default if no parameter set): The area of the contour, given by cv2.contourArea
            ContourOrderFeature.SIZE: The size of the contour measured in pixels. The "method" parameter of the 
                cv2.findContours function must be set to cv2.CHAIN_APPROX_NONE in order to get all the points of the contour

        Raises
        ------
        Nothing.
    """
    
    contoursNum = np.size(contours)
    contoursIndexesWithFeature = np.empty(contoursNum*3,dtype='i').reshape(contoursNum,3)
 
    for i in range(contoursNum):
        if contourFeature is ContourOrderFeature.AREA:
            contoursIndexesWithFeature[i,1] = cv2.contourArea(contours[i])
        else:
            contoursIndexesWithFeature[i,1] = np.size(contours[i])    
        contoursIndexesWithFeature[i,2] = i

    # Order the contoursIndexesWithFeature array by it's second column where the ordering 
    # feature of the contours is (AREA or SIZE)
    contoursIndexesWithFeatureOrder = contoursIndexesWithFeature[contoursIndexesWithFeature[:,1].argsort()]

    # Fill the first column with numbers from 0 to n (number of contours)
    contoursIndexesWithFeatureOrder[:,0] = range(contoursNum)

    return contoursIndexesWithFeatureOrder

def removeImagePerimeterContour(img, contours, percentArea = 0.8):
    """Given an image, its contours found by the cv2.findContours function, removes
        the first contour found whose area is larger than the percentage set of the image
        area set by the percentArea parameter
        Parameters
        ----------
        imb : ndarray
            Array containing the original image
        contours : ndarray
            Array returned by the function cv2.findContours containing all contours found on an image
        percentArea : float
            Percentage of area of the source img to determine the contour to remove if it's larger. By default 80% (Or 0.8)
            
        Raises
        ------
        Nothing.
    """

    height,width = img.shape

    contourNum = np.size(contours)
    contourFound = False
    imageArea = height * width

    for i in range(contourNum):
        if ((cv2.contourArea(contours[i]) / imageArea) > percentArea):
            contourFound = True
            break

    if contourFound:
        return np.delete(contours,i)
    else:
        return contours
