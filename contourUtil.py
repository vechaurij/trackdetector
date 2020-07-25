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

def getIndexOfContourClosestToPerimeter(img, contours, percentArea = 0.8):

    height,width = img.shape

    contourNum = np.size(contours)
    contourFound = False
    imageArea = height * width

    for i in range(contourNum):
        if ((cv2.contourArea(contours[i]) / imageArea) > percentArea):
            contourFound = True
            break

    if contourFound:
        return i
    else:
        return -1

def getIndexOfMostOuterContour(contours, hierarchy):

    contourNum = np.size(contours)

    for i in range(contourNum):
        if (hierarchy[0][i][0] == -1 and hierarchy[0][i][3] == -1):
            return i
    return -1

def getInnerAndOuterContours(contours, candidateTrackContours):
    """Given a set of candidate contours to be part of a track, 
        returns the inner and outer contours.
        The candidate contours to be part of a track are identified by
        the array candidateTrackContours, that contain the indexes of the candidate
        contours on the contours parameter.
        All candidate contours must be more or less concentric, as the function
        will return as inner contour, the contour with the smallest radius and as
        the outer contour the contour with the largest radius.
        ----------
        contours : ndarray
            Array of contours returned by the function cv2.findContours candidates
            to match a track, whre to find the inner and outer shape of the track.

        candidateTrackContours : ndarray
            Array of the indexes of contours candidates to be part of a track.

        Returns
        -------
            idxInner : int
                Integer with the index of the inner contour found on candidateTrackContours
            
            idxOuter : int
                Integer with the index of the outer contour found on candidateTrackContours

        Raises
        ------
        Nothing.
    """
    candidateTrackContoursNum = len(candidateTrackContours)

    centers = [None]*candidateTrackContoursNum
    radius = np.empty(candidateTrackContoursNum,dtype='f')

    for i in range(candidateTrackContoursNum):
        centers[i], radius[i] = cv2.minEnclosingCircle(contours[candidateTrackContours[i]])

    idxInner = radius.argmin()
    idxOuter = radius.argmax()

    print("Min radius: " + str(min(radius)) + " Index: " + str(idxInner))
    print("Max radius: " + str(max(radius)) + " Index: " + str(idxOuter))

    return idxInner.item(), idxOuter.item()