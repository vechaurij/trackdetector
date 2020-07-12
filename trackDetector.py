import cv2
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from kneebow.rotor import Rotor
from contourUtil import *


def findImageContours(imageFile):

    # Read the image from filesystem given by imageFile parameter
    img = cv2.imread(imageFile, cv2.IMREAD_UNCHANGED)

    # Set image size to work with
    imageSize = (800, 600)

    # Resize the read image to the working size defined
    imgResized = cv2.resize(img, imageSize)

    # Convert img to grey scale
    imgGrey = cv2.cvtColor(imgResized,cv2.COLOR_BGR2GRAY)

    # Set the kernel size for image blur
    ksize = (5, 5) 

    # Blur the image with the definez kernel size
    imgBlur = cv2.blur(imgGrey,ksize)

    # Set a threshold limit to transform the input image
    thresh = 80

    # Get threshold input image
    ret,imgThresh = cv2.threshold(imgBlur, thresh, 255, cv2.THRESH_BINARY)

    cv2.imshow("Threshold",imgThresh)
    cv2.waitKey()

    # Find contours on the thresholded image
    contours, hierarchy = cv2.findContours(imgThresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Remove the largest contour found if it could be the perimeter of the source image
    contours = removeImagePerimeterContour(imgThresh,contours)

    contourNum = np.size(contours)
    print("Contours found: " + str(contourNum))

    # Get an array of shape nx3 containing: number, area of the contour represented by the index and the contour index 
    contoursIndexesAndAreasOrderedByArea = getContoursIndexesOrderedByFeature(contours, ContourOrderFeature.AREA)

    # Get an array of shape nx3 containing: number, size of the contour represented by the index and the contour index 
    contoursIndexesAndSizesOrderedBySize = getContoursIndexesOrderedByFeature(contours, ContourOrderFeature.SIZE)

    print("Contours with indexes and areas: " + str(contoursIndexesAndAreasOrderedByArea))
    print("Contours with indexes and sizes: " + str(contoursIndexesAndSizesOrderedBySize))

    # Initialize the kneebow library to get the point of the curves of contur sizes or areas where they start to grow exponentialy
    rotor = Rotor()

    # Find the point of the curve containing all contour areas where it starts to grow exponentialy
    rotor.fit_rotate(contoursIndexesAndAreasOrderedByArea[:,0:2])
    elbowIndexAreas = rotor.get_elbow_index()

    # Find the point of the curve containing all contour sizes where it starts to grow exponentialy
    rotor.fit_rotate(contoursIndexesAndSizesOrderedBySize[:,0:2])
    elbowIndexSizes = rotor.get_elbow_index()

    print("Elbow Index for areas: " + str(elbowIndexAreas))
    print("Elbow Index for sizes: " + str(elbowIndexSizes))

    # The posssible contours candidates to be a track will start on the highest of the sizes or areas found
    # in order to discard false contours not being tracks but with enough size.
    highestElbow = elbowIndexAreas
    if (elbowIndexSizes > elbowIndexAreas):
        highestElbow = elbowIndexSizes

    # Create an empty image for drawing the contours
    imgContours = np.zeros(imgResized.shape)

    # Interate over all contours found but only draw those contours on the exponential grow part of the curves defined
    # by sizes and areas of the found contours
    for x in range(contourNum):
        
        print("Contour: " + str(contoursIndexesAndAreasOrderedByArea[x,2]) + " Area: " + str(contoursIndexesAndAreasOrderedByArea[x,1]) + "Size: " + str(contoursIndexesAndSizesOrderedBySize[x,1]))

        # Take into account those contours present on the exponential part of the curve defined by sizes or areas of found contours
        if (x > highestElbow):
            # Draw the contours on the empty image
            cv2.drawContours(imgContours, contours, contoursIndexesAndAreasOrderedByArea[x,2], (0,255,0), 3)

            # Display the contour over the empty image
            cv2.imshow("contoursFound",imgContours)
            cv2.waitKey()


# Parse argumetns from command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, 
                help = "Path to the input image filename")

args = vars(ap.parse_args())

findImageContours(args["image"])
