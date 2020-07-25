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
    ksize = (8, 8) 

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

    print("Hierarchy: " + str(hierarchy))
    print("Contours hierarchy shape: " + str(np.shape(hierarchy)))

    # Get the index of the largest contour found if it could be the perimeter of the source image
    perimeterContourIndex = getIndexOfContourClosestToPerimeter(imgThresh,contours)
    mostOuterContourIndex = getIndexOfMostOuterContour(contours, hierarchy)

    print("Moost Outer Contour Index: " + str(mostOuterContourIndex))

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

    highestElbow = elbowIndexAreas

    # Create an empty image for drawing the contours
    imgContours = np.zeros(imgResized.shape)

    candidateTrackContours = []

    # Interate over all contours found but only draw those contours on the exponential grow part of the curves defined
    # by sizes and areas of the found contours
    for x in range(contourNum):
        
        if (hierarchy[0][x][0] == -1 and hierarchy[0][x][3] == -1):
            print("-1 and -1 is in: " + str(x) + " - " + str(hierarchy[0][x]))

        # Take into account those contours present on the exponential part of the curve defined by sizes or areas of found contours
        if (x > highestElbow and contoursIndexesAndAreasOrderedByArea[x,2] != mostOuterContourIndex):
            # Draw the contours on the empty image
            #cv2.drawContours(imgContours, contours, contoursIndexesAndAreasOrderedByArea[x,2], (0,255,0), 3)

            print("Contour: " + str(contoursIndexesAndAreasOrderedByArea[x,2]) + " Area: " + str(contoursIndexesAndAreasOrderedByArea[x,1]) + "Size: " + str(contoursIndexesAndSizesOrderedBySize[x,1]) + " Hierarchy: " + str(hierarchy[0][contoursIndexesAndAreasOrderedByArea[x,2]]))

            candidateTrackContours.append(contoursIndexesAndAreasOrderedByArea[x,2])

            # Display the contour over the empty image
            #cv2.imshow("contoursFound",imgContours)
            #cv2.waitKey()
    
    candidateTrackContoursNum = len(candidateTrackContours)

    candidateContoursPoly = [None]*candidateTrackContoursNum
    candidateContours = [None]*candidateTrackContoursNum
    boundRect = [None]*candidateTrackContoursNum
    centers = [None]*candidateTrackContoursNum
    # radius = [None]*candidateTrackContoursNum
    radius = np.empty(candidateTrackContoursNum,dtype='f')



    for i in range(candidateTrackContoursNum):
        candidateContoursPoly[i] = cv2.approxPolyDP(contours[candidateTrackContours[i]], 5, True)
        candidateContours[i] = contours[candidateTrackContours[i]]
        boundRect[i] = cv2.boundingRect(candidateContoursPoly[i])
        centers[i], radius[i] = cv2.minEnclosingCircle(candidateContoursPoly[i])

    print("Bounding boxes centers: ")

    for i in range(candidateTrackContoursNum):

        #cv2.drawContours(imgContours, contours_poly, i, (255,255,0))
        #cv2.rectangle(imgContours, (int(boundRect[i][0]), int(boundRect[i][1])), \
        #  (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), (160,170,0), 2)
        cv2.circle(imgContours, (int(centers[i][0]), int(centers[i][1])), int(radius[i]), (14,100,244), 2)

        print("Center: " + str(i) + " " + str(centers[i]))
        print("Radius: " + str(i) + " " + str(radius[i]))

        # Display the contour over the empty image
        #cv2.imshow("contoursFound",imgContours)
        #cv2.waitKey()

    print("Min radius: " + str(min(radius)) + " Index: " + str(radius.argmin()))
    print("Max radius: " + str(max(radius)) + " Index: " + str(radius.argmax()))

    cv2.drawContours(imgContours, candidateContoursPoly, radius.argmin(), (255,255,0))
    cv2.drawContours(imgContours, candidateContoursPoly, radius.argmax(), (0,255,0))
    #cv2.drawContours(imgContours, candidateContours, radius.argmin(), (255,255,0), 2)
    #cv2.drawContours(imgContours, candidateContours, radius.argmax(), (0,255,0), 2)

    cv2.imshow("contoursFound",imgContours)
    cv2.waitKey()


# Parse argumetns from command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, 
                help = "Path to the input image filename")

args = vars(ap.parse_args())

findImageContours(args["image"])
