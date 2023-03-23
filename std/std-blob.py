import cv2 as cv
import numpy as np


img = cv.imread('298.jpg')
cv.imshow('image', img)
cv.waitKey(0)

params = cv.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 300

params.minThreshold = 1
params.maxThreshold = 200
params.thresholdStep = 10

params.minDistBetweenBlobs = 5

params.filterByColor = True
params.blobColor = 0

params.filterByConvexity = False
params.filterByInertia = False

detector = cv.SimpleBlobDetector_create(params)
keypoints = detector.detect(img)


img_keypoints = cv.drawKeypoints(img, keypoints, np.array(
    []), (255, 0, 0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv.imshow('image', img_keypoints)
cv.waitKey(0)
