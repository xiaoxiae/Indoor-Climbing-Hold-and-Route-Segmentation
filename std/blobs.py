import cv2 as cv
import numpy as np

from utils import *


img = cv.imread('298.jpg')
process_image(img, "blobs/1-original.png")

params = cv.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 500
params.maxArea = 100000

params.minThreshold = 1
params.maxThreshold = 200
params.thresholdStep = 10

params.filterByColor = False
params.blobColor = 0

params.filterByConvexity = False
params.filterByInertia = False

detector = cv.SimpleBlobDetector_create(params)
keypoints = detector.detect(img)

blur = cv.GaussianBlur(img, (13, 13), 0)
contours_img = sobel(blur)

img_keypoints = draw_keypoints(contours_img, keypoints, color=(255, 0, 0), thickness=2)
process_image(img_keypoints, "blobs/2-blobs.png")
