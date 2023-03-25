from utils import *


img = cv.imread('298.jpg')
process_image(img, "blobs/1-original.jpg")

params = cv.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 500
params.maxArea = 100000

params.minThreshold = 1
params.maxThreshold = 200
params.thresholdStep = 10

params.filterByColor = False
params.filterByConvexity = False
params.filterByInertia = False

detector = cv.SimpleBlobDetector_create(params)
keypoints = detector.detect(img)

img_keypoints = draw_keypoints(img, keypoints, color=(0, 255, 0), thickness=5)
process_image(img_keypoints, "blobs/2-blobs.jpg")
