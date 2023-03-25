from utils import *


img = cv.imread('298.jpg')
process_image(img, "blobs/1-original.jpg")

keypoints = detect_blobs(img)

draw_keypoints(img, keypoints)
process_image(img, "blobs/2-blobs.jpg")
