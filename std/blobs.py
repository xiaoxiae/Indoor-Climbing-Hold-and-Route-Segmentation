from utils import *


img = cv.imread('298.jpg')
process_image(img, "blobs/1-original.jpg")

keypoints = detect_blobs(img)
img_copy = img.copy()
draw_keypoints(img_copy, keypoints)
process_image(img_copy, "blobs/2-blobs.jpg")

keypoints = merge_blobs(keypoints)
img_copy = img.copy()
draw_keypoints(img_copy, keypoints)
process_image(img_copy, "blobs/3-blobs-merged.jpg")
