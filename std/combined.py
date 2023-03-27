from utils import *


img = cv.imread('298.jpg')

img_copy = img.copy()
process_image(img_copy, "combined/1-original.jpg")

# blobs
keypoints = detect_blobs(img)
keypoints = merge_blobs(keypoints)

# edges
blur = gaussian_blur(img)
edges = canny(blur)
contours = find_contours(edges)
contours = simplify_contours(contours)
contours = filter_size_contours(contours)
contours = filter_straight_contours(contours)

hold_approximations = detect_holds(img, keypoints, contours)

img_copy = img.copy()
draw_contour_boxes(img_copy, list(hold_approximations.values()), color=(0, 255, 0))
draw_contours(img_copy, list(hold_approximations.values()), (0, 128, 0))
process_image(img_copy, "combined/2-holds.jpg")
