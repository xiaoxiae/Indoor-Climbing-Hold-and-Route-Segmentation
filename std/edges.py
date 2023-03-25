from utils import *


img = cv.imread('298.jpg')
process_image(img, "edges/1-original.jpg")

blur = cv.GaussianBlur(img, (13, 13), 0)
process_image(blur, "edges/2-blur.jpg")

edges = cv.Canny(blur, 20, 25)
process_image(edges, "edges/3-edges.jpg")

contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours_img = img.copy()
cv.drawContours(contours_img, contours, -1, (0, 255, 0), thickness=5)
process_image(contours_img, "edges/4-contours.jpg")

contours = filter_size_contours(contours)
contours_img = img.copy()
cv.drawContours(contours_img, contours, -1, (0, 255, 0), thickness=5)
process_image(contours_img, "edges/5-contours-area-filter.jpg")

contours = filter_straight_contours(contours, max_avg_error=5)
contours_img = img.copy()
cv.drawContours(contours_img, contours, -1, (0, 255, 0), thickness=5)
process_image(contours_img, "edges/6-contours-straight-filter.jpg")
