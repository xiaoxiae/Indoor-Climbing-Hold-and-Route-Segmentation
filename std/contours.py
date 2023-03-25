import cv2 as cv
import numpy as np

from utils import *


img = cv.imread('298.jpg')
process_image(img, "edges/1-original.png", save=True)

blur = cv.GaussianBlur(img, (13, 13), 0)
process_image(blur, "edges/2-blur.png", save=True)

edges = cv.Canny(blur, 20, 25)
process_image(edges, "edges/3-edges.png", save=True)

contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours_img = sobel(blur)
cv.drawContours(contours_img, contours, -1, (0, 255, 0), thickness=2)
process_image(contours_img, "edges/4-contours.png", save=True)

contours = filter_size_contours(contours)
contours_img = sobel(blur)
cv.drawContours(contours_img, contours, -1, (0, 255, 0), thickness=2)
process_image(contours_img, "edges/5-contours-area-filter.png", save=True)

contours = filter_straight_contours(contours, max_avg_error=5)
contours_img = sobel(blur)
cv.drawContours(contours_img, contours, -1, (0, 255, 0), thickness=2)
process_image(contours_img, "edges/6-contours-straight-filter.png", save=True)
