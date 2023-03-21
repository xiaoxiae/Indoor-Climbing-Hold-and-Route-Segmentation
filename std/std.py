import cv2 as cv
import numpy as np
from shapely.geometry import Polygon
from matplotlib import pyplot as plt

img = cv.imread('298.jpg')
cv.imshow('image', img)
cv.waitKey(0)


def dist(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1/2)

def sobel(img, size=3, scale=1, delta=0):
    """Run a Sobel kernel on the input image."""
    grad_x = cv.Sobel(img, cv.CV_16S, 1, 0, ksize=size, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(img, cv.CV_16S, 0, 1, ksize=size, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return grad

def filter_contours(contours, min_points=3, min_bb_area=70, max_bb_size=700):
    """
    Filter out contours based on the number of their points and their bounding box area.

    @param min_points: the minimum number of points a contour can have
    @param min_bb_area: the minimum area of a bounding box of a contour
    @param max_bb_size: the maximum width/height of a bounding box of a contour
    """
    contours = list(contours)

    to_remove = []
    for i, c in enumerate(contours):
        if len(c) < min_points:
            to_remove.append(i)
            continue

        points = [j.tolist()[0] for j in c]
        p = Polygon(points)

        xl, yl, xh, yh = p.bounds

        w = abs(xl - xh)
        h = abs(yl - yh)

        if w * h < min_bb_area:
            to_remove.append(i)

        if w > max_bb_size or h > max_bb_size:
            to_remove.append(i)

    for r in reversed(to_remove):
        del contours[r]

    return contours


blur = cv.GaussianBlur(img, (13, 13), 0)
cv.imshow('image', blur)
cv.waitKey(0)

edges = cv.Canny(blur, 35, 35)
edges_softer = cv.Canny(blur, 10, 10)

cv.imshow('image', edges)
cv.waitKey(0)

#kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(9,9))
#dilated = cv.dilate(edges, kernel)
#
#cv.imshow('image', dilated)
# cv.waitKey(0)

#contours, _ = cv.findContours(dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#contours = list(contours)

contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = filter_contours(contours)

contours_softer, _ = cv.findContours(edges_softer, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours_softer = filter_contours(contours_softer)

#contours_img = sobel(blur)
contours_img = np.zeros(shape=[len(img), len(img[0]), 3], dtype=np.uint8)

#cv.drawContours(contours_img, contours_softer, -1, (0, 0, 255), thickness=2)
cv.drawContours(contours_img, contours, -1, (0, 255, 0), thickness=5)

cv.imshow('image', contours_img)
cv.waitKey(0)

kernel = np.ones((25,25),np.uint8)
closing = cv.morphologyEx(contours_img, cv.MORPH_CLOSE, kernel)

cv.imshow('image', closing)
cv.waitKey(0)

# TODO: now grow so the regions combine
