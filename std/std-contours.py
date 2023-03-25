import cv2 as cv
import numpy as np
from scipy.stats import linregress
from shapely.geometry import Polygon
from matplotlib import pyplot as plt


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

def filter_straight_contours(contours, debug_img, max_error=2.5):
    """
    Filter out contours that are too close to lines.

    TODO: remove debug_img

    @param max_rvalue: maximum Pearson correlation coefficient of the linear regression

    """
    contours = list(contours)

    def error_function(a, b, c):
        points = [j.tolist()[0] for j in c]

        error = 0
        for x, y in points:
            error += (a * x + b - y) ** 2

        return error / len(c)

    to_remove = []
    for i, c in enumerate(contours):
        points = np.asarray([j.tolist()[0] for j in c])

        x = points[:,0]
        y = points[:,1]

        result = linregress(x, y)

        a, b = result.slope, result.intercept

        p1 = (0, int(b))
        p2 = (len(debug_img[0]), int(a * len(debug_img[0]) + b))

        error = error_function(a, b, c)

        if error < max_error:
            to_remove.append(i)

    for r in reversed(to_remove):
        del contours[r]

    return contours

def filter_size_contours(contours, min_points=3, min_bb_area=125, max_bb_size=700):
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


img = cv.imread('298.jpg')
cv.imshow('image', img)
cv.waitKey(0)

blur = cv.GaussianBlur(img, (13, 13), 0)
cv.imshow('image', blur)
cv.waitKey(0)

edges = cv.Canny(blur, 20, 25)
cv.imshow('image', edges)
cv.waitKey(0)

contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours_img = sobel(blur)
cv.drawContours(contours_img, contours, -1, (0, 255, 0), thickness=2)
cv.imshow('image', contours_img)
cv.waitKey(0)

contours = filter_size_contours(contours)
contours_img = sobel(blur)
cv.drawContours(contours_img, contours, -1, (0, 255, 0), thickness=2)
cv.imshow('image', contours_img)
cv.waitKey(0)

contours = filter_straight_contours(contours, contours_img, max_error=5)
contours_img = sobel(blur)
cv.drawContours(contours_img, contours, -1, (0, 255, 0), thickness=2)
cv.imshow('image', contours_img)
cv.waitKey(0)
