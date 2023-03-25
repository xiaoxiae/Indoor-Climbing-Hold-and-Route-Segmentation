import os
import cv2 as cv
import numpy as np
from scipy.stats import linregress
from shapely.geometry import Polygon


def sobel(img, size=3, scale=1, delta=0):
    """Run a Sobel kernel on the input image."""
    grad_x = cv.Sobel(img, cv.CV_16S, 1, 0, ksize=size, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    grad_y = cv.Sobel(img, cv.CV_16S, 0, 1, ksize=size, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)

    grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    return grad

def filter_straight_contours(contours, max_avg_error=2.5):
    """
    Filter out contours that are too close to lines.

    @param max_avg_error: maximum average squared point-to-line error (in pixels)
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

        error = error_function(a, b, c)

        if error < max_avg_error:
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

def process_image(img, filename, save=True):
    """Display or save image."""
    d = os.path.dirname(filename)

    if not os.path.exists(d):
        os.mkdir(d)

    if save:
        cv.imwrite(filename, img)
    else:
        cv.imshow('image', img)
        cv.waitKey(0)

def draw_keypoints(img, keypoints, color=(255, 0, 0), thickness=2):
    for k in keypoints:
        x, y = k.pt
        cv.circle(img, (int(x), int(y)), int(k.size / 2), color=color, thickness=thickness)

    return img
