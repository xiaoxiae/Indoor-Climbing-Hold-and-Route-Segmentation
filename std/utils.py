import os
import sys
import cv2 as cv
import numpy as np
from scipy.stats import linregress
from shapely.geometry import Polygon

STROKE_COLOR = (0, 255, 0)
STROKE_THICKNESS = 5
SAVE = False
BLUR_SIZE = 13
CANNY = (20, 25)

def dist(p1, p2):
    """Return the Euclidean distance between p1 and p2."""
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** (1/2)

def contour_to_list(c):
    """Convert a contour to a list of (x, y) tuples."""
    l = []
    for j in c:
        l.append(j.tolist()[0])
    return l

def filter_straight_contours(contours, max_avg_error=5):
    """
    Filter out contours that are too close to lines.

    @param max_avg_error: maximum average squared point-to-line error (in pixels)
    """
    contours = list(contours)

    def error_function(a, b, c):
        error = 0
        for x, y in contour_to_list(c):
            error += (a * x + b - y) ** 2

        return error / len(c)

    to_remove = []
    for i, c in enumerate(contours):
        points = np.asarray(contour_to_list(c))

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

def process_image(img, filename, save=SAVE, scaling=0.5):
    """Display or save an image."""
    d = os.path.dirname(filename)

    if not os.path.exists(d):
        os.mkdir(d)

    if save:
        h, w = img.shape[:2]

        new_w, new_h = int(w * scaling), int(h * scaling)

        resized_img = cv.resize(img, (new_w, new_h), interpolation= cv.INTER_LINEAR)

        cv.imwrite(filename, resized_img)
    else:
        cv.imshow('image', img)
        cv.waitKey(0)

def draw_keypoints(img, keypoints, color=STROKE_COLOR, thickness=STROKE_THICKNESS):
    """Custom drawing of keypoints (since the OpenCV function doesn't support custom thickness)."""
    for k in keypoints:
        x, y = k.pt
        cv.circle(img, (int(x), int(y)), int(k.size / 2), color=color, thickness=thickness)

def draw_contours(img, contours, color=STROKE_COLOR, thickness=STROKE_THICKNESS):
    """cv.drawContours with sane default."""
    cv.drawContours(img, contours, -1, color=color, thickness=thickness)

def gaussian_blur(img, size=13):
    """cv.GaussianBlur with sane default."""
    return cv.GaussianBlur(img, (size, size), 0)

def canny(img, parameters=CANNY):
    """cv.Canny with sane default."""
    return cv.Canny(img, *parameters)

def find_contours(edges):
    """cv.findContours with sane default."""
    return cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

def threshold(img, start=0, end=255):
    """cv.threshold with sane default."""
    _, t = cv.threshold(img, start, end, cv.THRESH_BINARY)
    return t

def detect_blobs(img):
    """OpenCV simple blob detection with sane default."""
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
    return detector.detect(img)

def get_nearby_contours(point, contours, distance):
    """Return a list of contours any of whose points are close enough to the point."""
    def is_close(p, c):
        for pc in contour_to_list(c):
            if dist(p, pc) < distance:
                return True
        return False

    close = []
    for c in contours:
        if is_close(point, c):
            close.append(c)

    return close
