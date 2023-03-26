from utils import *


THRESHOLD_STEP = 5

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

# pre-compute thresholds and contours/edges since they're used repeatedly
thresholds = {}
for i in range(0, 255, THRESHOLD_STEP):
    t = threshold(blur, i, 255)
    thresholds[i] = []

    for c_t in t[:,:,0], t[:,:,1], t[:,:,2]:
        t_edges = canny(c_t)
        t_contours = find_contours(t_edges)
        t_contours = simplify_contours(t_contours)

        thresholds[i].append((t_edges, t_contours))

hold_approximations = {}
for k in keypoints:
    nearby_contours = get_nearby_contours(k.pt, contours, (k.size / 2))

    if len(nearby_contours) == 0:
        continue

    best_contour = None
    best_contour_error = float('inf')

    for i in range(0, 255, THRESHOLD_STEP):
        for t_edges, t_contours in thresholds[i]:
            closest_t_contour = get_closest_contour(k.pt, t_contours)

            if closest_t_contour is None:
                continue

            err = squared_contour_error(nearby_contours, closest_t_contour)

            if best_contour_error > err:
                best_contour_error = err
                best_contour = closest_t_contour

    if best_contour is None:
        continue

    hold_approximations[k] = best_contour

img_copy = img.copy()
draw_contours(img_copy, list(hold_approximations.values()))
process_image(img_copy, "combined/2-holds.jpg")
