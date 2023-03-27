from utils import *

images = ["../data/sm/298.jpg"]

annotations = []
for file in images:
    img = cv.imread(file)

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

    # holds
    hold_approximations = detect_holds(img, keypoints, contours)

    annotations.append(to_detectron_format(img, list(hold_approximations.values())))

# TODO: what to do now?
print(annotations)
