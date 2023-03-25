from utils import *


img = cv.imread('298.jpg')

# blobs
keypoints = detect_blobs(img)

# edges
blur = gaussian_blur(img)
edges = canny(blur)
contours, _ = find_contours(edges)
contours = filter_size_contours(contours)
contours = filter_straight_contours(contours)

for k in keypoints:
    img_copy = img.copy()

    nearby_contours = get_nearby_contours(k.pt, contours, (k.size / 2))

    if len(nearby_contours) == 0:
        continue

    draw_keypoints(img_copy, [k])
    draw_contours(img_copy, nearby_contours)

    process_image(img_copy, "combined/1-original.jpg")


#for i in range(0, 255, 1):
#    t = threshold(blur, i, 255)
#
#    r, g, b = t[:,:,0], t[:,:,1], t[:,:,2]
#
#    process_image(b, "combined/1-original.jpg")
