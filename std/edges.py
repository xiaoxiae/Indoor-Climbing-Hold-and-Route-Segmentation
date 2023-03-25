from utils import *


img = cv.imread('298.jpg')
process_image(img, "edges/1-original.jpg")

blur = gaussian_blur(img)
process_image(blur, "edges/2-blur.jpg")

edges = canny(blur)
process_image(edges, "edges/3-edges.jpg")

contours, _ = find_contours(edges)
contours_img = img.copy()
draw_contours(contours_img, contours)
process_image(contours_img, "edges/4-contours.jpg")

contours = filter_size_contours(contours)
contours_img = img.copy()
draw_contours(contours_img, contours)
process_image(contours_img, "edges/5-contours-area-filter.jpg")

contours = filter_straight_contours(contours)
contours_img = img.copy()
draw_contours(contours_img, contours)
process_image(contours_img, "edges/6-contours-straight-filter.jpg")
