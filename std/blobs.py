from utils import *


for image in ["../data/sm/298.jpg"]:
    img = cv.imread(image)
    process_image(img, "blobs/1-original.jpg", save=False)

    keypoints = detect_blobs(img)
    img_copy = img.copy()
    draw_keypoints(img_copy, keypoints)
    process_image(img_copy, "blobs/2-blobs.jpg", save=False)

    keypoints = merge_blobs(keypoints)
    img_copy = img.copy()
    draw_keypoints(img_copy, keypoints)
    process_image(img_copy, "blobs/3-blobs-merged.jpg", save=False)
