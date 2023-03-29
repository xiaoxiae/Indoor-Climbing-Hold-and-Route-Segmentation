from utils import *

from detectron2.evaluation.coco_evaluation import *

instances = []
for file in EVALUATION_DATA[:1]:
    stub = file[8:].replace("/", "-")

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

    instances.append(to_detectron_format(img, list(hold_approximations.values())))

for instance in instances:
    print(instances_to_coco_json(instance))

print(annotations)
#ev = COCOEvaluator()
