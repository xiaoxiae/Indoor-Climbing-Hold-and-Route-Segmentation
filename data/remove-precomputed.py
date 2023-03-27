"""A script for removing precomputed annotations (for exporting to Kaggle)."""
import json
import os

ANNOTATIONS = "bh/annotation.json"


def has_prelabeled(regions):
    for region in regions:
        if region["region_attributes"]["label_type"] == "prelabeled":
            return True
    return False


with open(ANNOTATIONS, 'r') as stream:
    try:
        data = json.loads(stream.read())
    except yaml.YAMLError as exc:
        print(exc)

img_data = data["_via_img_metadata"]

# the actual conversion
for i, file_from_key in enumerate(sorted(list(img_data))):
    to_delete = False

    if has_prelabeled(img_data[file_from_key]["regions"]):
        img_data[file_from_key]["regions"] = []

path, ext = os.path.splitext(ANNOTATIONS)

with open(path + "_out" + ext, "w") as f:
    f.write(json.dumps(data))
