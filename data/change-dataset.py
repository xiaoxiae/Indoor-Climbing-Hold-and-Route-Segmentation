"""A script for converting the images of one dataset to images of another (including
upscaling/downscaling). Done as mapping one-to-one via alphabetical sort."""
import json
import os

ANNOTATIONS = "sm/annotation.json"
FOLDER_TO = "/aux/Climber Tools/smichoff/boulder/whole/images/converted-downsized"
SCALE_FACTOR = 2


with open(ANNOTATIONS, 'r') as stream:
    try:
        data = json.loads(stream.read())
    except yaml.YAMLError as exc:
        print(exc)

if len(data["_via_img_metadata"]) != len(os.listdir(FOLDER_TO)):
    print("Mismatched to folder file counts and project files!")
    quit()

# clear filepaths
data["_via_settings"]['core']['filepath'] = {}

img_data = data["_via_img_metadata"]

# the actual conversion
files_to = sorted(list(os.listdir(FOLDER_TO)))
files_to_keys = []
files_from_keys = []
for i, file_from_key in enumerate(sorted(list(img_data))):
    file_to = files_to[i]
    file_to_path = os.path.join(FOLDER_TO, file_to)

    file_to_size = os.path.getsize(file_to_path)
    file_to_key = file_to + str(file_to_size)

    files_to_keys.append(file_to_key)

    img_data[file_to_key] = img_data[file_from_key]

    files_from_keys.append(file_from_key)

    img_data[file_to_key]["filename"] = file_to
    img_data[file_to_key]["size"] = file_to_size

    new_regions = []

    for region in img_data[file_to_key]["regions"]:
        for c in "xy":
            for i in range(len(region["shape_attributes"][f"all_points_{c}"])):
                v = int(region["shape_attributes"][f"all_points_{c}"][i] / SCALE_FACTOR)
                region["shape_attributes"][f"all_points_{c}"][i] = v

for k in files_from_keys:
    if k not in files_to_keys:
        del img_data[k]

data["_via_image_id_list"] = files_to_keys

path, ext = os.path.splitext(ANNOTATIONS)

with open(path + "_out" + ext, "w") as f:
    f.write(json.dumps(data))
