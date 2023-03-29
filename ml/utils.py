import json
import os
import shutil
from typing import List

import cv2
import detectron2.data.transforms as T
import numpy as np
import torch
from detectron2.structures import BoxMode
from imantics import Mask
from torch.utils.data import random_split
from detectron2.structures import polygons_to_bitmask
from tqdm import tqdm
import matplotlib.pyplot as plt


def cp_files(file_list: List[str], destination: str) -> None:
    for f in file_list:
        shutil.copyfile(f, os.path.join(destination, os.path.split(f)[-1]))


def move_train_test(
    img_dir: str,
    annotation_filename: str = "annotation.json",
    train_split: float = 0.8,
) -> None:
    annotation_json = os.path.join(img_dir, annotation_filename)
    dataset_filenames = []
    with open(annotation_json, "r") as f:
        data = json.load(f)
        img_annotations = data["_via_img_metadata"]
    for k, image in img_annotations.items():
        if not image["regions"]:
            # Skip images that have no region annotation, i.e. not annotated images
            continue
        if not all(
            [
                region["region_attributes"]["label_type"] == "handlabeled"
                for region in image["regions"]
            ]
        ):
            # Skip images which have regions that are not handlabeled
            continue
        dataset_filenames.append({k: image})
    train_size = int(train_split * len(dataset_filenames))
    test_size = len(dataset_filenames) - train_size
    train_dataset_files, test_dataset_files = random_split(
        dataset_filenames,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    train_dataset_filenames = [
        os.path.join(img_dir, values["filename"])
        for f in train_dataset_files
        for values in f.values()
    ]
    test_dataset_filenames = [
        os.path.join(img_dir, values["filename"])
        for f in test_dataset_files
        for values in f.values()
    ]
    os.makedirs(os.path.join(img_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(img_dir, "test"), exist_ok=True)
    cp_files(train_dataset_filenames, os.path.join(img_dir, "train"))
    cp_files(test_dataset_filenames, os.path.join(img_dir, "test"))
    with open(os.path.join(img_dir, "train", annotation_filename), "w") as f:
        data["_via_img_metadata"] = {
            k: v for f in train_dataset_files for k, v in f.items()
        }
        json.dump(data, f)
    with open(os.path.join(img_dir, "test", annotation_filename), "w") as f:
        data["_via_img_metadata"] = {
            k: v for f in test_dataset_files for k, v in f.items()
        }
        json.dump(data, f)


def create_dataset_dicts(
    img_dir: str, annotation_filename: str = "annotation.json", skip_no_routes=False
) -> List[dict]:
    hold_class_mapping = {"hold": 0, "volume": 1}
    annotation_json = os.path.join(img_dir, annotation_filename)
    with open(annotation_json, "r") as f:
        img_annotations = json.load(f)["_via_img_metadata"]
    dataset = []
    for id, image in enumerate(img_annotations.values()):
        if not image["regions"]:
            # Skip images that have no region annotation, i.e. not annotated images
            continue
        if not all(
            [
                region["region_attributes"]["label_type"] == "handlabeled"
                for region in image["regions"]
            ]
        ):
            # Skip images which have regions that are not handlabeled
            continue

        if skip_no_routes:
            if all(
                [
                    not "route_id" in region["region_attributes"]
                    for region in image["regions"]
                ]
            ):
                continue
        record = {}
        image_filename = os.path.join(img_dir, image["filename"])
        height, width = cv2.imread(image_filename).shape[:2]

        # Create dataset dict according to detectron2 dataset specification
        # https://detectron2.readthedocs.io/en/latest/tutorials/datasets.html#standard-dataset-dicts

        record["file_name"] = image_filename
        record["height"] = height
        record["width"] = width
        record["image_id"] = id

        annotation_objects = []
        for region in image["regions"]:
            anno = region["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            # Merge point lists and flatten to [x_1, y_1, ..., x_n y_n] format
            polygons = [(x, y) for (x, y) in zip(px, py)]
            polygons = np.ravel(polygons).tolist()
            annotation_obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [polygons],
                "category_id": hold_class_mapping[
                    region["region_attributes"]["hold_type"]
                ],
                "route_id": region["region_attributes"].get("route_id", None),
            }
            annotation_objects.append(annotation_obj)
        record["annotations"] = annotation_objects
        dataset.append(record)
    return dataset


def basic_augmentation(cfg):
    augs = [T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, sample_style="choice")]
    return augs


def create_train_augmentations(cfg):
    augs = [
        T.RandomApply(T.RandomCrop("relative", (0.5, 0.5))),
        T.ResizeShortestEdge(cfg.INPUT.MIN_SIZE_TRAIN, sample_style="choice"),
        T.RandomRotation([-20, 20]),
        T.RandomBrightness(0.75, 1.3),
        T.RandomContrast(0.75, 1.3),
        T.RandomSaturation(0.5, 2),
        T.RandomFlip(vertical=False),
    ]
    return augs


def annotate_polygons(
    predictor, img_dir: str, annotation_filename: str = "annotation.json"
):
    annotation_json = os.path.join(img_dir, annotation_filename)
    with open(annotation_json, "r") as f:
        data = json.load(f)
        img_annotations = data["_via_img_metadata"]
    for k, image in tqdm(img_annotations.items()):
        if image["regions"]:
            if not all(
                [
                    region["region_attributes"].get("label_type", "") != "prelabeled"
                    for region in image["regions"]
                ]
            ):
                # We only "relabel" prelabeled images
                continue
        image_filename = os.path.join(img_dir, image["filename"])
        im = cv2.imread(image_filename)
        outputs = predictor(im)
        regions = []
        for i in range(
            len(outputs["instances"])
        ):  # Workaround since Instances obj is not iterable
            poly = outputs["instances"][i]
            pred_mask = poly.pred_masks.cpu().squeeze().numpy().astype(np.uint8)
            polygons = Mask(pred_mask).polygons().points[0]
            # Reduce number of polygon points to make manual fixing easier
            epsilon = 0.01 * cv2.arcLength(polygons, True)
            polygons = cv2.approxPolyDP(polygons, epsilon, True)
            points_x = polygons[:, 0, 0].tolist()
            points_y = polygons[:, 0, 1].tolist()
            region_dict = {
                "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": points_x,
                    "all_points_y": points_y,
                },
                "region_attributes": {"label_type": "prelabeled"},
            }
            regions.append(region_dict)
        img_annotations[k]["regions"] = regions
    new_filename = "".join(
        annotation_filename.split(".")[:-1]
        + ["-prelabeled."]
        + annotation_filename.split(".")[-1:]
    )
    file_path = os.path.join(img_dir, new_filename)
    with open(file_path, "w") as f:
        json.dump(data, f)


def plot_routes(routes_dict, image_obj):
    figures_axes = []
    for route_id, holds in routes_dict.items():
        img_orig = cv2.imread(image_obj["file_name"])
        img = torch.tensor(img_orig).permute(2, 0, 1)
        final_bitmask = torch.zeros_like(img)
        for hold_idx in holds:
            poly = image_obj["annotations"][hold_idx]
            bitmask = torch.tensor(
                polygons_to_bitmask(
                    poly["segmentation"],
                    height=image_obj["height"],
                    width=image_obj["width"],
                )
            ).int()
            final_bitmask = final_bitmask + bitmask
        final_bitmask = final_bitmask.bool().float()
        masked_output = img * final_bitmask
        img_routes = masked_output.int().permute(1, 2, 0).cpu().detach().numpy()
        fig, ax = plt.subplots(ncols=2)
        img_routes = cv2.resize(
            img_routes, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR_EXACT
        )
        img_orig = cv2.resize(
            img_orig, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR_EXACT
        )
        ax[0].imshow(img_routes[:, :, ::-1])
        ax[0].axis("off")
        ax[1].imshow(img_orig[:, :, ::-1])
        ax[1].axis("off")
        figures_axes.append((fig, ax))
    return figures_axes


def plot_routes_instances(routes_dict, instances, img):
    figures_axes = []
    for route_id, holds in routes_dict.items():
        img_torch = torch.tensor(img).permute(2, 0, 1)
        final_bitmask = torch.zeros_like(img_torch)
        for hold_idx in holds:
            bitmask = instances[hold_idx].pred_masks.cpu().squeeze().int().float()
            final_bitmask = final_bitmask + bitmask
        final_bitmask = final_bitmask.bool().float()
        masked_output = img_torch * final_bitmask
        img_routes = masked_output.int().permute(1, 2, 0).cpu().detach().numpy()
        fig, ax = plt.subplots(ncols=2)
        img_routes = cv2.resize(
            img_routes, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR_EXACT
        )
        img_orig = cv2.resize(
            img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR_EXACT
        )
        ax[0].imshow(img_routes[:, :, ::-1])
        ax[0].axis("off")
        ax[1].imshow(img_orig[:, :, ::-1])
        ax[1].axis("off")
        figures_axes.append((fig, ax))
    return figures_axes


def instance_to_hold(instance, img, transforms, device):
    img_torch = torch.tensor(img).permute(2, 0, 1)
    pred_mask = instance.pred_masks.cpu().squeeze().int().float()
    box_coords = instance.pred_boxes.tensor.flatten().int()
    masked_img = img_torch * pred_mask
    hold = masked_img[:, box_coords[1] : box_coords[3], box_coords[0] : box_coords[2]]
    hold = transforms(hold)
    hold = hold.to(device)
    return hold
