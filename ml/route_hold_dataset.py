import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from detectron2.structures import polygons_to_bitmask
from torch.utils.data import Dataset


class RouteMatching(Dataset):
    def __init__(
        self, dataset_dict, transform=None, triplet=False, augment_colors=True
    ) -> None:
        super().__init__()
        self.orig_data = dataset_dict
        self.data = {}
        self.transform = transform
        self.triplet = triplet
        self.augment_colors = augment_colors
        for image_idx, image_obj in enumerate(self.orig_data):
            data_holds = []
            for hold_idx, poly in enumerate(image_obj["annotations"]):
                hold_data = {}
                if poly["category_id"] == 1:
                    # Volumes are considered part of the wall, not part of a route -> skip
                    continue
                if poly["route_id"] is None:
                    continue
                read_path = image_obj["file_name"].split("/")[:-1]
                hold_data["hold_file_name"] = os.path.join(
                    *read_path, "holds", f"{image_idx}-{hold_idx}.jpg"
                )
                hold_data["route_id"] = poly["route_id"]
                data_holds.append(hold_data)
            self.data[image_obj["file_name"]] = data_holds

    def __len__(self):
        return len([item for sublist in self.data.values() for item in sublist])

    def _load_hold(self, filename):
        hold = cv2.imread(filename)
        hold = torch.tensor(hold).permute(2, 0, 1)
        return hold

    def __getitem__(self, index):
        # weigh based on number of instances per image
        weights = np.array([len(sublist) for sublist in self.data.values()])
        weights = weights / self.__len__()

        random_image = random.choices(list(self.data.keys()), weights)[0]
        holds = self.data[random_image]

        random_hold_idx = random.randint(0, len(holds) - 1)
        random_hold = holds[random_hold_idx]

        holds_same_route = []
        for i, hold in enumerate(holds):
            if hold["route_id"] == random_hold["route_id"] and i != random_hold_idx:
                holds_same_route.append(hold)

        holds_different_route = [
            hold for hold in holds if hold["route_id"] != random_hold["route_id"]
        ]

        positive_hold = random.choice(holds_same_route)
        negative_hold = random.choice(holds_different_route)

        random_hold = self._load_hold(random_hold["hold_file_name"])
        positive_hold = self._load_hold(positive_hold["hold_file_name"])
        negative_hold = self._load_hold(negative_hold["hold_file_name"])

        if random.random() > 0.5 and self.augment_colors:
            brightness_value = random.uniform(0.75, 1.3)
            contrast_value = random.uniform(0.75, 1.3)
            saturation_value = random.uniform(0.5, 2)
            hue_value = random.uniform(-0.3, 0.3)
            random_hold = TF.adjust_brightness(random_hold, brightness_value)
            random_hold = TF.adjust_contrast(random_hold, contrast_value)
            random_hold = TF.adjust_saturation(random_hold, saturation_value)
            random_hold = TF.adjust_hue(random_hold, hue_value)

            positive_hold = TF.adjust_brightness(positive_hold, brightness_value)
            positive_hold = TF.adjust_contrast(positive_hold, contrast_value)
            positive_hold = TF.adjust_saturation(positive_hold, saturation_value)
            positive_hold = TF.adjust_hue(positive_hold, hue_value)

            negative_hold = TF.adjust_brightness(negative_hold, brightness_value)
            negative_hold = TF.adjust_contrast(negative_hold, contrast_value)
            negative_hold = TF.adjust_saturation(negative_hold, saturation_value)
            negative_hold = TF.adjust_hue(negative_hold, hue_value)

        if self.transform:
            random_hold = self.transform(random_hold)
            positive_hold = self.transform(positive_hold)
            negative_hold = self.transform(negative_hold)

        if self.triplet:
            return random_hold, positive_hold, negative_hold

        if index % 2 == 0:
            target = torch.tensor(1, dtype=torch.float)
            return random_hold, positive_hold, target
        else:
            target = torch.tensor(0, dtype=torch.float)
            return random_hold, negative_hold, target


def create_hold_dataset(d_dict):
    for image_idx, image_obj in enumerate(d_dict):
        img = cv2.imread(image_obj["file_name"])
        write_path = image_obj["file_name"].split("/")[:-1]
        write_path = os.path.join(*write_path, "holds")
        os.makedirs(write_path, exist_ok=True)
        img = torch.tensor(img).permute(2, 0, 1)
        for hold_idx, poly in enumerate(image_obj["annotations"]):
            if poly["category_id"] == 1:
                # Volumes are considered part of the wall, not part of a route -> skip
                continue
            if poly["route_id"] is None:
                continue
            bitmask = torch.tensor(
                polygons_to_bitmask(
                    poly["segmentation"],
                    height=image_obj["height"],
                    width=image_obj["width"],
                )
            )
            masked_output = img * bitmask.int().float()
            box_coords = poly["bbox"]
            hold = masked_output[
                :, box_coords[1] : box_coords[3], box_coords[0] : box_coords[2]
            ]
            hold = hold.permute(1, 2, 0).numpy()
            cv2.imwrite(os.path.join(write_path, f"{image_idx}-{hold_idx}.jpg"), hold)
