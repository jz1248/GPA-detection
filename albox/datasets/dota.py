#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
`DOTA Dataset <https://captain-whu.github.io/DOTA/dataset.html>`_ support for PyTorch

Created on 2021/11/8

@author: Jonathan Zhang <jz1248it@gmail.com>
"""
import os

import torch

from typing import Optional, Callable, AnyStr, Tuple, List

from tqdm import tqdm
import cv2

from albox.datasets.base import AlboxBaseObjectDetectionDataset, ODDatasetItem, ODTarget
from albox.utils import file_support
from albox.utils import const

# Define classes of DOTA V2 dataset
const.DATASET_DOTA_V2_CLASSES = \
    ["__background__", "plane", "ship", "storage-tank", "baseball-diamond", "tennis-court", "basketball-court",
     "ground-track-field", "harbor", "bridge", "large-vehicle", "small-vehicle", "helicopter", "roundabout",
     "soccer-ball-field", "swimming-pool", "container-crane", "airport", "helipad"]


class DOTADatasetV2(AlboxBaseObjectDetectionDataset):
    """
    `DOTA Dataset <https://captain-whu.github.io/DOTA/dataset.html>`_ for Albox
    """

    img_file_ext = "png"
    annotation_file_ext = "txt"

    LEGAL_PHASE = ["train", "val"]

    def __init__(self, root: str,
                 phase: str = 'train',
                 transforms: Optional[Callable] = None,
                 bbox_type: AnyStr = 'hbb',
                 verbose: bool = False):
        assert bbox_type in ["hbb", "obb"], ValueError(f"Unrecognized bounding box type: {bbox_type}")
        super().__init__(root, const.DATASET_DOTA_V2_CLASSES, transforms, bbox_type, verbose)

        assert phase in self.LEGAL_PHASE
        self.phase = phase
        self.repr_dict["Phase"] = phase

        self.path_img = dict()
        self.path_img["train"] = os.path.join(self.root, "train", "images")
        self.path_img["val"] = os.path.join(self.root, "val", "images")

        self.path_annotation = dict()
        self.path_annotation["train"] = os.path.join(self.root, "train", "labelTxt-v2.0",
                                                     "DOTA-v2.0_train_hbb" if self.bbox_type == "hbb" else "DOTA-v2.0_train")
        self.path_annotation["val"] = os.path.join(self.root, "val", "labelTxt-v2.0",
                                                   "DOTA-v2.0_val_hbb" if self.bbox_type == "hbb" else "DOTA-v2.0_val")

        self.img_index = dict()
        self.img_index[self.phase] = file_support.get_file_path_with_ext(self.path_img[self.phase], self.img_file_ext)

        # define accessed annotation cache for faster loading and statistics
        self._annotation_cache = dict()
        self._annotation_cache[self.phase] = [None for _ in range(len(self.img_index[self.phase]))]
        self._img_sizes = dict()
        self._img_sizes[self.phase] = [None for _ in range(len(self.img_index[self.phase]))]
        self._total_annotation_count = None
        self._label_annotation_count = None
        self.repr_dict["Total annotations"] = self.get_num_annotations()
        self.repr_dict["Number of annotations for each class"] = \
            ", ".join([f"{self.classes[i]}({self._label_annotation_count[i]})" for i in range(1, len(self.classes))])
        self.repr_dict["Max number ground truth boxes"] = \
            max([(len(t["labels"]), t["image_name"]) for t in self._annotation_cache[self.phase]], key=lambda tu: tu[0])

    def __getitem__(self, index: int) -> ODDatasetItem:
        if self.bbox_type == "hbb":
            img, target = self._get_hbb_sample(index)
        elif self.bbox_type == "obb":
            img, target = self._get_obb_sample(index)
        else:
            raise ValueError(f"Unrecognized bounding box type: {self.bbox_type}")

        return img, target

    def __len__(self) -> int:
        return len(self.img_index[self.phase])

    def get_image_sizes(self) -> List[Tuple[int]]:
        if None not in self._img_sizes[self.phase]:
            return self._img_sizes[self.phase]

        iterator = range(len(self))
        iterator = tqdm(iterator, "Getting image sizes") if self._verbose else iterator
        for i in iterator:
            if self._img_sizes[self.phase][i] is None:
                img = self._load_image(i)
                self._img_sizes[self.phase][i] = img.shape
        return self._img_sizes[self.phase]

    def get_all_annotations(self):
        self._cache_all_annotations()
        return self._annotation_cache[self.phase]

    def get_image_path(self, index: int) -> str:
        img_index = self.img_index[self.phase][index]
        return os.path.join(self.path_img[self.phase], f"{img_index}{os.extsep}{self.img_file_ext}")

    def _load_image(self, index: int):
        # load image
        img_path = img_path = self.get_image_path(index)
        img = cv2.imread(img_path)
        img = img[..., ::-1].copy()    # convert BGR to RGB
        return img

    def _get_hbb_sample(self, index: int) -> ODDatasetItem:
        """Get sample in horizontal bounding box format"""
        img = self._load_image(index)

        # load annotations from cache
        target = self._load_annotation(index)

        # apply transforms
        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def get_num_annotations(self, index: int = None, label: AnyStr = None) -> int:
        if index is not None and label is not None:
            raise ValueError("Argument `index` and `label` are mutual exclusive.")
        self._cache_all_annotations()   # make sure that all annotations are cached
        if index is not None:
            return self._label_annotation_count[index]
        if label is not None:
            return self._label_annotation_count[self.classes.index(label)]
        return self._total_annotation_count

    def _cache_all_annotations(self):
        """Cache annotations of all images"""
        if None not in self._annotation_cache[self.phase]:
            return

        for index, img_index in enumerate(
                tqdm(self.img_index[self.phase], "Loading annotations")
                if self._verbose else self.img_index[self.phase]):
            self._load_annotation(index)

    def _load_annotation(self, index: int) -> ODTarget:
        """Load annotations from disk and save to cache. Statistics are done at the same time."""
        img_index = self.img_index[self.phase][index]
        target = self._annotation_cache[self.phase][index]
        if not target:
            # cache missing. load from disk
            if not self._total_annotation_count:
                self._total_annotation_count = 0
            if not self._label_annotation_count:
                self._label_annotation_count = [0 for _ in range(len(self.classes))]
            # read from disk
            annotation_path = self._concat_annotation_path(img_index)
            annotation = file_support.read_txt_file_by_lines(annotation_path)

            # parse annotations and save to cache
            if self.bbox_type == "hbb":
                target = self._parse_hbb_annotation(annotation, int(img_index[1:]), img_index)
            else:
                # TODO: parsing obb annotations
                raise NotImplementedError(f"Oriented bounding box format for DOTA V2 is not supported")
            self._annotation_cache[self.phase][index] = target

            # count annotation number for each label and for total
            self._total_annotation_count += len(target["labels"])
            for label in target["labels"]:
                self._label_annotation_count[label] += 1
        return target

    def _concat_annotation_path(self, img_index: str) -> str:
        return os.path.join(self.path_annotation[self.phase], f"{img_index}{os.extsep}{self.annotation_file_ext}")

    def _parse_hbb_annotation(self, content: List[AnyStr], image_id: int, image_name: str) -> ODTarget:
        target = dict()
        boxes, labels, area, difficulty = [], [], [], []
        meta_dict = dict()

        for i, c in enumerate(content):
            if ":" in c:
                # handle meta data
                meta_item = c.split(":")
                meta_dict[meta_item[0]] = meta_item[1]
            else:
                # handle bbox
                l = c.split()
                x1, y1, x2, y2, x3, y3, x4, y4, category, difficult = l
                x1, y1, x3, y3 = float(x1), float(y1), float(x3), float(y3)
                boxes.append([x1, y1, x3, y3])
                try:
                    label_id = self.classes.index(category)
                except ValueError:
                    print(f"Unrecognized category: {category}, line: {i+1}. This object will be ignored.")
                    continue
                labels.append(label_id)
                area.append((x3 - x1) * (y3 - y1))
                difficulty.append(int(difficult))

        # common dict items
        target["boxes"] = torch.FloatTensor(boxes)
        target["labels"] = torch.LongTensor(labels)
        target["area"] = torch.FloatTensor(area)
        target["image_id"] = torch.LongTensor([image_id])
        target["image_name"] = image_name
        target["iscrowd"] = torch.zeros(len(labels), dtype=torch.uint8)
        target["difficulty"] = torch.ByteTensor(difficulty)

        # custom dict items
        for k, v in meta_dict.items():
            target[k] = v
        return target

    def _get_obb_sample(self, index: int) -> ODDatasetItem:
        """Get sample in oriented bounding box format"""
        # TODO: implement oriented bounding box support for DOTA dataset
        raise NotImplementedError(f"Oriented bounding box format for DOTA V2 is not supported")
