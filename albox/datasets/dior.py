#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/11/9

@author: Jonathan Zhang <jz1248it@gmail.com>
"""
import os
from typing import Any, Dict, Optional, Callable, AnyStr, List, Tuple

from tqdm import tqdm

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

import cv2
import torch

from albox.datasets.base import AlboxBaseObjectDetectionDataset, ODTarget, ODDatasetItem
from albox.utils import const, file_support

const.DATASET_DIOR_CLASSES = \
    ["__background__", "airplane", "airport", "baseballfield", "basketballcourt", "bridge", "chimney", "dam",
     "Expressway-Service-area", "Expressway-toll-station", "golffield", "groundtrackfield", "harbor", "overpass",
     "ship", "stadium", "storagetank", "tenniscourt", "trainstation", "vehicle", "windmill"]


class DIORDataset(AlboxBaseObjectDetectionDataset):
    """
    `DIOR Dataset <http://www.escience.cn/people/gongcheng/DIOR.html>`_ for Albox
    """

    img_file_ext = "jpg"
    annotation_file_ext = "xml"

    LEGAL_PHASE = ["train", "val", "test"]

    def __init__(self, root: str, phase: str = 'train', transforms: Optional[Callable] = None, verbose: bool = False):
        super().__init__(root, const.DATASET_DIOR_CLASSES, transforms, "hbb", verbose)
        assert phase in self.LEGAL_PHASE
        self.phase = phase
        self.repr_dict["Phase"] = phase

        self.path_img = dict()
        self.path_img["train"] = os.path.join(self.root, "JPEGImages-trainval")
        self.path_img["val"] = os.path.join(self.root, "JPEGImages-trainval")
        self.path_img["test"] = os.path.join(self.root, "JPEGImages-test")

        self.path_annotation = dict()
        self.path_annotation["train"] = os.path.join(self.root, "Annotations")
        self.path_annotation["val"] = os.path.join(self.root, "Annotations")
        self.path_annotation["test"] = os.path.join(self.root, "Annotations")

        self.img_index = dict()
        self.img_index[self.phase] = file_support.read_txt_file_by_lines(
            os.path.join(self.root, "ImageSets", "Main", f"{self.phase}.txt"))

        # cache accessed annotation for faster loading and statistics
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

        img = self._load_image(index)
        # load annotations from cache
        target = self._load_annotation(index)

        # apply transforms
        if self.transforms:
            img, target = self.transforms(img, target)

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

    def get_num_annotations(self, index: int = None, label: AnyStr = None) -> int:
        if index is not None and label is not None:
            raise ValueError("Argument `index` and `label` are mutual exclusive.")
        self._cache_all_annotations()   # make sure that all annotations are cached
        if index is not None:
            return self._label_annotation_count[index]
        if label is not None:
            return self._label_annotation_count[self.classes.index(label)]
        return self._total_annotation_count

    def get_image_path(self, index: int) -> str:
        img_index = self.img_index[self.phase][index]
        return os.path.join(self.path_img[self.phase], f"{img_index}{os.extsep}{self.img_file_ext}")

    def _cache_all_annotations(self):
        """Cache annotations of all images"""
        if None not in self._annotation_cache[self.phase]:
            return

        for index, img_index in enumerate(
                tqdm(self.img_index[self.phase], "Loading annotations")
                if self._verbose else self.img_index[self.phase]):
            self._load_annotation(index)

    def _load_image(self, index: int):
        # load image
        img_path = self.get_image_path(index)
        img = cv2.imread(img_path)
        img = img[..., ::-1].copy()    # convert BGR to RGB
        return img

    def _load_annotation(self, index: int) -> ODTarget:
        """Load annotations from disk and save to cache. Statistics are done at the same time."""
        img_index = self.img_index[self.phase][index]
        target = self._annotation_cache[self.phase][index]
        if not target:
            if not self._total_annotation_count:
                self._total_annotation_count = 0
            if not self._label_annotation_count:
                self._label_annotation_count = [0 for _ in range(len(self.classes))]

            # cache missing. load from disk
            annotation_path = self._concat_annotation_path(img_index)

            # parse annotations and save to cache
            target = self._parse_annotation(annotation_path, int(img_index), img_index)
            assert target is not None
            self._annotation_cache[self.phase][index] = target

            # count annotation number for each label and for total
            self._total_annotation_count += len(target["labels"])
            for label in target["labels"]:
                self._label_annotation_count[label] += 1
        return target

    def _concat_annotation_path(self, img_index: str) -> str:
        return os.path.join(self.path_annotation[self.phase], f"{img_index}{os.extsep}{self.annotation_file_ext}")

    def _parse_annotation(self, path: str, image_id: int, image_name: str) -> ODTarget:
        target = dict()
        boxes, labels, area, difficulty = [], [], [], []

        tree = ET.ElementTree(file=path)
        for obj in tree.findall("object"):
            label = obj.find("name").text
            bbox = obj.find("bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            labels.append(self.classes.index(label))
            boxes.append([xmin, ymin, xmax, ymax])
            area.append((xmax - xmin) * (ymax - ymin))

        # common dict items
        target["boxes"] = torch.FloatTensor(boxes)
        target["labels"] = torch.LongTensor(labels)
        target["area"] = torch.FloatTensor(area)
        target["image_id"] = torch.LongTensor([image_id])
        target["image_name"] = image_name
        target["iscrowd"] = torch.zeros(len(labels), dtype=torch.uint8)
        target["difficulty"] = torch.zeros(len(labels), dtype=torch.uint8)

        return target

