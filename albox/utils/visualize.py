#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/11/27

@author: Jonathan Zhang <jz1248it@gmail.com>
"""
import random
from typing import Tuple

import numpy as np
import cv2
from datasets.base import AlboxBaseObjectDetectionDataset


def visualize_detection_dataset(dataset: AlboxBaseObjectDetectionDataset):
    print(dataset)
    color_maps = [generate_color() for i in range(len(dataset.classes) - 1)]
    for img, target in dataset:
        img = np.copy(img)
        for i in range(len(target["labels"])):
            x1, y1, x2, y2 = target["boxes"][i].int().tolist()
            cv2.rectangle(img, (x1, y1), (x2, y2), color=color_maps[target["labels"][i].item()-1], thickness=2)
        cv2.imshow(target["image_name"], img)
        cv2.waitKey(0)


def generate_color() -> Tuple[int, int, int]:
    return random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)
