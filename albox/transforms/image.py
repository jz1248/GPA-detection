#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/12/1

@author: Jonathan Zhang <jz1248it@gmail.com>
"""
from typing import Any

import torch
from torchvision.transforms import transforms as T
from PIL import Image

from albox.datasets.base import ODTarget, ODDatasetItem
from albox.transforms.base import AlboxBaseObjectDetectionTransform

Tensor = torch.Tensor


class ToTensor(AlboxBaseObjectDetectionTransform):

    def __init__(self):
        self.to_tensor = T.ToTensor()

    def __call__(self, img: Any, target: ODTarget) -> ODDatasetItem:
        return self.to_tensor(img), target


class Resize(AlboxBaseObjectDetectionTransform):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.resize = T.Resize(size, interpolation)

    def __call__(self, img: Any, target: ODTarget) -> ODDatasetItem:
        img_resized = self.resize(img)
        if isinstance(img, torch.Tensor):
            # C, H, W
            ratio_x = img_resized.shape[2] / img.shape[2]
            ratio_y = img_resized.shape[1] / img.shape[1]
        else:
            # PIL: H, W, C
            ratio_x = img_resized.shape[1] / img.shape[1]
            ratio_y = img_resized.shape[0] / img.shape[0]
        # resize boxes
        resized_boxes = [[b[0] * ratio_x, b[1] * ratio_y, b[2] * ratio_x, b[3] * ratio_y]
                         for b in target["boxes"].tolist()]
        resized_area = [ratio_y * ratio_x * a for a in target["area"].tolist()]
        target["boxes"] = torch.tensor(resized_boxes)
        target["area"] = torch.tensor(resized_area)
        return img_resized, target


class Normalize(AlboxBaseObjectDetectionTransform):

    def __init__(self, mean, std, inplace=False):
        self.normalize = T.Normalize(mean, std, inplace)

    def __call__(self, img: Any, target: ODTarget) -> ODDatasetItem:
        return self.normalize(img), target
