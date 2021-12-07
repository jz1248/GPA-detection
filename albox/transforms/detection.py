#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/11/27

@author: Jonathan Zhang <jz1248it@gmail.com>
"""
from typing import Any

from albox.datasets.base import ODTarget, ODDatasetItem
from albox.transforms.base import AlboxBaseObjectDetectionTransform


class RandomScale(AlboxBaseObjectDetectionTransform):

    def __init__(self, min_scale: int, max_scale: int):
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, img: Any, target: ODTarget) -> ODDatasetItem:
        raise NotImplementedError()
