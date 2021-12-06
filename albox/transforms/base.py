#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/11/26

@author: Jonathan Zhang <jz1248it@gmail.com>
"""
import abc
from abc import ABCMeta

from typing import Any, List
from albox.datasets.base import ODTarget, ODDatasetItem


class AlboxBaseObjectDetectionTransform(metaclass=ABCMeta):
    """
    Base class for object detection data transform
    """

    @abc.abstractmethod
    def __call__(self, img: Any, target: ODTarget) -> ODDatasetItem:
        raise NotImplementedError(f"{self.__class__.__name__} not implemented.")


class Compose(AlboxBaseObjectDetectionTransform):

    def __init__(self, *transforms: AlboxBaseObjectDetectionTransform):
        self.transforms = transforms

    def __call__(self, img: Any, target: ODTarget) -> ODDatasetItem:
        for transform in self.transforms:
            img, target = transform(img, target)

        return img, target
