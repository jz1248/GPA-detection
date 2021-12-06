#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/11/8

@author: Jonathan Zhang <jz1248it@gmail.com>
"""
import abc
import sys
from typing import Any, Dict, Optional, AnyStr, List, Tuple, Callable

# from torchvision.datasets.vision import VisionDataset
from torch.utils.data.dataset import Dataset

ODTarget = Dict[AnyStr, Any]
ODDatasetItem = Tuple[Any, ODTarget]


class AlboxBaseObjectDetectionDataset(Dataset, metaclass=abc.ABCMeta):
    """
    Base class for object detection dataset.
    """

    def __init__(self, root: Optional[AnyStr],
                 classes: List[AnyStr],
                 transforms: Optional[Callable] = None,
                 bbox_type: AnyStr = 'hbb',
                 verbose: bool = False):
        assert classes[0] == '__background__'
        # super().__init__(root, transforms)
        self.root = root
        self.transforms = transforms
        self.bbox_type = bbox_type
        self.classes = classes
        self._verbose = verbose

        self.phase = "unspecified_phase"
        self.dataset_name = self.__class__.__name__
        self.repr_dict = dict()

    @abc.abstractmethod
    def __getitem__(self, index: int) -> ODDatasetItem:
        """
        Returns an image (in numpy/PIL H, W, C format) and its targets (annotations represented by dict)

        target dict:

        - boxes (FloatTensor[N, 4]) [x0, y0, x1, y1]
        - labels (Int64Tensor[N])
        - image_id (Int64Tensor[1])
        - area (Tensor[N]): area of each box
        - iscrowd (UInt8Tensor[N])
        - difficulty (UInt8Tensor[N])

        dict format is inspired by
            `this tutorial <https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html#defining-the-dataset>`_

        Args:
            index:

        Returns:
            (image, targets): image and a dict contains annotation info
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    def __repr__(self) -> str:
        head = "Dataset " + self.dataset_name
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]

        body += self.extra_repr().splitlines()

        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def get_num_classes(self) -> int:
        return len(self.classes) - 1

    @abc.abstractmethod
    def get_num_annotations(self, index: int = None, label: AnyStr = None) -> int:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_image_sizes(self) -> List[Tuple[int]]:
        """Returns a size list of every image in (H, W, C) format"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_image_path(self, index: int) -> str:
        """Returns absolute path of image at index"""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_all_annotations(self):
        raise NotImplementedError()

    def extra_repr(self) -> str:
        s = f"Classes ({self.get_num_classes()}): {self.classes[1:] if self.classes else 'Undefined'}"
        for k, v in self.repr_dict.items():
            s += '\n' + f"{k}: {v}"
        return s

    def _verbose_print(self, *content: Any, stream=sys.stdout):
        """Print to stream only if in verbose mode"""
        if self._verbose:
            print(*content, file=stream)

    def collate_fn(self) -> Callable:
        return collate_fn_for_detection


def collate_fn_for_detection(batch):
    return tuple(zip(*batch))

