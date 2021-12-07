#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Adapter datasets for transfer learning

Created on 2021/11/28

@author: Jonathan Zhang <jz1248it@gmail.com>
"""
import os
from typing import Dict, Optional, Callable, List, AnyStr, Tuple

import torch
from tqdm import tqdm

from albox.datasets.base import AlboxBaseObjectDetectionDataset, ODDatasetItem
from albox.utils import file_support


class DomainAdaptationDatasetAdapterForDetection:
    """
    Domain adaptation dataset adapter for object detection

    This class combines source and target datasets,
    provides a higher level abstraction of dataset used for domain adaptation task
    """

    class _DomainAdaptationSubDatasetForDetection(AlboxBaseObjectDetectionDataset):
        def __init__(self, dataset: AlboxBaseObjectDetectionDataset,
                     classes: List[AnyStr],
                     cache_dir: Optional[str] = None,
                     verbose: bool = False):
            super().__init__(None, classes, bbox_type=dataset.bbox_type, verbose=verbose)
            self.dataset = dataset
            self.dataset_name = f"{self.dataset.dataset_name} for domain adaptation"
            self.img_file_ext = self.dataset.img_file_ext
            self.annotation_file_ext = self.dataset.annotation_file_ext

            # statistics
            self._total_annotation_count = 0

            # filter all annotations
            self._active_original_index = []    # map from filtered dataset index to original dataset index
            self._annotation_cache = []
            self._annotation_cache_file_name = \
                f"cache_DA_{dataset.__class__.__name__}_{self.dataset.phase}_{self.get_num_classes()}_{'_'.join(classes[1:])}.pkl"
            file_support.mkdir(cache_dir)
            self._annotation_cache_file = os.path.join(cache_dir, self._annotation_cache_file_name)

            self._image_sizes = None

            if os.path.exists(self._annotation_cache_file):
                # load cache from disk
                cache_dict = file_support.pickle_load(self._annotation_cache_file)
                self._active_original_index = cache_dict["index"]
                self._annotation_cache = cache_dict["cache"]
                self._total_annotation_count = cache_dict["total_count"]
                self._image_sizes = cache_dict["image_sizes"]

                self._verbose_print(f"Loaded {self.dataset.__class__.__name__} annotation for DA from disk cache.")
            else:
                # filter original dataset
                self._verbose_print(f"{self.dataset.__class__.__name__} {self.dataset.phase} annotation for DA is not found at {os.path.abspath(self._annotation_cache_file)}. "
                                    f"Try filtering annotations now.")
                iterator = self.dataset
                iterator = tqdm(iterator, desc=f"Filtering annotations for {self.dataset.__class__.__name__}")\
                    if self._verbose else iterator
                for index, item in enumerate(iterator):
                    img, target = item
                    boxes, labels, area, iscrowd, difficulty = \
                        target["boxes"].tolist(), target["labels"].tolist(), target["area"].tolist(), \
                        target["iscrowd"].tolist(), target["difficulty"].tolist()

                    boxes_new, labels_new, area_new, iscrowd_new, difficulty_new = [], [], [], [], []

                    for i, label in enumerate(labels):
                        label_name = self.dataset.classes[label]
                        if label_name in self.classes:
                            boxes_new.append(boxes[i])
                            labels_new.append(self.classes.index(label_name))
                            area_new.append(area[i])
                            iscrowd_new.append(iscrowd)
                            difficulty_new.append(difficulty)

                    if len(labels_new) == 0:
                        # ignore image with no annotations
                        continue

                    target_new = dict()

                    target_new["boxes"] = torch.FloatTensor(boxes_new)
                    target_new["labels"] = torch.LongTensor(labels_new)
                    target_new["area"] = torch.FloatTensor(area_new)
                    target_new["image_id"] = target["image_id"]
                    target_new["image_name"] = target["image_name"]
                    target_new["iscrowd"] = torch.ByteTensor(iscrowd_new)
                    target_new["difficulty"] = torch.ByteTensor(difficulty_new)

                    # save the mapping relation between self index and original dataset index
                    self._active_original_index.append(index)
                    self._annotation_cache.append(target_new)

                    self._total_annotation_count += len(labels_new)

                # save cache to disk if cache dir is configured and legal
                cache_dict = dict()
                cache_dict["cache"] = self._annotation_cache
                cache_dict["index"] = self._active_original_index
                cache_dict["total_count"] = self._total_annotation_count
                if self._image_sizes is None:
                    self.get_image_sizes()
                cache_dict["image_sizes"] = self._image_sizes
                file_support.pickle_dump(cache_dict, self._annotation_cache_file)
                self._verbose_print(f"Saved {self.dataset.phase} annotations to disk cache: {os.path.abspath(self._annotation_cache_file)}")

            self.repr_dict["Total annotations"] = self.get_num_annotations()
            self.repr_dict["Number of annotations for each class"] = \
                ", ".join([f"{cls}({self.get_num_annotations(label=cls)})" for cls in self.classes[1:]])

        def __getitem__(self, index: int) -> ODDatasetItem:
            original_index = self._active_original_index[index]
            return self.dataset[original_index][0], self._annotation_cache[index]

        def __len__(self) -> int:
            return len(self._active_original_index)

        def get_image_sizes(self) -> List[Tuple[int]]:
            if self._image_sizes is None:
                original_image_sizes = self.dataset.get_image_sizes()
                self._image_sizes = [original_image_sizes[self._active_original_index[i]] for i in range(len(self))]
            return self._image_sizes

        def get_all_annotations(self):
            return self._annotation_cache

        def get_num_annotations(self, index: int = None, label: AnyStr = None) -> int:
            if index is not None and label is not None:
                raise ValueError("Argument `index` and `label` are mutual exclusive.")
            if index is not None:
                return self.dataset.get_num_annotations(label=self.classes[index])
            if label is not None:
                return self.dataset.get_num_annotations(label=label)
            return self._total_annotation_count

        def get_image_path(self, index: int) -> str:
            return self.dataset.get_image_path(self._active_original_index[index])

    def __init__(self, source_dataset: AlboxBaseObjectDetectionDataset,
                 target_dataset: AlboxBaseObjectDetectionDataset,
                 classes_map_dict: Optional[Dict[str, str]] = None,
                 classes_map_fn: Optional[Callable[[List[str], List[str]], List[str]]] = None,
                 work_dir: str = None,
                 verbose: bool = False):
        self._verbose = verbose
        self._source_classes, self._target_classes = \
            self._determine_classes(source_dataset.classes, target_dataset.classes, classes_map_dict, classes_map_fn)
        if self._verbose:
            print(f"Determined classes: \n\tSource {self._source_classes}\n\tTarget {self._target_classes}")
        self._source_dataset = self._DomainAdaptationSubDatasetForDetection(
            source_dataset, self._source_classes, work_dir, verbose=verbose)
        self._target_dataset = self._DomainAdaptationSubDatasetForDetection(
            target_dataset, self._target_classes, work_dir, verbose=verbose)

    def __getitem__(self, index: int) -> ODDatasetItem:
        pass

    def __len__(self) -> int:
        pass

    def __repr__(self) -> str:
        return f"Domain adaptation dataset adapter.\nSource: {self._source_dataset}\nTarget: {self._target_dataset}"

    def source(self) -> _DomainAdaptationSubDatasetForDetection:
        """Returns filtered source dataset"""
        return self._source_dataset

    def target(self) -> _DomainAdaptationSubDatasetForDetection:
        return self._target_dataset

    def _determine_classes(self, source: List[str],
                           target: List[str],
                           classes_map_dict: Optional[Dict[str, str]],
                           classes_map_fn: Optional[Callable[[List[str], List[str]], Tuple[List[str], List[str]]]]
                           ) -> Tuple[List[str], List[str]]:
        """
        Determine which classes are reserved according to classes map.
        If no classes map are given, reserve the intersect of classes of source and target dataset.

        The first element of the list returned by `classes_map_fn` must be '__background__'
        """
        if classes_map_dict is not None and classes_map_fn is not None:
            raise ValueError("Argument `classes_map_dict` and `classes_map_fn` are mutual exclusive.")

        if classes_map_dict is not None:
            # TODO: check if dict is 1v1 map
            assert len(classes_map_dict) > 0
            for k in classes_map_dict.keys():
                assert k in source, "All keys must be source dataset classes."
            for v in classes_map_dict.values():
                assert v in target, "All values must be target dataset classes."

            source_classes, target_classes = [], []
            for k, v in classes_map_dict.items():
                # common_class_label = f"src_{k}_tgt_{v}"
                source_classes.append(k)
                target_classes.append(v)

            source_classes.sort()
            target_classes = [classes_map_dict[l] for l in source_classes]
            if source_classes[0] != "__background__":
                source_classes.insert(0, "__background__")
            if target_classes[0] != "__background__":
                target_classes.insert(0, "__background__")
            return source_classes, target_classes

        if classes_map_fn is not None:
            source_classes, target_classes = classes_map_fn(source, target)
            source_classes.sort()
            target_classes.sort()
            assert len(source_classes) > 0 and len(source_classes) == len(target_classes), \
                "Datasets for domain adaptation task must have the same number of classes"
            assert source_classes[0] == "__background__"
            assert target_classes[0] == "__background__"

            return source_classes, target_classes

        # default policy: pick up classes which has the same name
        common_classes = list(set(source).intersection(set(target)))
        common_classes.sort()
        if common_classes[0] != "__background__":
            common_classes.insert(0, "__background__")
        return common_classes, common_classes
