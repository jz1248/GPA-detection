from __future__ import print_function
from __future__ import absolute_import
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import xml.dom.minidom as minidom

import os
# import PIL
import numpy as np
import scipy.sparse
import subprocess
import math
import glob
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
import pickle

import albox.datasets.base
from .imdb import imdb
from .imdb import ROOT_DIR
from . import ds_utils
from .voc_eval import voc_eval

from typing import List

AlboxDataset = albox.datasets.base.AlboxBaseObjectDetectionDataset

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from model.utils.config import cfg

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

# <<<< obsolete

# define the cityscapes dataset for vehicle detection
class albox_od(imdb):
    def __init__(self, albox_dataset: AlboxDataset, image_set, num_shot=None):
        imdb.__init__(self, 'albox_' + image_set)
        self.albox_dataset = albox_dataset
        self.num_shot = num_shot
        self._image_set = image_set
        self._data_path = albox_dataset.root
        self._classes = albox_dataset.classes
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = f'.{albox_dataset.img_file_ext}'
        # self._image_index = self._load_image_set_index()
        # Default to roidb handler
        # self._roidb_handler = self.selective_search_roidb
        self._roidb_handler = self.gt_roidb
        self._salt = str(uuid.uuid4())
        self._comp_id = 'comp4'

        # PASCAL specific config options
        self.config = {'cleanup': True,
                       'use_salt': False,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.albox_dataset.get_image_path(i)

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def _get_widths(self):
        return [s[1] for s in self.albox_dataset.get_image_sizes()]

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        # if self.num_shot is None:
        #     cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        # else:
        #     cache_file = os.path.join(self.cache_path, self.name + '_tgt_gt_roidb.pkl')

        # if os.path.exists(cache_file):
        #     with open(cache_file, 'rb') as fid:
        #         roidb = pickle.load(fid, encoding='iso-8859-1')
        #     print('{} gt roidb loaded from {}'.format(self.name, cache_file))
        #     return roidb
        all_annotations = self.albox_dataset.get_all_annotations()

        gt_roidb = self._convert_albox_annotations_to_roidb(all_annotations)

        return gt_roidb

    def _convert_albox_annotations_to_roidb(self, annotations) -> List:
        roidb = []
        if self.num_shot is not None:
            annotations = annotations[:self.num_shot]
        for anno in annotations:
            num_objs = len(anno["labels"])
            boxes = anno["boxes"].int().numpy()
            gt_classes = anno["labels"].int().numpy()
            overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
            for ix, cls in enumerate(gt_classes):
                overlaps[ix, cls] = 1.0
            overlaps = scipy.sparse.csr_matrix(overlaps)
            # "Seg" area for pascal is just the box area
            seg_areas = anno["area"].numpy()
            ishards = anno["difficulty"].int().numpy()
            roidb.append({'boxes': boxes,
                          'gt_classes': gt_classes,
                          'gt_ishard': ishards,
                          'gt_overlaps': overlaps,
                          'flipped': False,
                          'seg_areas': seg_areas})
        return roidb

    # def _load_image_set_index(self):
    #     """
    #     Load the indexes listed in this dataset's image set file.
    #     """
    #     image_index = self.albox_dataset.img_index[self.albox_dataset.phase]
    #     if self.num_shot is not None:
    #         image_index = image_index[:self.num_shot]
    #
    #     return image_index
