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
import podm.podm
import scipy.sparse
import subprocess
import math
import glob
import uuid
import scipy.io as sio
import xml.etree.ElementTree as ET
import pickle

from tqdm import tqdm

import albox.datasets.base
from .imdb import imdb
from .imdb import ROOT_DIR
from . import ds_utils
from .voc_eval import voc_eval

from podm.podm import get_pascal_voc_metrics
from podm.podm import BoundingBox, MetricPerClass

from typing import List, Dict, Any

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
def _is_box_in_region(box, region):
    x1, y1, x2, y2 = box
    l, t, r, b = region
    return l <= x1 and t <= y1 and x2 < r and y2 < b


class albox_od(imdb):
    def __init__(self, albox_dataset: AlboxDataset, image_set, num_shot=None):
        print(albox_dataset)
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

        self.origin_roidb = []
        self.extended_roidb = []
        self.extended_roi_entry_origin_index = []
        self.image_sizes = None

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
        return self.albox_dataset.get_image_path(self._get_origin_image_index(i))

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def _get_origin_image_index(self, i):
        if self.roidb[i]['flipped']:
            assert len(self.roidb) % 2 == 0
            return self._get_origin_image_index(i-len(self.roidb)//2)
        if i < len(self.origin_roidb):
            # origin
            return i
        else:
            # extended
            return self.extended_roi_entry_origin_index[i-len(self.origin_roidb)]

    def _get_widths(self):
        origin_widths = [s[1] for s in self.albox_dataset.get_image_sizes()]
        all_width = []
        all_width.extend(origin_widths)
        all_width.extend([e["clip_region_width"] for e in self.extended_roidb])
        return all_width

    def append_flipped_images(self):
        super().append_flipped_images()
        if self.image_sizes is None:
            self.get_image_sizes()
        self.image_sizes = self.image_sizes * 2

    def get_image_sizes(self):
        if self.image_sizes is None:
            origin_sizes = self.albox_dataset.get_image_sizes()
            all_sizes = []
            all_sizes.extend(origin_sizes)
            all_sizes.extend([(e["clip_region_height"], e["clip_region_width"], origin_sizes[self._get_origin_image_index(i)][2])
                          for i, e in enumerate(self.extended_roidb)])
            self.image_sizes = all_sizes
        return self.image_sizes

    @property
    def num_images(self):
        return len(self.roidb)

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
        image_sizes = self.albox_dataset.get_image_sizes()

        if self.num_shot is not None:
            all_annotations = all_annotations[:self.num_shot]
        if self.num_shot is not None:
            image_sizes = image_sizes[:self.num_shot]
        gt_roidb = self._convert_albox_annotations_to_roidb(all_annotations)

        assert len(image_sizes) == len(gt_roidb)

        extended_roidb, extended_roi_entry_origin_index = self._clip_huge_images(gt_roidb, image_sizes, 800)
        print('Clip check finished.')

        self.origin_roidb = gt_roidb
        self.extended_roidb = extended_roidb
        self.extended_roi_entry_origin_index = extended_roi_entry_origin_index

        all_roidb = []
        all_roidb.extend(self.origin_roidb)
        all_roidb.extend(self.extended_roidb)
        return all_roidb

    def _clip_huge_images(self, gt_roidb, image_sizes, thresh):
        new_roi_entries = []
        extended_roi_entry_origin_index = []    # 扩展的 roi entry 所对应的原图像的序号
        for i, (img_size, roi) in tqdm(enumerate(zip(image_sizes, gt_roidb)), desc='Checking if need clipping'):
            h, w, c = img_size
            if (min(h, w) > thresh):
                # need clip
                roi_entries = self._clip_image_roi_entry(img_size, roi, thresh)
                new_roi_entries.extend(roi_entries)
                extended_roi_entry_origin_index.extend([i]*len(roi_entries))

        return new_roi_entries, extended_roi_entry_origin_index

    def _clip_image_roi_entry(self, img_size, roi_entry, thresh) -> List[Dict[str, Any]]:
        extended_roi_entries = []
        h, w, c = img_size
        size_h = int(h / (h // thresh + 1))
        size_w = int(w / (w // thresh + 1))

        # make cut points
        cut_h = list(range(0, h - size_h + 1, size_h))
        cut_w = list(range(0, w - size_w + 1, size_w))

        # make region
        cut_h.append(h)
        cut_w.append(w)
        regions = []
        for i in range(len(cut_h) - 1):
            for j in range(len(cut_w) - 1):
                regions.append((cut_w[j], cut_h[i], cut_w[j+1], cut_h[i+1]))

        # filter boxes
        origin_boxes = roi_entry["boxes"].copy()
        origin_label = roi_entry["gt_classes"].copy()
        origin_ishards = roi_entry["gt_ishard"].copy()
        origin_seg_areas = roi_entry["seg_areas"].copy()
        for region in regions:
            left, top = region[0], region[1]
            filtered_indices = []
            for i, box in enumerate(origin_boxes):
                if _is_box_in_region(box, region):
                    filtered_indices.append(i)
            if len(filtered_indices) == 0:
                continue
            filtered_indices = np.array(filtered_indices)
            filtered_boxes = origin_boxes[filtered_indices, :]

            # reassign coordinates
            oldx1 = filtered_boxes[:, 0].copy()
            oldy1 = filtered_boxes[:, 1].copy()
            oldx2 = filtered_boxes[:, 2].copy()
            oldy2 = filtered_boxes[:, 3].copy()
            filtered_boxes[:, 0] = oldx1 - left
            filtered_boxes[:, 1] = oldy1 - top
            filtered_boxes[:, 2] = oldx2 - left
            filtered_boxes[:, 3] = oldy2 - top

            # # calculate filpped boxes
            # flipped_boxes = filtered_boxes.copy()
            # oldx1 = flipped_boxes[:, 0].copy()
            # oldx2 = flipped_boxes[:, 2].copy()
            # flipped_boxes[:, 0] = (region[2] - region[0]) - oldx2 - 1
            # flipped_boxes[:, 2] = (region[2] - region[0]) - oldx1 - 1

            filtered_label = origin_label[filtered_indices]
            filtered_ishards = origin_ishards[filtered_indices]
            filtered_seg_areas = origin_seg_areas[filtered_indices]
            num_objs = len(filtered_label)
            filtered_overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
            for ix, cls in enumerate(filtered_label):
                filtered_overlaps[ix, cls] = 1.0
            filtered_overlaps = scipy.sparse.csr_matrix(filtered_overlaps)

            extended_roi_entries.append({'boxes': filtered_boxes,
                          'gt_classes': filtered_label,
                          'gt_ishard': filtered_ishards,
                          'gt_overlaps': filtered_overlaps,
                          'flipped': False,
                          'clip': True,
                          'clip_region': region,
                          'clip_region_width': region[2] - region[0],
                          'clip_region_height': region[3] - region[1],
                          # 'flipped_boxes': flipped_boxes,
                          'seg_areas': filtered_seg_areas})

        return extended_roi_entries

    def _convert_albox_annotations_to_roidb(self, annotations) -> List:
        roidb = []
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
                          'clip': False,
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

    def evaluate_detections(self, all_boxes, output_dir, epoch = 10):
        gt_boxes: List[BoundingBox] = []
        pred_boxes: List[BoundingBox] = []

        # process prediction boxes
        for i in range(1, self.num_classes):
            label = self.classes[i]
            for idx_img, pred_mat in enumerate(all_boxes[i]):
                num_pred_boxes = pred_mat.shape[0]
                for idx_box in range(num_pred_boxes):
                    score = pred_mat[idx_box, 4]
                    x1, y1, x2, y2 = pred_mat[idx_box, :4]
                    bbox = BoundingBox(str(idx_img), label, x1, y1, x2, y2, score)
                    pred_boxes.append(bbox)

        roidb = self.roidb
        for idx_img, anno in enumerate(roidb):
            boxes = anno['boxes']
            gt_classes = anno['gt_classes']
            num_gt_boxes = gt_classes.size
            for idx_gt_box in range(num_gt_boxes):
                assert boxes[idx_gt_box].size == 4
                label = self.classes[gt_classes[idx_gt_box]]
                x1, y1, x2, y2 = boxes[idx_gt_box]
                gt_bbox = BoundingBox(str(idx_img), label, x1, y1, x2, y2)
                gt_boxes.append(gt_bbox)

        result = get_pascal_voc_metrics(gt_boxes, pred_boxes)
        result_path = os.path.join(output_dir, f"eval_result_epoch_{epoch}.pkl")
        with open(result_path, 'wb') as f:
            pickle.dump(result, f)

        ap_dict = dict()
        map = MetricPerClass.get_mAP(result)
        for label, m in result.items():
            ap_dict[label] = m.ap

        print(f"----Evaluation result for epoch {epoch}-----")
        for k, v in ap_dict.items():
            print(f'AP for {k}: \t{v}')
        print(f'mAP: {map}')
        print("---------End evaluation result---------")
