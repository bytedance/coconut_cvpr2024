# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/dataset_mappers/coco_instance_new_baseline_dataset_mapper.py
# modified by Qihang Yu
import copy
import logging

import numpy as np
import torch
import random

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Boxes, Instances, polygons_to_bitmask

from pycocotools import mask as coco_mask
import pycocotools.mask as mask_util

import os

__all__ = ["InstancekMaXDeepLabDatasetMapper_nocopypaste"]


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

def build_transform_gen(cfg, is_train, scale_ratio=1.0):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    image_size = cfg.INPUT.IMAGE_SIZE
    assert is_train

    min_scale = cfg.INPUT.MIN_SCALE * scale_ratio
    max_scale = cfg.INPUT.MAX_SCALE * scale_ratio


    augmentation = [
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size[0], target_width=image_size[1]
        ),
        ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT),
        T.RandomCrop(crop_type="absolute", crop_size=(image_size[0], image_size[1])),
        T.RandomFlip(),
    ]

    return augmentation


class InstancekMaXDeepLabDatasetMapper_nocopypaste:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by kMaX-DeepLab.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        tfm_gens_copy_paste,
        image_format,
        image_size,
        dataset_name
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            tfm_gens_copy_paste: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`
            image_size: expected image size
        """
        self.tfm_gens = tfm_gens
        self.tfm_gens_copy_paste = tfm_gens_copy_paste
        if is_train:
            logging.getLogger(__name__).info(
                "[InstancekMaXDeepLabDatasetMapper] Full TransformGens used in training: {}, {}".format(
                    str(self.tfm_gens), str(self.tfm_gens_copy_paste)
                )
            )
        else:
            logging.getLogger(__name__).info(
                "[InstancekMaXDeepLabDatasetMapper] Full TransformGens used in testing: {}".format(
                    str(self.tfm_gens)
                )
            )
        self.img_format = image_format
        self.is_train = is_train
        self.image_size = image_size
        self.dataset_name = dataset_name


    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)
        tfm_gens_copy_paste = build_transform_gen(cfg, is_train, scale_ratio=0.5)
        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "tfm_gens_copy_paste": tfm_gens_copy_paste,
            "image_format": cfg.INPUT.FORMAT,
            "image_size": cfg.INPUT.IMAGE_SIZE,
            "dataset_name": {
                'coco_instance_lsj': 'coco',
                'ade20k_instance_lsj': 'ade20k',
                'cityscapes_instance_lsj': 'cityscapes',
            }[cfg.INPUT.DATASET_MAPPER_NAME]
        }
        return ret

    def read_dataset_dict(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = np.ascontiguousarray(image.transpose(2, 0, 1))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        # We pad the image manually, for copy-paste purpose.
        padded_image = np.zeros((3, self.image_size[0], self.image_size[1]), dtype=dataset_dict["image"].dtype)
        new_h, new_w = dataset_dict["image"].shape[1:]
        offset_h, offset_w = 0, 0 # following the d2 panoptic deeplab implementaiton to only perform bottom/right padding.
        padded_image[:, offset_h:offset_h+new_h, offset_w:offset_w+new_w] = dataset_dict["image"]
        dataset_dict["image"] = torch.as_tensor(padded_image)
        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                # Let's always keep mask
                # if not self.mask_on:
                #     anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            # NOTE: does not support BitMask due to augmentation
            # Current BitMask cannot handle empty objects
            instances = utils.annotations_to_instances(annos, image_shape)
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # Need to filter empty instances first (due to augmentation)
            instances = utils.filter_empty_instances(instances)
            # Generate masks from polygon
            h, w = instances.image_size
            # image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float)
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                # padding
                padded_gt_masks = torch.zeros((gt_masks.shape[0], self.image_size[0], self.image_size[1]), dtype=gt_masks.dtype)
                is_real_pixels = np.zeros((self.image_size[0], self.image_size[1]), dtype=np.bool)
                padded_gt_masks[:, offset_h:offset_h+new_h, offset_w:offset_w+new_w] = gt_masks
                is_real_pixels[offset_h:offset_h+new_h, offset_w:offset_w+new_w] = True
                instances.gt_masks = padded_gt_masks#[:, ::4, ::4]
                dataset_dict["is_real_pixels"] = torch.as_tensor(is_real_pixels)
            
            dataset_dict["instances"] = instances
        return dataset_dict

    def __call__(self, dataset_dict):
        res = self.read_dataset_dict(dataset_dict)
        return res