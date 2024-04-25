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

__all__ = ["InstancekMaXDeepLabDatasetMapper"]


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


class InstancekMaXDeepLabDatasetMapper:
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

        dataset_root = os.getenv("DETECTRON2_DATASETS", "datasets")
        if dataset_name == 'coco':
            image_dir = os.path.join(dataset_root, "coco/train2017")
            json_file = os.path.join(dataset_root, "coco/annotations/instances_train2017.json")
            from detectron2.data.datasets import coco
            self.dataset_dict_all = coco.load_coco_json(
                json_file=json_file, image_root=image_dir, dataset_name='coco_2017_train')

        elif dataset_name == 'ade20k':
            image_dir = os.path.join(dataset_root, "ADEChallengeData2016/images/training")
            json_file = os.path.join(dataset_root, "ADEChallengeData2016/ade20k_instance_train.json")
            from ..datasets import register_ade20k_instance
            from detectron2.data.datasets import coco
            self.dataset_dict_all = coco.load_coco_json(
                json_file=json_file, image_root=image_dir, dataset_name='ade20k_instance_train')
        elif dataset_name == 'cityscapes':
            image_dir = os.path.join(dataset_root, "cityscapes/leftImg8bit/train")
            gt_dir = os.path.join(dataset_root, "cityscapes/gtFine/train")
            from detectron2.data.datasets import cityscapes
            self.dataset_dict_all = cityscapes.load_cityscapes_instances(
                image_dir, gt_dir, from_json=True, to_polygons=True
            )

        if self.dataset_name in ['coco', 'ade20k']:
            if self.dataset_name in ['coco', 'ade20k']:
                from detectron2.data.build import filter_images_with_only_crowd_annotations
                self.dataset_dict_all = filter_images_with_only_crowd_annotations(self.dataset_dict_all)
        self.filename2idx = {}
        for idx, dataset_dict in enumerate(self.dataset_dict_all):
            self.filename2idx[os.path.splitext(dataset_dict["file_name"].split('/')[-1])[0]] = idx

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

    def read_dataset_dict(self, dataset_dict, is_copy_paste=False):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if not is_copy_paste:
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        else:
            image, transforms = T.apply_transform_gens(self.tfm_gens_copy_paste, image)
        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = np.ascontiguousarray(image.transpose(2, 0, 1))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        # We pad the image manually, for copy-paste purpose.
        padded_image = np.zeros((3, self.image_size[0], self.image_size[1]), dtype=dataset_dict["image"].dtype)
        new_h, new_w = dataset_dict["image"].shape[1:]
        offset_h, offset_w = 0, 0 # following the d2 panoptic deeplab implementaiton to only perform bottom/right padding.
        padded_image[:, offset_h:offset_h+new_h, offset_w:offset_w+new_w] = dataset_dict["image"]
        dataset_dict["image"] = padded_image
        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                # Let's always keep mask
                # if not self.mask_on:
                #     anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            # annos = [
            #     utils.transform_instance_annotations(obj, transforms, image_shape)
            #     for obj in dataset_dict.pop("annotations")
            #     if obj.get("iscrowd", 0) == 0
            # ]

            annos=[]

            for obj in dataset_dict.pop("annotations"):
                if obj.get("iscrowd", 0) == 0 and 'segmentation' in obj:
                    # print(obj.keys())
                    annos.append(utils.transform_instance_annotations(obj, transforms, image_shape))


            if self.dataset_name in ["coco",]:
                # NOTE: does not support BitMask due to augmentation
                # Current BitMask cannot handle empty objects
                instances = utils.annotations_to_instances(annos, image_shape)
                # if not instances.has("gt_masks"):
                #     print('### no gt masks ####')
                #     print(annos)
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
                gt_masks = instances.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                gt_classes = None # already in instance
            elif self.dataset_name in ["ade20k", "cityscapes"]:
                instances = Instances(image_shape)
                segms = [obj["segmentation"] for obj in annos]
                masks = []
                for segm in segms:
                    if isinstance(segm, list):
                        # polygon
                        masks.append(polygons_to_bitmask(segm, *image.shape[:2]))
                    elif isinstance(segm, dict):
                        # COCO RLE
                        masks.append(mask_util.decode(segm))
                    elif isinstance(segm, np.ndarray):
                        assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                            segm.ndim
                        )
                        # mask array
                        masks.append(segm)
                    else:
                        raise ValueError(
                            "Cannot convert segmentation of type '{}' to BitMasks!"
                            "Supported types are: polygons as list[list[float] or ndarray],"
                            " COCO-style RLE as a dict, or a binary segmentation mask "
                            " in a 2D numpy array of shape HxW.".format(type(segm))
                        )

                if len(masks) == 0:
                    # Some image does not have annotation (all ignored)
                    gt_masks = torch.zeros((0, image_shape[0], image_shape[1]))
                else:
                    gt_masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks], dim=0)
                gt_classes = [int(obj["category_id"]) for obj in annos]
                gt_classes = torch.tensor(gt_classes, dtype=torch.int64)
            else:
                raise NotImplementedError

            # padding
            padded_gt_masks = torch.zeros((gt_masks.shape[0], self.image_size[0], self.image_size[1]), dtype=gt_masks.dtype)
            is_real_pixels = np.zeros((self.image_size[0], self.image_size[1]), dtype=np.bool_)
            padded_gt_masks[:, offset_h:offset_h+new_h, offset_w:offset_w+new_w] = gt_masks
            is_real_pixels[offset_h:offset_h+new_h, offset_w:offset_w+new_w] = True
            instances.gt_masks = padded_gt_masks
            if gt_classes is not None:
                instances.gt_classes = gt_classes
            dataset_dict["instances"] = instances
            dataset_dict["is_real_pixels"] = is_real_pixels

            return dataset_dict

        # This should never happen.
        raise NotImplementedError

    def call_copypaste(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # Read main image.
        dataset_dict = self.read_dataset_dict(dataset_dict, is_copy_paste=False)
        # Read copy-paste image.
        # We use the last number as a bias to random number, in case same random numbers are generated across devices.
        main_image_idx = self.filename2idx[os.path.splitext(dataset_dict["file_name"].split('/')[-1])[0]]
        random_image_idx = main_image_idx + random.randint(0, len(self.dataset_dict_all) - 1)
        random_image_idx = random_image_idx % len(self.dataset_dict_all)
        dataset_dict_copy_paste = copy.deepcopy(self.dataset_dict_all[random_image_idx])
        dataset_dict_copy_paste = self.read_dataset_dict(dataset_dict_copy_paste, is_copy_paste=True)

        height, width = dataset_dict["instances"].gt_masks.shape[-2], dataset_dict["instances"].gt_masks.shape[-1]
        # gt_masks: N x H x W
        # Copy data_dict_copy_paste onto data_dict. 0 means keep original pixel, 1 means use copy-paste pixel.
        copy_paste_masks = np.zeros((height, width))

        # we copy all instances (thing) from copy_paste_image to main_image.
        all_ids = list(range(dataset_dict_copy_paste["instances"].gt_masks.shape[0]))
        random.shuffle(all_ids)
        keep_number = random.randint(0, len(all_ids))
        for i in range(keep_number):
            copy_paste_masks[dataset_dict_copy_paste["instances"].gt_masks[all_ids[i]] > 0] = 1.0
        # We merge the image and copy-paste image based on the copy-paste mask.
        dataset_dict["image"] = (dataset_dict["image"] * (1.0 - copy_paste_masks).astype(dataset_dict["image"].dtype) +
                                 dataset_dict_copy_paste["image"] * copy_paste_masks.astype(dataset_dict["image"].dtype))
        dataset_dict["image"] = torch.as_tensor(dataset_dict["image"])

        dataset_dict["is_real_pixels"] = (dataset_dict["is_real_pixels"] * (1.0 - copy_paste_masks).astype(dataset_dict["is_real_pixels"].dtype) +
                                 dataset_dict_copy_paste["is_real_pixels"] * copy_paste_masks.astype(dataset_dict["is_real_pixels"].dtype))
        dataset_dict["is_real_pixels"] = torch.as_tensor(dataset_dict["is_real_pixels"])

        # remove all pixels that are overwritten.
        new_gt_masks = dataset_dict["instances"].gt_masks.numpy()
        new_gt_masks = np.concatenate([new_gt_masks * (1.0 - copy_paste_masks).astype(new_gt_masks.dtype),
                        dataset_dict_copy_paste["instances"].gt_masks.numpy() * copy_paste_masks.astype(new_gt_masks.dtype)], axis=0)
        #new_gt_masks = new_gt_masks[:, ::4, ::4]
        new_gt_classes = np.concatenate([dataset_dict["instances"].gt_classes.numpy(), dataset_dict_copy_paste["instances"].gt_classes.numpy()], axis=0)
        # filter empty masks.
        classes = []
        masks = []
        valid_pixel_num = 0
        for i in range(new_gt_masks.shape[0]):
            valid_pixel_num_ = new_gt_masks[i].sum()
            valid_pixel_num += valid_pixel_num_
            if valid_pixel_num_ > 0:
                classes.append(new_gt_classes[i])
                masks.append(new_gt_masks[i])

        image_shape = new_gt_masks.shape[1:]  # h, w
        instances = Instances(image_shape)
        classes = np.array(classes)
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)        
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros((0, new_gt_masks.shape[1], new_gt_masks.shape[2]))
            instances.gt_boxes = Boxes(torch.zeros((0, 4)))
        else:
            masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
            instances.gt_masks = masks.tensor
            instances.gt_boxes = masks.get_bounding_boxes()

        dataset_dict["instances"] = instances
        dataset_dict["valid_pixel_num"] = valid_pixel_num
        return dataset_dict

    def __call__(self, dataset_dict):
        res = self.call_copypaste(dataset_dict)
        while ("instances" in res and res["instances"].gt_masks.shape[0] == 0) or ("valid_pixel_num" in res and res["valid_pixel_num"] <= 4096):
            # this gt is empty or contains too many void pixels, let's re-generate one.
            main_image_idx = self.filename2idx[os.path.splitext(dataset_dict["file_name"].split('/')[-1])[0]]
            random_image_idx = main_image_idx + random.randint(0, len(self.dataset_dict_all) - 1)
            random_image_idx = random_image_idx % len(self.dataset_dict_all)
            dataset_dict = self.dataset_dict_all[random_image_idx]
            res = self.call_copypaste(dataset_dict)

        return res