# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/data/dataset_mappers/coco_panoptic_new_baseline_dataset_mapper.py
# modified by Qihang Yu
import copy
import logging

import numpy as np
import torch
import random
import json

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Boxes, Instances

from fvcore.transforms.transform import PadTransform

__all__ = ["COCOPanoptickMaXDeepLabDatasetMapper"]


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

    # Augmnetation order majorlly follows deeplab2: resize -> autoaug (color jitter) -> random pad/crop -> flip
    # But we alter it to  resize -> color jitter -> flip -> pad/crop, as random pad is not supported in detectron2.
    # The order of flip and pad/crop does not matter as we are doing random padding/crop anyway.
    augmentation = [
        # Unlike deeplab2 in tf, here the interp will be done in uin8 instead of float32.
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size[0], target_width=image_size[1]
        ), # perofrm on uint8 or float32
        ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT), # performed on uint8
        # Unlike deeplab2 in tf, here the padding value for image is 128, for label is 255. Besides, padding here will only pad right and bottom.
        # T.FixedSizeCrop(crop_size=(image_size, image_size)),

        # We only perform crop, and do padding manually as the padding value matters. This will crop the image to min(h, image_size).
        T.RandomCrop(crop_type="absolute", crop_size=(image_size[0], image_size[1])),
        T.RandomFlip(),
    ]

    return augmentation


# This is specifically designed for the COCO dataset.
class COCOPanoptickMaXDeepLabDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by kMaX-DeepLab.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

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
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            crop_gen: crop augmentation
            tfm_gens: data augmentation
            tfm_gens_copy_paste: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        self.tfm_gens_copy_paste = tfm_gens_copy_paste
        if is_train:
            logging.getLogger(__name__).info(
                "[COCOPanopticDeepLab2DatasetMapper] Full TransformGens used in training: {}, {}".format(
                    str(self.tfm_gens), str(self.tfm_gens_copy_paste)
                )
            )
        else:
            logging.getLogger(__name__).info(
                "[COCOPanopticDeepLab2DatasetMapper] Full TransformGens used in testing: {}".format(
                    str(self.tfm_gens)
                )
            )
        self.img_format = image_format
        self.is_train = is_train
        self.image_size = image_size


        # TODO(qihangyu): Better way to implement this copy paste augmentation.
        image_dir = "./datasets/coco/train2017"
        gt_dir = "./datasets/coco/panoptic_train2017"
        semseg_dir = "./datasets/coco/panoptic_semseg_train2017"
        json_file = "./datasets/coco/annotations/panoptic_train2017.json"
        from ..datasets import register_coco_panoptic_annos_semseg
        meta_data = register_coco_panoptic_annos_semseg.get_metadata()
        self.dataset_dict_all = register_coco_panoptic_annos_semseg.load_coco_panoptic_json(
            json_file, image_dir, gt_dir, semseg_dir, meta_data
        )
        self.filename2idx = {}
        for idx, dataset_dict in enumerate(self.dataset_dict_all):
            self.filename2idx[dataset_dict["file_name"].split('/')[-1].replace('.jpg', '')] = idx


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
            "image_size": cfg.INPUT.IMAGE_SIZE
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

        dataset_dict["image"] = np.ascontiguousarray(image.transpose(2, 0, 1))

        if not self.is_train:
            # If this is for test, we can directly return the unpadded image, as the padding
            # will be handled by size_divisibility
            dataset_dict.pop("annotations", None)
            return dataset_dict, None

        # We pad the image manually, for copy-paste purpose.
        padded_image = np.zeros((3, self.image_size[0], self.image_size[1]), dtype=dataset_dict["image"].dtype)
        new_h, new_w = dataset_dict["image"].shape[1:]
        #offset_h, offset_w = np.random.randint(0, self.image_size[0] - new_h + 1), np.random.randint(0, self.image_size[1] - new_w + 1)
        offset_h, offset_w = 0, 0 # following the d2 panoptic deeplab implementaiton to only perform bottom/right padding.
        padded_image[:, offset_h:offset_h+new_h, offset_w:offset_w+new_w] = dataset_dict["image"]
        dataset_dict["image"] = padded_image
        if "pan_seg_file_name" in dataset_dict:
            pan_seg_gt = utils.read_image(dataset_dict.pop("pan_seg_file_name"), "RGB")

            # apply the same transformation to panoptic segmentation
            pan_seg_gt = transforms.apply_segmentation(pan_seg_gt)

            from panopticapi.utils import rgb2id

            pan_seg_gt = rgb2id(pan_seg_gt) # int32 # H x W
            # similarily, we manually pad the label, and we use label -1 to indicate those padded pixels.
            # In this way, we can masking out the padded pixels values to -1 after normalization, which aligns the
            # behavior between training and testing.
            padded_pan_seg_gt = np.zeros((self.image_size[0], self.image_size[1]), dtype=pan_seg_gt.dtype)
            is_real_pixels = np.zeros((self.image_size[0], self.image_size[1]), dtype=np.bool)
            padded_pan_seg_gt[offset_h:offset_h+new_h, offset_w:offset_w+new_w] = pan_seg_gt
            is_real_pixels[offset_h:offset_h+new_h, offset_w:offset_w+new_w] = True
            dataset_dict["is_real_pixels"] = is_real_pixels
            pan_seg_gt = padded_pan_seg_gt
            return dataset_dict, pan_seg_gt

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
        dataset_dict, pan_seg_gt = self.read_dataset_dict(dataset_dict, is_copy_paste=False)
        # Read copy-paste image.
        # It should be sometinng like xxx/xxx/xxx/000000139.jpg, etc. we use the last number as a bias to random number.
        main_image_idx = self.filename2idx[dataset_dict["file_name"].split('/')[-1].replace('.jpg', '')]
        random_image_idx = main_image_idx + random.randint(0, len(self.dataset_dict_all) - 1)
        random_image_idx = random_image_idx % len(self.dataset_dict_all)
        dataset_dict_copy_paste = copy.deepcopy(self.dataset_dict_all[random_image_idx])
        dataset_dict_copy_paste, pan_seg_gt_copy_paste = self.read_dataset_dict(dataset_dict_copy_paste, is_copy_paste=True)

        # Copy data_dict_copy_paste onto data_dict. 0 means keep original pixel, 1 means use copy-paste pixel.
        copy_paste_masks = np.zeros((pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))

        segments_info_copy_paste = dataset_dict_copy_paste["segments_info"]
        all_ids = []
        thing_ids = []
        for segment_info_copy_paste in segments_info_copy_paste:
            class_id = segment_info_copy_paste["category_id"]
            if not segment_info_copy_paste["iscrowd"]:
                # -1 is reserved for padded pixels.
                if segment_info_copy_paste["id"] in [-1, 0]:
                    print(segment_info_copy_paste)
                    raise ValueError("id should not be -1, 0")
                all_ids.append(segment_info_copy_paste["id"])
                if segment_info_copy_paste["isthing"]: # All thing classes are copy-pasted.
                    thing_ids.append(segment_info_copy_paste["id"])

        # Shuffle and randomly select kept label ids.
        random.shuffle(all_ids)
        keep_number = random.randint(0, len(all_ids))

        for index, label_id in enumerate(all_ids):
            # randomly copy labels, but keep all thing classes.
            if index < keep_number or label_id in thing_ids:
                copy_paste_masks[pan_seg_gt_copy_paste == label_id] = 1

        # We merge the image and copy-paste image based on the copy-paste mask.
        # 3 x H x W
        dataset_dict["image"] = (dataset_dict["image"] * (1.0 - copy_paste_masks).astype(dataset_dict["image"].dtype) +
                                 dataset_dict_copy_paste["image"] * copy_paste_masks.astype(dataset_dict["image"].dtype))
        dataset_dict["image"] = torch.as_tensor(dataset_dict["image"])

        dataset_dict["is_real_pixels"] = (dataset_dict["is_real_pixels"] * (1.0 - copy_paste_masks).astype(dataset_dict["is_real_pixels"].dtype) +
                                 dataset_dict_copy_paste["is_real_pixels"] * copy_paste_masks.astype(dataset_dict["is_real_pixels"].dtype))
        dataset_dict["is_real_pixels"] = torch.as_tensor(dataset_dict["is_real_pixels"])
        # We set all ids in copy-paste image to be negative, so that there will be no overlap between original id and copy-paste id.
        pan_seg_gt_copy_paste = -pan_seg_gt_copy_paste
        pan_seg_gt = (pan_seg_gt * (1.0 - copy_paste_masks).astype(pan_seg_gt.dtype) +
                       pan_seg_gt_copy_paste * copy_paste_masks.astype(pan_seg_gt.dtype))

        # We use 4x downsampled gt for final supervision.
        pan_seg_gt = pan_seg_gt[::4, ::4]

        sem_seg_gt = -np.ones_like(pan_seg_gt) # H x W, init with -1

        # We then process the obtained pan_seg_gt to training format.
        image_shape = dataset_dict["image"].shape[1:]  # h, w
        segments_info = dataset_dict["segments_info"]
        instances = Instances(image_shape)
        classes = []
        masks = []
        valid_pixel_num = 0
        # As the two images may share same stuff classes, we use a dict to track existing stuff and merge them.
        stuff_class_to_idx = {}
        for segment_info in segments_info:
            class_id = segment_info["category_id"]
            if not segment_info["iscrowd"]:
                # -1 is reserved to indicate padded pixels.
                if segment_info["id"] in [-1, 0]:
                    print(segment_info)
                    raise ValueError("id should not be -1, 0")
                binary_mask = (pan_seg_gt == segment_info["id"])
                # As it is possible that some masks are removed during the copy-paste process, we need
                # to double check if the maks exists.
                valid_pixel_num_ = binary_mask.sum()
                valid_pixel_num += valid_pixel_num_
                if valid_pixel_num_ > 0:
                    sem_seg_gt[binary_mask] = class_id      
                    if not segment_info["isthing"]:
                        # For original image, stuff should only appear once.
                        if class_id in stuff_class_to_idx:
                            raise ValueError('class_id should not already be in stuff_class_to_idx!')
                        else:
                            stuff_class_to_idx[class_id] = len(masks)
                    classes.append(class_id)
                    masks.append(binary_mask)

        for segment_info in segments_info_copy_paste:
            class_id = segment_info["category_id"]
            if not segment_info["iscrowd"]:
                # -1 is reserved to indicate padded pixels.
                if segment_info["id"] in [-1, 0]:
                    print(segment_info)
                    raise ValueError("id should not be -1, 0")
                # Note that copy-paste id is negative.
                binary_mask = (pan_seg_gt == -segment_info["id"])
                valid_pixel_num_ = binary_mask.sum()
                valid_pixel_num += valid_pixel_num_
                if valid_pixel_num_ > 0:
                    sem_seg_gt[binary_mask] = class_id
                    if not segment_info["isthing"]:
                        # The stuff in copy-paste image already appeared in original image.
                        if class_id in stuff_class_to_idx:
                            # Merge into original stuff masks. 
                            masks[stuff_class_to_idx[class_id]] = np.logical_or(masks[stuff_class_to_idx[class_id]], binary_mask)
                            continue
                        else:
                            stuff_class_to_idx[class_id] = len(masks)
                    classes.append(class_id)
                    masks.append(binary_mask)

        classes = np.array(classes)
        instances.gt_classes = torch.tensor(classes, dtype=torch.int64)
        sem_seg_gt = torch.tensor(sem_seg_gt, dtype=torch.int64)
        
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros((0, pan_seg_gt.shape[-2], pan_seg_gt.shape[-1]))
            instances.gt_boxes = Boxes(torch.zeros((0, 4)))
        else:
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
            )
            instances.gt_masks = masks.tensor
            instances.gt_boxes = masks.get_bounding_boxes()

        dataset_dict["instances"] = instances
        dataset_dict["sem_seg_gt"] = sem_seg_gt
        dataset_dict["valid_pixel_num"] = valid_pixel_num
        return dataset_dict

    def __call__(self, dataset_dict):
        res = self.call_copypaste(dataset_dict)
        while ("instances" in res and res["instances"].gt_masks.shape[0] == 0) or ("valid_pixel_num" in res and res["valid_pixel_num"] <= 4096):
            # this gt is empty or contains too many void pixels, let's re-generate one.
            main_image_idx = self.filename2idx[dataset_dict["file_name"].split('/')[-1].replace('.jpg', '')]
            random_image_idx = main_image_idx + random.randint(0, len(self.dataset_dict_all) - 1)
            random_image_idx = random_image_idx % len(self.dataset_dict_all)
            dataset_dict = self.dataset_dict_all[random_image_idx]
            res = self.call_copypaste(dataset_dict)

        return res
