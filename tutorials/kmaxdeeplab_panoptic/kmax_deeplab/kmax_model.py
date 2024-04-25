# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py
# Reference: https://github.com/google-research/deeplab2/blob/main/model/kmax_deeplab.py
# Reference: https://github.com/google-research/deeplab2/blob/main/model/post_processor/max_deeplab.py
# Modified by Qihang Yu

from typing import Tuple, List

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from torch.cuda.amp import autocast
from .coco_meta import COCO_META


@META_ARCH_REGISTRY.register()
class kMaXDeepLab(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        class_threshold_thing: float,
        class_threshold_stuff: float,
        overlap_threshold: float,
        reorder_class_weight: float,
        reorder_mask_weight: float,
        thing_area_limit: int,
        stuff_area_limit: int,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        input_shape: List[int]
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.class_threshold_thing = class_threshold_thing
        self.class_threshold_stuff = class_threshold_stuff
        self.reorder_class_weight = reorder_class_weight
        self.reorder_mask_weight = reorder_mask_weight
        self.thing_area_limit = thing_area_limit
        self.stuff_area_limit = stuff_area_limit
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        self.input_shape = input_shape

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.KMAX_DEEPLAB.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.KMAX_DEEPLAB.NO_OBJECT_WEIGHT
        share_final_matching = cfg.MODEL.KMAX_DEEPLAB.SHARE_FINAL_MATCHING

        # loss weights
        class_weight = cfg.MODEL.KMAX_DEEPLAB.CLASS_WEIGHT
        dice_weight = cfg.MODEL.KMAX_DEEPLAB.DICE_WEIGHT
        mask_weight = cfg.MODEL.KMAX_DEEPLAB.MASK_WEIGHT
        insdis_weight = cfg.MODEL.KMAX_DEEPLAB.INSDIS_WEIGHT
        aux_semantic_weight = cfg.MODEL.KMAX_DEEPLAB.AUX_SEMANTIC_WEIGHT

        # building criterion
        matcher = HungarianMatcher()

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight,
        "loss_pixel_insdis": insdis_weight, "loss_aux_semantic": aux_semantic_weight}

        if deep_supervision:
            dec_layers = sum(cfg.MODEL.KMAX_DEEPLAB.TRANS_DEC.DEC_LAYERS)
            aux_weight_dict = {}
            for i in range(dec_layers):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]
        if insdis_weight > 0:
            losses += ["pixels"]
        if aux_semantic_weight > 0:
            losses += ["aux_semantic"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            share_final_matching=share_final_matching,
            pixel_insdis_temperature=cfg.MODEL.KMAX_DEEPLAB.PIXEL_INSDIS_TEMPERATURE,
            pixel_insdis_sample_k=cfg.MODEL.KMAX_DEEPLAB.PIXEL_INSDIS_SAMPLE_K,
            aux_semantic_temperature=cfg.MODEL.KMAX_DEEPLAB.AUX_SEMANTIC_TEMPERATURE,
            aux_semantic_sample_k=cfg.MODEL.KMAX_DEEPLAB.AUX_SEMANTIC_SAMPLE_K
        ) 

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.KMAX_DEEPLAB.TRANS_DEC.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.KMAX_DEEPLAB.TEST.OBJECT_MASK_THRESHOLD,
            "class_threshold_thing": cfg.MODEL.KMAX_DEEPLAB.TEST.CLASS_THRESHOLD_THING,
            "class_threshold_stuff": cfg.MODEL.KMAX_DEEPLAB.TEST.CLASS_THRESHOLD_STUFF,
            "overlap_threshold": cfg.MODEL.KMAX_DEEPLAB.TEST.OVERLAP_THRESHOLD,
            "reorder_class_weight": cfg.MODEL.KMAX_DEEPLAB.TEST.REORDER_CLASS_WEIGHT,
            "reorder_mask_weight": cfg.MODEL.KMAX_DEEPLAB.TEST.REORDER_MASK_WEIGHT,
            "thing_area_limit": cfg.MODEL.KMAX_DEEPLAB.TEST.THING_AREA_LIMIT,
            "stuff_area_limit": cfg.MODEL.KMAX_DEEPLAB.TEST.STUFF_AREA_LIMIT,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.KMAX_DEEPLAB.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.KMAX_DEEPLAB.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.KMAX_DEEPLAB.TEST.PANOPTIC_ON
                or cfg.MODEL.KMAX_DEEPLAB.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.KMAX_DEEPLAB.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.KMAX_DEEPLAB.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.KMAX_DEEPLAB.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "input_shape": cfg.INPUT.IMAGE_SIZE
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        if "is_real_pixels" in batched_inputs[0]:
            is_real_pixels = [x["is_real_pixels"] for x in batched_inputs]
            # Set all padded pixel values to 0.
            images = [x * y.to(x) for x, y in zip(images, is_real_pixels)]

        # We perform zero padding to ensure input shape equal to self.input_shape.
        # The padding is done on the right and bottom sides. 
        for idx in range(len(images)):
            cur_height, cur_width = images[idx].shape[-2:]
            padding = (0, max(0, self.input_shape[1] - cur_width), 0, max(0, self.input_shape[0] - cur_height), 0, 0)
            images[idx] = F.pad(images[idx], padding, value=0)
        images = ImageList.from_tensors(images, -1)

        if self.training:
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                gt_semantic = [x["sem_seg_gt"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, gt_semantic, images)
            else:
                targets = None

        features = self.backbone(images.tensor)
        # import pdb; pdb.set_trace()
        outputs = self.sem_seg_head(features)

        if self.training:

            with autocast(enabled=False):
                # bipartite matching-based loss
                for output_key in ["pixel_feature", "pred_masks", "pred_logits", "aux_semantic_pred"]:
                    if output_key in outputs:
                        outputs[output_key] = outputs[output_key].float()
                for i in range(len(outputs["aux_outputs"])):
                    for output_key in ["pixel_feature", "pred_masks", "pred_logits"]:
                        outputs["aux_outputs"][i][output_key] = outputs["aux_outputs"][i][output_key].float()
                
                losses = self.criterion(outputs, targets)

                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                        # remove this loss if not specified in `weight_dict`
                        losses.pop(k)
                return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            # import pdb; pdb.set_trace()

            align_corners = (images.tensor.shape[-1] % 2 == 1) 
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=align_corners,
            )

            del outputs

            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                cur_image = input_per_image["image"].to(self.device)
                processed_results.append({})
                scale_factor = max(images.tensor.shape[-2:]) / max(height, width)
                ori_height, ori_width = round(height * scale_factor), round(width * scale_factor)
                mask_pred_result = mask_pred_result[:, :ori_height, :ori_width].expand(1, -1, -1, -1)
                cur_image = cur_image[:, :ori_height, :ori_width].expand(1, -1, -1, -1)
                mask_pred_result = F.interpolate(
                    mask_pred_result, size=(height, width), mode="bilinear", align_corners=align_corners
                )[0]
                cur_image = F.interpolate(
                    cur_image.float(), size=(height, width), mode="bilinear", align_corners=align_corners
                )[0].to(torch.uint8)

                if self.sem_seg_postprocess_before_inference:
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r
                    processed_results[-1]["original_image"] = cur_image
                
                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["instances"] = instance_r

                processed_results[-1]['features']=features
            return processed_results

    def prepare_targets(self, targets, targets_semantic, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image, semantic_gt_mask in zip(targets, targets_semantic):
            # pad gt
            gt_masks = targets_per_image.gt_masks
            #padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            #padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks

            #padded_semantic_masks = -torch.ones((h_pad, w_pad), dtype=semantic_gt_mask.dtype, device=semantic_gt_mask.device)
            #padded_semantic_masks[: semantic_gt_mask.shape[0], : semantic_gt_mask.shape[1]] = semantic_gt_mask

            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": gt_masks,
                    "semantic_masks": semantic_gt_mask
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        # For cls prob, we exluced the void class following
        # https://github.com/google-research/deeplab2/blob/main/model/post_processor/max_deeplab.py#L199
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = F.softmax(mask_pred, dim=0)
        # Mask2Former combines the soft prob to obtain sem results
        # In kMaX, the mask logits is argmax'ed.
        # https://github.com/google-research/deeplab2/blob/main/model/post_processor/max_deeplab.py#L315
        mask_pred = (mask_pred >= mask_pred.max(dim=0, keepdim=True)[0]).float()
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        # mask_cls: N x C
        # mask_pred: N x H x W
        # some hyper-params
        
        num_mask_slots = mask_pred.shape[0]
        cls_threshold_thing = self.class_threshold_thing
        cls_threshold_stuff = self.class_threshold_stuff
        object_mask_threshold = self.object_mask_threshold
        overlap_threshold = self.overlap_threshold
        reorder_class_weight = self.reorder_class_weight
        reorder_mask_weight = self.reorder_mask_weight

        # https://github.com/google-research/deeplab2/blob/main/model/post_processor/max_deeplab.py#L675
        # https://github.com/google-research/deeplab2/blob/main/model/post_processor/max_deeplab.py#L199
        # import pdb; pdb.set_trace()
        cls_scores, cls_labels = F.softmax(mask_cls, dim=-1)[..., :-1].max(-1) # N
        mask_scores = F.softmax(mask_pred, dim=0)
        binary_masks = mask_scores > object_mask_threshold # N x H x W
        mask_scores_flat = mask_scores.flatten(1) # N x HW
        binary_masks_flat = binary_masks.flatten(1).float() # N x HW
        pixel_number_flat = binary_masks_flat.sum(1) # N
        mask_scores_flat = (mask_scores_flat * binary_masks_flat).sum(1) / torch.clamp(pixel_number_flat, min=1.0) # N

        reorder_score = (cls_scores ** reorder_class_weight) * (mask_scores_flat ** reorder_mask_weight) # N
        reorder_indices = torch.argsort(reorder_score, dim=-1, descending=True)

        panoptic_seg = torch.zeros((mask_pred.shape[1], mask_pred.shape[2]),
         dtype=torch.int32, device=mask_pred.device)
        segments_info = []

        current_segment_id = 0
        stuff_memory_list = {}
        confidence=0
        for i in range(num_mask_slots):
            cur_idx = reorder_indices[i].item() # 1
            cur_binary_mask = binary_masks[cur_idx] # H x W
            cur_cls_score = cls_scores[cur_idx].item() # 1 
            cur_cls_label = cls_labels[cur_idx].item() # 1
            
            is_thing = cur_cls_label in self.metadata.thing_dataset_id_to_contiguous_id.values()

            ## get stuff
            # if not is_thing:
            #     continue
            
            is_confident = (is_thing and cur_cls_score > cls_threshold_thing) or (
                (not is_thing) and cur_cls_score > cls_threshold_stuff)

            original_pixel_number = cur_binary_mask.float().sum()
            new_binary_mask = torch.logical_and(cur_binary_mask, (panoptic_seg == 0))
            new_pixel_number = new_binary_mask.float().sum()
            is_not_overlap_too_much = new_pixel_number > (original_pixel_number * overlap_threshold)

            # Filter by area size.
            if (is_thing and new_pixel_number < self.thing_area_limit) or (
                not is_thing and new_pixel_number < self.stuff_area_limit):
                continue

            if is_confident and is_not_overlap_too_much:
                # merge stuff regions
                # import pdb; pdb.set_trace()
                # if COCO_META[cur_cls_label+1]['name']=='cardboard':
                #     confidence=cur_cls_score
                #     print(COCO_META[cur_cls_label+1])
                #     print('confidence:', cur_cls_score)
                if not is_thing:
                    if int(cur_cls_label) in stuff_memory_list.keys():
                        panoptic_seg[new_binary_mask] = stuff_memory_list[int(cur_cls_label)]
                        continue
                    else:
                        stuff_memory_list[int(cur_cls_label)] = current_segment_id + 1

                current_segment_id += 1                
                panoptic_seg[new_binary_mask] = current_segment_id

                segments_info.append(
                    {
                        "id": current_segment_id,
                        "isthing": bool(is_thing),
                        "category_id": int(cur_cls_label),
                        "confidence": cur_cls_score,
                    }
                )

        return panoptic_seg, segments_info


    # This is not carefully aligned!
    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        mask_pred = mask_pred.softmax(dim=0)
        # [Q, K]
        scores = F.softmax(mask_cls[:, :-1], dim=-1)
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // self.sem_seg_head.num_classes
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > self.object_mask_threshold).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        # result.confidence=confidence
        return result
