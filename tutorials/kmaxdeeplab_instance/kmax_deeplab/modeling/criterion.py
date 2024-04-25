# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/criterion.py
# Reference: https://github.com/google-research/deeplab2/blob/main/model/loss/max_deeplab_loss.py
# Modified by Qihang Yu

import torch
import torch.nn.functional as F
from torch import nn

from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)

from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
_SOFTMAX_MASKING_CONSTANT = -99999.0

# https://www.tensorflow.org/api_docs/python/tf/math/divide_no_nan
def divide_no_nan(x: torch.Tensor, y: torch.Tensor):
    return torch.nan_to_num(x / y, nan=0.0, posinf=0.0, neginf=0.0)


# https://github.com/google-research/deeplab2/blob/main/model/loss/max_deeplab_loss.py#L50
def _gumbel_topk_sample(logits: torch.Tensor, k: int):
    """Samples k points from the softmax distribution with Gumbel-Top-k trick."""
    # Note that torch.rand is [0, 1), we need to make it (0, 1) to ensure the log is valid.
    gumbel_noise = torch.rand(size=logits.shape, dtype=logits.dtype, device=logits.device)
    gumbel_noise = -torch.log(-torch.log(gumbel_noise))
    _, indices = torch.topk(logits + gumbel_noise, k)
    return indices


# https://github.com/google-research/deeplab2/blob/main/model/loss/max_deeplab_loss.py#L576
def pixelwise_insdis_loss(
    pixel_feature: torch.Tensor,
    gt_mask: torch.Tensor,
    sample_temperature: float,
    sample_k: int,
    instance_discrimination_temperature: float,
    pixel_gt_void_mask: torch.Tensor,
    inverse_gt_mask_area: torch.Tensor
    ):
    
    # pixel_feature: B x C x H x W
    # gt_mask: B x N x H x W
    pixel_feature = pixel_feature.flatten(2) # B x C x HW
    gt_mask = gt_mask.flatten(2) # B x N x HW
    pixel_gt_void_mask = pixel_gt_void_mask.flatten(1) # B x HW
    inverse_gt_mask_area = inverse_gt_mask_area.flatten(1) # B x HW

    sample_logits = torch.log(inverse_gt_mask_area) * sample_temperature # B x HW
    # sample_logits.masked_fill_(pixel_gt_void_mask, float('-inf'))
    sample_logits += pixel_gt_void_mask.to(sample_logits) * _SOFTMAX_MASKING_CONSTANT

    sample_indices = _gumbel_topk_sample(sample_logits, sample_k) # B x K
    # Sample ground truth one-hot encodings and compute gt_similarity.
    pixel_gt_sampled_feature = torch.gather(gt_mask, dim=2, index=sample_indices.unsqueeze(1).repeat(1, gt_mask.shape[1], 1)) # B x N x K
    sampled_gt_similarity = torch.einsum('bnk,bnj->bkj', pixel_gt_sampled_feature, pixel_gt_sampled_feature) # B x K x K

    # Normalize the ground truth similarity into a distribution (sum to 1).
    pixel_normalizing_constant = sampled_gt_similarity.sum(dim=1, keepdim=True) # B x 1 x K
    sampled_gt_similarity /= torch.clamp(pixel_normalizing_constant, min=1.0) # B x K x K

    # Sample predicted features and compute pred_similarity.
    pixel_pred_sampled_feature = torch.gather(pixel_feature, dim=2, index=sample_indices.unsqueeze(1).repeat(1, pixel_feature.shape[1], 1)) # B x C x K
    sampled_pred_similarity = torch.einsum('bck,bcj->bkj', pixel_pred_sampled_feature, pixel_pred_sampled_feature) # B x K x K
    sampled_pred_similarity /= instance_discrimination_temperature # B x K x K
    loss = F.cross_entropy(sampled_pred_similarity, sampled_gt_similarity, reduction="none") # B x K

    num_non_zero = (loss != 0.0).to(loss).sum(-1) # B
    num_non_zero = torch.clamp(num_non_zero, min=1.0)
    loss_sum_per_sample = loss.sum(-1) # B
    return divide_no_nan(loss_sum_per_sample, num_non_zero).mean() # 1


def aux_semantic_loss(
    pred_semantic_logits: torch.Tensor,
    ground_truth_semantic: torch.Tensor,
    sample_temperature: float,
    sample_k: int,
    pixel_gt_void_mask: torch.Tensor,
    inverse_gt_mask_area: torch.Tensor,
    num_classes: int):

    # The pred maybe in lower resolution, we downsample gt beforehand.
    if pred_semantic_logits.shape[-2:] != ground_truth_semantic.shape[-2:]:
        assert (ground_truth_semantic.shape[-1] - 1) // (pred_semantic_logits.shape[-1] - 1) == (ground_truth_semantic.shape[-2] - 1) // (pred_semantic_logits.shape[-2] - 1)
        stride = (ground_truth_semantic.shape[-1] - 1) // (pred_semantic_logits.shape[-1] - 1)
        ground_truth_semantic = ground_truth_semantic[:, ::stride, ::stride]
        pixel_gt_void_mask = pixel_gt_void_mask[:, ::stride, ::stride]
        inverse_gt_mask_area = inverse_gt_mask_area[:, ::stride, ::stride]

    pred_semantic_logits = pred_semantic_logits.flatten(2) # B x C x HW
    ground_truth_semantic = ground_truth_semantic.flatten(1) # B x HW
    pixel_gt_void_mask = pixel_gt_void_mask.flatten(1) # B x HW
    inverse_gt_mask_area = inverse_gt_mask_area.flatten(1) # B x HW
    if sample_k == 0:
        # This falls back to normal cross-entropy loss
        sampled_ground_truth_semantic = ground_truth_semantic # B x HW
        sampled_pred_semantic_logits = pred_semantic_logits # B x C x HW
    else:
        sample_logits = torch.log(inverse_gt_mask_area) * sample_temperature # B x HW
        sample_logits += pixel_gt_void_mask.to(sample_logits) * _SOFTMAX_MASKING_CONSTANT
        sample_indices = _gumbel_topk_sample(sample_logits, sample_k) # B x K
        sampled_ground_truth_semantic = torch.gather(ground_truth_semantic, dim=1, index=sample_indices) # B x K
        sampled_pred_semantic_logits = torch.gather(pred_semantic_logits, dim=2, index=sample_indices.unsqueeze(1).repeat(1, pred_semantic_logits.shape[1], 1)) # B x C x K
    # ignore the class index num_classes.
    keep_mask = (sampled_ground_truth_semantic != num_classes) # B x K
    loss = F.cross_entropy(sampled_pred_semantic_logits, sampled_ground_truth_semantic, ignore_index=num_classes, reduction='none') # B x K
    loss = loss * keep_mask.to(loss)
    num_non_zero = (loss != 0.0).to(loss).sum(-1) # B
    num_non_zero = torch.clamp(num_non_zero, min=1.0)
    loss_sum_per_sample = loss.sum(-1) # B
    return divide_no_nan(loss_sum_per_sample, num_non_zero).mean() # 1


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float    
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, share_final_matching,
                 pixel_insdis_temperature=1.5, pixel_insdis_sample_k=4096,
                 aux_semantic_temperature=2.0, aux_semantic_sample_k=4096,
                 masking_void_pixel=True,
                 num_points=12544, oversample_ratio=3.0, importance_sample_ratio=0.75):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.share_final_matching = share_final_matching
        self.pixel_insdis_temperature = pixel_insdis_temperature
        self.pixel_insdis_sample_k = pixel_insdis_sample_k
        self.aux_semantic_temperature = aux_semantic_temperature
        self.aux_semantic_sample_k = aux_semantic_sample_k
        self.masking_void_pixel = masking_void_pixel

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = 12544 #num_points
        self.oversample_ratio = 3.0 #oversample_ratio
        self.importance_sample_ratio = 0.75 #importance_sample_ratio

    def loss_labels(self, outputs, targets):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"] # B x N x C
        target_classes = targets["labels"] # B x N
        losses = {"loss_ce": F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)}
        return losses
    
    def loss_masks(self, outputs, targets):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_masks = outputs["pred_masks"][targets["src_idx"]]
        target_masks = targets["matched_target_masks"]
        num_masks = targets["num_masks"]

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=True,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=True,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def loss_pixels(self, outputs, targets):
        pixel_feature = outputs["pixel_feature"]
        target_masks = targets["masks"]
        pixel_gt_void_mask = targets["pixel_gt_void_mask"]
        inverse_gt_mask_area = targets["inverse_gt_mask_area"]

        losses = {"loss_pixel_insdis": pixelwise_insdis_loss(
            pixel_feature=pixel_feature,
            gt_mask=target_masks,
            sample_temperature=self.pixel_insdis_temperature,
            sample_k=self.pixel_insdis_sample_k,
            instance_discrimination_temperature=0.3,
            pixel_gt_void_mask=pixel_gt_void_mask,
            inverse_gt_mask_area=inverse_gt_mask_area
            )}

        del target_masks
        return losses

    def loss_semantic(self, outputs, targets):
        pred_semantic_logits = outputs["aux_semantic_pred"]
        ground_truth_semantic = targets["ground_truth_semantic"]
        pixel_gt_void_mask = targets["pixel_gt_void_mask"]
        inverse_gt_mask_area = targets["inverse_gt_mask_area"]

        losses = {"loss_aux_semantic": aux_semantic_loss(
            pred_semantic_logits=pred_semantic_logits,
            ground_truth_semantic=ground_truth_semantic,
            sample_temperature=self.aux_semantic_temperature,
            sample_k=self.aux_semantic_sample_k,
            pixel_gt_void_mask=pixel_gt_void_mask,
            inverse_gt_mask_area=inverse_gt_mask_area,
            num_classes=self.num_classes
        )}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


    def get_loss(self, loss, outputs, targets):
        loss_map = {
            'labels': self.loss_labels,
            'masks': self.loss_masks,
            'pixels': self.loss_pixels,
            'aux_semantic': self.loss_semantic,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets)

    @torch.no_grad()
    def process_gt(self, outputs, targets, indices, process_semantic=False):
        # Permute&Pad Pred&GT for loss compuation.
        # By controling process_gt, we can share the matching results for all preds.
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_masks = outputs["pred_masks"] # B x N x H x W

        target_masks = [t["masks"] for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(target_masks).decompose()
        target_masks = target_masks.to(src_masks) # B x N x H x W
        matched_target_masks = target_masks[tgt_idx] # M x H x W
        target_masks = target_masks[:, :, ::4, ::4]

        # pixel_gt_void_mask is used to indicate those pixels without labels.
        pixel_gt_void_mask = (target_masks.sum(1) < 1) # B x H x W
   
        # inverse_gt_mask_area is used to sample pixels.
        mask_gt_area = target_masks.sum(2).sum(2) # B x N
        pixel_gt_area = torch.einsum('bnhw,bn->bhw', target_masks, mask_gt_area) # B x H x W
        inverse_gt_mask_area = (pixel_gt_area.shape[1] * pixel_gt_area.shape[2]) / torch.clamp(pixel_gt_area, min=1.0) # B x H x W

        src_logits = outputs["pred_logits"] # B x N x C
        # Pad and permute the target_classes to B x N
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        # This serves as a padding.
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        # We put real GT to those corresponds to src_idx, and put void into other places.
        target_classes[src_idx] = target_classes_o

        
        processed_gt = {"src_idx": src_idx,
                        "matched_target_masks": matched_target_masks,
                        "masks": target_masks, "labels": target_classes,
                        "pixel_gt_void_mask": pixel_gt_void_mask,
                        "inverse_gt_mask_area": inverse_gt_mask_area,}
    
        if process_semantic and "semantic_masks" in targets[0]:
            # To obtain semantic gt
            ground_truth_semantic = [t["semantic_masks"] for t in targets]
            ground_truth_semantic = torch.stack(ground_truth_semantic, dim=0) # B x H x W
            # self.num_classes is set to ignore label
            ground_truth_semantic[ground_truth_semantic==-1] = self.num_classes
            processed_gt.update({"ground_truth_semantic": ground_truth_semantic[:, ::4, ::4]})

        return processed_gt


    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_masks = sum(len(t["labels"]) for t in targets)
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()

        # Pad GT to the same number of prediction.
        processed_targets = self.process_gt(outputs, targets, indices, process_semantic=False)
        # Compute all the requested losses
        processed_targets["num_masks"] = num_masks
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, processed_targets))

        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                # We share matching results across predictions.
                if not self.share_final_matching:
                    indices = self.matcher(aux_outputs, targets)
                    processed_targets = self.process_gt(aux_outputs, targets, indices, process_semantic=False)
                    processed_targets["num_masks"] = num_masks
                for loss in self.losses:
                    if loss in ['aux_semantic']:
                        # Only for final output.
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, processed_targets)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)