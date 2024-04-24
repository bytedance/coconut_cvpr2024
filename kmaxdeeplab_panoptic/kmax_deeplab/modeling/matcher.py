# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/matcher.py
# Reference: https://github.com/google-research/deeplab2/blob/main/model/loss/max_deeplab_loss.py
# Modified by Qihang Yu

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.cuda.amp import autocast
import numpy as np


# https://github.com/google-research/deeplab2/blob/c4a533c14fac1a1071a6d24c5379c31a69a3e5e6/model/loss/max_deeplab_loss.py#L158
@torch.no_grad()
def compute_mask_similarity(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    denominator_epsilon = 1e-5
    inputs = F.softmax(inputs, dim=0)
    inputs = inputs.flatten(1) # N x HW

    pixel_gt_non_void_mask = (targets.sum(0, keepdim=True) > 0).to(inputs) # 1xHW
    inputs = inputs * pixel_gt_non_void_mask

    intersection = torch.einsum("nc,mc->nm", inputs, targets)
    denominator = (inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]) / 2.0
    return intersection / (denominator + denominator_epsilon)


# https://github.com/google-research/deeplab2/blob/c4a533c14fac1a1071a6d24c5379c31a69a3e5e6/model/loss/max_deeplab_loss.py#L941
@torch.no_grad()
def compute_class_similarity(inputs: torch.Tensor, targets: torch.Tensor):
    pred_class_prob = inputs.softmax(-1)[..., :-1] # exclude the void class
    return pred_class_prob[:, targets]


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """More memory-friendly matching"""
        bs, num_queries = outputs["pred_logits"].shape[:2]

        indices = []
        matched_dice = []
        matched_cls_prob = []
        # Iterate through batch size
        for b in range(bs):
            with autocast(enabled=False):
                class_similarity = compute_class_similarity(outputs["pred_logits"][b].float(), targets[b]["labels"])
            out_mask = outputs["pred_masks"][b].flatten(1)  # [num_queries, H_pred, W_pred]
            # gt masks are already padded when preparing target
            tgt_mask = targets[b]["masks"].to(out_mask).flatten(1)
            with autocast(enabled=False):
                mask_similarity = compute_mask_similarity(out_mask.float(), tgt_mask.float())
            
            # Final cost matrix
            C = - mask_similarity * class_similarity
            C = C.reshape(num_queries, -1).cpu() # N x M , N = num_queries, M = num_gt

            # the assignment will be truncated to a square matrix.
            row_ind, col_ind = linear_sum_assignment(C)
            matched_dice.append(mask_similarity[row_ind, col_ind].detach())
            matched_cls_prob.append(class_similarity[row_ind, col_ind].detach())
            indices.append((row_ind, col_ind)) # row_ind and col_ind, row_ind = 0,1,2,3,...,N-1, col_ind = a,b,c,d,...

        indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]
        
        return indices, matched_dice, matched_cls_prob
    

    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets)

    def __repr__(self, _repr_indent=4):
        head = "Matcher " + self.__class__.__name__
        body = []
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)