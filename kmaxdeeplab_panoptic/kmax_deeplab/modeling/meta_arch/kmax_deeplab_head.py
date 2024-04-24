# Reference: https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/meta_arch/mask_former_head.py
# Modified by Qihang Yu

from typing import Dict

from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.kmax_transformer_decoder import build_transformer_decoder


# https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/modeling/pixel_decoder/fpn.py#L21
def build_pixel_decoder(cfg, input_shape):
    """
    Build a pixel decoder from `cfg.MODEL.KMAX_DEEPLAB.PIXEL_DEC.NAME`.
    """
    name = cfg.MODEL.KMAX_DEEPLAB.PIXEL_DEC.NAME
    model = SEM_SEG_HEADS_REGISTRY.get(name)(cfg, input_shape)
    forward_features = getattr(model, "forward_features", None)
    if not callable(forward_features):
        raise ValueError(
            "Only SEM_SEG_HEADS with forward_features method can be used as pixel decoder. "
            f"Please implement forward_features for {name} to only return mask features."
        )
    return model


@SEM_SEG_HEADS_REGISTRY.register()
class kMaXDeepLabHead(nn.Module):

    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        num_classes: int,
        pixel_decoder: nn.Module,
        loss_weight: float = 1.0,
        ignore_value: int = -1,
        # extra parameters
        transformer_predictor: nn.Module,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]

        self.ignore_value = ignore_value
        self.common_stride = 4
        self.loss_weight = loss_weight

        self.pixel_decoder = pixel_decoder
        self.predictor = transformer_predictor

        self.num_classes = num_classes

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        # figure out in_channels to transformer predictor
        return {
            "input_shape": {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.KMAX_DEEPLAB.PIXEL_DEC.IN_FEATURES
            },
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "pixel_decoder": build_pixel_decoder(cfg, input_shape),
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "transformer_predictor": build_transformer_decoder(cfg, input_shape),
        }

    def forward(self, features):
        return self.layers(features)

    def layers(self, features):
        panoptic_features, semantic_features, multi_scale_features = self.pixel_decoder.forward_features(features)
        predictions = self.predictor(multi_scale_features, panoptic_features, semantic_features)
        return predictions
