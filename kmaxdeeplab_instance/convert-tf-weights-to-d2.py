import tensorflow as tf
import pickle as pkl
import sys

import torch
import numpy as np

ADE20K_150_CATEGORIES = [
    {"color": [120, 120, 120], "id": 0, "isthing": 0, "name": "wall"},
    {"color": [180, 120, 120], "id": 1, "isthing": 0, "name": "building"},
    {"color": [6, 230, 230], "id": 2, "isthing": 0, "name": "sky"},
    {"color": [80, 50, 50], "id": 3, "isthing": 0, "name": "floor"},
    {"color": [4, 200, 3], "id": 4, "isthing": 0, "name": "tree"},
    {"color": [120, 120, 80], "id": 5, "isthing": 0, "name": "ceiling"},
    {"color": [140, 140, 140], "id": 6, "isthing": 0, "name": "road, route"},
    {"color": [204, 5, 255], "id": 7, "isthing": 1, "name": "bed"},
    {"color": [230, 230, 230], "id": 8, "isthing": 1, "name": "window "},
    {"color": [4, 250, 7], "id": 9, "isthing": 0, "name": "grass"},
    {"color": [224, 5, 255], "id": 10, "isthing": 1, "name": "cabinet"},
    {"color": [235, 255, 7], "id": 11, "isthing": 0, "name": "sidewalk, pavement"},
    {"color": [150, 5, 61], "id": 12, "isthing": 1, "name": "person"},
    {"color": [120, 120, 70], "id": 13, "isthing": 0, "name": "earth, ground"},
    {"color": [8, 255, 51], "id": 14, "isthing": 1, "name": "door"},
    {"color": [255, 6, 82], "id": 15, "isthing": 1, "name": "table"},
    {"color": [143, 255, 140], "id": 16, "isthing": 0, "name": "mountain, mount"},
    {"color": [204, 255, 4], "id": 17, "isthing": 0, "name": "plant"},
    {"color": [255, 51, 7], "id": 18, "isthing": 1, "name": "curtain"},
    {"color": [204, 70, 3], "id": 19, "isthing": 1, "name": "chair"},
    {"color": [0, 102, 200], "id": 20, "isthing": 1, "name": "car"},
    {"color": [61, 230, 250], "id": 21, "isthing": 0, "name": "water"},
    {"color": [255, 6, 51], "id": 22, "isthing": 1, "name": "painting, picture"},
    {"color": [11, 102, 255], "id": 23, "isthing": 1, "name": "sofa"},
    {"color": [255, 7, 71], "id": 24, "isthing": 1, "name": "shelf"},
    {"color": [255, 9, 224], "id": 25, "isthing": 0, "name": "house"},
    {"color": [9, 7, 230], "id": 26, "isthing": 0, "name": "sea"},
    {"color": [220, 220, 220], "id": 27, "isthing": 1, "name": "mirror"},
    {"color": [255, 9, 92], "id": 28, "isthing": 0, "name": "rug"},
    {"color": [112, 9, 255], "id": 29, "isthing": 0, "name": "field"},
    {"color": [8, 255, 214], "id": 30, "isthing": 1, "name": "armchair"},
    {"color": [7, 255, 224], "id": 31, "isthing": 1, "name": "seat"},
    {"color": [255, 184, 6], "id": 32, "isthing": 1, "name": "fence"},
    {"color": [10, 255, 71], "id": 33, "isthing": 1, "name": "desk"},
    {"color": [255, 41, 10], "id": 34, "isthing": 0, "name": "rock, stone"},
    {"color": [7, 255, 255], "id": 35, "isthing": 1, "name": "wardrobe, closet, press"},
    {"color": [224, 255, 8], "id": 36, "isthing": 1, "name": "lamp"},
    {"color": [102, 8, 255], "id": 37, "isthing": 1, "name": "tub"},
    {"color": [255, 61, 6], "id": 38, "isthing": 1, "name": "rail"},
    {"color": [255, 194, 7], "id": 39, "isthing": 1, "name": "cushion"},
    {"color": [255, 122, 8], "id": 40, "isthing": 0, "name": "base, pedestal, stand"},
    {"color": [0, 255, 20], "id": 41, "isthing": 1, "name": "box"},
    {"color": [255, 8, 41], "id": 42, "isthing": 1, "name": "column, pillar"},
    {"color": [255, 5, 153], "id": 43, "isthing": 1, "name": "signboard, sign"},
    {
        "color": [6, 51, 255],
        "id": 44,
        "isthing": 1,
        "name": "chest of drawers, chest, bureau, dresser",
    },
    {"color": [235, 12, 255], "id": 45, "isthing": 1, "name": "counter"},
    {"color": [160, 150, 20], "id": 46, "isthing": 0, "name": "sand"},
    {"color": [0, 163, 255], "id": 47, "isthing": 1, "name": "sink"},
    {"color": [140, 140, 140], "id": 48, "isthing": 0, "name": "skyscraper"},
    {"color": [250, 10, 15], "id": 49, "isthing": 1, "name": "fireplace"},
    {"color": [20, 255, 0], "id": 50, "isthing": 1, "name": "refrigerator, icebox"},
    {"color": [31, 255, 0], "id": 51, "isthing": 0, "name": "grandstand, covered stand"},
    {"color": [255, 31, 0], "id": 52, "isthing": 0, "name": "path"},
    {"color": [255, 224, 0], "id": 53, "isthing": 1, "name": "stairs"},
    {"color": [153, 255, 0], "id": 54, "isthing": 0, "name": "runway"},
    {"color": [0, 0, 255], "id": 55, "isthing": 1, "name": "case, display case, showcase, vitrine"},
    {
        "color": [255, 71, 0],
        "id": 56,
        "isthing": 1,
        "name": "pool table, billiard table, snooker table",
    },
    {"color": [0, 235, 255], "id": 57, "isthing": 1, "name": "pillow"},
    {"color": [0, 173, 255], "id": 58, "isthing": 1, "name": "screen door, screen"},
    {"color": [31, 0, 255], "id": 59, "isthing": 0, "name": "stairway, staircase"},
    {"color": [11, 200, 200], "id": 60, "isthing": 0, "name": "river"},
    {"color": [255, 82, 0], "id": 61, "isthing": 0, "name": "bridge, span"},
    {"color": [0, 255, 245], "id": 62, "isthing": 1, "name": "bookcase"},
    {"color": [0, 61, 255], "id": 63, "isthing": 0, "name": "blind, screen"},
    {"color": [0, 255, 112], "id": 64, "isthing": 1, "name": "coffee table"},
    {
        "color": [0, 255, 133],
        "id": 65,
        "isthing": 1,
        "name": "toilet, can, commode, crapper, pot, potty, stool, throne",
    },
    {"color": [255, 0, 0], "id": 66, "isthing": 1, "name": "flower"},
    {"color": [255, 163, 0], "id": 67, "isthing": 1, "name": "book"},
    {"color": [255, 102, 0], "id": 68, "isthing": 0, "name": "hill"},
    {"color": [194, 255, 0], "id": 69, "isthing": 1, "name": "bench"},
    {"color": [0, 143, 255], "id": 70, "isthing": 1, "name": "countertop"},
    {"color": [51, 255, 0], "id": 71, "isthing": 1, "name": "stove"},
    {"color": [0, 82, 255], "id": 72, "isthing": 1, "name": "palm, palm tree"},
    {"color": [0, 255, 41], "id": 73, "isthing": 1, "name": "kitchen island"},
    {"color": [0, 255, 173], "id": 74, "isthing": 1, "name": "computer"},
    {"color": [10, 0, 255], "id": 75, "isthing": 1, "name": "swivel chair"},
    {"color": [173, 255, 0], "id": 76, "isthing": 1, "name": "boat"},
    {"color": [0, 255, 153], "id": 77, "isthing": 0, "name": "bar"},
    {"color": [255, 92, 0], "id": 78, "isthing": 1, "name": "arcade machine"},
    {"color": [255, 0, 255], "id": 79, "isthing": 0, "name": "hovel, hut, hutch, shack, shanty"},
    {"color": [255, 0, 245], "id": 80, "isthing": 1, "name": "bus"},
    {"color": [255, 0, 102], "id": 81, "isthing": 1, "name": "towel"},
    {"color": [255, 173, 0], "id": 82, "isthing": 1, "name": "light"},
    {"color": [255, 0, 20], "id": 83, "isthing": 1, "name": "truck"},
    {"color": [255, 184, 184], "id": 84, "isthing": 0, "name": "tower"},
    {"color": [0, 31, 255], "id": 85, "isthing": 1, "name": "chandelier"},
    {"color": [0, 255, 61], "id": 86, "isthing": 1, "name": "awning, sunshade, sunblind"},
    {"color": [0, 71, 255], "id": 87, "isthing": 1, "name": "street lamp"},
    {"color": [255, 0, 204], "id": 88, "isthing": 1, "name": "booth"},
    {"color": [0, 255, 194], "id": 89, "isthing": 1, "name": "tv"},
    {"color": [0, 255, 82], "id": 90, "isthing": 1, "name": "plane"},
    {"color": [0, 10, 255], "id": 91, "isthing": 0, "name": "dirt track"},
    {"color": [0, 112, 255], "id": 92, "isthing": 1, "name": "clothes"},
    {"color": [51, 0, 255], "id": 93, "isthing": 1, "name": "pole"},
    {"color": [0, 194, 255], "id": 94, "isthing": 0, "name": "land, ground, soil"},
    {
        "color": [0, 122, 255],
        "id": 95,
        "isthing": 1,
        "name": "bannister, banister, balustrade, balusters, handrail",
    },
    {
        "color": [0, 255, 163],
        "id": 96,
        "isthing": 0,
        "name": "escalator, moving staircase, moving stairway",
    },
    {
        "color": [255, 153, 0],
        "id": 97,
        "isthing": 1,
        "name": "ottoman, pouf, pouffe, puff, hassock",
    },
    {"color": [0, 255, 10], "id": 98, "isthing": 1, "name": "bottle"},
    {"color": [255, 112, 0], "id": 99, "isthing": 0, "name": "buffet, counter, sideboard"},
    {
        "color": [143, 255, 0],
        "id": 100,
        "isthing": 0,
        "name": "poster, posting, placard, notice, bill, card",
    },
    {"color": [82, 0, 255], "id": 101, "isthing": 0, "name": "stage"},
    {"color": [163, 255, 0], "id": 102, "isthing": 1, "name": "van"},
    {"color": [255, 235, 0], "id": 103, "isthing": 1, "name": "ship"},
    {"color": [8, 184, 170], "id": 104, "isthing": 1, "name": "fountain"},
    {
        "color": [133, 0, 255],
        "id": 105,
        "isthing": 0,
        "name": "conveyer belt, conveyor belt, conveyer, conveyor, transporter",
    },
    {"color": [0, 255, 92], "id": 106, "isthing": 0, "name": "canopy"},
    {
        "color": [184, 0, 255],
        "id": 107,
        "isthing": 1,
        "name": "washer, automatic washer, washing machine",
    },
    {"color": [255, 0, 31], "id": 108, "isthing": 1, "name": "plaything, toy"},
    {"color": [0, 184, 255], "id": 109, "isthing": 0, "name": "pool"},
    {"color": [0, 214, 255], "id": 110, "isthing": 1, "name": "stool"},
    {"color": [255, 0, 112], "id": 111, "isthing": 1, "name": "barrel, cask"},
    {"color": [92, 255, 0], "id": 112, "isthing": 1, "name": "basket, handbasket"},
    {"color": [0, 224, 255], "id": 113, "isthing": 0, "name": "falls"},
    {"color": [112, 224, 255], "id": 114, "isthing": 0, "name": "tent"},
    {"color": [70, 184, 160], "id": 115, "isthing": 1, "name": "bag"},
    {"color": [163, 0, 255], "id": 116, "isthing": 1, "name": "minibike, motorbike"},
    {"color": [153, 0, 255], "id": 117, "isthing": 0, "name": "cradle"},
    {"color": [71, 255, 0], "id": 118, "isthing": 1, "name": "oven"},
    {"color": [255, 0, 163], "id": 119, "isthing": 1, "name": "ball"},
    {"color": [255, 204, 0], "id": 120, "isthing": 1, "name": "food, solid food"},
    {"color": [255, 0, 143], "id": 121, "isthing": 1, "name": "step, stair"},
    {"color": [0, 255, 235], "id": 122, "isthing": 0, "name": "tank, storage tank"},
    {"color": [133, 255, 0], "id": 123, "isthing": 1, "name": "trade name"},
    {"color": [255, 0, 235], "id": 124, "isthing": 1, "name": "microwave"},
    {"color": [245, 0, 255], "id": 125, "isthing": 1, "name": "pot"},
    {"color": [255, 0, 122], "id": 126, "isthing": 1, "name": "animal"},
    {"color": [255, 245, 0], "id": 127, "isthing": 1, "name": "bicycle"},
    {"color": [10, 190, 212], "id": 128, "isthing": 0, "name": "lake"},
    {"color": [214, 255, 0], "id": 129, "isthing": 1, "name": "dishwasher"},
    {"color": [0, 204, 255], "id": 130, "isthing": 1, "name": "screen"},
    {"color": [20, 0, 255], "id": 131, "isthing": 0, "name": "blanket, cover"},
    {"color": [255, 255, 0], "id": 132, "isthing": 1, "name": "sculpture"},
    {"color": [0, 153, 255], "id": 133, "isthing": 1, "name": "hood, exhaust hood"},
    {"color": [0, 41, 255], "id": 134, "isthing": 1, "name": "sconce"},
    {"color": [0, 255, 204], "id": 135, "isthing": 1, "name": "vase"},
    {"color": [41, 0, 255], "id": 136, "isthing": 1, "name": "traffic light"},
    {"color": [41, 255, 0], "id": 137, "isthing": 1, "name": "tray"},
    {"color": [173, 0, 255], "id": 138, "isthing": 1, "name": "trash can"},
    {"color": [0, 245, 255], "id": 139, "isthing": 1, "name": "fan"},
    {"color": [71, 0, 255], "id": 140, "isthing": 0, "name": "pier"},
    {"color": [122, 0, 255], "id": 141, "isthing": 0, "name": "crt screen"},
    {"color": [0, 255, 184], "id": 142, "isthing": 1, "name": "plate"},
    {"color": [0, 92, 255], "id": 143, "isthing": 1, "name": "monitor"},
    {"color": [184, 255, 0], "id": 144, "isthing": 1, "name": "bulletin board"},
    {"color": [0, 133, 255], "id": 145, "isthing": 0, "name": "shower"},
    {"color": [255, 214, 0], "id": 146, "isthing": 1, "name": "radiator"},
    {"color": [25, 194, 194], "id": 147, "isthing": 1, "name": "glass, drinking glass"},
    {"color": [102, 255, 0], "id": 148, "isthing": 1, "name": "clock"},
    {"color": [92, 0, 255], "id": 149, "isthing": 1, "name": "flag"},
]

CITYSCAPES_CATEGORIES = [
    {"color": (128, 64, 128), "isthing": 0, "id": 7, "trainId": 0, "name": "road"},
    {"color": (244, 35, 232), "isthing": 0, "id": 8, "trainId": 1, "name": "sidewalk"},
    {"color": (70, 70, 70), "isthing": 0, "id": 11, "trainId": 2, "name": "building"},
    {"color": (102, 102, 156), "isthing": 0, "id": 12, "trainId": 3, "name": "wall"},
    {"color": (190, 153, 153), "isthing": 0, "id": 13, "trainId": 4, "name": "fence"},
    {"color": (153, 153, 153), "isthing": 0, "id": 17, "trainId": 5, "name": "pole"},
    {"color": (250, 170, 30), "isthing": 0, "id": 19, "trainId": 6, "name": "traffic light"},
    {"color": (220, 220, 0), "isthing": 0, "id": 20, "trainId": 7, "name": "traffic sign"},
    {"color": (107, 142, 35), "isthing": 0, "id": 21, "trainId": 8, "name": "vegetation"},
    {"color": (152, 251, 152), "isthing": 0, "id": 22, "trainId": 9, "name": "terrain"},
    {"color": (70, 130, 180), "isthing": 0, "id": 23, "trainId": 10, "name": "sky"},
    {"color": (220, 20, 60), "isthing": 1, "id": 24, "trainId": 11, "name": "person"},
    {"color": (255, 0, 0), "isthing": 1, "id": 25, "trainId": 12, "name": "rider"},
    {"color": (0, 0, 142), "isthing": 1, "id": 26, "trainId": 13, "name": "car"},
    {"color": (0, 0, 70), "isthing": 1, "id": 27, "trainId": 14, "name": "truck"},
    {"color": (0, 60, 100), "isthing": 1, "id": 28, "trainId": 15, "name": "bus"},
    {"color": (0, 80, 100), "isthing": 1, "id": 31, "trainId": 16, "name": "train"},
    {"color": (0, 0, 230), "isthing": 1, "id": 32, "trainId": 17, "name": "motorcycle"},
    {"color": (119, 11, 32), "isthing": 1, "id": 33, "trainId": 18, "name": "bicycle"},
]



def load_tf_weights(ckpt_path):
    # https://stackoverflow.com/questions/40118062/how-to-read-weights-saved-in-tensorflow-checkpoint-file
    from tensorflow.python.training import py_checkpoint_reader
    reader = py_checkpoint_reader.NewCheckpointReader(ckpt_path)
    state_dict = {}
    for k in reader.get_variable_to_shape_map():
        if '.OPTIMIZER_SLOT' in k or 'optimizer' in k or '_CHECKPOINTABLE_OBJECT_GRAPH' in k or 'save_counter' in k or 'global_step' in k:
            continue
        v = reader.get_tensor(k)
        state_dict[k.replace('/.ATTRIBUTES/VARIABLE_VALUE', '')] = v
    for k in sorted(state_dict.keys()):
        print(k, state_dict[k].shape)
    return state_dict

def map_bn(name1, name2):
    res = {}
    res[name1 + '/gamma'] = name2 + ".weight"
    res[name1 + '/beta'] = name2 + ".bias"
    res[name1 + '/moving_mean'] = name2 + ".running_mean"
    res[name1 + '/moving_variance'] = name2 + ".running_var"
    return res


def map_conv(name1, name2, dw=False, bias=False):
    res = {}
    if dw:
        res[name1 + '/depthwise_kernel'] = name2 + ".weight"
    else:
        res[name1 + '/kernel'] = name2 + ".weight"
    if bias:
        res[name1 + '/bias'] = name2 + ".bias"
    return res


def tf_2_torch_mapping_r50():
    res = {}
    res.update(map_conv('encoder/_stem/_conv', 'backbone.stem.conv1'))
    res.update(map_bn('encoder/_stem/_batch_norm', 'backbone.stem.conv1.norm'))
    block_num = {2: 3, 3: 4, 4: 6, 5: 3}
    for stage_idx in range(2, 6):
        for block_idx in range(1, block_num[stage_idx] + 1):
            res.update(map_conv(f'encoder/_stage{stage_idx}/_block{block_idx}/_conv1_bn_act/_conv',
             f'backbone.res{stage_idx}.{block_idx-1}.conv1'))
            res.update(map_bn(f'encoder/_stage{stage_idx}/_block{block_idx}/_conv1_bn_act/_batch_norm',
             f'backbone.res{stage_idx}.{block_idx-1}.conv1.norm'))
            res.update(map_conv(f'encoder/_stage{stage_idx}/_block{block_idx}/_conv2_bn_act/_conv',
             f'backbone.res{stage_idx}.{block_idx-1}.conv2'))
            res.update(map_bn(f'encoder/_stage{stage_idx}/_block{block_idx}/_conv2_bn_act/_batch_norm',
             f'backbone.res{stage_idx}.{block_idx-1}.conv2.norm'))
            res.update(map_conv(f'encoder/_stage{stage_idx}/_block{block_idx}/_conv3_bn/_conv',
             f'backbone.res{stage_idx}.{block_idx-1}.conv3'))
            res.update(map_bn(f'encoder/_stage{stage_idx}/_block{block_idx}/_conv3_bn/_batch_norm',
             f'backbone.res{stage_idx}.{block_idx-1}.conv3.norm'))
            res.update(map_conv(f'encoder/_stage{stage_idx}/_block{block_idx}/_shortcut/_conv',
             f'backbone.res{stage_idx}.{block_idx-1}.shortcut'))
            res.update(map_bn(f'encoder/_stage{stage_idx}/_block{block_idx}/_shortcut/_batch_norm',
             f'backbone.res{stage_idx}.{block_idx-1}.shortcut.norm'))
    return res

def tf_2_torch_mapping_convnext():
    res = {}
    for i in range(4):
        if i == 0:
            res.update(map_conv(f'encoder/downsample_layers/{i}/layer_with_weights-0',
                f'backbone.downsample_layers.{i}.0', bias=True))
            res.update(map_bn(f'encoder/downsample_layers/{i}/layer_with_weights-1',
                f'backbone.downsample_layers.{i}.1'))
        else:
            res.update(map_conv(f'encoder/downsample_layers/{i}/layer_with_weights-1',
                f'backbone.downsample_layers.{i}.1', bias=True))
            res.update(map_bn(f'encoder/downsample_layers/{i}/layer_with_weights-0',
                f'backbone.downsample_layers.{i}.0'))
    
    block_num = {0: 3, 1: 3, 2: 27, 3: 3}
    for stage_idx in range(4):
        for block_idx in range(block_num[stage_idx]):
            res.update(map_conv(f'encoder/stages/{stage_idx}/layer_with_weights-{block_idx}/depthwise_conv',
                f'backbone.stages.{stage_idx}.{block_idx}.dwconv', bias=True))
            res.update(map_bn(f'encoder/stages/{stage_idx}/layer_with_weights-{block_idx}/norm',
                f'backbone.stages.{stage_idx}.{block_idx}.norm'))
            res.update(map_conv(f'encoder/stages/{stage_idx}/layer_with_weights-{block_idx}/pointwise_conv1',
                f'backbone.stages.{stage_idx}.{block_idx}.pwconv1', bias=True))
            res.update(map_conv(f'encoder/stages/{stage_idx}/layer_with_weights-{block_idx}/pointwise_conv2',
                f'backbone.stages.{stage_idx}.{block_idx}.pwconv2', bias=True))
            res[f'encoder/stages/{stage_idx}/layer_with_weights-{block_idx}/layer_scale'] = f'backbone.stages.{stage_idx}.{block_idx}.gamma'
    
    return res

def tf_2_torch_mapping_pixel_dec():
    res = {}
    for i in range(5):
        res.update(map_bn(f'pixel_decoder/_backbone_norms/{i}', f'sem_seg_head.pixel_decoder._in_norms.{i}'))
    
    for i in range(4):
        res.update(map_conv(f'pixel_decoder/_skip_connections/{i}/_resized_conv_bn1/_conv',
         f'sem_seg_head.pixel_decoder._resized_fuses.{i}._conv_bn_low.conv'))
        res.update(map_bn(f'pixel_decoder/_skip_connections/{i}/_resized_conv_bn1/_batch_norm',
         f'sem_seg_head.pixel_decoder._resized_fuses.{i}._conv_bn_low.norm'))
        res.update(map_conv(f'pixel_decoder/_skip_connections/{i}/_resized_conv_bn2/_conv',
         f'sem_seg_head.pixel_decoder._resized_fuses.{i}._conv_bn_high.conv'))
        res.update(map_bn(f'pixel_decoder/_skip_connections/{i}/_resized_conv_bn2/_batch_norm',
         f'sem_seg_head.pixel_decoder._resized_fuses.{i}._conv_bn_high.norm'))

    num_blocks = {0: 1, 1:5, 2:1, 3:1, 4:1}
    for stage_idx in range(5):
        for block_idx in range(1, 1+num_blocks[stage_idx]):
            res.update(map_conv(f'pixel_decoder/_stages/{stage_idx}/_block{block_idx}/_shortcut/_conv',
                f'sem_seg_head.pixel_decoder._stages.{stage_idx}._blocks.{block_idx-1}._shortcut.conv'))
            res.update(map_bn(f'pixel_decoder/_stages/{stage_idx}/_block{block_idx}/_shortcut/_batch_norm',
                f'sem_seg_head.pixel_decoder._stages.{stage_idx}._blocks.{block_idx-1}._shortcut.norm'))
            res.update(map_conv(f'pixel_decoder/_stages/{stage_idx}/_block{block_idx}/_conv1_bn_act/_conv',
                f'sem_seg_head.pixel_decoder._stages.{stage_idx}._blocks.{block_idx-1}._conv1_bn_act.conv'))
            res.update(map_bn(f'pixel_decoder/_stages/{stage_idx}/_block{block_idx}/_conv1_bn_act/_batch_norm',
                f'sem_seg_head.pixel_decoder._stages.{stage_idx}._blocks.{block_idx-1}._conv1_bn_act.norm'))
            res.update(map_conv(f'pixel_decoder/_stages/{stage_idx}/_block{block_idx}/_conv3_bn/_conv',
                f'sem_seg_head.pixel_decoder._stages.{stage_idx}._blocks.{block_idx-1}._conv3_bn.conv'))
            res.update(map_bn(f'pixel_decoder/_stages/{stage_idx}/_block{block_idx}/_conv3_bn/_batch_norm',
                f'sem_seg_head.pixel_decoder._stages.{stage_idx}._blocks.{block_idx-1}._conv3_bn.norm'))
            if stage_idx <= 1:
                for attn in ['height', 'width']:
                    res.update(map_bn(f'pixel_decoder/_stages/{stage_idx}/_block{block_idx}/_attention/_{attn}_axis/_batch_norm_qkv',
                    f'sem_seg_head.pixel_decoder._stages.{stage_idx}._blocks.{block_idx-1}._attention._{attn}_axis._batch_norm_qkv'))
                    res.update(map_bn(f'pixel_decoder/_stages/{stage_idx}/_block{block_idx}/_attention/_{attn}_axis/_batch_norm_retrieved_output',
                    f'sem_seg_head.pixel_decoder._stages.{stage_idx}._blocks.{block_idx-1}._attention._{attn}_axis._batch_norm_retrieved_output'))
                    res.update(map_bn(f'pixel_decoder/_stages/{stage_idx}/_block{block_idx}/_attention/_{attn}_axis/_batch_norm_similarity',
                    f'sem_seg_head.pixel_decoder._stages.{stage_idx}._blocks.{block_idx-1}._attention._{attn}_axis._batch_norm_similarity'))
                    res[f'pixel_decoder/_stages/{stage_idx}/_block{block_idx}/_attention/_{attn}_axis/_key_rpe/embeddings'] = (
                        f'sem_seg_head.pixel_decoder._stages.{stage_idx}._blocks.{block_idx-1}._attention._{attn}_axis._key_rpe._embeddings.weight')
                    res[f'pixel_decoder/_stages/{stage_idx}/_block{block_idx}/_attention/_{attn}_axis/_query_rpe/embeddings'] = (
                        f'sem_seg_head.pixel_decoder._stages.{stage_idx}._blocks.{block_idx-1}._attention._{attn}_axis._query_rpe._embeddings.weight')
                    res[f'pixel_decoder/_stages/{stage_idx}/_block{block_idx}/_attention/_{attn}_axis/_value_rpe/embeddings'] = (
                        f'sem_seg_head.pixel_decoder._stages.{stage_idx}._blocks.{block_idx-1}._attention._{attn}_axis._value_rpe._embeddings.weight')
                    res[f'pixel_decoder/_stages/{stage_idx}/_block{block_idx}/_attention/_{attn}_axis/qkv_kernel'] = (
                        f'sem_seg_head.pixel_decoder._stages.{stage_idx}._blocks.{block_idx-1}._attention._{attn}_axis.qkv_transform.conv.weight')
            else:
                res.update(map_conv(f'pixel_decoder/_stages/{stage_idx}/_block{block_idx}/_conv2_bn_act/_conv',
                    f'sem_seg_head.pixel_decoder._stages.{stage_idx}._blocks.{block_idx-1}._conv2_bn_act.conv'))
                res.update(map_bn(f'pixel_decoder/_stages/{stage_idx}/_block{block_idx}/_conv2_bn_act/_batch_norm',
                    f'sem_seg_head.pixel_decoder._stages.{stage_idx}._blocks.{block_idx-1}._conv2_bn_act.norm'))
    return res     


def tf_2_torch_mapping_predcitor(prefix_tf, prefix_torch):
    res = {}
    res.update(map_bn(prefix_tf + 'pixel_space_feature_batch_norm',
        prefix_torch + '_pixel_space_head_last_convbn.norm'))
    res[prefix_tf + 'pixel_space_head/conv_block/_conv1_bn_act/_depthwise/_depthwise_conv/depthwise_kernel'] = (
        prefix_torch + '_pixel_space_head_conv0bnact.conv.weight'
    )
    res.update(map_bn(prefix_tf + 'pixel_space_head/conv_block/_conv1_bn_act/_depthwise/_batch_norm',
        prefix_torch + '_pixel_space_head_conv0bnact.norm'))
    res.update(map_conv(prefix_tf + 'pixel_space_head/conv_block/_conv1_bn_act/_pointwise/_conv',
        prefix_torch + '_pixel_space_head_conv1bnact.conv'))
    res.update(map_bn(prefix_tf + 'pixel_space_head/conv_block/_conv1_bn_act/_pointwise/_batch_norm',
        prefix_torch + '_pixel_space_head_conv1bnact.norm'))
    res.update(map_conv(prefix_tf + 'pixel_space_head/final_conv',
        prefix_torch + '_pixel_space_head_last_convbn.conv', bias=True))
    res.update(map_bn(prefix_tf + 'pixel_space_mask_batch_norm',
        prefix_torch + '_pixel_space_mask_batch_norm'))
    res.update(map_conv(prefix_tf + 'transformer_class_head/_conv',
        prefix_torch + '_transformer_class_head.conv', bias=True))
    res.update(map_conv(prefix_tf + 'transformer_mask_head/_conv',
        prefix_torch + '_transformer_mask_head.conv'))
    res.update(map_bn(prefix_tf + 'transformer_mask_head/_batch_norm',
        prefix_torch + '_transformer_mask_head.norm'))
    
    return res


def tf_2_torch_mapping_trans_dec():
    res = {}

    res.update(map_bn('transformer_decoder/_class_embedding_projection/_batch_norm',
        'sem_seg_head.predictor._class_embedding_projection.norm'))
    res.update(map_conv('transformer_decoder/_class_embedding_projection/_conv',
        'sem_seg_head.predictor._class_embedding_projection.conv'))
    res.update(map_bn('transformer_decoder/_mask_embedding_projection/_batch_norm',
        'sem_seg_head.predictor._mask_embedding_projection.norm'))
    res.update(map_conv('transformer_decoder/_mask_embedding_projection/_conv',
        'sem_seg_head.predictor._mask_embedding_projection.conv'))

    res['transformer_decoder/cluster_centers'] = 'sem_seg_head.predictor._cluster_centers.weight'

    res.update(tf_2_torch_mapping_predcitor(
            prefix_tf = '',
            prefix_torch = 'sem_seg_head.predictor._predcitor.'
        ))
    for kmax_idx in range(6):
        res.update(tf_2_torch_mapping_predcitor(
            prefix_tf = f'transformer_decoder/_kmax_decoder/{kmax_idx}/_block1_transformer/_auxiliary_clustering_predictor/_',
            prefix_torch = f'sem_seg_head.predictor._kmax_transformer_layers.{kmax_idx}._predcitor.'
        ))
        common_prefix_tf = f'transformer_decoder/_kmax_decoder/{kmax_idx}/_block1_transformer/'
        common_prefix_torch = f'sem_seg_head.predictor._kmax_transformer_layers.{kmax_idx}.'
        res.update(map_bn(common_prefix_tf + '_kmeans_memory_batch_norm_retrieved_value',
            common_prefix_torch + '_kmeans_query_batch_norm_retrieved_value'))
        res.update(map_bn(common_prefix_tf + '_kmeans_memory_conv3_bn/_batch_norm',
            common_prefix_torch + '_kmeans_query_conv3_bn.norm'))
        res.update(map_conv(common_prefix_tf + '_kmeans_memory_conv3_bn/_conv',
            common_prefix_torch + '_kmeans_query_conv3_bn.conv'))
        res.update(map_bn(common_prefix_tf + '_memory_attention/_batch_norm_retrieved_value',
            common_prefix_torch + '_query_self_attention._batch_norm_retrieved_value'))
        res.update(map_bn(common_prefix_tf + '_memory_attention/_batch_norm_similarity',
            common_prefix_torch + '_query_self_attention._batch_norm_similarity'))

        res.update(map_bn(common_prefix_tf + '_memory_conv1_bn_act/_batch_norm',
            common_prefix_torch + '_query_conv1_bn_act.norm'))
        res.update(map_conv(common_prefix_tf + '_memory_conv1_bn_act/_conv',
            common_prefix_torch + '_query_conv1_bn_act.conv'))

        res.update(map_bn(common_prefix_tf + '_memory_conv3_bn/_batch_norm',
            common_prefix_torch + '_query_conv3_bn.norm'))
        res.update(map_conv(common_prefix_tf + '_memory_conv3_bn/_conv',
            common_prefix_torch + '_query_conv3_bn.conv'))

        res.update(map_bn(common_prefix_tf + '_memory_ffn_conv1_bn_act/_batch_norm',
            common_prefix_torch + '_query_ffn_conv1_bn_act.norm'))
        res.update(map_conv(common_prefix_tf + '_memory_ffn_conv1_bn_act/_conv',
            common_prefix_torch + '_query_ffn_conv1_bn_act.conv'))

        res.update(map_bn(common_prefix_tf + '_memory_ffn_conv2_bn/_batch_norm',
            common_prefix_torch + '_query_ffn_conv2_bn.norm'))
        res.update(map_conv(common_prefix_tf + '_memory_ffn_conv2_bn/_conv',
            common_prefix_torch + '_query_ffn_conv2_bn.conv'))

        res.update(map_bn(common_prefix_tf + '_memory_qkv_conv_bn/_batch_norm',
            common_prefix_torch + '_query_qkv_conv_bn.norm'))
        res.update(map_conv(common_prefix_tf + '_memory_qkv_conv_bn/_conv',
            common_prefix_torch + '_query_qkv_conv_bn.conv'))

        res.update(map_bn(common_prefix_tf + '_pixel_conv1_bn_act/_batch_norm',
            common_prefix_torch + '_pixel_conv1_bn_act.norm'))
        res.update(map_conv(common_prefix_tf + '_pixel_conv1_bn_act/_conv',
            common_prefix_torch + '_pixel_conv1_bn_act.conv'))

        res.update(map_bn(common_prefix_tf + '_pixel_v_conv_bn/_batch_norm',
            common_prefix_torch + '_pixel_v_conv_bn.norm'))
        res.update(map_conv(common_prefix_tf + '_pixel_v_conv_bn/_conv',
            common_prefix_torch + '_pixel_v_conv_bn.conv'))

    return res


def tf_2_torch_mapping_aux_semanic_dec():
    res = {}
    res.update(map_conv('semantic_decoder/_aspp/_conv_bn_act/_conv',
             'sem_seg_head.predictor._auxiliary_semantic_predictor._aspp._aspp_conv0.conv'))
    res.update(map_bn('semantic_decoder/_aspp/_conv_bn_act/_batch_norm',
             'sem_seg_head.predictor._auxiliary_semantic_predictor._aspp._aspp_conv0.norm'))
    
    res.update(map_conv('semantic_decoder/_aspp/_aspp_pool/_conv_bn_act/_conv',
             'sem_seg_head.predictor._auxiliary_semantic_predictor._aspp._aspp_pool.conv'))
    res.update(map_bn('semantic_decoder/_aspp/_aspp_pool/_conv_bn_act/_batch_norm',
             'sem_seg_head.predictor._auxiliary_semantic_predictor._aspp._aspp_pool.norm'))

    res.update(map_conv('semantic_decoder/_aspp/_proj_conv_bn_act/_conv',
             'sem_seg_head.predictor._auxiliary_semantic_predictor._aspp._proj_conv_bn_act.conv'))
    res.update(map_bn('semantic_decoder/_aspp/_proj_conv_bn_act/_batch_norm',
             'sem_seg_head.predictor._auxiliary_semantic_predictor._aspp._proj_conv_bn_act.norm'))
    for i in range(1, 4):
        res.update(map_conv(f'semantic_decoder/_aspp/_aspp_conv{i}/_conv_bn_act/_conv',
             f'sem_seg_head.predictor._auxiliary_semantic_predictor._aspp._aspp_conv{i}.conv'))
        res.update(map_bn(f'semantic_decoder/_aspp/_aspp_conv{i}/_conv_bn_act/_batch_norm',
             f'sem_seg_head.predictor._auxiliary_semantic_predictor._aspp._aspp_conv{i}.norm'))
    
    res.update({
        'semantic_decoder/_fusion_conv1/_conv1_bn_act/_depthwise/_depthwise_conv/depthwise_kernel':
         'sem_seg_head.predictor._auxiliary_semantic_predictor._low_level_fusion_os8_conv0_bn_act.conv.weight'})
    res.update(map_bn('semantic_decoder/_fusion_conv1/_conv1_bn_act/_depthwise/_batch_norm',
             'sem_seg_head.predictor._auxiliary_semantic_predictor._low_level_fusion_os8_conv0_bn_act.norm'))
    res.update({
        'semantic_decoder/_fusion_conv1/_conv1_bn_act/_pointwise/_conv/kernel':
         'sem_seg_head.predictor._auxiliary_semantic_predictor._low_level_fusion_os8_conv1_bn_act.conv.weight'})
    res.update(map_bn('semantic_decoder/_fusion_conv1/_conv1_bn_act/_pointwise/_batch_norm',
             'sem_seg_head.predictor._auxiliary_semantic_predictor._low_level_fusion_os8_conv1_bn_act.norm'))
    
    res.update({
        'semantic_decoder/_fusion_conv2/_conv1_bn_act/_depthwise/_depthwise_conv/depthwise_kernel':
         'sem_seg_head.predictor._auxiliary_semantic_predictor._low_level_fusion_os4_conv0_bn_act.conv.weight'})
    res.update(map_bn('semantic_decoder/_fusion_conv2/_conv1_bn_act/_depthwise/_batch_norm',
             'sem_seg_head.predictor._auxiliary_semantic_predictor._low_level_fusion_os4_conv0_bn_act.norm'))
    res.update({
        'semantic_decoder/_fusion_conv2/_conv1_bn_act/_pointwise/_conv/kernel':
         'sem_seg_head.predictor._auxiliary_semantic_predictor._low_level_fusion_os4_conv1_bn_act.conv.weight'})
    res.update(map_bn('semantic_decoder/_fusion_conv2/_conv1_bn_act/_pointwise/_batch_norm',
             'sem_seg_head.predictor._auxiliary_semantic_predictor._low_level_fusion_os4_conv1_bn_act.norm'))

    res.update({
        'semantic_decoder/_low_level_conv1/_conv/kernel':
         'sem_seg_head.predictor._auxiliary_semantic_predictor._low_level_projection_os8.conv.weight'})
    res.update(map_bn('semantic_decoder/_low_level_conv1/_batch_norm',
             'sem_seg_head.predictor._auxiliary_semantic_predictor._low_level_projection_os8.norm'))
    res.update({
        'semantic_decoder/_low_level_conv2/_conv/kernel':
         'sem_seg_head.predictor._auxiliary_semantic_predictor._low_level_projection_os4.conv.weight'})
    res.update(map_bn('semantic_decoder/_low_level_conv2/_batch_norm',
             'sem_seg_head.predictor._auxiliary_semantic_predictor._low_level_projection_os4.norm'))
    

    res.update({
        'semantic_head_without_last_layer/_conv1_bn_act/_depthwise/_depthwise_conv/depthwise_kernel':
         'sem_seg_head.predictor._auxiliary_semantic_predictor.conv_block_0.conv.weight'})
    res.update(map_bn('semantic_head_without_last_layer/_conv1_bn_act/_depthwise/_batch_norm',
             'sem_seg_head.predictor._auxiliary_semantic_predictor.conv_block_0.norm'))
    res.update({
        'semantic_head_without_last_layer/_conv1_bn_act/_pointwise/_conv/kernel':
         'sem_seg_head.predictor._auxiliary_semantic_predictor.conv_block_1.conv.weight'})
    res.update(map_bn('semantic_head_without_last_layer/_conv1_bn_act/_pointwise/_batch_norm',
             'sem_seg_head.predictor._auxiliary_semantic_predictor.conv_block_1.norm'))

    res.update({
        'semantic_last_layer/kernel':
         'sem_seg_head.predictor._auxiliary_semantic_predictor.final_conv.conv.weight'})
    res.update({
        'semantic_last_layer/bias':
         'sem_seg_head.predictor._auxiliary_semantic_predictor.final_conv.conv.bias'})
    return res


# python3 convert-tf-weights-to-d2.py kmax_resnet50_coco_train/ckpt-150000 tf_kmax_r50.pkl

if __name__ == "__main__":
    input = sys.argv[1]

    state_dict = load_tf_weights(input)
    #exit()

    state_dict_torch = {}

    mapping_key = {}
    if 'resnet50' in input:
        mapping_key.update(tf_2_torch_mapping_r50())
    elif 'convnext' in input:
        mapping_key.update(tf_2_torch_mapping_convnext())
    mapping_key.update(tf_2_torch_mapping_pixel_dec())
    mapping_key.update(tf_2_torch_mapping_trans_dec())

    mapping_key.update(tf_2_torch_mapping_aux_semanic_dec())

    for k in state_dict.keys():
        value = state_dict[k]
        k2 = mapping_key[k]
        rank = len(value.shape)

        if '_batch_norm_retrieved_output' in k2 or '_batch_norm_similarity' in k2 or '_batch_norm_retrieved_value' in k2:
            value = np.reshape(value, [-1])
        elif 'qkv_transform.conv.weight' in k2:
            # (512, 1024) -> (1024, 512, 1)
            value = np.transpose(value, (1, 0))[:, :, None]
        elif '_cluster_centers.weight' in k2:
            # (1, 128, 256) -> (256, 128)
            value = np.transpose(value[0], (1, 0))
        elif '_pixel_conv1_bn_act.conv.weight' in k2:
            # (1, 512, 256) -> (256, 512, 1, 1)
            value = np.transpose(value, (2, 1, 0))[:, :, :, None]
        elif '_pixel_v_conv_bn.conv.weight' in k2:
            # (1, 256, 256) -> (256, 256, 1, 1) 
            value = np.transpose(value, (2, 1, 0))[:, :, :, None]
        elif '_pixel_space_head_conv0bnact.conv.weight' in k2:
            # (5, 5, 256, 1) -> (256, 1, 5, 5)
            value = np.transpose(value, (2, 3, 0, 1))
        elif '/layer_scale' in k:
            value = np.reshape(value, [-1])
        elif 'pwconv1.weight' in k2 or 'pwconv2.weight' in k2:
            # (128, 512) -> (512, 128)
            value = np.transpose(value, (1, 0))
        elif ('_low_level_fusion_os4_conv0_bn_act.conv.weight' in k2 
        or '_low_level_fusion_os8_conv0_bn_act.conv.weight' in k2 
        or 'sem_seg_head.predictor._auxiliary_semantic_predictor.conv_block_0.conv.weight' in k2):
            value = np.transpose(value, (2, 3, 0, 1))
        else:
            if rank == 1: # bias, norm etc
                pass
            elif rank == 2: # _query_rpe
                pass
            elif rank == 3: # conv 1d kernel, etc
                value = np.transpose(value, (2, 1, 0))
            elif rank == 4: # conv 2d kernel, etc
                value = np.transpose(value, (3, 2, 0, 1))

        classifer_key = '_transformer_class_head.conv.'
        if classifer_key in k2:
            # For cityscsapes, the tf uses [thing_classes, stuff_classes, void] 
            # and we need to transpose to align to torch format.
            if value.shape[0] in [20, 151]: # cityscapes
                if value.shape[0] == 20:
                    META_INFO = CITYSCAPES_CATEGORIES
                elif value.shape[0] == 151:
                    META_INFO = ADE20K_150_CATEGORIES

                shuffled_index = []
                mapping = []
                inverse_mapping = [-1] * (value.shape[0] - 1)
                for i in range(value.shape[0] - 1):
                    is_thing = META_INFO[i]['isthing']
                    if is_thing:
                        mapping.append(i)
                        inverse_mapping[i] = len(mapping) - 1
                for i in range(value.shape[0] - 1):
                    is_thing = META_INFO[i]['isthing']
                    if not is_thing:
                        mapping.append(i)
                        inverse_mapping[i] = len(mapping) - 1

                print('classifer_key', value.shape)
                value = np.concatenate([value[:-1][inverse_mapping], value[-1:]], axis=0)
        state_dict_torch[k2] = value


    res = {"model": state_dict_torch, "__author__": "third_party", "matching_heuristics": True}

    with open(sys.argv[2], "wb") as f:
        pkl.dump(res, f)