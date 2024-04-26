import os
import copy
import json
# import mmcv
import cv2
import numpy as np
# import pdb
import pycocotools.mask as mask_utils
from panopticapi.utils import rgb2id
# from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
from tqdm import tqdm
# from coco_meta import COCO_META 
from skimage import measure
# from itertools import groupby
import argparse
from PIL import Image


def get_parser():
    parser = argparse.ArgumentParser(description='extract corners from annotation')
    parser.add_argument('--mask-dir', type=str, help='panoptic segmentation mask dir')
    parser.add_argument('--image-dir', type=str, help='image dir')
    parser.add_argument('--panseg-info', type=str, help='input panseg info json file')
    parser.add_argument('--output', type=str, help='output json file path for instances')
    args = parser.parse_args()
    return args


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_rle(binary_mask):
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    # Ensure the binary mask array has dtype 'uint8'
    binary_mask = binary_mask.astype(np.uint8)
    # Convert the binary mask to RLE using pycocotools
    rle = mask_utils.encode(np.asfortranarray(binary_mask))
    rle['counts'] = rle['counts'].decode('ascii')
    return rle

def binary_mask_to_polygon(binary_mask, tolerance=2):
    """Converts a binary mask to COCO polygon representation

    Args:
        binary_mask: a 2D binary numpy array where '1's represent the object
        tolerance: Maximum distance from original points of polygon to approximated
            polygonal chain. If tolerance is 0, the original coordinate array is returned.

    """
    polygons = []
    # pad mask to close contours of shapes which start and end at an edge
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)

    if len(contours)==1:
        contours = np.subtract(contours, 1)
    else:
        contours = [np.subtract(contour, 1) for contour in contours]

        
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        # after padding and subtracting 1 we may get -0.5 points in our segmentation 
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)

    return polygons



if __name__ == "__main__":
    args = get_parser()
    mask_dir=args.mask_dir
    img_dir=args.image_dir
    panseg_info=args.panseg_info
    output=args.output

    # init box_id or instance id
    box_id=0

    img_list=[]
    anno_list=[]
    
    # load panseg infos to load mask with segids
    with open(panseg_info,'r') as f:
        panseg_info=json.load(f)

    for anno in tqdm(panseg_info['annotations']):
        img_id=anno['image_id']
        img_id=f'{img_id:012d}'

        segments_info=anno['segments_info']

        img_dict={}
        img_dict["license"]=3
        image_path=f'{img_dir}/{img_id}.jpg'
        img_dict["file_name"]=f'{img_id}.jpg'
        image=cv2.imread(image_path)
        img_shape=image.shape
        img_dict["height"]=img_shape[0]
        img_dict["width"]=img_shape[1]
        img_dict['id']=int(img_id)
        img_dict['date_captured']= '2013-11-14 16:03:19'
        img_dict['coco_url']= f'http://images.cocodataset.org/val2017/{img_id}.jpg'
        img_list.append(img_dict)

        mask_path=f'{mask_dir}/{img_id}.png'
        if not os.path.exists(mask_path):
            continue

        panoptic_orig = np.asarray(Image.open(mask_path), dtype=np.uint32)
        panseg = rgb2id(panoptic_orig)

        for segment_info in segments_info:
            # skip stuff classes
            if not segment_info['isthing']:
                continue
                
            box_id += 1
            segment_id=segment_info['id']
            
            binary_mask=np.array(panseg==segment_id,dtype=np.int32)

            item={}
            item['image_id']=int(img_id)
            item['category_id']=segment_info['category_id']
            item['id']=box_id
            item['iscrowd']=segment_info['iscrowd']       
            item['area']=np.count_nonzero(np.array(binary_mask>0,dtype=np.float32))

            segmentation = np.where(binary_mask>0)

            if len(segmentation) != 0 and len(segmentation[1]) != 0 and len(segmentation[0]) != 0:
                x_min = float(np.min(segmentation[1]))
                x_max = float(np.max(segmentation[1]))
                y_min = float(np.min(segmentation[0]))
                y_max = float(np.max(segmentation[0]))

        
            if segment_info['iscrowd'] or True:
                segmentations=binary_mask_to_rle(binary_mask)
            else:
                segmentations=binary_mask_to_polygon(binary_mask)

            if len(segmentations)==0:
                continue

            item['segmentation']=segmentations
            item['bbox'] = [x_min,  y_min, x_max-x_min, y_max-y_min]
            
            anno_list.append(item)


    out_annos={}
    out_annos['images']=img_list
    out_annos['annotations']=anno_list

    # copy paste the informtion from COCO instances json files

    out_annos['categories']=[{'supercategory': 'person', 'isthing': 1, 'id': 1, 'name': 'person'}, {'supercategory': 'vehicle', 'isthing': 1, 'id': 2, 'name': 'bicycle'}, {'supercategory': 'vehicle', 'isthing': 1, 'id': 3, 'name': 'car'}, {'supercategory': 'vehicle', 'isthing': 1, 'id': 4, 'name': 'motorcycle'}, {'supercategory': 'vehicle', 'isthing': 1, 'id': 5, 'name': 'airplane'}, {'supercategory': 'vehicle', 'isthing': 1, 'id': 6, 'name': 'bus'}, {'supercategory': 'vehicle', 'isthing': 1, 'id': 7, 'name': 'train'}, {'supercategory': 'vehicle', 'isthing': 1, 'id': 8, 'name': 'truck'}, {'supercategory': 'vehicle', 'isthing': 1, 'id': 9, 'name': 'boat'}, {'supercategory': 'outdoor', 'isthing': 1, 'id': 10, 'name': 'traffic light'}, {'supercategory': 'outdoor', 'isthing': 1, 'id': 11, 'name': 'fire hydrant'}, {'supercategory': 'outdoor', 'isthing': 1, 'id': 13, 'name': 'stop sign'}, {'supercategory': 'outdoor', 'isthing': 1, 'id': 14, 'name': 'parking meter'}, {'supercategory': 'outdoor', 'isthing': 1, 'id': 15, 'name': 'bench'}, {'supercategory': 'animal', 'isthing': 1, 'id': 16, 'name': 'bird'}, {'supercategory': 'animal', 'isthing': 1, 'id': 17, 'name': 'cat'}, {'supercategory': 'animal', 'isthing': 1, 'id': 18, 'name': 'dog'}, {'supercategory': 'animal', 'isthing': 1, 'id': 19, 'name': 'horse'}, {'supercategory': 'animal', 'isthing': 1, 'id': 20, 'name': 'sheep'}, {'supercategory': 'animal', 'isthing': 1, 'id': 21, 'name': 'cow'}, {'supercategory': 'animal', 'isthing': 1, 'id': 22, 'name': 'elephant'}, {'supercategory': 'animal', 'isthing': 1, 'id': 23, 'name': 'bear'}, {'supercategory': 'animal', 'isthing': 1, 'id': 24, 'name': 'zebra'}, {'supercategory': 'animal', 'isthing': 1, 'id': 25, 'name': 'giraffe'}, {'supercategory': 'accessory', 'isthing': 1, 'id': 27, 'name': 'backpack'}, {'supercategory': 'accessory', 'isthing': 1, 'id': 28, 'name': 'umbrella'}, {'supercategory': 'accessory', 'isthing': 1, 'id': 31, 'name': 'handbag'}, {'supercategory': 'accessory', 'isthing': 1, 'id': 32, 'name': 'tie'}, {'supercategory': 'accessory', 'isthing': 1, 'id': 33, 'name': 'suitcase'}, {'supercategory': 'sports', 'isthing': 1, 'id': 34, 'name': 'frisbee'}, {'supercategory': 'sports', 'isthing': 1, 'id': 35, 'name': 'skis'}, {'supercategory': 'sports', 'isthing': 1, 'id': 36, 'name': 'snowboard'}, {'supercategory': 'sports', 'isthing': 1, 'id': 37, 'name': 'sports ball'}, {'supercategory': 'sports', 'isthing': 1, 'id': 38, 'name': 'kite'}, {'supercategory': 'sports', 'isthing': 1, 'id': 39, 'name': 'baseball bat'}, {'supercategory': 'sports', 'isthing': 1, 'id': 40, 'name': 'baseball glove'}, {'supercategory': 'sports', 'isthing': 1, 'id': 41, 'name': 'skateboard'}, {'supercategory': 'sports', 'isthing': 1, 'id': 42, 'name': 'surfboard'}, {'supercategory': 'sports', 'isthing': 1, 'id': 43, 'name': 'tennis racket'}, {'supercategory': 'kitchen', 'isthing': 1, 'id': 44, 'name': 'bottle'}, {'supercategory': 'kitchen', 'isthing': 1, 'id': 46, 'name': 'wine glass'}, {'supercategory': 'kitchen', 'isthing': 1, 'id': 47, 'name': 'cup'}, {'supercategory': 'kitchen', 'isthing': 1, 'id': 48, 'name': 'fork'}, {'supercategory': 'kitchen', 'isthing': 1, 'id': 49, 'name': 'knife'}, {'supercategory': 'kitchen', 'isthing': 1, 'id': 50, 'name': 'spoon'}, {'supercategory': 'kitchen', 'isthing': 1, 'id': 51, 'name': 'bowl'}, {'supercategory': 'food', 'isthing': 1, 'id': 52, 'name': 'banana'}, {'supercategory': 'food', 'isthing': 1, 'id': 53, 'name': 'apple'}, {'supercategory': 'food', 'isthing': 1, 'id': 54, 'name': 'sandwich'}, {'supercategory': 'food', 'isthing': 1, 'id': 55, 'name': 'orange'}, {'supercategory': 'food', 'isthing': 1, 'id': 56, 'name': 'broccoli'}, {'supercategory': 'food', 'isthing': 1, 'id': 57, 'name': 'carrot'}, {'supercategory': 'food', 'isthing': 1, 'id': 58, 'name': 'hot dog'}, {'supercategory': 'food', 'isthing': 1, 'id': 59, 'name': 'pizza'}, {'supercategory': 'food', 'isthing': 1, 'id': 60, 'name': 'donut'}, {'supercategory': 'food', 'isthing': 1, 'id': 61, 'name': 'cake'}, {'supercategory': 'furniture', 'isthing': 1, 'id': 62, 'name': 'chair'}, {'supercategory': 'furniture', 'isthing': 1, 'id': 63, 'name': 'couch'}, {'supercategory': 'furniture', 'isthing': 1, 'id': 64, 'name': 'potted plant'}, {'supercategory': 'furniture', 'isthing': 1, 'id': 65, 'name': 'bed'}, {'supercategory': 'furniture', 'isthing': 1, 'id': 67, 'name': 'dining table'}, {'supercategory': 'furniture', 'isthing': 1, 'id': 70, 'name': 'toilet'}, {'supercategory': 'electronic', 'isthing': 1, 'id': 72, 'name': 'tv'}, {'supercategory': 'electronic', 'isthing': 1, 'id': 73, 'name': 'laptop'}, {'supercategory': 'electronic', 'isthing': 1, 'id': 74, 'name': 'mouse'}, {'supercategory': 'electronic', 'isthing': 1, 'id': 75, 'name': 'remote'}, {'supercategory': 'electronic', 'isthing': 1, 'id': 76, 'name': 'keyboard'}, {'supercategory': 'electronic', 'isthing': 1, 'id': 77, 'name': 'cell phone'}, {'supercategory': 'appliance', 'isthing': 1, 'id': 78, 'name': 'microwave'}, {'supercategory': 'appliance', 'isthing': 1, 'id': 79, 'name': 'oven'}, {'supercategory': 'appliance', 'isthing': 1, 'id': 80, 'name': 'toaster'}, {'supercategory': 'appliance', 'isthing': 1, 'id': 81, 'name': 'sink'}, {'supercategory': 'appliance', 'isthing': 1, 'id': 82, 'name': 'refrigerator'}, {'supercategory': 'indoor', 'isthing': 1, 'id': 84, 'name': 'book'}, {'supercategory': 'indoor', 'isthing': 1, 'id': 85, 'name': 'clock'}, {'supercategory': 'indoor', 'isthing': 1, 'id': 86, 'name': 'vase'}, {'supercategory': 'indoor', 'isthing': 1, 'id': 87, 'name': 'scissors'}, {'supercategory': 'indoor', 'isthing': 1, 'id': 88, 'name': 'teddy bear'}, {'supercategory': 'indoor', 'isthing': 1, 'id': 89, 'name': 'hair drier'}, {'supercategory': 'indoor', 'isthing': 1, 'id': 90, 'name': 'toothbrush'}, {'supercategory': 'textile', 'isthing': 0, 'id': 92, 'name': 'banner'}, {'supercategory': 'textile', 'isthing': 0, 'id': 93, 'name': 'blanket'}, {'supercategory': 'building', 'isthing': 0, 'id': 95, 'name': 'bridge'}, {'supercategory': 'raw-material', 'isthing': 0, 'id': 100, 'name': 'cardboard'}, {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 107, 'name': 'counter'}, {'supercategory': 'textile', 'isthing': 0, 'id': 109, 'name': 'curtain'}, {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 112, 'name': 'door-stuff'}, {'supercategory': 'floor', 'isthing': 0, 'id': 118, 'name': 'floor-wood'}, {'supercategory': 'plant', 'isthing': 0, 'id': 119, 'name': 'flower'}, {'supercategory': 'food-stuff', 'isthing': 0, 'id': 122, 'name': 'fruit'}, {'supercategory': 'ground', 'isthing': 0, 'id': 125, 'name': 'gravel'}, {'supercategory': 'building', 'isthing': 0, 'id': 128, 'name': 'house'}, {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 130, 'name': 'light'}, {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 133, 'name': 'mirror-stuff'}, {'supercategory': 'structural', 'isthing': 0, 'id': 138, 'name': 'net'}, {'supercategory': 'textile', 'isthing': 0, 'id': 141, 'name': 'pillow'}, {'supercategory': 'ground', 'isthing': 0, 'id': 144, 'name': 'platform'}, {'supercategory': 'ground', 'isthing': 0, 'id': 145, 'name': 'playingfield'}, {'supercategory': 'ground', 'isthing': 0, 'id': 147, 'name': 'railroad'}, {'supercategory': 'water', 'isthing': 0, 'id': 148, 'name': 'river'}, {'supercategory': 'ground', 'isthing': 0, 'id': 149, 'name': 'road'}, {'supercategory': 'building', 'isthing': 0, 'id': 151, 'name': 'roof'}, {'supercategory': 'ground', 'isthing': 0, 'id': 154, 'name': 'sand'}, {'supercategory': 'water', 'isthing': 0, 'id': 155, 'name': 'sea'}, {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 156, 'name': 'shelf'}, {'supercategory': 'ground', 'isthing': 0, 'id': 159, 'name': 'snow'}, {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 161, 'name': 'stairs'}, {'supercategory': 'building', 'isthing': 0, 'id': 166, 'name': 'tent'}, {'supercategory': 'textile', 'isthing': 0, 'id': 168, 'name': 'towel'}, {'supercategory': 'wall', 'isthing': 0, 'id': 171, 'name': 'wall-brick'}, {'supercategory': 'wall', 'isthing': 0, 'id': 175, 'name': 'wall-stone'}, {'supercategory': 'wall', 'isthing': 0, 'id': 176, 'name': 'wall-tile'}, {'supercategory': 'wall', 'isthing': 0, 'id': 177, 'name': 'wall-wood'}, {'supercategory': 'water', 'isthing': 0, 'id': 178, 'name': 'water-other'}, {'supercategory': 'window', 'isthing': 0, 'id': 180, 'name': 'window-blind'}, {'supercategory': 'window', 'isthing': 0, 'id': 181, 'name': 'window-other'}, {'supercategory': 'plant', 'isthing': 0, 'id': 184, 'name': 'tree-merged'}, {'supercategory': 'structural', 'isthing': 0, 'id': 185, 'name': 'fence-merged'}, {'supercategory': 'ceiling', 'isthing': 0, 'id': 186, 'name': 'ceiling-merged'}, {'supercategory': 'sky', 'isthing': 0, 'id': 187, 'name': 'sky-other-merged'}, {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 188, 'name': 'cabinet-merged'}, {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 189, 'name': 'table-merged'}, {'supercategory': 'floor', 'isthing': 0, 'id': 190, 'name': 'floor-other-merged'}, {'supercategory': 'ground', 'isthing': 0, 'id': 191, 'name': 'pavement-merged'}, {'supercategory': 'solid', 'isthing': 0, 'id': 192, 'name': 'mountain-merged'}, {'supercategory': 'plant', 'isthing': 0, 'id': 193, 'name': 'grass-merged'}, {'supercategory': 'ground', 'isthing': 0, 'id': 194, 'name': 'dirt-merged'}, {'supercategory': 'raw-material', 'isthing': 0, 'id': 195, 'name': 'paper-merged'}, {'supercategory': 'food-stuff', 'isthing': 0, 'id': 196, 'name': 'food-other-merged'}, {'supercategory': 'building', 'isthing': 0, 'id': 197, 'name': 'building-other-merged'}, {'supercategory': 'solid', 'isthing': 0, 'id': 198, 'name': 'rock-merged'}, {'supercategory': 'wall', 'isthing': 0, 'id': 199, 'name': 'wall-other-merged'}, {'supercategory': 'textile', 'isthing': 0, 'id': 200, 'name': 'rug-merged'}]
    out_annos['licenses']=[{'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/', 'id': 1, 'name': 'Attribution-NonCommercial-ShareAlike License'}, {'url': 'http://creativecommons.org/licenses/by-nc/2.0/', 'id': 2, 'name': 'Attribution-NonCommercial License'}, {'url': 'http://creativecommons.org/licenses/by-nc-nd/2.0/', 'id': 3, 'name': 'Attribution-NonCommercial-NoDerivs License'}, {'url': 'http://creativecommons.org/licenses/by/2.0/', 'id': 4, 'name': 'Attribution License'}, {'url': 'http://creativecommons.org/licenses/by-sa/2.0/', 'id': 5, 'name': 'Attribution-ShareAlike License'}, {'url': 'http://creativecommons.org/licenses/by-nd/2.0/', 'id': 6, 'name': 'Attribution-NoDerivs License'}, {'url': 'http://flickr.com/commons/usage/', 'id': 7, 'name': 'No known copyright restrictions'}, {'url': 'http://www.usa.gov/copyright.shtml', 'id': 8, 'name': 'United States Government Work'}]
    out_annos['info']={'description': 'COCO 2018 Panoptic Dataset', 'url': 'http://cocodataset.org', 'version': '1.0', 'year': 2018, 'contributor': 'https://arxiv.org/abs/1801.00868', 'date_created': '2018-06-01 00:00:00.0'}
    
    with open(args.output,'w') as f:
        json.dump(out_annos,f,indent=4)
