import json
import os
from tqdm import tqdm
from panopticapi.utils import rgb2id,id2rgb
from PIL import Image
import numpy as np
import cv2 


def get_args():
    import argparse
    parser = argparse.ArgumentParser(description='extract bbox from panseg json')
    parser.add_argument('--panseg_json',type=str, help='panoptic info json path')
    parser.add_argument('--panseg_dir',type=str, help='panoptic mask dir')
    parser.add_argument('--out_json',type=str, help='output json path')
    args = parser.parse_args()
    return args


def main():
    out_anno=[]

    args=get_args()

    with open(args.panseg_json,'r') as f:
        panseg_info=json.load(f)

    panseg_dir=args.panseg_dir

    for anno_info in tqdm(panseg_info['annotations']):
        img_id=anno_info['image_id']
        img_name=anno_info['file_name']
        panseg_path=os.path.join(panseg_dir,img_name)
        if not os.path.exists(panseg_path):
            continue
        # read panseg image
        panoptic_orig = np.asarray(Image.open(panseg_path), dtype=np.uint32)
        panoptic_seg_orig = rgb2id(panoptic_orig).astype(np.int32)

        bbox_infos=[]

        for segment_info in anno_info['segments_info']:
            # skip stuff classes
            if not segment_info['isthing']:
                continue
            binary_mask=panoptic_seg_orig==segment_info['id']

            if np.sum(binary_mask)==0:
                continue
            # bbox: x, y, w, h
            bbox=cv2.boundingRect(binary_mask.astype(np.uint8))
            # bbox: left, top, right, bottom
            bbox=[bbox[0],bbox[1],bbox[0]+bbox[2],bbox[1]+bbox[3]]

            bbox_infos.append({
                'bbox':bbox,
                'category_id':segment_info['category_id'],
                'id':segment_info['id'],
                'isthing':segment_info['isthing']
            })

        out_anno.append({
            'image_id':img_id,
            'file_name':img_name,
            'bbox_infos':bbox_infos
        })


    with open(args.output_json,'w') as f:
        json.dump(out_anno,f)


if __name__=='__main__':
    main()


    


        

