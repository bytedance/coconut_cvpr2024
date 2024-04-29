import os
import json
from datasets import load_dataset, Image
import argparse
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser(description='download coconut from huggingface')
    parser.add_argument('--split', default='relabeled_coco_val', choices=['relabeled_coco_val','coconut_s','coconut_b'], help='coconut dataset split')
    parser.add_argument('--output_dir', default='coconut_datasets', help='output directory')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    split = args.split
    output_dir = args.output_dir
    dataset_name=f"xdeng77/{split}"
    dataset = load_dataset(dataset_name)

    # create output folder
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # create output json file path
    output_json_file=f"{output_dir}/{split}.json"
    output_annos={}

    # create output mask folder
    output_mask_dir=f"{output_dir}/{split}"
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)


    # collect items from huggingface dataset: annotations (segments_info) and image infos
    output_annotations=[]
    output_img_infos=[]

    print("Saving dataset to local path......")
    for item in tqdm(dataset["train"]):

        anno_info=item["segments_info"]
        img_id=anno_info['file_name'].split('.')[0]

        # save PIL image object to disk
        mask_path= f"{output_mask_dir}/{img_id}.png"
        mask=item['mask'].save(mask_path)

        # save anno info to output_annotations
        output_annotations.append(anno_info)

        # save image info to output_img_infos
        output_img_infos.append(item['image_info'])

    output_json={}
    output_json['images']=output_img_infos
    output_json['annotations']=output_annotations


    output_json['categories']=[{'supercategory': 'person', 'isthing': 1, 'id': 1, 'name': 'person'}, {'supercategory': 'vehicle', 'isthing': 1, 'id': 2, 'name': 'bicycle'}, {'supercategory': 'vehicle', 'isthing': 1, 'id': 3, 'name': 'car'}, {'supercategory': 'vehicle', 'isthing': 1, 'id': 4, 'name': 'motorcycle'}, {'supercategory': 'vehicle', 'isthing': 1, 'id': 5, 'name': 'airplane'}, {'supercategory': 'vehicle', 'isthing': 1, 'id': 6, 'name': 'bus'}, {'supercategory': 'vehicle', 'isthing': 1, 'id': 7, 'name': 'train'}, {'supercategory': 'vehicle', 'isthing': 1, 'id': 8, 'name': 'truck'}, {'supercategory': 'vehicle', 'isthing': 1, 'id': 9, 'name': 'boat'}, {'supercategory': 'outdoor', 'isthing': 1, 'id': 10, 'name': 'traffic light'}, {'supercategory': 'outdoor', 'isthing': 1, 'id': 11, 'name': 'fire hydrant'}, {'supercategory': 'outdoor', 'isthing': 1, 'id': 13, 'name': 'stop sign'}, {'supercategory': 'outdoor', 'isthing': 1, 'id': 14, 'name': 'parking meter'}, {'supercategory': 'outdoor', 'isthing': 1, 'id': 15, 'name': 'bench'}, {'supercategory': 'animal', 'isthing': 1, 'id': 16, 'name': 'bird'}, {'supercategory': 'animal', 'isthing': 1, 'id': 17, 'name': 'cat'}, {'supercategory': 'animal', 'isthing': 1, 'id': 18, 'name': 'dog'}, {'supercategory': 'animal', 'isthing': 1, 'id': 19, 'name': 'horse'}, {'supercategory': 'animal', 'isthing': 1, 'id': 20, 'name': 'sheep'}, {'supercategory': 'animal', 'isthing': 1, 'id': 21, 'name': 'cow'}, {'supercategory': 'animal', 'isthing': 1, 'id': 22, 'name': 'elephant'}, {'supercategory': 'animal', 'isthing': 1, 'id': 23, 'name': 'bear'}, {'supercategory': 'animal', 'isthing': 1, 'id': 24, 'name': 'zebra'}, {'supercategory': 'animal', 'isthing': 1, 'id': 25, 'name': 'giraffe'}, {'supercategory': 'accessory', 'isthing': 1, 'id': 27, 'name': 'backpack'}, {'supercategory': 'accessory', 'isthing': 1, 'id': 28, 'name': 'umbrella'}, {'supercategory': 'accessory', 'isthing': 1, 'id': 31, 'name': 'handbag'}, {'supercategory': 'accessory', 'isthing': 1, 'id': 32, 'name': 'tie'}, {'supercategory': 'accessory', 'isthing': 1, 'id': 33, 'name': 'suitcase'}, {'supercategory': 'sports', 'isthing': 1, 'id': 34, 'name': 'frisbee'}, {'supercategory': 'sports', 'isthing': 1, 'id': 35, 'name': 'skis'}, {'supercategory': 'sports', 'isthing': 1, 'id': 36, 'name': 'snowboard'}, {'supercategory': 'sports', 'isthing': 1, 'id': 37, 'name': 'sports ball'}, {'supercategory': 'sports', 'isthing': 1, 'id': 38, 'name': 'kite'}, {'supercategory': 'sports', 'isthing': 1, 'id': 39, 'name': 'baseball bat'}, {'supercategory': 'sports', 'isthing': 1, 'id': 40, 'name': 'baseball glove'}, {'supercategory': 'sports', 'isthing': 1, 'id': 41, 'name': 'skateboard'}, {'supercategory': 'sports', 'isthing': 1, 'id': 42, 'name': 'surfboard'}, {'supercategory': 'sports', 'isthing': 1, 'id': 43, 'name': 'tennis racket'}, {'supercategory': 'kitchen', 'isthing': 1, 'id': 44, 'name': 'bottle'}, {'supercategory': 'kitchen', 'isthing': 1, 'id': 46, 'name': 'wine glass'}, {'supercategory': 'kitchen', 'isthing': 1, 'id': 47, 'name': 'cup'}, {'supercategory': 'kitchen', 'isthing': 1, 'id': 48, 'name': 'fork'}, {'supercategory': 'kitchen', 'isthing': 1, 'id': 49, 'name': 'knife'}, {'supercategory': 'kitchen', 'isthing': 1, 'id': 50, 'name': 'spoon'}, {'supercategory': 'kitchen', 'isthing': 1, 'id': 51, 'name': 'bowl'}, {'supercategory': 'food', 'isthing': 1, 'id': 52, 'name': 'banana'}, {'supercategory': 'food', 'isthing': 1, 'id': 53, 'name': 'apple'}, {'supercategory': 'food', 'isthing': 1, 'id': 54, 'name': 'sandwich'}, {'supercategory': 'food', 'isthing': 1, 'id': 55, 'name': 'orange'}, {'supercategory': 'food', 'isthing': 1, 'id': 56, 'name': 'broccoli'}, {'supercategory': 'food', 'isthing': 1, 'id': 57, 'name': 'carrot'}, {'supercategory': 'food', 'isthing': 1, 'id': 58, 'name': 'hot dog'}, {'supercategory': 'food', 'isthing': 1, 'id': 59, 'name': 'pizza'}, {'supercategory': 'food', 'isthing': 1, 'id': 60, 'name': 'donut'}, {'supercategory': 'food', 'isthing': 1, 'id': 61, 'name': 'cake'}, {'supercategory': 'furniture', 'isthing': 1, 'id': 62, 'name': 'chair'}, {'supercategory': 'furniture', 'isthing': 1, 'id': 63, 'name': 'couch'}, {'supercategory': 'furniture', 'isthing': 1, 'id': 64, 'name': 'potted plant'}, {'supercategory': 'furniture', 'isthing': 1, 'id': 65, 'name': 'bed'}, {'supercategory': 'furniture', 'isthing': 1, 'id': 67, 'name': 'dining table'}, {'supercategory': 'furniture', 'isthing': 1, 'id': 70, 'name': 'toilet'}, {'supercategory': 'electronic', 'isthing': 1, 'id': 72, 'name': 'tv'}, {'supercategory': 'electronic', 'isthing': 1, 'id': 73, 'name': 'laptop'}, {'supercategory': 'electronic', 'isthing': 1, 'id': 74, 'name': 'mouse'}, {'supercategory': 'electronic', 'isthing': 1, 'id': 75, 'name': 'remote'}, {'supercategory': 'electronic', 'isthing': 1, 'id': 76, 'name': 'keyboard'}, {'supercategory': 'electronic', 'isthing': 1, 'id': 77, 'name': 'cell phone'}, {'supercategory': 'appliance', 'isthing': 1, 'id': 78, 'name': 'microwave'}, {'supercategory': 'appliance', 'isthing': 1, 'id': 79, 'name': 'oven'}, {'supercategory': 'appliance', 'isthing': 1, 'id': 80, 'name': 'toaster'}, {'supercategory': 'appliance', 'isthing': 1, 'id': 81, 'name': 'sink'}, {'supercategory': 'appliance', 'isthing': 1, 'id': 82, 'name': 'refrigerator'}, {'supercategory': 'indoor', 'isthing': 1, 'id': 84, 'name': 'book'}, {'supercategory': 'indoor', 'isthing': 1, 'id': 85, 'name': 'clock'}, {'supercategory': 'indoor', 'isthing': 1, 'id': 86, 'name': 'vase'}, {'supercategory': 'indoor', 'isthing': 1, 'id': 87, 'name': 'scissors'}, {'supercategory': 'indoor', 'isthing': 1, 'id': 88, 'name': 'teddy bear'}, {'supercategory': 'indoor', 'isthing': 1, 'id': 89, 'name': 'hair drier'}, {'supercategory': 'indoor', 'isthing': 1, 'id': 90, 'name': 'toothbrush'}, {'supercategory': 'textile', 'isthing': 0, 'id': 92, 'name': 'banner'}, {'supercategory': 'textile', 'isthing': 0, 'id': 93, 'name': 'blanket'}, {'supercategory': 'building', 'isthing': 0, 'id': 95, 'name': 'bridge'}, {'supercategory': 'raw-material', 'isthing': 0, 'id': 100, 'name': 'cardboard'}, {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 107, 'name': 'counter'}, {'supercategory': 'textile', 'isthing': 0, 'id': 109, 'name': 'curtain'}, {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 112, 'name': 'door-stuff'}, {'supercategory': 'floor', 'isthing': 0, 'id': 118, 'name': 'floor-wood'}, {'supercategory': 'plant', 'isthing': 0, 'id': 119, 'name': 'flower'}, {'supercategory': 'food-stuff', 'isthing': 0, 'id': 122, 'name': 'fruit'}, {'supercategory': 'ground', 'isthing': 0, 'id': 125, 'name': 'gravel'}, {'supercategory': 'building', 'isthing': 0, 'id': 128, 'name': 'house'}, {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 130, 'name': 'light'}, {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 133, 'name': 'mirror-stuff'}, {'supercategory': 'structural', 'isthing': 0, 'id': 138, 'name': 'net'}, {'supercategory': 'textile', 'isthing': 0, 'id': 141, 'name': 'pillow'}, {'supercategory': 'ground', 'isthing': 0, 'id': 144, 'name': 'platform'}, {'supercategory': 'ground', 'isthing': 0, 'id': 145, 'name': 'playingfield'}, {'supercategory': 'ground', 'isthing': 0, 'id': 147, 'name': 'railroad'}, {'supercategory': 'water', 'isthing': 0, 'id': 148, 'name': 'river'}, {'supercategory': 'ground', 'isthing': 0, 'id': 149, 'name': 'road'}, {'supercategory': 'building', 'isthing': 0, 'id': 151, 'name': 'roof'}, {'supercategory': 'ground', 'isthing': 0, 'id': 154, 'name': 'sand'}, {'supercategory': 'water', 'isthing': 0, 'id': 155, 'name': 'sea'}, {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 156, 'name': 'shelf'}, {'supercategory': 'ground', 'isthing': 0, 'id': 159, 'name': 'snow'}, {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 161, 'name': 'stairs'}, {'supercategory': 'building', 'isthing': 0, 'id': 166, 'name': 'tent'}, {'supercategory': 'textile', 'isthing': 0, 'id': 168, 'name': 'towel'}, {'supercategory': 'wall', 'isthing': 0, 'id': 171, 'name': 'wall-brick'}, {'supercategory': 'wall', 'isthing': 0, 'id': 175, 'name': 'wall-stone'}, {'supercategory': 'wall', 'isthing': 0, 'id': 176, 'name': 'wall-tile'}, {'supercategory': 'wall', 'isthing': 0, 'id': 177, 'name': 'wall-wood'}, {'supercategory': 'water', 'isthing': 0, 'id': 178, 'name': 'water-other'}, {'supercategory': 'window', 'isthing': 0, 'id': 180, 'name': 'window-blind'}, {'supercategory': 'window', 'isthing': 0, 'id': 181, 'name': 'window-other'}, {'supercategory': 'plant', 'isthing': 0, 'id': 184, 'name': 'tree-merged'}, {'supercategory': 'structural', 'isthing': 0, 'id': 185, 'name': 'fence-merged'}, {'supercategory': 'ceiling', 'isthing': 0, 'id': 186, 'name': 'ceiling-merged'}, {'supercategory': 'sky', 'isthing': 0, 'id': 187, 'name': 'sky-other-merged'}, {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 188, 'name': 'cabinet-merged'}, {'supercategory': 'furniture-stuff', 'isthing': 0, 'id': 189, 'name': 'table-merged'}, {'supercategory': 'floor', 'isthing': 0, 'id': 190, 'name': 'floor-other-merged'}, {'supercategory': 'ground', 'isthing': 0, 'id': 191, 'name': 'pavement-merged'}, {'supercategory': 'solid', 'isthing': 0, 'id': 192, 'name': 'mountain-merged'}, {'supercategory': 'plant', 'isthing': 0, 'id': 193, 'name': 'grass-merged'}, {'supercategory': 'ground', 'isthing': 0, 'id': 194, 'name': 'dirt-merged'}, {'supercategory': 'raw-material', 'isthing': 0, 'id': 195, 'name': 'paper-merged'}, {'supercategory': 'food-stuff', 'isthing': 0, 'id': 196, 'name': 'food-other-merged'}, {'supercategory': 'building', 'isthing': 0, 'id': 197, 'name': 'building-other-merged'}, {'supercategory': 'solid', 'isthing': 0, 'id': 198, 'name': 'rock-merged'}, {'supercategory': 'wall', 'isthing': 0, 'id': 199, 'name': 'wall-other-merged'}, {'supercategory': 'textile', 'isthing': 0, 'id': 200, 'name': 'rug-merged'}]
    output_json['licenses']=[{'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/', 'id': 1, 'name': 'Attribution-NonCommercial-ShareAlike License'}, {'url': 'http://creativecommons.org/licenses/by-nc/2.0/', 'id': 2, 'name': 'Attribution-NonCommercial License'}, {'url': 'http://creativecommons.org/licenses/by-nc-nd/2.0/', 'id': 3, 'name': 'Attribution-NonCommercial-NoDerivs License'}, {'url': 'http://creativecommons.org/licenses/by/2.0/', 'id': 4, 'name': 'Attribution License'}, {'url': 'http://creativecommons.org/licenses/by-sa/2.0/', 'id': 5, 'name': 'Attribution-ShareAlike License'}, {'url': 'http://creativecommons.org/licenses/by-nd/2.0/', 'id': 6, 'name': 'Attribution-NoDerivs License'}, {'url': 'http://flickr.com/commons/usage/', 'id': 7, 'name': 'No known copyright restrictions'}, {'url': 'http://www.usa.gov/copyright.shtml', 'id': 8, 'name': 'United States Government Work'}]
    output_json['info']={'description': 'COCO 2018 Panoptic Dataset', 'url': 'http://cocodataset.org', 'version': '1.0', 'year': 2018, 'contributor': 'https://arxiv.org/abs/1801.00868', 'date_created': '2018-06-01 00:00:00.0'}

    with open(output_json_file, 'w') as f:
        json.dump(output_json, f, indent=4)

    print(f"Downloaded {args.split} successfully!")
    print(f"Data saved at {args.output_dir}.")


if __name__ == '__main__':
    main()



