# Train 
## COCONut-S and B
COCONut-S consists of COCO [train2017](http://images.cocodataset.org/zips/train2017.zip), and B consists of both COCO train and [unlabeled](http://images.cocodataset.org/zips/unlabeled2017.zip) set.
You should download the images from [COCO](https://cocodataset.org/#download) and our COCONut annotations from Kaggle or Huggingface.
We follow detectron2 definition of the dataset. The dataset should be organized as follow:
```
datasets
└── coco
    ├── annotations 
    │   └── panoptic_train2017.json # coconut-b.json / coconut-s.json
    ├── panoptic_train2017  # coconut-b / coconut-s
    ├── train2017 # original COCO dataset train and unlabeled set images / original COCO train set.
```

It is noted that the folder names are fixed regardless using COCONut-S or COCONut-B. You can modify it in the detectron2 definition, but so far we don't support it.

## COCONut-L
COCONut-Large consists of three subsets from COCO train2017, COCO unlabeled set and subsets from Objects365. To use the COCONut-Large panoptic masks, you should follow the steps below:
1. Download the panoptic masks and annotation json file from [huggingface](https://huggingface.co/datasets/xdeng77/coconut_large/tree/main)
2. Download the images from [Objects365](https://data.baai.ac.cn/details/Objects365_2020). The images are organized using patches, please download the corresponding raw patches: patch25,patch32,patch35,patch40 from the official website.
3. Follow the instruction to set up COCONut-B, which is used to build COCONut-L. The folder organization should be as follow:
 ```
datasets
└── coco
    ├── annotations 
    │   └── panoptic_train2017.json # coconut-b.json
    ├── panoptic_train2017  # coconut-b
    ├── train2017 # original COCO dataset train and unlabeled set images
```
4. Link the Objects365 images and panoptic masks to the coco/train_2017 and coco/panoptic_train2017 respectively using the dataset path of COCONut-B.
```
objects365/images ----> coco/train2017
object365/panoptic_masks ----> coco/panoptic_train2017
```
5. Merge the object365 json files to COCONut-B json files using the 'merged.py' script. Then it is ready to be used.


# Eval
### relabeled COCO-val
Our relabeled COCO-val is similar to be used in COCO val but only replacing our annotations. Similarly, you can download it from Kaggle or Huggingface.
The dataset should be organized as follow:
```
datasets
└── coco
    ├── annotations 
    │   └── panoptic_val2017.json # relabeled_coco_val.json
    ├── panoptic_val2017  # relabeled coco val
    ├── val2017 # original COCO val set
```
## COCONut-val
1. Similar to COCONut-Large, images need to be downloaded from Ojbects365, we provide a [link](https://drive.google.com/file/d/1-wzLtddJucBVBJ67ailLrfMNmLGFag4i/view?usp=sharing) to download the selected val set images. Link the image to COCO val2017.
2. Download the panoptic masks from [huggingface](https://huggingface.co/datasets/xdeng77/coconut_val). Link the panoptic masks to COCO panoptci_val2017.
3. Then merge the downloaded COCONut-val and relabeled COCO-val jsons using ```merged.py```.
