## Installation

### Requirements
- Linux or macOS with Python ≥ 3.8
- CUDA>=11.7, lower CUDA versions may result in not successfully built on detectron2
- `pip install -r requirements.txt`



### Example virtualenv environment setup for kMaX-DeepLab
```bash
pip3 install virtualenv
python3 -m virtualenv kmax_deeplab --python=python3
source kmax_deeplab/bin/activate

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -r requirements.txt
unzip detectron2.zip
cd detectron2
pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git
```

### Example test model checkpoint of kMaX-DeepLab
1. Download the [checkpoint](https://drive.google.com/file/d/14S2QrJqnlbeSK2qMyD3i01eoup4XsVOq/view?usp=drive_link).


2. Dataset preperation and structure for evaluation as below. You need to download ['relabeled_COCO_val'](https://www.kaggle.com/datasets/xueqingdeng/coconut/) and rename it to 'panoptic_val2017', and rename the corresponding json file and download the images from [COCO dataset](http://images.cocodataset.org/zips/val2017.zip).
```
datasets
└── coco
    ├── annotations 
    │   └── panoptic_val2017.json # relabeled_coco_val.json
    ├── panoptic_val2017  # relabeled_coco_val
    ├── val2017 # original COCO dataset val set images
```


3. Use the script below to evaluate the model.

```bash
export DETECTRON2_DATASETS=YOUR_DATA_PATH
python3 train_net.py --num-gpus 8 --dist-url tcp://127.0.0.1:9999 \
--config-file configs/coco/panoptic-segmentation/kmax_convnext_large.yaml \
--eval-only MODEL.WEIGHTS YOUR_MODEL_PATH
```


3. The provided checkpoint should give a PQ of 64.4 on relabeled COCO-val.

### Example demo 

Use the following script to demo images in a directory.
```bash
python3 demo.py --config-file ../configs/coco/panoptic-segmentation/kmax_convnext_large.yaml     \
--input YOUR_IMG_FOLDER_PATH \
--output  OUTPUT_VIS_FOLDER  \
--opts MODEL.WEIGHTS  YOUR_MODEL_PATH
```
