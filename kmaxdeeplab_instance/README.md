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
pip install pycocotools
```

### Example test model checkpoint of kMaX-DeepLab
1. Download the [checkpoint]().


2. Dataset preperation and structure for evaluation as below. You need to download ['relabeled_COCO_val'](https://www.kaggle.com/datasets/xueqingdeng/coconut/) and  json file and download the images from [COCO dataset](http://images.cocodataset.org/zips/val2017.zip).


3. Convert the downloaded panoptic segmentations to instances by using the following script.
```
python create_instance_by_panseg.py --mask-dir  YOUR_PANSEG_MASK_DIR
--image-dir YOUR_COCO_IMGAGE_DIR
--panseg-info YOUR_PANSEG_INFO_JSON_FILE
--output YOUR_OUTPUT_PATH
```

4. Prepare the dataset structure.
```
datasets
└── coco
    ├── annotations 
    │   └── instances_val2017.json # relabeled_coco_val.json
    ├── val2017 # original COCO dataset val set images
```


5. Use the script below to evaluate the model.

```bash
export DETECTRON2_DATASETS=YOUR_DATA_PATH
python3 train_net.py --num-gpus 8 --dist-url tcp://127.0.0.1:9999 \
--config-file configs/coco/instance_segmentation/kmax_convnext_large.yaml \
--eval-only MODEL.WEIGHTS YOUR_MODEL_PATH
```


### Distributed training
Need to set up your environment variables to run the training script below. 
```bash
export DETECTRON2_DATASETS=YOUR_DATASET_PATH
python3 train_net.py --num-gpus 8 --num-machines $WORKER_NUM \
--machine-rank $WORKER_ID --dist-url tcp://$WORKER_0_HOST:$port \
--config-file configs/coco/instance_segmentation/kmax_convnext_large.yaml 
```
