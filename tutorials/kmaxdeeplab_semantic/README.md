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
```

### Example test model checkpoint of kMaX-DeepLab
1. Download the [checkpoint](https://drive.google.com/file/d/1TCXCVh3dBUuF_7htUND72TWXMe4sP1UZ/view?usp=sharing).


2. Dataset preperation and structure for evaluation as below. You need to download ['relabeled_COCO_val'](https://www.kaggle.com/datasets/xueqingdeng/coconut/) and  json file and download the images from [COCO dataset](http://images.cocodataset.org/zips/val2017.zip).


3. Convert the downloaded panoptic segmentations to instances by using the following script.
```
pip install git+https://github.com/cocodataset/panopticapi.git
python prepare_coco_panoptic_semseg.py
```
This script will automatically detect your DETECTRON2_DATASETS path, the default is "./datasets/coco".

If you want to change your folder path, please export your dataset path using the script as below.
```
export DETECTRON2_DATASETS=YOUR_DATA_PATH
```

4. Prepare the dataset structure.
```
datasets
└── coco
    ├── panoptic_semseg_val2017  # converted semantic segmentation masks
    ├── val2017 # original COCO dataset val set images
```


5. Use the script below to evaluate the model.

```bash
export DETECTRON2_DATASETS=YOUR_DATA_PATH
python3 train_net.py --num-gpus 8 --dist-url tcp://127.0.0.1:9999 \
--config-file configs/coco/semantic_segmentation/kmax_convnext_large.yaml \
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

### Model zoo
<table>
<thead>
  <tr>
    <th></th>
    <th></th>
    <th>COCO-val</th>
    <th>relabeled COCO-val</th>
    <th>COCONut-val</th>
    <th></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>backbone</td>
    <td>training set</td>
    <td>mIoU</td>
    <td>mIoU</td>
    <td>mIoU</td>
    <td>model</td>
  </tr>
  <tr>
    <td rowspan="3">Swin-L</td>
    <td>COCO</td>
    <td>67.1</td>
    <td>70.9</td>
    <td>68.1</td>
    <td><a href="https://drive.google.com/file/d/1F9N8_B_nb3TkqpzCy6TMfaepXpLVpaqo/view?usp=drive_link" target="_blank" rel="noopener noreferrer">download</a></td>
  </tr>
  <tr>
    <td>COCONut-S</td>
    <td>66.1</td>
    <td>71.9</td>
    <td>69.9</td>
    <td><a href="https://drive.google.com/file/d/1G25n1fkRv2tPb5O_YbHqxHNf2ICxwV_n/view?usp=drive_link" target="_blank" rel="noopener noreferrer">download</a></td>
  </tr>
  <tr>
    <td>COCONut-B</td>
    <td>67.4</td>
    <td>72.4</td>
    <td>71.3</td>
    <td><a href="https://drive.google.com/file/d/1DhxcFOldzmx64qc_zFvdXf3lyC5bQtfX/view?usp=drive_link" target="_blank" rel="noopener noreferrer">download</a></td>
  </tr>
</tbody>
</table>
