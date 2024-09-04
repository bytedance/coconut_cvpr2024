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

### Distributed training
Need to set up your environment variables to run the training script below. 
```bash
export DETECTRON2_DATASETS=YOUR_DATASET_PATH
python3 train_net.py --num-gpus 8 --num-machines $WORKER_NUM \
--machine-rank $WORKER_ID --dist-url tcp://$WORKER_0_HOST:$port \
--config-file configs/coco/panoptic-segmentation/kmax_convnext_large.yaml
```

## Explore COCONut-Large
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


### Explore COCONut-val
1. Similar to COCONut-Large, images need to be downloaded from Ojbects365, we provide a [link](https://drive.google.com/file/d/1-wzLtddJucBVBJ67ailLrfMNmLGFag4i/view?usp=sharing) to download the selected val set images. Link the image to COCO val2017.
2. Download the panoptic masks from [huggingface](https://huggingface.co/datasets/xdeng77/coconut_val). Link the panoptic masks to COCO panoptci_val2017.
3. Then merge the downloaded COCONut-val and relabeled COCO-val jsons using ```merged.py```.

### Model zoo

<table class="tg">
<thead>
  <tr>
    <th class="tg-0lax"></th>
    <th class="tg-0lax"></th>
    <th class="tg-pb0m" colspan="3">coco-val</th>
    <th class="tg-pb0m" colspan="3">relabeled coco-val</th>
    <th class="tg-pb0m" colspan="3">coconut-val</th>
    <th class="tg-baqh" colspan="2">checkpoint</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0lax">backbone</td>
    <td class="tg-0lax">training set</td>
    <td class="tg-baqh">PQ</td>
    <td class="tg-baqh">AP_mask</td>
    <td class="tg-baqh">mIoU</td>
    <td class="tg-baqh">PQ</td>
    <td class="tg-baqh">AP_mask</td>
    <td class="tg-baqh">mIoU</td>
    <td class="tg-baqh">PQ</td>
    <td class="tg-baqh">AP_mask</td>
    <td class="tg-baqh">mIoU</td>
    <td class="tg-0lax">gdrive</td>
    <td class="tg-0lax">huggingface</td>
  </tr>
  <tr>
    <td class="tg-nrix" rowspan="4">ResNet 50</td>
    <td class="tg-0lax">COCO</td>
    <td class="tg-pb0m">53.3</td>
    <td class="tg-pb0m">39.6</td>
    <td class="tg-pb0m">61.7</td>
    <td class="tg-pb0m">55.1</td>
    <td class="tg-pb0m">40.6</td>
    <td class="tg-pb0m">63.9</td>
    <td class="tg-pb0m">53.1</td>
    <td class="tg-pb0m">37.1</td>
    <td class="tg-pb0m">62.5</td>
    <td class="tg-0lax"><a href="https://drive.google.com/file/d/1EyTbKUnFjUOEo57YZMawfl51LUkkLwXa/view?usp=drive_link" target="_blank" rel="noopener noreferrer">download</a></td>
    <td class="tg-0lax"><a href="https://huggingface.co/xdeng77/kmaxdeeplab_panoptic">download</a></td>
  </tr>
  <tr>
    <td class="tg-0lax">COCONut-S</td>
    <td class="tg-baqh">51.7</td>
    <td class="tg-baqh">37.5</td>
    <td class="tg-baqh">59.4</td>
    <td class="tg-baqh">58.9</td>
    <td class="tg-baqh">44.4</td>
    <td class="tg-baqh">64.4</td>
    <td class="tg-baqh">56.7</td>
    <td class="tg-baqh">41.2</td>
    <td class="tg-baqh">63.6</td>
    <td class="tg-0lax"><a href="https://drive.google.com/file/d/1MPZJVIIs-F6AF8bSZo2wJXqlvO1k0Nrj/view?usp=drive_link" target="_blank" rel="noopener noreferrer">download</a></td>
    <td class="tg-0lax"><a href="https://huggingface.co/xdeng77/kmaxdeeplab_panoptic">download</a></td>
  </tr>
  <tr>
    <td class="tg-0lax">COCONut-B</td>
    <td class="tg-baqh">53.4</td>
    <td class="tg-baqh">39.3</td>
    <td class="tg-baqh">62.6</td>
    <td class="tg-baqh">60.2</td>
    <td class="tg-baqh">45.2</td>
    <td class="tg-baqh">65.7</td>
    <td class="tg-baqh">58.1</td>
    <td class="tg-baqh">42.9</td>
    <td class="tg-baqh">64.7</td>
    <td class="tg-0lax"><a href="https://drive.google.com/file/d/1EW07Wg9pMpmlA2G9WT-P5ttgahkrfgJz/view?usp=drive_link" target="_blank" rel="noopener noreferrer">download</a></td>
    <td class="tg-0lax"><a href="https://huggingface.co/xdeng77/kmaxdeeplab_panoptic">download</a></td>
  </tr>
</tbody>
</table>


<table>
<thead>
  <tr>
    <th></th>
    <th></th>
    <th colspan="3">coco-val</th>
    <th colspan="3">relabeled coco-val</th>
    <th colspan="3">coconut-val</th>
    <th colspan="2">checkpoint</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>backbone</td>
    <td>training set</td>
    <td>PQ</td>
    <td>AP_mask</td>
    <td>mIoU</td>
    <td>PQ</td>
    <td>AP_mask</td>
    <td>mIoU</td>
    <td>PQ</td>
    <td>AP_mask</td>
    <td>mIoU</td>
    <td>gdrive</td>
    <td>huggingface</td>
  </tr>
  <tr>
    <td rowspan="4">ConvNeXt-Large</td>
    <td>COCO</td>
    <td>57.9</td>
    <td>45.0</td>
    <td>66.9</td>
    <td>60.4</td>
    <td>46.4</td>
    <td>69.9</td>
    <td>58.3</td>
    <td>44.1</td>
    <td>66.4</td>
    <td><a href="https://drive.google.com/file/d/1JWwQY_VPCVKrmDhROHalYpXwpQoPJUqz/view?usp=drive_link" target="_blank" rel="noopener noreferrer">download</a></td>
    <td><a href="https://huggingface.co/xdeng77/kmaxdeeplab_panoptic">download</a></td>
  </tr>
  <tr>
    <td>COCONut-S</td>
    <td>55.9</td>
    <td>41.9</td>
    <td>66.1</td>
    <td>64.4</td>
    <td>50.8</td>
    <td>71.4</td>
    <td>59.4</td>
    <td>45.7</td>
    <td>67.8</td>
    <td><a href="https://drive.google.com/file/d/14S2QrJqnlbeSK2qMyD3i01eoup4XsVOq/view?usp=drive_link" target="_blank" rel="noopener noreferrer">download</a></td>
    <td><a href="https://huggingface.co/xdeng77/kmaxdeeplab_panoptic">download</a></td>
  </tr>
  <tr>
    <td>COCONut-B</td>
    <td>57.8</td>
    <td>44.8</td>
    <td>66.6</td>
    <td>64.9</td>
    <td>51.2</td>
    <td>71.8</td>
    <td>61.3</td>
    <td>46.5</td>
    <td>69.5</td>
    <td><a href="https://drive.google.com/file/d/12Fdmbyz-0jIDtj6swtJQzsuIk8LWEAw0/view?usp=drive_link" target="_blank" rel="noopener noreferrer">download</a></td>
    <td><a href="https://huggingface.co/xdeng77/kmaxdeeplab_panoptic">download</a></td>
  </tr>
     <tr>
    <td>COCONut-L</td>
    <td>57.9</td>
    <td>45.1</td>
    <td>67.1</td>
    <td>65.0</td>
    <td>51.1</td>
    <td>71.9</td>
    <td>62.4</td>
    <td>47.5</td>
    <td>69.9</td>
    <td><a href="https://drive.google.com/file/d/1RoMrqqeNpJBrujsm8l3LW_eYMMdEuGTv/view?usp=sharing" target="_blank" rel="noopener noreferrer">download</a></td>
    <td><a href="https://huggingface.co/xdeng77/kmaxdeeplab_panoptic">download</a></td>
  </tr>
</tbody>
</table>
