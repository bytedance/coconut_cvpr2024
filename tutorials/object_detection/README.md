## Installation

We introduce the environment set up for evaluation and training for object detection on COCONut. More details refer to [DETA repo](https://github.com/jozhang97/DETA/tree/master?tab=readme-ov-file).

### Requirements
- Linux or macOS with Python ≥ 3.7
- `pip install -r requirements.txt`


### Example virtualenv environment setup for kMaX-DeepLab
tested on `torch==1.13.0+cu11.7`
```bash
pip3 install virtualenv
python3 -m virtualenv deta --python=python3
source deta/bin/activate

pip install torch==1.13.1 torchvision==0.14.1 
git clone https://github.com/jozhang97/DETA.git
pip install -r requirements.txt

# Compiling CUDA operators
cd ./models/ops
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

### Example test model checkpoint of DETA 
1. Prepare dataset, the dataset structure should be as follow:
```
datasets
└── coco
    ├── annotations 
    │   └── instances_val2017.json # relabeled_coco_val.json
    ├── val2017 # original COCO dataset val set images
```

2. Use the script below to evaluate the model.
```bash
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/deta_swin_ft.sh \
--eval --coco_path YOUR_COCO_DATASET_PATH \ # for example, ./datasets/coco
--resume YOUR_MODEL_CHECKPOINT
```

### Distributed training
Need to set up your environment variables to run the training script below. 
```bash
MASTER_ADDR=$WORKER_0_HOST_IP NODE_RANK=$NODE_ID GPUS_PER_NODE=8 \
./tools/run_dist_launch.sh 16 ./configs/deta_swin_ft.sh \
--coco_path YOUR_COCO_DATASET_PATH
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
    <td>AP_box</td>
    <td>AP_box</td>
    <td>AP_box</td>
    <td>model</td>
  </tr>
  <tr>
    <td rowspan="3">Swin-L</td>
    <td>COCO</td>
    <td>26.8</td>
    <td>16.8</td>
    <td>34.1</td>
    <td><a href="https://drive.google.com/file/d/1WMlOPexz1RwdjO2vpRgmR8Trq01kk7hm/view?usp=drive_link" target="_blank" rel="noopener noreferrer">download</a></td>
  </tr>
  <tr>
    <td>COCONut-S</td>
    <td>27.3</td>
    <td>17.3</td>
    <td>33.8</td>
    <td><a href="https://drive.google.com/file/d/1aRPJYH0sAMeVwxOOmNInnOzAq9hnAtf2/view?usp=drive_link" target="_blank" rel="noopener noreferrer">download</a></td>
  </tr>
  <tr>
    <td>COCONut-B</td>
    <td>27.4</td>
    <td>17.4</td>
    <td>33.7</td>
    <td><a href="https://drive.google.com/file/d/1g5l4IUJy-19rJZoJbSK5EVAShYHEhEJm/view?usp=drive_link" target="_blank" rel="noopener noreferrer">download</a></td>
  </tr>
</tbody>
</table>