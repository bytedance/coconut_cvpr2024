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

