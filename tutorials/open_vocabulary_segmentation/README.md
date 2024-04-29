## Installation

### Requirements
- Linux or macOS with Python ≥ 3.8
- CUDA>=11.7, lower CUDA versions may result in not successfully built on detectron2
- Mask2Former

This document provides a brief intro of the usage of FC-CLIP.

Please see [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md) for full usage.

### Example virtualenv environment setup for kMaX-DeepLab
```bash
pip3 install virtualenv
python3 -m virtualenv fc-clip --python=python3
source fc-clip/bin/activate

# recommened pytorch version, others may not work
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# get fc-clip repo and set up environment
git clone https://github.com/bytedance/fc-clip.git
pip install -r requirements.txt

# it is recommend to use our provided local detectron2.zip to set up detectron2
unzip detectron2.zip
cd detectron2
pip install -e .

# panotic api
pip install git+https://github.com/cocodataset/panopticapi.git

# install the multi-scale deformable conv
cd fcclip/modeling/pixel_decoder/ops
pip install -3 . # it is recommended to use pip install instead of sh make.sh which does not work any more.

```

### Inference Demo with Pre-trained Models

```
cd demo/
python demo.py \
  --input YOUR_IMG_1.jpg YOUR_IMG_2.jpg \
  [--other-options]
  --opts MODEL.WEIGHTS YOUR_MODEL_PATH
```
The configs are made for training, therefore we need to specify `MODEL.WEIGHTS` to a model from model zoo for evaluation.
This command will run the inference and show visualizations in an OpenCV window.

More details refer to the official repo of [fc-clip](https://github.com/bytedance/fc-clip.git).


### Training & Evaluation in Command Line

To train a model with "train_net.py", first
setup the corresponding datasets following
[datasets/README.md](https://github.com/bytedance/fc-clip/blob/main/datasets/README.md),
then run:
```
python train_net.py --num-gpus 8 \
  --config-file configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_ade20k.yaml
```


To evaluate a model's performance, use
```
python train_net.py \
  --config-file configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_ade20k.yaml \
  --eval-only MODEL.WEIGHTS /path/to/checkpoint_file
```
