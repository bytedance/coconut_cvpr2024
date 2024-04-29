# [CVPR2024] 🥥COCONut: Crafting the Future of Segmentation Datasets with Exquisite Annotations in the Era of ✨Big Data✨
Xueqing Deng, Qihang Yu, Peng Wang, Xiaohui Shen, Liang-Chieh Chen

[![Dataset](https://img.shields.io/badge/Dataset-Access-<COLOR>)](https://www.kaggle.com/datasets/xueqingdeng/coconut/)
[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://xdeng7.github.io/coconut.github.io/)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2404.08639)
[![Full Paper](https://img.shields.io/badge/Full_Paper-Read-0000FF.svg)](coconut_arxiv.pdf)


## 🚀 Contributions

#### 🔥 1st large-scale human verified dataset for segmentation, more info can be found at our [website](https://xdeng7.github.io/coconut.github.io/).

#### 🔥 COCONut is now available at [Kaggle](https://www.kaggle.com/datasets/xueqingdeng/coconut/), welcome to download!

<p>
<img src="static/vis_masks_video_tasks_v2.gif" alt="teaser" width=90% height=90%>
</p>


## 📢 News!
* 4/28: COCONut is back to [huggingface](https://huggingface.co/collections/xdeng77/coconut-dataset-661da98608dd378c816a4398). [relabeled_coco_val](https://huggingface.co/datasets/xdeng77/relabeled_coco_val) and [coconut_s](https://huggingface.co/datasets/xdeng77/coconut_s) are available. COCONut-B is uploading, should arrive soon.
* 4/25: Tutorial on visualizing COCONut panoptic masks using detectron2. Turn the black mask image into overlayed colorful mask.
* 4/24: Collected FAQs are out, please check them before you leave any issue.
* 4/22: Tutorial on instance segmentation is out! More are coming!
* 4/19: Tutorial on panoptic segmentation is out!
* 4/16: COCONut is available at Kaggle! No need to merge COCONut-B from COCONut-S, update a version of ready-to-use.
* 4/15: COCONut is higlighted by AK's [daily paper](https://huggingface.co/papers?date=2024-04-15)!
* 4/15: Huggingface download links are temporarily closed.

### TODO
- [x] Huggingface dataset preview
- [ ] Convert the annotation to semantic segmentation and object detection.
- [ ] Release COCONut-L and COCONut-val by the end of April

## Dataset Splits
Splits    |  #images | #masks | images | annotations
----------|----------|--------|--------|-------------
COCONut-S | 118K     | 1.54M  | [download](http://images.cocodataset.org/zips/train2017.zip) | [Kaggle](https://www.kaggle.com/datasets/xueqingdeng/coconut)
COCONut-B | 242K     | 2.78M  | [download](http://images.cocodataset.org/zips/unlabeled2017.zip) | [Kaggle](https://www.kaggle.com/datasets/xueqingdeng/coconut)
COCONut-L | 358K     | 4.75M  | [coming]() | [coming]()
relabeled-COCO-val | 5K | 67K | [download](http://images.cocodataset.org/zips/val2017.zip) | [Kaggle](https://www.kaggle.com/datasets/xueqingdeng/coconut)
COCONut-val | 25K     | 437K  | [coming]() | [coming]()



## Get Started



We only provide the annotation, for those who are interested to use our annotation will need to download the images from the links: [COCONut-S images](http://images.cocodataset.org/zips/train2017.zip), [COCONut-B images](http://images.cocodataset.org/zips/unlabeled2017.zip) and [relabeled COCO-val images](http://images.cocodataset.org/zips/val2017.zip).

### 🔗[Kaggle download link](https://www.kaggle.com/datasets/xueqingdeng/coconut/)
  
You can use the web UI to download the dataset directly on Kaggle.

If you find our dataset useful, we really appreciate if you can upvote our dataset on Kaggle, 

### 🔗[Huggingface dataset preview](https://huggingface.co/collections/xdeng77/coconut-dataset-661da98608dd378c816a4398)
  
Directly download the data from huggingface or git clone the huggingface dataset repo will result in invalid data structure.

We recommend you to use our provided download script to download the dataset from huggingface.
```
python download_coconut.py # default split: relabeled_coco_val
```

You can switch to download coconut_s by adding "--split coconut_s" to the command.
```
python download_coconut.py --split coconut_s
```

The data will saved at "./coconut_datasets" by default, you can change it to your preferred path by adding "--output_dir YOUR_DATA_PATH".

## Tutorials
 * [visualization on COCONut panoptic masks](tutorials/visualization/demo.ipynb)

 * [panoptic segmentation](tutorials/kmaxdeeplab_panoptic/README.md)

 * [instance segmentation](tutorials/kmaxdeeplab_instance/README.md)



## FAQ

We summarize the common issues in [FAQ.md](FAQ.md), please check this out before you create any new issues.


## Bibtex  
If you find our dataset useful, please cite:
```
@inproceedings{coconut2024cvpr,
  author    = {Xueqing Deng, Qihang Yu, Peng Wang, Xiaohui Shen, Liang-Chieh Chen},
  title     = {COCONut: Modernizing COCO Segmentation},
  booktitle   = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2024},
```

## More visualization on COCONut annotation
<p>
<img src="static/vis_simple_mask.png" alt="vis1" width=90% height=90%>
</p>
<p>
<img src="static/vis_dense_mask.png" alt="vis2" width=90% height=90%>
</p>

## Terms of use
We follow the same license as [COCO dataset for images](https://cocodataset.org/#termsofuse). For COCONut's annotations, non-commercial use are allowed. 

## Acknowledgement
* [kMax-DeepLab](https://github.com/bytedance/kmax-deeplab.git)
* [FC-CLIP](https://github.com/bytedance/fc-clip.git)
* [detectron2](https://github.com/facebookresearch/detectron2.git)

