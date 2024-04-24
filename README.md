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
* 4/22: Tutorials on panoptic and instance segmentation are out! More are coming!
* 4/16: COCONut is available at Kaggle! No need to merge COCONut-B from COCONut-S, update a version of ready-to-use.
* 4/15: COCONut is higlighted by AK's [daily paper](https://huggingface.co/papers?date=2024-04-15)!
* 4/15: Huggingface download links are temporarily closed.

### TODO
- [ ] Huggingface dataset preview
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


Tutorials on [panoptic segmentation](kmaxdeeplab_panoptic/README.md), [instance segmentation](kmaxdeeplab_instance/README.md) with kMaX-DeepLab.



## FAQ

We summarize the common issues in [FAQ.md](FAQ.md), please check this out before you create any new issues.


#### Bibtex  
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

