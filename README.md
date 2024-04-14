# [CVPR2024] 🥥COCONut: Crafting the Future of Segmentation Datasets with Exquisite Annotations in the Era of ✨Big Data✨
Xueqing Deng, Qihang Yu, Peng Wang, Xiaohui Shen, Liang-Chieh Chen

#### 🔥 1st large-scale human verified dataset for segmentation, more info can be found at our [website](https://xdeng7.github.io/coconut.github.io/).

#### 🔥 Download and play with our dataset at [huggingface](https://huggingface.co/datasets/xdeng77/coconut_cvpr2024/tree/main)!

<p>
<img src="static/vis_masks_video_tasks_v2.gif" alt="teaser" width=90% height=90%>
</p>

### TODO
- [ ] Huggingface dataset preview
- [ ] Release code to merge dataset split
- [ ] Convert the annotation to instance/semantic segmentation and object detection.
- [ ] Release COCONut-L and COCONut-val by the end of April



## Dataset usage

[COCONut-S](https://huggingface.co/datasets/xdeng77/coconut_cvpr2024/tree/main), [COCONut-B](https://huggingface.co/datasets/xdeng77/coconut_cvpr2024/tree/main), [relabeled-COCO-val](https://huggingface.co/datasets/xdeng77/coconut_cvpr2024/tree/main) and [annotation informations](https://huggingface.co/datasets/xdeng77/coconut_cvpr2024/tree/main)

We only provide the annotation in our huggingface, for those who are interested to use our annotation will need to download the images from the links: [COCONut-S images](http://images.cocodataset.org/zips/train2017.zip), [COCONut-B images](http://images.cocodataset.org/zips/unlabeled2017.zip) and [relabeled COCO-val images](http://images.cocodataset.org/zips/val2017.zip).

  
### Example dataset download scripts to build COCONut-S panoptic train dataset:

1. Download the [panoptic masks](https://huggingface.co/datasets/xdeng77/coconut_cvpr2024/tree/main) and [panoptic segments info](https://huggingface.co/datasets/xdeng77/coconut_cvpr2024/tree/main) from huggingface.
2. Download the train set images from [COCO dataset](http://images.cocodataset.org/zips/train2017.zip).
3. To use the dataset loader from detectron2, you will need to rename the mask folder to 'panoptic_train2017' and annotation file to 'panoptic_train2017.json'.

The dataset structure should be as follow:


```
COCONut-S
  train2017
  panoptic_train2017
  annotations
    panoptic_train2017.json
```

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

## Acknowledgement
* [kMax-DeepLab](https://github.com/bytedance/kmax-deeplab.git)
* [FC-CLIP](https://github.com/bytedance/fc-clip.git)
* [detectron2](https://github.com/facebookresearch/detectron2.git)

