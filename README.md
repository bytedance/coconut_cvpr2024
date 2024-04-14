# [CVPR2024] 🥥COCONut: Crafting the Future of Segmentation Datasets with Exquisite Annotations in the Era of ✨Big Data✨
Xueqing Deng, Qihang Yu, Peng Wang, Xiaohui Shen, Liang-Chieh Chen
<p>
<img src="static/vis_masks_video_tasks_v2.gif" alt="teaser" width=90% height=90%>
</p>

### TODO
- [ ] Huggingface dataset preview
- [ ] Release code to merge dataset split
- [ ] Convert the annotation to instance/semantic segmentation and object detection.
- [ ] Release COCONut-L and COCONut-val by the end of April



## Dataset download

[COCONut-S](https://drive.google.com/file/d/1HJ06YvA-id5ot1owQgWXPcy3Iba0eWl3/view?usp=drive_link), [COCONut-B](https://drive.google.com/file/d/1SfV2lJStzJeyxbxeKfT7b_teiqt3UxXv/view?usp=drive_link), [relabeled-COCO-val](https://drive.google.com/file/d/14FvDOMMYiaMF_PQXBJLz83cywzYwj4mN/view?usp=drive_link) and [panoptic segments informations](https://drive.google.com/drive/folders/1ced4zcGF0G-nZmjJ8CPWhAz643ne-OHZ?usp=drive_link)



## Dataset usage
* Download the images from [COCO dataset](https://cocodataset.org/#download)
* Download our panoptic masks from [COCONut dataset](https://drive.google.com/drive/folders/1313Uf2LyRKu2czHzmOfDNJepObCUwqRY?usp=drive_link) and the corresponding [panoptic info](https://drive.google.com/drive/folders/1ced4zcGF0G-nZmjJ8CPWhAz643ne-OHZ?usp=drive_link).

  
### Example dataset download scripts to build COCONut-S panoptic train dataset:

1. Download the [panoptic masks](https://drive.google.com/file/d/1HJ06YvA-id5ot1owQgWXPcy3Iba0eWl3/view?usp=drive_link) and [panoptic sements info](https://drive.google.com/file/d/1fTHBZIvz005UBYrkr97yipzUrjeZyh1I/view?usp=drive_link) from google drive.
2. Download the train set images from [COCO dataset](http://images.cocodataset.org/zips/train2017.zip).

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

