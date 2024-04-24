# FAQ

________________________________________________________________________________

**Q1: Why the downloaded images are all black?**

A: There are colors in the image but hard to be noticed. The reason is that we did not 100% align the colors in panopticapi. But using panopticapi utils, you can convert these almost black masks into overlayed colored masks using detectron2. We show an example visualization script using detectron2 below to visualize our masks.


```
from panopticapi import rgb2id

from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2.data import MetadataCatalog

from PIL import Image
from numpy import np
from coco_meta import COCO_META

import torch


# create a visualizer using COCO meta
instance_mode=ColorMode.IMAGE
metadata=MetadataCatalog.get(
            "coco_2017_val_panoptic" 
        )
visualizer = Visualizer(image, metadata, instance_mode=instance_mode)


# load COCO image


# load panoptic mask
input_panoptic=YOUR_PANOPTIC_MASK_PATH
panoptic_orig = np.asarray(Image.open(input_panoptic), dtype=np.uint32)
panoptic_seg = rgb2id(panoptic_orig).astype(np.int32)


# create category ids to continuous ids mapper from COCO_META 
catid2cont_id={}
for i,meta in enumerate(COCO_META[1:]):
    catid2cont_id[meta['id']]=i

# convert category ids to continuous category ids using the defined mapper
# need to split the segments_info from the annotation json file first by filtering the image_id
for segment in segments_info:
	# replace with the new continus id
	segment['category_id']=catid2cont_id[segment['category_id']]


# visualize output
vis_output = visualizer.draw_panoptic_seg_predictions(
    torch.from_numpy(panoptic_seg), segments_info
)

vis_output.save(YOUR_OUTPUT_PATH)

```