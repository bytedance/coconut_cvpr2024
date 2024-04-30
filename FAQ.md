# FAQ

________________________________________________________________________________

**Q1: Why the downloaded images are all black?**

A: There are colors in the image but hard to be noticed. The reason is that we did not 100% align the colors in panopticapi. But using panopticapi utils, you can convert these almost black masks into overlayed colored masks using detectron2. We show an example visualization script using detectron2 below to visualize our masks.

Please refer to [visualization](tutorials/visualization/demo.ipynb) for more details.


**Q2: Where is the annotations for instances that can be used by pycocotools**
A: We don't provide direct download link for the instance masks. Instead, we provide a script to convert the instance masks for 'thing' from our panoptic segmentation masks to the instance polygons that can be loaded by pycocotools. Please refer to [kmaxdeeplab_instance](tutorials/kmaxdeeplab_instance) step 3 for details.


**Q3: Getting wrong instance masks visualized from the convertion script using panoptic masks?**
A: Please check our [visualization tutorial](tutorials/kmaxdeeplab_instance/vis_converted_instance.ipynb) using pycocotools. 