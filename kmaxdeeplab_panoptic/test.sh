python3 train_net.py --num-gpus 8 --dist-url tcp://127.0.0.1:9999 \
--config-file configs/coco/panoptic-segmentation/kmax_convnext_large.yaml \
--eval-only MODEL.WEIGHTS ./kmax_convnext_large_coconut_s.pkl