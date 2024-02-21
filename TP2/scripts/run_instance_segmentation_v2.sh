python main.py --multirun \
    segmentation_method=instance_segmentation \
    segmentation_method.threshold=0.5 \
    segmentation_method.mask_threshold=0.7 \
    segmentation_method.model=maskrcnn_resnet50_fpn_v2 \
    segmentation_method.device=cuda \
    batch_size=2
