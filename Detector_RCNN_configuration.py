class detectorConfig:
    min_size = 800
    max_size = 1000
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]

    # anchor parameters
    anchor_size = [64, 128, 256]
    anchor_ratio = [0.5, 1, 2.0]

    # roi align parameters
    roi_out_size = [7, 7]
    roi_sample_rate = 2

    # rpn process parameters
    rpn_pre_nms_top_n_train = 2000
    rpn_post_nms_top_n_train = 2000

    rpn_pre_nms_top_n_test = 1000
    rpn_post_nms_top_n_test = 1000

    rpn_nms_thresh = 0.7
    rpn_fg_iou_thresh = 0.7
    rpn_bg_iou_thresh = 0.3
    rpn_batch_size_per_image = 256
    rpn_positive_fraction = 0.5

    # remove low threshold target
    box_score_thresh = 0.015
    box_nms_thresh = 0.5
    box_detections_per_img = 100
    box_fg_iou_thresh = 0.5
    box_bg_iou_thresh = 0.5
    box_batch_size_per_image = 512
    box_positive_fraction = 0.25
    bbox_reg_weights = None
