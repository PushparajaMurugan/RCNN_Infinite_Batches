import torch.nn as nn
from torchvision import ops
import torchvision

from backbone.mobilenet import MobileNetV2
from backbone.vision_transformer import CvT
from backbone.resnet50_fpn_model import *
from backbone.Efficientnet import EfficientNet
from backbone.nfnets.model import NFNet
from utils.anchor_utils import AnchorsGenerator
from utils.faster_rcnn_utils import FasterRCNN, FastRCNNPredictor


def create_model(backbone_network, cfg, num_classes, backbone_pretrained_weights=None):

    if backbone_network == 'mobilenet':
        backbone = MobileNetV2(weights_path=backbone_pretrained_weights).features
        backbone.out_channels = 1280

    elif backbone_network == 'resnet18':
        resnet18 = torchvision.models.resnet18(pretrained=False)
        modules = list(resnet18.children())[:-2]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 512

    elif backbone_network == 'shufflenet':
        shufflenet = torchvision.models.shufflenet_v2_x1_0(pretrained=False)
        modules = list(shufflenet.children())[:-1]
        backbone = nn.Sequential(*modules)
        backbone.out_channels = 1024

    elif backbone_network == 'efficientnet-b0':
        backbone = EfficientNet.from_name('efficientnet-b0', condconv_num_expert=-1, norm_layer=None)
        backbone.out_channels = 1280

    elif backbone_network == 'transformer-CvT':
        backbone = CvT(
            num_classes=1000,
            s1_emb_dim=64,  # stage 1 - dimension
            s1_emb_kernel=7,  # stage 1 - conv kernel
            s1_emb_stride=4,  # stage 1 - conv stride
            s1_proj_kernel=3,  # stage 1 - attention ds-conv kernel size
            s1_kv_proj_stride=2,  # stage 1 - attention key / value projection stride
            s1_heads=1,  # stage 1 - heads
            s1_depth=1,  # stage 1 - depth
            s1_mlp_mult=4,  # stage 1 - feedforward expansion factor
            s2_emb_dim=192,  # stage 2 - (same as above)
            s2_emb_kernel=3,
            s2_emb_stride=2,
            s2_proj_kernel=3,
            s2_kv_proj_stride=2,
            s2_heads=3,
            s2_depth=2,
            s2_mlp_mult=4,
            s3_emb_dim=384,  # stage 3 - (same as above)
            s3_emb_kernel=3,
            s3_emb_stride=2,
            s3_proj_kernel=3,
            s3_kv_proj_stride=2,
            s3_heads=4,
            s3_depth=10,
            s3_mlp_mult=4,
            dropout=0.
        )

        backbone.out_channels = 384

    elif backbone_network == 'nfnet-f0':
        backbone = NFNet(variant='F0', activation='gelu', stochdepth_rate=0.25, alpha=0.2, se_ratio=0.5)
        backbone.out_channels = 3072

    else:
        raise NotImplemented

    anchor_sizes = tuple((f,) for f in cfg.anchor_size)
    aspect_ratios = tuple((f,) for f in cfg.anchor_ratio) * len(anchor_sizes)
    anchor_generator = AnchorsGenerator(sizes=anchor_sizes,
                                        aspect_ratios=aspect_ratios)

    roi_pooler = ops.MultiScaleRoIAlign(featmap_names=['0'],  # roi pooling in which resolution feature
                                        output_size=cfg.roi_out_size,  # roi_pooling output feature size
                                        sampling_ratio=cfg.roi_sample_rate)  # sampling_ratio

    model = FasterRCNN(backbone=backbone, num_classes=num_classes,
                       # transform parameters
                       min_size=cfg.min_size, max_size=cfg.max_size,
                       image_mean=cfg.image_mean, image_std=cfg.image_std,
                       # rpn parameters
                       rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler,
                       rpn_pre_nms_top_n_train=cfg.rpn_pre_nms_top_n_train,
                       rpn_pre_nms_top_n_test=cfg.rpn_pre_nms_top_n_test,
                       rpn_post_nms_top_n_train=cfg.rpn_post_nms_top_n_train,
                       rpn_post_nms_top_n_test=cfg.rpn_post_nms_top_n_test,
                       rpn_nms_thresh=cfg.rpn_nms_thresh,
                       rpn_fg_iou_thresh=cfg.rpn_fg_iou_thresh,
                       rpn_bg_iou_thresh=cfg.rpn_bg_iou_thresh,
                       rpn_batch_size_per_image=cfg.rpn_batch_size_per_image,
                       rpn_positive_fraction=cfg.rpn_positive_fraction,
                       # Box parameters
                       box_head=None, box_predictor=None,

                       # remove low threshold target
                       box_score_thresh=cfg.box_score_thresh,
                       box_nms_thresh=cfg.box_nms_thresh,
                       box_detections_per_img=cfg.box_detections_per_img,
                       box_fg_iou_thresh=cfg.box_fg_iou_thresh,
                       box_bg_iou_thresh=cfg.box_bg_iou_thresh,
                       box_batch_size_per_image=cfg.box_batch_size_per_image,
                       box_positive_fraction=cfg.box_positive_fraction,
                       bbox_reg_weights=cfg.bbox_reg_weights
                       )

    return model
