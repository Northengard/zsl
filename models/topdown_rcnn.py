from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as func
import torchvision
from torchvision.ops import MultiScaleRoIAlign

from .roi_heads import RoIHeads

import losses
# from torchvision.models.detection.anchor_utils import AnchorGenerator
# from torchvision.models.detection.rpn import RegionProposalNetwork, RPNHead


# def get_rpn(out_channels):
#     anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
#     aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
#     rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
#     return RegionProposalNetwork(anchor_generator=rpn_anchor_generator,
#                                  head=RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0]),
#                                  nms_thresh=0.7, fg_iou_thresh=0.7, bg_iou_thresh=0.3,
#                                  positive_fraction=0.5, score_thresh=0.0,
#                                  pre_nms_top_n=dict(training=2000, testing=1000),
#                                  post_nms_top_n=dict(training=2000, testing=1000),
#                                  batch_size_per_image=512)


def topdown_rcnn(config):
    num_classes = config.LOSS.PARAMS.NUM_CLS
    embedding_size = config.MODEL.PARAMS.VECTOR_SIZE
    embeddings_loss_function = getattr(losses, config.LOSS.NAME)(config)
    return TopDownRCNN(num_classes, embedding_size, embeddings_loss_function)


def get_roi(out_channels, num_classes, embedding_size, embeddings_loss_function):
    box_score_thresh = 0.05
    box_nms_thresh = 0.5
    box_detections_per_img = 100
    box_fg_iou_thresh = 0.5
    box_bg_iou_thresh = 0.5
    box_batch_size_per_image = 512
    box_positive_fraction = 0.25
    bbox_reg_weights = None

    box_roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)
    resolution = box_roi_pool.output_size[0]
    representation_size = 1024
    box_head = TwoMLPHead(
        out_channels * resolution ** 2,
        representation_size)

    representation_size = 1024
    box_predictor = FastRCNNPredictor(
        representation_size,
        num_classes + 1,
        embedding_size)

    return RoIHeads(box_roi_pool, box_head, box_predictor,
                    box_fg_iou_thresh, box_bg_iou_thresh,
                    box_batch_size_per_image, box_positive_fraction,
                    bbox_reg_weights, box_score_thresh,
                    box_nms_thresh, box_detections_per_img,
                    embeddings_loss_function=embeddings_loss_function)


class TopDownRCNN(nn.Module):
    def __init__(self, num_classes, embedding_size, embeddings_loss_function):
        super(TopDownRCNN, self).__init__()
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
        self.preprocess = model.transform
        self.backbone = model.backbone
        out_channels = self.backbone.out_channels
        self.rpn = model.rpn
        self.roi_heads = get_roi(out_channels, num_classes, embedding_size, embeddings_loss_function)

    def forward(self, images, targets=None):
        """
        Args:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).
        """
        if not self.training and targets is None:
            raise ValueError(f'targets cannot be None, when model is in training mode')
        images, targets = self.preprocess(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, targets)

        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        # detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        return detections, losses


class TwoMLPHead(nn.Module):
    """
    Standard heads for FPN-based models
    Args:
        in_channels (int): number of input channels
        representation_size (int): size of the intermediate representation
    """

    def __init__(self, in_channels, representation_size):
        super(TwoMLPHead, self).__init__()

        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1)

        x = func.relu(self.fc6(x))
        x = func.relu(self.fc7(x))

        return x


class FastRCNNPredictor(nn.Module):
    """
    Standard classification + bounding box regression layers
    for Fast R-CNN.
    Args:
        in_channels (int): number of input channels
        num_classes (int): number of output classes (including background)
    """

    def __init__(self, in_channels, num_classes, embedding_size):
        super(FastRCNNPredictor, self).__init__()
        self.embedder = nn.Linear(in_channels, embedding_size)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        embeddings = self.embedder(x)
        bbox_deltas = self.bbox_pred(x)

        return embeddings, bbox_deltas
