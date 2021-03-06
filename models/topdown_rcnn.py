from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as func
import torchvision
from torchvision.ops import MultiScaleRoIAlign
from .general import ResNetBlock

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
    norm_vectors = config.MODEL.NORM_VECTORS
    return TopDownRCNN(config.MODEL.PARAMS, norm_vectors, num_classes, embedding_size, embeddings_loss_function)


def get_roi(cfg, norm_vectors, out_channels, num_classes, embedding_size, embeddings_loss_function):
    box_score_thresh = cfg.BBOX.SCORE_THRSH
    box_nms_thresh = cfg.BBOX.NMS_THRSH
    box_detections_per_img = cfg.BBOX.DET_PER_IMH
    box_fg_iou_thresh = cfg.BBOX.FG_IOU_THRSH
    box_bg_iou_thresh = cfg.BBOX.BG_IOU_THRSH
    box_batch_size_per_image = cfg.BBOX.BS_PER_IMG
    box_positive_fraction = cfg.BBOX.POS_FRAQ
    bbox_reg_weights = cfg.BBOX.REG_W

    box_roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2)
    resolution = box_roi_pool.output_size[0]
    representation_size = cfg.BBOX.REPR_SIZE
    extended_mlp = cfg.EXTENDED_TWOMLP
    extendend_emb_head = cfg.EXTENDED_EMB_HEAD
    box_head = TwoMLPHead(
        extended_mlp,
        out_channels, resolution ** 2,
        representation_size)

    box_predictor = FastRCNNPredictor(
        norm_vectors,
        extendend_emb_head,
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
    def __init__(self, cfg, norm_vectors, num_classes, embedding_size, embeddings_loss_function):
        super(TopDownRCNN, self).__init__()
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
        self.preprocess = model.transform
        self.norm_vectors = norm_vectors
        self.backbone = model.backbone
        out_channels = self.backbone.out_channels
        self.rpn = model.rpn
        self.roi_heads = get_roi(cfg, norm_vectors, out_channels, num_classes, embedding_size, embeddings_loss_function)

    @property
    def support_matrix(self):
        return self.roi_heads.support_matrix

    @support_matrix.setter
    def support_matrix(self, matrix):
        self.roi_heads.support_matrix = matrix

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
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        if self.training and targets is None:
            raise ValueError(f'targets cannot be None, when model is in training mode')
        images, targets = self.preprocess(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, targets)

        # halt = torch.stack([torch.isnan(pl) for pl in proposal_losses.values()], 0)
        # halt = halt.sum()

        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        detections = self.preprocess.postprocess(detections, images.image_sizes, original_image_sizes)

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

    def __init__(self, extended_mlp, in_channels, resolution, representation_size):
        super(TwoMLPHead, self).__init__()
        self.extended_mlp = extended_mlp
        if extended_mlp:
            self.res_block = ResNetBlock(in_channels, in_channels, expand=2, stride=1)
        self.fc6 = nn.Linear(in_channels * resolution, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        if self.extended_mlp:
            x = self.res_block(x)
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

    def __init__(self, norm_vectors, extendend_emb_head, in_channels, num_classes, embedding_size):
        super(FastRCNNPredictor, self).__init__()
        self.extendend_emb_head = extendend_emb_head
        self.num_classes = num_classes
        self.norm_vectors = norm_vectors
        self.embedding_size = embedding_size
        if extendend_emb_head:
            self.embedder = nn.Sequential(nn.Linear(in_channels, in_channels * 2),
                                          nn.Linear(in_channels * 2, in_channels),
                                          nn.Linear(in_channels, embedding_size))
        else:
            self.embedder = nn.Linear(in_channels, embedding_size)
        self.bbox_pred = nn.Linear(in_channels, 4)

    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        embeddings = self.embedder(x)
        if self.norm_vectors:
            embeddings /= torch.norm(embeddings, p=2, dim=-1).detach()[..., None]
        bbox_deltas = self.bbox_pred(x)

        return embeddings, bbox_deltas
