import torch
import torch.nn as nn
import torch.nn.functional as F

from sgg.modeling.core.obb_ops import obb2xyxy


class RotatedRPNHead(nn.Module):
    """
    Refactored Rotated RPN head without cfg / init_cfg / registry usage.

    This module keeps the core head structure, forward behavior, loss API,
    and proposal generation API close to the original implementation, while
    delegating framework-specific training and postprocessing behaviors to
    externally provided components.

    Required external components:
        - anchor_generator
        - bbox_coder
        - train_target_assigner
        - cls_loss_fn
        - bbox_loss_fn
        - nms_fn
        - utilities for anchor/level reshaping

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels in the hidden conv layer.
        num_anchors (int): Number of anchors per spatial position.
        cls_out_channels (int): Number of classification output channels.
            For sigmoid RPN this is usually 1.
        version (str): Angle version used by mmrotate helper functions.
        use_sigmoid_cls (bool): Whether the classification branch uses sigmoid.
        sampling (bool): Whether training uses positive/negative sampling.
        reg_decoded_bbox (bool): Whether bbox loss is applied on decoded boxes.
        nms_pre (int): Number of candidates to keep before NMS.
        min_bbox_size (float): Minimum width/height for proposals.
        max_per_img (int): Maximum number of proposals kept after NMS.
        anchor_generator (object): External anchor generator. Must provide
            ``num_levels`` and ``grid_priors(featmap_sizes, device=...)``.
        bbox_coder (object): External bbox coder. Must provide ``encode`` and
            ``decode``.
        train_target_assigner (object): External training target builder. Must
            provide ``get_targets(...)`` for multi-image target generation.
        cls_loss_fn (callable): Classification loss function.
        bbox_loss_fn (callable): Bounding box regression loss function.
        nms_fn (callable): Post-NMS function with signature
            ``nms_fn(boxes, scores, level_ids, iou_threshold)``.
        nms_iou_threshold (float): IoU threshold passed to ``nms_fn``.
    """

    def __init__(
        self,
        in_channels,
        feat_channels=256,
        num_anchors=1,
        cls_out_channels=1,
        version="oc",
        use_sigmoid_cls=True,
        sampling=True,
        reg_decoded_bbox=False,
        nms_pre=2000,
        min_bbox_size=0,
        max_per_img=1000,
        anchor_generator=None,
        bbox_coder=None,
        train_target_assigner=None,
        cls_loss_fn=None,
        bbox_loss_fn=None,
        nms_fn=None,
        nms_iou_threshold=0.7,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.num_anchors = num_anchors
        self.cls_out_channels = cls_out_channels
        self.version = version
        self.use_sigmoid_cls = use_sigmoid_cls
        self.sampling = sampling
        self.reg_decoded_bbox = reg_decoded_bbox

        self.nms_pre = nms_pre
        self.min_bbox_size = min_bbox_size
        self.max_per_img = max_per_img
        self.nms_iou_threshold = nms_iou_threshold

        self.anchor_generator = anchor_generator
        self.bbox_coder = bbox_coder
        self.train_target_assigner = train_target_assigner
        self.cls_loss_fn = cls_loss_fn
        self.bbox_loss_fn = bbox_loss_fn
        self.nms_fn = nms_fn

        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.rpn_conv = nn.Conv2d(self.in_channels, self.feat_channels, 3, padding=1)
        self.rpn_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            1,
        )
        self.rpn_reg = nn.Conv2d(self.feat_channels, self.num_anchors * 6, 1)

    def init_weights(self):
        """Initialize convolution weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_single(self, x):
        """
        Forward one feature level.

        Args:
            x (Tensor): Input feature map of shape (N, C, H, W).

        Returns:
            tuple[Tensor, Tensor]:
                - cls_score: shape (N, num_anchors * cls_out_channels, H, W)
                - bbox_pred: shape (N, num_anchors * 6, H, W)
        """
        x = x.to(torch.float32)
        x = self.rpn_conv(x)
        x = F.relu(x, inplace=True)
        cls_score = self.rpn_cls(x)
        bbox_pred = self.rpn_reg(x)
        return cls_score, bbox_pred

    def forward(self, feats):
        """
        Forward all feature levels.

        Args:
            feats (list[Tensor]): Multi-level features.

        Returns:
            tuple[list[Tensor], list[Tensor]]:
                Classification scores and bbox predictions of all levels.
        """
        cls_scores = []
        bbox_preds = []
        for feat in feats:
            cls_score, bbox_pred = self.forward_single(feat)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred)
        return cls_scores, bbox_preds

    def loss_single(
        self,
        cls_score,
        bbox_pred,
        anchors,
        labels,
        label_weights,
        bbox_targets,
        bbox_weights,
        num_total_samples,
    ):
        """Compute classification and bbox loss of one scale level."""
        if self.cls_loss_fn is None or self.bbox_loss_fn is None:
            raise RuntimeError("cls_loss_fn and bbox_loss_fn must be provided.")

        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.cls_loss_fn(
            cls_score,
            labels,
            label_weights,
            avg_factor=num_total_samples,
        )

        bbox_targets = bbox_targets.reshape(-1, 6)
        bbox_weights = bbox_weights.reshape(-1, 6)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 6)

        if self.reg_decoded_bbox:
            anchors = anchors.reshape(-1, 4)
            bbox_pred = self.bbox_coder.decode(anchors, bbox_pred)

        loss_bbox = self.bbox_loss_fn(
            bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples,
        )
        return loss_cls, loss_bbox

    def loss(self, cls_scores, bbox_preds, gt_bboxes, img_metas, gt_bboxes_ignore=None):
        """
        Compute training losses.

        This method keeps the original public API style, but delegates target
        generation to ``train_target_assigner``.
        """
        if self.anchor_generator is None:
            raise RuntimeError("anchor_generator must be provided.")
        if self.train_target_assigner is None:
            raise RuntimeError("train_target_assigner must be provided.")

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.anchor_generator.get_anchors(
            featmap_sizes,
            img_metas,
            device=device,
        )

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        target_results = self.train_target_assigner.get_targets(
            anchor_list=anchor_list,
            valid_flag_list=valid_flag_list,
            gt_bboxes_list=gt_bboxes,
            img_metas=img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=None,
            label_channels=label_channels,
            unmap_outputs=True,
        )

        if target_results is None:
            return None

        (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = target_results

        num_total_samples = (num_total_pos + num_total_neg) if self.sampling else num_total_pos

        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        concat_anchor_list = [torch.cat(per_img_anchors) for per_img_anchors in anchor_list]
        all_anchor_list = self.train_target_assigner.images_to_levels(
            concat_anchor_list,
            num_level_anchors,
        )

        losses_cls = []
        losses_bbox = []
        for args in zip(
            cls_scores,
            bbox_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
        ):
            loss_cls, loss_bbox = self.loss_single(*args, num_total_samples=num_total_samples)
            losses_cls.append(loss_cls)
            losses_bbox.append(loss_bbox)

        return {
            "loss_rpn_cls": losses_cls,
            "loss_rpn_bbox": losses_bbox,
        }

    def get_bboxes(self, cls_scores, bbox_preds, img_metas, rescale=False, with_nms=True):
        """
        Convert head outputs into proposal boxes for a batch.
        """
        assert with_nms, "with_nms must be True for RPN proposal generation."
        assert len(cls_scores) == len(bbox_preds)
        if self.anchor_generator is None:
            raise RuntimeError("anchor_generator must be provided.")

        num_levels = len(cls_scores)
        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_priors(featmap_sizes, device=device)

        result_list = []
        for img_id, _ in enumerate(img_metas):
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            bbox_pred_list = [bbox_preds[i][img_id].detach() for i in range(num_levels)]
            img_shape = img_metas[img_id]["img_shape"]
            scale_factor = img_metas[img_id].get("scale_factor", None)
            proposals = self._get_bboxes_single(
                cls_scores=cls_score_list,
                bbox_preds=bbox_pred_list,
                mlvl_anchors=mlvl_anchors,
                img_shape=img_shape,
                scale_factor=scale_factor,
                rescale=rescale,
            )
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self, cls_scores, bbox_preds, mlvl_anchors, img_shape, scale_factor, rescale=False):
        """
        Convert outputs of one image into proposals.
        """
        if self.bbox_coder is None:
            raise RuntimeError("bbox_coder must be provided.")
        if self.nms_fn is None:
            raise RuntimeError("nms_fn must be provided.")

        level_ids = []
        mlvl_scores = []
        mlvl_bbox_preds = []
        mlvl_valid_anchors = []

        for idx, _ in enumerate(cls_scores):
            rpn_cls_score = cls_scores[idx]
            rpn_bbox_pred = bbox_preds[idx]
            assert rpn_cls_score.size()[-2:] == rpn_bbox_pred.size()[-2:]

            rpn_cls_score = rpn_cls_score.permute(1, 2, 0)
            if self.use_sigmoid_cls:
                rpn_cls_score = rpn_cls_score.reshape(-1)
                scores = rpn_cls_score.sigmoid()
            else:
                rpn_cls_score = rpn_cls_score.reshape(-1, 2)
                scores = rpn_cls_score.softmax(dim=1)[:, 0]

            rpn_bbox_pred = rpn_bbox_pred.permute(1, 2, 0).reshape(-1, 6)
            anchors = mlvl_anchors[idx]

            if self.nms_pre > 0 and scores.shape[0] > self.nms_pre:
                ranked_scores, rank_inds = scores.sort(descending=True)
                topk_inds = rank_inds[: self.nms_pre]
                scores = ranked_scores[: self.nms_pre]
                rpn_bbox_pred = rpn_bbox_pred[topk_inds, :]
                anchors = anchors[topk_inds, :]

            mlvl_scores.append(scores)
            mlvl_bbox_preds.append(rpn_bbox_pred)
            mlvl_valid_anchors.append(anchors)
            level_ids.append(scores.new_full((scores.size(0),), idx, dtype=torch.long))

        scores = torch.cat(mlvl_scores)
        anchors = torch.cat(mlvl_valid_anchors)
        rpn_bbox_pred = torch.cat(mlvl_bbox_preds)
        proposals = self.bbox_coder.decode(anchors, rpn_bbox_pred, max_shape=img_shape)
        ids = torch.cat(level_ids)

        if self.min_bbox_size > 0:
            w = proposals[:, 2]
            h = proposals[:, 3]
            valid_mask = (w >= self.min_bbox_size) & (h >= self.min_bbox_size)
            if not valid_mask.all():
                proposals = proposals[valid_mask]
                scores = scores[valid_mask]
                ids = ids[valid_mask]

        if proposals.numel() == 0:
            return proposals.new_zeros((0, 6))

        hproposals = obb2xyxy(proposals, self.version)
        dets, keep = self.nms_fn(
            hproposals,
            scores,
            ids,
            iou_threshold=self.nms_iou_threshold,
        )
        dets = torch.cat([proposals, scores[:, None]], dim=1)
        dets = dets[keep]
        return dets[: self.max_per_img]
