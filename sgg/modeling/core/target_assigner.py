import torch

from .obb_ops import obb2xyxy


class RotatedRPNTargetAssigner:
    """
    Adapter that keeps the original target-building behavior, but moves all
    training framework dependencies behind explicit constructor arguments.

    Required external callables / objects:
        - anchor_inside_flags_fn
        - images_to_levels_fn
        - unmap_fn
        - assigner
        - sampler
        - bbox_coder

    Notes:
        This class still depends on an external assigner and sampler. That is
        intentional: those are the parts you said you are willing to provide as
        part of your training framework.
    """

    def __init__(
        self,
        version="oc",
        allowed_border=0,
        pos_weight=-1,
        sampling=True,
        reg_decoded_bbox=False,
        num_classes=1,
        bbox_coder=None,
        assigner=None,
        sampler=None,
        anchor_inside_flags_fn=None,
        images_to_levels_fn=None,
        unmap_fn=None,
    ):
        self.version = version
        self.allowed_border = allowed_border
        self.pos_weight = pos_weight
        self.sampling = sampling
        self.reg_decoded_bbox = reg_decoded_bbox
        self.num_classes = num_classes

        self.bbox_coder = bbox_coder
        self.assigner = assigner
        self.sampler = sampler
        self.anchor_inside_flags_fn = anchor_inside_flags_fn
        self.images_to_levels_fn = images_to_levels_fn
        self.unmap_fn = unmap_fn

    def images_to_levels(self, targets, num_level_anchors):
        if self.images_to_levels_fn is None:
            raise RuntimeError("images_to_levels_fn must be provided.")
        return self.images_to_levels_fn(targets, num_level_anchors)

    def _get_targets_single(
        self,
        flat_anchors,
        valid_flags,
        gt_bboxes,
        gt_bboxes_ignore,
        gt_labels,
        img_meta,
        label_channels=1,
        unmap_outputs=True,
    ):
        if self.anchor_inside_flags_fn is None:
            raise RuntimeError("anchor_inside_flags_fn must be provided.")
        if self.unmap_fn is None:
            raise RuntimeError("unmap_fn must be provided.")
        if self.assigner is None or self.sampler is None or self.bbox_coder is None:
            raise RuntimeError("assigner, sampler, and bbox_coder must be provided.")

        inside_flags = self.anchor_inside_flags_fn(
            flat_anchors,
            valid_flags,
            img_meta["img_shape"][:2],
            self.allowed_border,
        )
        if not inside_flags.any():
            return (None,) * 7

        anchors = flat_anchors[inside_flags, :]
        gt_hbboxes = obb2xyxy(gt_bboxes, self.version)

        assign_result = self.assigner.assign(
            anchors,
            gt_hbboxes,
            gt_bboxes_ignore,
            None if self.sampling else gt_labels,
        )
        sampling_result = self.sampler.sample(assign_result, anchors, gt_hbboxes)

        if gt_bboxes.numel() == 0:
            sampling_result.pos_gt_bboxes = gt_bboxes.new(
                (0, gt_bboxes.size(-1))
            ).zero_()
        else:
            sampling_result.pos_gt_bboxes = gt_bboxes[
                sampling_result.pos_assigned_gt_inds, :
            ]

        num_valid_anchors = anchors.shape[0]
        bbox_targets = anchors.new_zeros((anchors.size(0), 6))
        bbox_weights = anchors.new_zeros((anchors.size(0), 6))
        labels = anchors.new_full((num_valid_anchors,), self.num_classes, dtype=torch.long)
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        if len(pos_inds) > 0:
            if not self.reg_decoded_bbox:
                pos_bbox_targets = self.bbox_coder.encode(
                    sampling_result.pos_bboxes,
                    sampling_result.pos_gt_bboxes,
                )
            else:
                pos_bbox_targets = sampling_result.pos_gt_bboxes

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels is None:
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]

            label_weights[pos_inds] = 1.0 if self.pos_weight <= 0 else self.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            labels = self.unmap_fn(labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = self.unmap_fn(label_weights, num_total_anchors, inside_flags)
            bbox_targets = self.unmap_fn(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = self.unmap_fn(bbox_weights, num_total_anchors, inside_flags)

        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            pos_inds,
            neg_inds,
            sampling_result,
        )

    def get_targets(
        self,
        anchor_list,
        valid_flag_list,
        gt_bboxes_list,
        img_metas,
        gt_bboxes_ignore_list=None,
        gt_labels_list=None,
        label_channels=1,
        unmap_outputs=True,
    ):
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        concat_anchor_list = []
        concat_valid_flag_list = []
        for i in range(num_imgs):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
            concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))

        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]

        results = [
            self._get_targets_single(
                concat_anchor_list[i],
                concat_valid_flag_list[i],
                gt_bboxes_list[i],
                gt_bboxes_ignore_list[i],
                gt_labels_list[i],
                img_metas[i],
                label_channels=label_channels,
                unmap_outputs=unmap_outputs,
            )
            for i in range(num_imgs)
        ]

        all_labels = [r[0] for r in results]
        all_label_weights = [r[1] for r in results]
        all_bbox_targets = [r[2] for r in results]
        all_bbox_weights = [r[3] for r in results]
        pos_inds_list = [r[4] for r in results]
        neg_inds_list = [r[5] for r in results]

        if any(labels is None for labels in all_labels):
            return None

        num_total_pos = sum(max(inds.numel(), 1) for inds in pos_inds_list)
        num_total_neg = sum(max(inds.numel(), 1) for inds in neg_inds_list)

        labels_list = self.images_to_levels(all_labels, num_level_anchors)
        label_weights_list = self.images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = self.images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list = self.images_to_levels(all_bbox_weights, num_level_anchors)

        return (
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        )
