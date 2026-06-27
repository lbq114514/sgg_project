import math
import torch

from ..core.obb_ops import norm_angle, obb2poly, obb2xyxy, poly2obb


class HorizontalBoxCoder:
    """
    Standard HBB coder for boxes in xyxy format.
    """

    def __init__(self, weights=(1.0, 1.0, 1.0, 1.0), bbox_xform_clip=math.log(1000.0 / 16)):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip

    def encode(self, reference_boxes, proposals):
        """
        Args:
            reference_boxes (Tensor): gt boxes, shape (N, 4)
            proposals (Tensor): anchors / proposals, shape (N, 4)
        """
        wx, wy, ww, wh = self.weights

        ex_widths = proposals[:, 2] - proposals[:, 0]
        ex_heights = proposals[:, 3] - proposals[:, 1]
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights

        gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0]
        gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1]
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights

        eps = torch.finfo(ex_widths.dtype).eps
        ex_widths = ex_widths.clamp(min=eps)
        ex_heights = ex_heights.clamp(min=eps)
        gt_widths = gt_widths.clamp(min=eps)
        gt_heights = gt_heights.clamp(min=eps)

        dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        dw = ww * torch.log(gt_widths / ex_widths)
        dh = wh * torch.log(gt_heights / ex_heights)

        return torch.stack((dx, dy, dw, dh), dim=1)

    def decode(self, rel_codes, boxes, max_shape=None):
        """
        Args:
            rel_codes (Tensor): shape (N, 4)
            boxes (Tensor): anchors / proposals, shape (N, 4)
        """
        boxes = boxes.to(rel_codes.dtype)

        wx, wy, ww, wh = self.weights

        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        dx = rel_codes[:, 0] / wx
        dy = rel_codes[:, 1] / wy
        dw = rel_codes[:, 2] / ww
        dh = rel_codes[:, 3] / wh

        dw = torch.clamp(dw, max=self.bbox_xform_clip)
        dh = torch.clamp(dh, max=self.bbox_xform_clip)

        pred_ctr_x = dx * widths + ctr_x
        pred_ctr_y = dy * heights + ctr_y
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        x1 = pred_ctr_x - 0.5 * pred_w
        y1 = pred_ctr_y - 0.5 * pred_h
        x2 = pred_ctr_x + 0.5 * pred_w
        y2 = pred_ctr_y + 0.5 * pred_h

        decoded = torch.stack((x1, y1, x2, y2), dim=1)

        if max_shape is not None:
            if len(max_shape) >= 2:
                h, w = max_shape[:2]
                decoded[:, 0].clamp_(min=0, max=w)
                decoded[:, 2].clamp_(min=0, max=w)
                decoded[:, 1].clamp_(min=0, max=h)
                decoded[:, 3].clamp_(min=0, max=h)

        return decoded


class RotatedBoxCoder:
    """
    Simple OBB coder for boxes in xywha format.

    Format:
        (cx, cy, w, h, angle)

    Note:
        Angle convention must be consistent across the whole project.
    """

    def __init__(
        self,
        weights=(1.0, 1.0, 1.0, 1.0, 1.0),
        bbox_xform_clip=math.log(1000.0 / 16),
        angle_normalize=True,
    ):
        self.weights = weights
        self.bbox_xform_clip = bbox_xform_clip
        self.angle_normalize = angle_normalize

    def _normalize_angle(self, da):
        # normalize to [-pi, pi)
        return (da + math.pi) % (2 * math.pi) - math.pi

    def encode(self, reference_boxes, proposals):
        """
        Args:
            reference_boxes (Tensor): gt rotated boxes, shape (N, 5)
            proposals (Tensor): anchors / proposals, shape (N, 5)
        """
        wx, wy, ww, wh, wa = self.weights

        px, py, pw, ph, pa = proposals.unbind(dim=1)
        gx, gy, gw, gh, ga = reference_boxes.unbind(dim=1)

        eps = torch.finfo(pw.dtype).eps
        pw = pw.clamp(min=eps)
        ph = ph.clamp(min=eps)
        gw = gw.clamp(min=eps)
        gh = gh.clamp(min=eps)

        dx = wx * (gx - px) / pw
        dy = wy * (gy - py) / ph
        dw = ww * torch.log(gw / pw)
        dh = wh * torch.log(gh / ph)

        da = ga - pa
        if self.angle_normalize:
            da = self._normalize_angle(da)
        da = wa * da

        return torch.stack((dx, dy, dw, dh, da), dim=1)

    def decode(self, rel_codes, boxes, max_shape=None):
        """
        Args:
            rel_codes (Tensor): shape (N, 5)
            boxes (Tensor): anchors / proposals, shape (N, 5)
        """
        boxes = boxes.to(rel_codes.dtype)

        wx, wy, ww, wh, wa = self.weights

        px, py, pw, ph, pa = boxes.unbind(dim=1)
        dx, dy, dw, dh, da = rel_codes.unbind(dim=1)

        dx = dx / wx
        dy = dy / wy
        dw = torch.clamp(dw / ww, max=self.bbox_xform_clip)
        dh = torch.clamp(dh / wh, max=self.bbox_xform_clip)
        da = da / wa

        gx = dx * pw + px
        gy = dy * ph + py
        gw = torch.exp(dw) * pw
        gh = torch.exp(dh) * ph
        ga = da + pa

        if self.angle_normalize:
            ga = self._normalize_angle(ga)

        decoded = torch.stack((gx, gy, gw, gh, ga), dim=1)

        if max_shape is not None and len(max_shape) >= 2:
            h, w = max_shape[:2]
            decoded[:, 0].clamp_(min=0, max=w)
            decoded[:, 1].clamp_(min=0, max=h)

        return decoded


class UnifiedBoxCoder:
    """
    Wrapper box coder.

    Args:
        mode (str): "hbb", "obb", "midpointoffset", or "delta_xywha_obb"
    """

    def __init__(self, mode="hbb", **kwargs):
        if mode not in ("hbb", "obb", "midpointoffset", "delta_xywha_obb"):
            raise ValueError(f"Unsupported mode: {mode}")

        self.mode = mode
        if mode == "hbb":
            self.coder = HorizontalBoxCoder(**kwargs)
        elif mode == "obb":
            self.coder = RotatedBoxCoder(**kwargs)
        elif mode == "midpointoffset":
            self.coder = MidpointOffsetBoxCoder(**kwargs)
        else:
            self.coder = DeltaXYWHAOBBoxCoder(**kwargs)

    def encode(self, bboxes, gt_bboxes):
        """
        Public interface aligned with mmdet/mmrotate heads.

        Args:
            bboxes (Tensor): Source boxes / anchors.
            gt_bboxes (Tensor): Target boxes.
        """
        if self.mode in ("midpointoffset", "delta_xywha_obb"):
            return self.coder.encode(bboxes, gt_bboxes)
        return self.coder.encode(gt_bboxes, bboxes)

    def decode(self, bboxes, pred_bboxes, max_shape=None):
        """
        Public interface aligned with mmdet/mmrotate heads.

        Args:
            bboxes (Tensor): Source boxes / anchors.
            pred_bboxes (Tensor): Encoded deltas.
        """
        if self.mode in ("midpointoffset", "delta_xywha_obb"):
            return self.coder.decode(bboxes, pred_bboxes, max_shape=max_shape)
        return self.coder.decode(pred_bboxes, bboxes, max_shape=max_shape)


class DeltaXYWHAOBBoxCoder:
    """
    Delta coder for rotated boxes in ``xywha`` format.

    This mirrors mmrotate's ``DeltaXYWHAOBBoxCoder`` but stays dependency-light
    and follows the project's degree-valued angle convention.
    """

    def __init__(
        self,
        target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
        target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
        angle_range="oc",
        norm_factor=None,
        edge_swap=False,
        proj_xy=False,
        add_ctr_clamp=False,
        ctr_clamp=32,
    ):
        self.means = target_means
        self.stds = target_stds
        self.angle_range = angle_range
        self.norm_factor = norm_factor
        self.edge_swap = edge_swap
        self.proj_xy = proj_xy
        self.add_ctr_clamp = add_ctr_clamp
        self.ctr_clamp = ctr_clamp

    def encode(self, bboxes, gt_bboxes):
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == 5
        assert gt_bboxes.size(-1) == 5
        return self._bbox2delta(
            bboxes,
            gt_bboxes,
            self.means,
            self.stds,
            self.angle_range,
            self.norm_factor,
            self.edge_swap,
            self.proj_xy,
        )

    def decode(self, bboxes, pred_bboxes, max_shape=None, wh_ratio_clip=16 / 1000):
        assert pred_bboxes.size(0) == bboxes.size(0)
        return self._delta2bbox(
            bboxes,
            pred_bboxes,
            self.means,
            self.stds,
            max_shape,
            wh_ratio_clip,
            self.add_ctr_clamp,
            self.ctr_clamp,
            self.angle_range,
            self.norm_factor,
            self.edge_swap,
            self.proj_xy,
        )

    @staticmethod
    def _bbox2delta(
        proposals,
        gt,
        means,
        stds,
        angle_range,
        norm_factor,
        edge_swap,
        proj_xy,
    ):
        proposals = proposals.float()
        gt = gt.float()

        px, py, pw, ph, pa = proposals.unbind(dim=-1)
        gx, gy, gw, gh, ga = gt.unbind(dim=-1)

        eps = torch.finfo(proposals.dtype).eps
        pw = pw.clamp(min=eps)
        ph = ph.clamp(min=eps)
        gw = gw.clamp(min=eps)
        gh = gh.clamp(min=eps)

        if proj_xy:
            pa_rad = torch.deg2rad(pa)
            dx = (torch.cos(pa_rad) * (gx - px) + torch.sin(pa_rad) * (gy - py)) / pw
            dy = (-torch.sin(pa_rad) * (gx - px) + torch.cos(pa_rad) * (gy - py)) / ph
        else:
            dx = (gx - px) / pw
            dy = (gy - py) / ph

        if edge_swap:
            dtheta1 = norm_angle(ga - pa, angle_range)
            dtheta2 = norm_angle(ga - pa + 90.0, angle_range)
            abs_dtheta1 = torch.abs(dtheta1)
            abs_dtheta2 = torch.abs(dtheta2)
            use_primary = abs_dtheta1 < abs_dtheta2
            gw_regular = torch.where(use_primary, gw, gh)
            gh_regular = torch.where(use_primary, gh, gw)
            da = torch.where(use_primary, dtheta1, dtheta2)
            dw = torch.log(gw_regular / pw)
            dh = torch.log(gh_regular / ph)
        else:
            da = norm_angle(ga - pa, angle_range)
            dw = torch.log(gw / pw)
            dh = torch.log(gh / ph)

        if norm_factor is not None:
            da = da / (norm_factor * 180.0)

        deltas = torch.stack([dx, dy, dw, dh, da], dim=-1)
        means = deltas.new_tensor(means).unsqueeze(0)
        stds = deltas.new_tensor(stds).unsqueeze(0)
        return deltas.sub_(means).div_(stds)

    @staticmethod
    def _delta2bbox(
        rois,
        deltas,
        means,
        stds,
        max_shape,
        wh_ratio_clip,
        add_ctr_clamp,
        ctr_clamp,
        angle_range,
        norm_factor,
        edge_swap,
        proj_xy,
    ):
        means = deltas.new_tensor(means).view(1, -1).repeat(1, deltas.size(1) // 5)
        stds = deltas.new_tensor(stds).view(1, -1).repeat(1, deltas.size(1) // 5)
        denorm_deltas = deltas * stds + means

        dx = denorm_deltas[:, 0::5]
        dy = denorm_deltas[:, 1::5]
        dw = denorm_deltas[:, 2::5]
        dh = denorm_deltas[:, 3::5]
        da = denorm_deltas[:, 4::5]

        if norm_factor is not None:
            da = da * (norm_factor * 180.0)

        px = rois[:, 0].unsqueeze(1).expand_as(dx)
        py = rois[:, 1].unsqueeze(1).expand_as(dy)
        pw = rois[:, 2].unsqueeze(1).expand_as(dw)
        ph = rois[:, 3].unsqueeze(1).expand_as(dh)
        pa = rois[:, 4].unsqueeze(1).expand_as(da)

        dx_width = pw * dx
        dy_height = ph * dy
        max_ratio = abs(math.log(wh_ratio_clip))

        if add_ctr_clamp:
            dx_width = torch.clamp(dx_width, min=-ctr_clamp, max=ctr_clamp)
            dy_height = torch.clamp(dy_height, min=-ctr_clamp, max=ctr_clamp)
            dw = torch.clamp(dw, max=max_ratio)
            dh = torch.clamp(dh, max=max_ratio)
        else:
            dw = dw.clamp(min=-max_ratio, max=max_ratio)
            dh = dh.clamp(min=-max_ratio, max=max_ratio)

        gw = pw * dw.exp()
        gh = ph * dh.exp()

        if proj_xy:
            pa_rad = torch.deg2rad(pa)
            gx = dx * pw * torch.cos(pa_rad) - dy * ph * torch.sin(pa_rad) + px
            gy = dx * pw * torch.sin(pa_rad) + dy * ph * torch.cos(pa_rad) + py
        else:
            gx = px + dx_width
            gy = py + dy_height

        ga = norm_angle(pa + da, angle_range)

        if max_shape is not None:
            if len(max_shape) >= 2:
                h, w = max_shape[:2]
                gx = gx.clamp(min=0, max=w - 1)
                gy = gy.clamp(min=0, max=h - 1)

        if edge_swap:
            use_regular = gw > gh
            w_regular = torch.where(use_regular, gw, gh)
            h_regular = torch.where(use_regular, gh, gw)
            theta_regular = torch.where(use_regular, ga, ga + 90.0)
            theta_regular = norm_angle(theta_regular, angle_range)
            return torch.stack([gx, gy, w_regular, h_regular, theta_regular], dim=-1).view_as(deltas)

        return torch.stack([gx, gy, gw, gh, ga], dim=-1).view_as(deltas)


class MidpointOffsetBoxCoder:
    """
    6-dim rotated box coder used by Oriented RPN.

    This matches the midpoint-offset representation in mmrotate:
    anchors are horizontal boxes in ``xyxy`` and targets are rotated boxes
    in ``xywha``. The encoded delta is ``(dx, dy, dw, dh, da, db)``.
    """

    def __init__(
        self,
        target_means=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        target_stds=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        angle_range="oc",
        wh_ratio_clip=16 / 1000,
    ):
        self.means = target_means
        self.stds = target_stds
        self.version = angle_range
        self.wh_ratio_clip = wh_ratio_clip

    def encode(self, bboxes, gt_bboxes):
        """
        Args:
            bboxes (Tensor): Horizontal anchors, shape (N, 4).
            gt_bboxes (Tensor): Rotated gt boxes, shape (N, 5).
        """
        assert bboxes.size(0) == gt_bboxes.size(0)
        assert bboxes.size(-1) == 4
        assert gt_bboxes.size(-1) == 5
        return self._bbox2delta(bboxes, gt_bboxes)

    def decode(self, bboxes, pred_bboxes, max_shape=None):
        """
        Args:
            bboxes (Tensor): Horizontal anchors, shape (N, 4).
            pred_bboxes (Tensor): Midpoint deltas, shape (N, 6).
        """
        assert pred_bboxes.size(0) == bboxes.size(0)
        assert bboxes.size(-1) == 4
        assert pred_bboxes.size(-1) == 6
        return self._delta2bbox(bboxes, pred_bboxes, max_shape=max_shape)

    def _bbox2delta(self, proposals, gt):
        proposals = proposals.float()
        gt = gt.float()

        px = (proposals[..., 0] + proposals[..., 2]) * 0.5
        py = (proposals[..., 1] + proposals[..., 3]) * 0.5
        pw = proposals[..., 2] - proposals[..., 0]
        ph = proposals[..., 3] - proposals[..., 1]

        hbb = obb2xyxy(gt, self.version)
        poly = obb2poly(gt, self.version)
        gx = (hbb[..., 0] + hbb[..., 2]) * 0.5
        gy = (hbb[..., 1] + hbb[..., 3]) * 0.5
        gw = hbb[..., 2] - hbb[..., 0]
        gh = hbb[..., 3] - hbb[..., 1]

        x_coor, y_coor = poly[:, 0::2], poly[:, 1::2]
        y_min, _ = torch.min(y_coor, dim=1, keepdim=True)
        x_max, _ = torch.max(x_coor, dim=1, keepdim=True)

        top_x = x_coor.clone()
        top_x[torch.abs(y_coor - y_min) > 0.1] = -1000
        ga, _ = torch.max(top_x, dim=1)

        right_y = y_coor.clone()
        right_y[torch.abs(x_coor - x_max) > 0.1] = -1000
        gb, _ = torch.max(right_y, dim=1)

        dx = (gx - px) / pw
        dy = (gy - py) / ph
        dw = torch.log(gw / pw)
        dh = torch.log(gh / ph)
        da = (ga - gx) / gw
        db = (gb - gy) / gh
        deltas = torch.stack([dx, dy, dw, dh, da, db], dim=-1)

        means = deltas.new_tensor(self.means).unsqueeze(0)
        stds = deltas.new_tensor(self.stds).unsqueeze(0)
        return deltas.sub_(means).div_(stds)

    def _delta2bbox(self, rois, deltas, max_shape=None):
        means = deltas.new_tensor(self.means).repeat(1, deltas.size(1) // 6)
        stds = deltas.new_tensor(self.stds).repeat(1, deltas.size(1) // 6)
        denorm_deltas = deltas * stds + means
        dx = denorm_deltas[:, 0::6]
        dy = denorm_deltas[:, 1::6]
        dw = denorm_deltas[:, 2::6]
        dh = denorm_deltas[:, 3::6]
        da = denorm_deltas[:, 4::6]
        db = denorm_deltas[:, 5::6]

        max_ratio = abs(math.log(self.wh_ratio_clip))
        dw = dw.clamp(min=-max_ratio, max=max_ratio)
        dh = dh.clamp(min=-max_ratio, max=max_ratio)

        px = ((rois[:, 0] + rois[:, 2]) * 0.5).unsqueeze(1).expand_as(dx)
        py = ((rois[:, 1] + rois[:, 3]) * 0.5).unsqueeze(1).expand_as(dy)
        pw = (rois[:, 2] - rois[:, 0]).unsqueeze(1).expand_as(dw)
        ph = (rois[:, 3] - rois[:, 1]).unsqueeze(1).expand_as(dh)

        gw = pw * dw.exp()
        gh = ph * dh.exp()
        gx = px + pw * dx
        gy = py + ph * dy

        x1 = gx - gw * 0.5
        y1 = gy - gh * 0.5
        x2 = gx + gw * 0.5
        y2 = gy + gh * 0.5

        da = da.clamp(min=-0.5, max=0.5)
        db = db.clamp(min=-0.5, max=0.5)
        ga = gx + da * gw
        ga_alt = gx - da * gw
        gb = gy + db * gh
        gb_alt = gy - db * gh
        polys = torch.stack([ga, y1, x2, gb, ga_alt, y2, x1, gb_alt], dim=-1)

        center = torch.stack([gx, gy, gx, gy, gx, gy, gx, gy], dim=-1)
        center_polys = polys - center
        diag_len = torch.sqrt(
            center_polys[..., 0::2] * center_polys[..., 0::2]
            + center_polys[..., 1::2] * center_polys[..., 1::2]
        )
        max_diag_len, _ = torch.max(diag_len, dim=-1, keepdim=True)
        diag_scale_factor = max_diag_len / diag_len
        center_polys = center_polys * diag_scale_factor.repeat_interleave(2, dim=-1)
        rectpolys = center_polys + center
        obboxes = poly2obb(rectpolys, self.version)

        if max_shape is not None and len(max_shape) >= 2:
            h, w = max_shape[:2]
            obboxes[:, 0].clamp_(min=0, max=w)
            obboxes[:, 1].clamp_(min=0, max=h)

        return obboxes
