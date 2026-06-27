import torch

from .boxlist import BoxList

try:
    from mmcv.ops import box_iou_rotated, nms_rotated
except Exception:
    box_iou_rotated = None
    nms_rotated = None

try:
    from mmrotate.core import obb2xyxy
except Exception:
    obb2xyxy = None


def _nms_hbb(boxes: torch.Tensor, scores: torch.Tensor, nms_thresh: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        current = order[0]
        keep.append(int(current.item()))
        if order.numel() == 1:
            break
        cur_box = boxes[current : current + 1]
        other = boxes[order[1:]]
        lt = torch.max(cur_box[:, None, :2], other[None, :, :2])
        rb = torch.min(cur_box[:, None, 2:], other[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        inter = wh[..., 0] * wh[..., 1]
        area1 = ((cur_box[:, 2] - cur_box[:, 0]).clamp(min=0) * (cur_box[:, 3] - cur_box[:, 1]).clamp(min=0))
        area2 = ((other[:, 2] - other[:, 0]).clamp(min=0) * (other[:, 3] - other[:, 1]).clamp(min=0))
        union = area1[:, None] + area2[None, :] - inter + 1e-6
        iou = (inter / union).squeeze(0)
        order = order[1:][iou <= nms_thresh]
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def _assert_same_image_size(boxlist1, boxlist2):
    if boxlist1.size != boxlist2.size:
        raise ValueError("BoxLists should have the same image size")


def boxlist_iou(boxlist1, boxlist2, mode="auto"):
    """
    Compute IoU between two BoxLists.

    Args:
        boxlist1 (BoxList)
        boxlist2 (BoxList)
        mode (str):
            "auto", "hbb", or "obb"

    Returns:
        Tensor:
            IoU matrix of shape (N, M)
    """
    _assert_same_image_size(boxlist1, boxlist2)

    if mode == "auto":
        mode = "obb" if (boxlist1.mode == "xywha" and boxlist2.mode == "xywha") else "hbb"

    if mode == "hbb":
        b1 = boxlist1.convert("xyxy").bbox
        b2 = boxlist2.convert("xyxy").bbox

        area1 = boxlist1.convert("xyxy").area()
        area2 = boxlist2.convert("xyxy").area()

        lt = torch.max(b1[:, None, :2], b2[:, :2])
        rb = torch.min(b1[:, None, 2:], b2[:, 2:])
        wh = (rb - lt).clamp(min=0)

        inter = wh[:, :, 0] * wh[:, :, 1]
        union = area1[:, None] + area2 - inter
        return inter / union.clamp(min=1e-6)

    elif mode == "obb":
        if box_iou_rotated is None:
            raise ImportError(
                "mmcv.ops.box_iou_rotated is required for OBB IoU"
            )
        if boxlist1.mode != "xywha" or boxlist2.mode != "xywha":
            raise ValueError("OBB IoU requires both BoxLists in 'xywha' mode")

        return box_iou_rotated(boxlist1.bbox, boxlist2.bbox)

    else:
        raise ValueError(f"Unsupported mode: {mode}")


def boxlist_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores", mode="auto"):
    """
    Apply NMS to a BoxList.

    Args:
        boxlist (BoxList)
        nms_thresh (float)
        max_proposals (int)
        score_field (str)
        mode (str):
            "auto", "hbb", or "obb"
    """
    if len(boxlist) == 0:
        return boxlist

    scores = boxlist.get_field(score_field)

    if mode == "auto":
        mode = "obb" if boxlist.mode == "xywha" else "hbb"

    if mode == "hbb":
        boxes = boxlist.convert("xyxy").bbox
        keep = _nms_hbb(boxes, scores, nms_thresh)

    elif mode == "obb":
        if nms_rotated is None:
            raise ImportError(
                "mmcv.ops.nms_rotated is required for OBB NMS"
            )
        if boxlist.mode != "xywha":
            raise ValueError("OBB NMS requires BoxList in 'xywha' mode")

        dets, keep = nms_rotated(boxlist.bbox, scores, nms_thresh)

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if max_proposals > 0:
        keep = keep[:max_proposals]

    return boxlist[keep]


def remove_small_boxes(boxlist, min_size, mode="auto"):
    """
    Remove boxes with either width or height smaller than min_size.
    """
    if len(boxlist) == 0:
        return boxlist

    if mode == "auto":
        mode = "obb" if boxlist.mode == "xywha" else "hbb"

    if mode == "hbb":
        boxes = boxlist.convert("xyxy").bbox
        ws = boxes[:, 2] - boxes[:, 0]
        hs = boxes[:, 3] - boxes[:, 1]

    elif mode == "obb":
        if boxlist.mode != "xywha":
            raise ValueError("OBB mode requires BoxList in 'xywha' mode")
        _, _, ws, hs, _ = boxlist.bbox.unbind(dim=1)

    else:
        raise ValueError(f"Unsupported mode: {mode}")

    keep = (ws >= min_size) & (hs >= min_size)
    return boxlist[keep]


def cat_boxlist(boxlists):
    """
    Concatenate a list of BoxLists with the same size and mode.
    """
    if len(boxlists) == 0:
        raise ValueError("cat_boxlist expects a non-empty list")

    size = boxlists[0].size
    mode = boxlists[0].mode

    for boxlist in boxlists:
        if boxlist.size != size:
            raise ValueError("All BoxLists must have the same image size")
        if boxlist.mode != mode:
            raise ValueError("All BoxLists must have the same mode")

    bbox = torch.cat([b.bbox for b in boxlists], dim=0)
    cat_boxes = BoxList(bbox, size, mode)

    fields = boxlists[0].fields()
    for field in fields:
        data = [b.get_field(field) for b in boxlists]
        if torch.is_tensor(data[0]):
            cat_data = torch.cat(data, dim=0)
        else:
            cat_data = sum(data, [])
        cat_boxes.add_field(field, cat_data)

    return cat_boxes


def obb_to_hbb(boxlist, version="oc"):
    """
    Convert an OBB BoxList (xywha) into an HBB BoxList (xyxy).
    """
    if boxlist.mode != "xywha":
        raise ValueError("obb_to_hbb expects BoxList in 'xywha' mode")
    if obb2xyxy is None:
        raise ImportError("mmrotate.core.obb2xyxy is required for OBB->HBB conversion")

    hbb = obb2xyxy(boxlist.bbox, version)
    new_boxlist = BoxList(hbb, boxlist.size, mode="xyxy")

    for k, v in boxlist.extra_fields.items():
        new_boxlist.add_field(k, v)

    return new_boxlist
