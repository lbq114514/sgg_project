from ..layers.nms import batched_nms as _batched_nms


def batched_nms(boxes, scores, level_ids, iou_threshold=0.7, mode="hbb"):
    """
    Thin compatibility wrapper over ``sgg.modeling.layers.nms.batched_nms``.

    Args:
        boxes (Tensor):
            - ``mode="hbb"``: shape (N, 4) in ``xyxy`` format.
            - ``mode="obb"``: shape (N, 5) in ``xywha`` format.
        scores (Tensor): Scores of shape (N,).
        level_ids (Tensor): Feature-level ids or labels of shape (N,).
        iou_threshold (float): IoU threshold for NMS.
        mode (str): ``"hbb"`` or ``"obb"``.

    Returns:
        tuple[Tensor, Tensor]:
            - dets: ``(K, 5)`` for HBB or ``(K, 6)`` for OBB
            - keep: kept indices in the original input
    """
    return _batched_nms(
        boxes=boxes,
        scores=scores,
        idxs=level_ids,
        iou_threshold=iou_threshold,
        mode=mode,
    )
