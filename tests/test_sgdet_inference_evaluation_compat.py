import torch
import pytest

from sgg.config.defaults import get_default_cfg
from sgg.evaluation.sgg_eval import _PairAccuracyMetric, _pairwise_iou
from sgg.modeling.core.obb_ops import set_boxlist_angle_unit
from sgg.modeling.roi_heads.relation_inference import obj_prediction_nms
from sgg.modeling.detectors.sgdet_detection_cache import (
    CACHE_VERSION,
    load_sgdet_detection,
    serialize_sgdet_detection,
)
from sgg.structures.boxes import BoxList


def test_late_object_nms_assigns_overlapping_boxes_to_next_best_class():
    proposal = BoxList(
        torch.tensor(
            [
                [16.0, 16.0, 12.0, 6.0, 0.0],
                [16.0, 16.0, 12.0, 6.0, 0.0],
            ]
        ),
        (32, 32),
        "xywha",
    )
    set_boxlist_angle_unit(proposal, "radian")
    boxes_per_cls = proposal.bbox[:, None, :].expand(-1, 3, -1).clone()
    logits = torch.tensor(
        [
            [-5.0, 5.0, 3.0],
            [-5.0, 4.0, 3.5],
        ]
    )

    labels = obj_prediction_nms(proposal, boxes_per_cls, logits, nms_thresh=0.3)
    assert torch.equal(labels, torch.tensor([1, 2]))


def test_rotated_iou_respects_radian_metadata():
    angle = torch.tensor(torch.pi / 4)
    boxes1 = torch.tensor([[0.0, 0.0, 10.0, 2.0, angle]])
    boxes2 = torch.tensor([[0.0, 0.0, 10.0, 2.0, -angle]])

    radian_iou = float(_pairwise_iou(boxes1, boxes2, "radian", "radian")[0, 0])
    incorrectly_degree_iou = float(_pairwise_iou(boxes1, boxes2, "degree", "degree")[0, 0])
    assert radian_iou < 0.3
    assert incorrectly_degree_iou > 0.8


def test_pair_accuracy_aggregates_hits_and_counts_across_images():
    metric = _PairAccuracyMetric((1000,))
    pair = torch.tensor([[0, 1]], dtype=torch.long)
    metric.add("sgcls", [[0]], pair, {(0, 1)}, gt_rel_count=1)
    metric.add("sgcls", [[]], pair, {(0, 1)}, gt_rel_count=9)
    assert metric.finalize()[1000] == 0.1


def test_default_relation_late_nms_threshold_matches_toolkit():
    cfg = get_default_cfg()
    assert cfg["TEST"]["RELATION"]["LATER_NMS_PREDICTION_THRES"] == 0.3


def test_cache_v5_preserves_detector_nms_fields_and_rejects_v4(tmp_path):
    cfg = get_default_cfg()
    cfg["MODEL"]["TASK"] = "sgdet"
    proposal = BoxList(
        torch.tensor([[8.0, 8.0, 4.0, 2.0, 0.0]]), (16, 16), "xywha"
    )
    set_boxlist_angle_unit(proposal, "radian")
    proposal.add_field("pred_labels", torch.tensor([1]))
    proposal.add_field("pred_scores", torch.tensor([0.8]))
    proposal.add_field("detector_nms_labels", torch.tensor([0]))
    proposal.add_field("detector_nms_scores", torch.tensor([0.9]))
    payload = serialize_sgdet_detection(
        proposal,
        cfg=cfg,
        split="test",
        image_id=1,
        raw_size=(16, 16),
        relation_size=(16, 16),
    )
    assert payload["cache_version"] == CACHE_VERSION == 5
    assert "detector_nms_labels" in payload["fields"]
    assert "detector_nms_scores" in payload["fields"]

    payload["cache_version"] = 4
    path = tmp_path / "old.pt"
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="Unsupported sgdet detection cache version"):
        load_sgdet_detection(path, cfg=cfg, device="cpu")
