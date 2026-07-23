import importlib.util

import torch

from sgg.config.defaults import get_default_cfg
from sgg.modeling.core.obb_ops import set_boxlist_angle_unit
from sgg.modeling.detectors.scene_graph_detector import SceneGraphDetector
from sgg.modeling.detectors.sgdet_detection_cache import original_sgdet_score_keep
from sgg.structures.boxes import BoxList


def _synthetic_sgdet_cfg():
    cfg = get_default_cfg()
    cfg["MODEL"]["TASK"] = "sgdet"
    cfg["MODEL"]["BOX_MODE"] = "obb"
    cfg["MODEL"]["OBB_ANGLE_UNIT"] = "radian"
    cfg["MODEL"]["USE_COMBINED_ROI_HEADS"] = False
    cfg["MODEL"]["BACKBONE"] = {"NAME": "simple_cnn"}
    cfg["MODEL"]["NECK"] = {"NAME": ""}
    cfg["MODEL"]["ROI_EXTRACTOR"].update(
        {
            "NAME": "rotated_roi_extractor",
            "OUT_CHANNELS": 256,
            "FEATURE_KEY": "p2",
            "FEATMAP_NAMES": ["p2"],
            "FEATMAP_STRIDES": [4],
            "POOL_SIZE": 7,
            "FEATURE_STRIDE": 4,
            "ANGLE_VERSION": "le90",
            "SPATIAL_SCALE": 0.25,
        }
    )
    cfg["MODEL"]["RPN_HEAD"] = {
        "NAME": "oriented_rpn_head",
        "IN_CHANNELS": 256,
        "FEAT_CHANNELS": 256,
        "NUM_ANCHORS": 3,
        "CLS_OUT_CHANNELS": 1,
        "VERSION": "le90",
        "USE_SIGMOID_CLS": True,
    }
    cfg["MODEL"]["ROI_BOX_HEAD"]["MLP_HEAD_DIM"] = 128
    cfg["MODEL"]["ROI_BOX_HEAD"]["NUM_CLASSES"] = 49
    cfg["MODEL"]["NUM_CLASSES"] = 49
    cfg["MODEL"]["SGDET_COMPAT"].update(
        {
            "ENABLED": True,
            "USE_D2": False,
            "RPN_ANCHOR_SIZES": [32],
            "RCNN_SCORE_THRESH": 0.001,
            "RCNN_MAX_PER_IMG": 10,
            "ADD_GTBOX_TO_PROPOSAL_IN_TRAIN": True,
        }
    )
    return cfg


def test_sgdet_compat_uses_multilevel_midpoint_rpn_and_preserves_object_logits():
    model = SceneGraphDetector(_synthetic_sgdet_cfg()).eval()
    with torch.inference_mode():
        image = torch.randn(1, 3, 64, 64)
        features = model._extract_features(image)
        proposals = model._sgdet_rpn_proposals(features, (64, 64))
        detections = model._sgdet_roi_detections(features, proposals)

    assert model.rpn_head.anchor_generator is model.sgdet_rpn_anchor_generator
    assert model.rpn_head.bbox_coder is model.sgdet_rpn_bbox_coder
    assert proposals[0].bbox.size(1) == 5
    assert detections[0].get_field("predict_logits").shape[1] == 49
    assert detections[0].get_field("boxes_per_cls").shape[1:] == (49, 5)
    assert detections[0].has_field("detector_nms_labels")
    assert detections[0].has_field("detector_nms_scores")
    assert tuple(model.roi_head.bbox_head.bbox_coder.stds) == (
        0.1,
        0.1,
        0.2,
        0.2,
        0.1,
    )


def test_sgdet_compat_uses_original_zero_center_anchor_offset():
    model = SceneGraphDetector(_synthetic_sgdet_cfg()).eval()
    generator = model.sgdet_rpn_anchor_generator
    assert generator.offset == 0.0

    # The first source mmdet anchor is centered on feature-grid coordinate
    # (0, 0), rather than the half-stride coordinate (2, 2) formerly used by
    # this project at P2/stride 4.
    anchors = generator.grid_priors([(1, 1)], device=torch.device("cpu"))[0]
    centers = 0.5 * (anchors[:, :2] + anchors[:, 2:])
    assert torch.allclose(centers, torch.zeros_like(centers))


def test_sgdet_compat_injects_gt_boxes_and_attaches_match_labels():
    model = SceneGraphDetector(_synthetic_sgdet_cfg()).train()
    image = torch.randn(1, 3, 64, 64)
    target = BoxList(torch.tensor([[32.0, 32.0, 16.0, 8.0, 0.0]]), (64, 64), "xywha")
    set_boxlist_angle_unit(target, "radian")
    target.add_field("labels", torch.tensor([5], dtype=torch.long))
    target.add_field("pair_labels", torch.zeros((1, 1), dtype=torch.long))
    target.add_field("relation_triplets", torch.zeros((0, 3), dtype=torch.long))

    with torch.no_grad():
        features = model._extract_features(image)
    proposals, losses = model._prepare_train_proposals(features, image, [target])
    proposal = proposals[0]
    assert proposal.get_field("is_gt_injected").any()
    assert int(proposal.get_field("labels")[-1]) == 5
    assert all(float(value) == 0.0 for value in losses.values())


def test_sgdet_large_image_patch_schedule_matches_original_d1_d2_roles():
    model = SceneGraphDetector(_synthetic_sgdet_cfg()).eval()
    model.sgdet_use_d2 = True
    model.sgdet_d2_scale = 0.5
    model.patch_auto_min_size = 16
    model.patch_max_pyramid_layers = 4
    calls = []

    def fake_detect(image, original_size, *, use_d2=False, full_resolution_image=None):
        calls.append((tuple(image.shape[-2:]), original_size, use_d2))
        empty = BoxList(image.new_zeros((0, 5)), original_size, "xywha")
        set_boxlist_angle_unit(empty, "radian")
        return empty

    model._detect_single_scale_patches = fake_detect
    result = model._detect_single_image_multiscale(torch.randn(3, 64, 64))

    # Source ``batch`` omits 64x64, then runs d1 at 32x32 and d2 at 16x16.
    assert calls == [((32, 32), (64, 64), False), ((16, 16), (64, 64), True)]
    assert len(result) == 0


def test_sgdet_patch_relocation_transforms_boxes_per_class_with_main_bbox():
    model = SceneGraphDetector(_synthetic_sgdet_cfg()).eval()
    bbox = torch.tensor([[10.0, 20.0, 8.0, 4.0, 0.25]])
    detection = BoxList(bbox.clone(), (64, 64), "xywha")
    set_boxlist_angle_unit(detection, "radian")
    detection.add_field(
        "boxes_per_cls",
        bbox[:, None, :].expand(-1, 49, -1).clone(),
    )

    relocated = model._relocate_patch_detection(
        detection,
        {"x0": 32, "y0": 16},
        scaled_size=(128, 128),
        original_size=(256, 256),
    )
    expected = torch.tensor([84.0, 72.0, 16.0, 8.0, 0.25])
    assert torch.allclose(relocated.bbox[0], expected)
    assert torch.allclose(relocated.get_field("boxes_per_cls")[0, 0], expected)
    assert torch.allclose(
        relocated.get_field("boxes_per_cls"),
        relocated.bbox[:, None, :].expand(-1, 49, -1),
    )


def test_sgdet_scale2_trigger_runs_four_original_resolution_d1_patches():
    model = SceneGraphDetector(_synthetic_sgdet_cfg()).eval()
    model.patch_size = (16, 16)
    model.patch_batch_size_large = 4
    branch_calls = []

    model._extract_features = lambda batch: {"p2": batch}

    def fake_branch(features, image_hw, device, *, use_d2):
        branch_calls.append((features["p2"].shape[0], use_d2))
        outputs = []
        for _ in range(features["p2"].shape[0]):
            result = model._empty_sgdet_boxlist((image_hw[1], image_hw[0]), device)
            outputs.append(result)
        return outputs

    model._detect_sgdet_detector_branch = fake_branch
    outputs = model._detect_scale2_full_resolution_refinements(
        torch.randn(3, 64, 64),
        [{"x0": 0, "y0": 0, "width": 16, "height": 16}],
        scaled_size=(32, 32),
        original_size=(64, 64),
    )
    assert len(outputs) == 4
    assert branch_calls == [(4, False)]


def test_cross_patch_nms_uses_detector_bucket_not_relation_argmax_label():
    model = SceneGraphDetector(_synthetic_sgdet_cfg()).eval()
    boxes = torch.tensor(
        [[16.0, 16.0, 12.0, 6.0, 0.0], [16.0, 16.0, 12.0, 6.0, 0.0]]
    )
    detection = BoxList(boxes, (32, 32), "xywha")
    set_boxlist_angle_unit(detection, "radian")
    logits = torch.zeros((2, 49))
    logits[:, 1] = 5.0  # relation-view argmax is identical for both boxes
    detection.add_field("predict_logits", logits)
    detection.add_field("pred_labels", torch.tensor([1, 1]))
    detection.add_field("pred_scores", torch.tensor([0.9, 0.8]))
    detection.add_field("detector_nms_labels", torch.tensor([0, 1]))
    detection.add_field("detector_nms_scores", torch.tensor([0.9, 0.8]))
    detection.add_field("boxes_per_cls", boxes[:, None, :].expand(-1, 49, -1).clone())

    merged = model._merge_patch_detections([detection], (32, 32))
    assert len(merged) == 2
    assert torch.equal(merged.get_field("detector_nms_labels"), torch.tensor([0, 1]))
    assert torch.equal(merged.get_field("pred_labels"), torch.tensor([1, 1]))


def test_sgdet_cross_patch_merge_does_not_apply_second_global_detection_cap():
    model = SceneGraphDetector(_synthetic_sgdet_cfg()).eval()
    model.sgdet_rcnn_max_per_img = 3
    boxes = torch.tensor(
        [
            [5.0, 5.0, 2.0, 2.0, 0.0],
            [15.0, 5.0, 2.0, 2.0, 0.0],
            [25.0, 5.0, 2.0, 2.0, 0.0],
            [35.0, 5.0, 2.0, 2.0, 0.0],
            [45.0, 5.0, 2.0, 2.0, 0.0],
            [55.0, 5.0, 2.0, 2.0, 0.0],
        ]
    )
    detection = BoxList(boxes, (64, 64), "xywha")
    set_boxlist_angle_unit(detection, "radian")
    scores = torch.linspace(0.99, 0.90, len(boxes))
    labels = torch.ones((len(boxes),), dtype=torch.long)
    logits = torch.zeros((len(boxes), 49))
    detection.add_field("pred_scores", scores)
    detection.add_field("scores", scores)
    detection.add_field("pred_labels", labels)
    detection.add_field("predict_logits", logits)
    detection.add_field("boxes_per_cls", boxes[:, None, :].expand(-1, 49, -1).clone())

    merged = model._merge_patch_detections([detection], (64, 64))
    assert len(merged) == len(boxes)
    assert len(merged) > model.sgdet_rcnn_max_per_img


def test_original_sgdet_score_filter_uses_high_threshold_then_class_fallback():
    scores = torch.tensor([0.9, 0.4, 0.25, 0.21, 0.05])
    labels = torch.tensor([1, 1, 2, 2, 3])
    thresholds = [0.3, 0.2, 0.1, 0.001, 0.00001]

    # At 0.3 only class 1 survives, so the source route falls back to 0.2,
    # where classes 1 and 2 are both present.
    keep = original_sgdet_score_keep(scores, labels, thresholds)
    assert torch.equal(keep, torch.tensor([0, 1, 2, 3]))


def test_original_sgdet_score_filter_is_only_applied_to_final_merge():
    model = SceneGraphDetector(_synthetic_sgdet_cfg()).eval()
    boxes = torch.tensor(
        [
            [5.0, 5.0, 2.0, 2.0, 0.0],
            [15.0, 5.0, 2.0, 2.0, 0.0],
            [25.0, 5.0, 2.0, 2.0, 0.0],
        ]
    )
    detection = BoxList(boxes, (32, 32), "xywha")
    set_boxlist_angle_unit(detection, "radian")
    scores = torch.tensor([0.8, 0.25, 0.15])
    labels = torch.tensor([1, 2, 2])
    logits = torch.zeros((3, 49))
    detection.add_field("pred_scores", scores)
    detection.add_field("scores", scores)
    detection.add_field("pred_labels", labels)
    detection.add_field("predict_logits", logits)
    detection.add_field("boxes_per_cls", boxes[:, None, :].expand(-1, 49, -1).clone())

    per_scale = model._merge_patch_detections([detection], (32, 32))
    final = model._merge_patch_detections(
        [detection], (32, 32), apply_original_score_filter=True
    )
    assert len(per_scale) == 3
    assert len(final) == 2
    assert torch.all(final.get_field("detector_nms_scores") >= 0.2)


def test_sgdet_raw_detector_boxes_are_mapped_to_relation_view_coordinates():
    model = SceneGraphDetector(_synthetic_sgdet_cfg()).eval()
    raw_target = BoxList(torch.tensor([[32.0, 32.0, 16.0, 8.0, 0.0]]), (64, 64), "xywha")
    relation_target = BoxList(
        torch.tensor([[16.0, 16.0, 8.0, 4.0, 0.0]]), (32, 32), "xywha"
    )
    set_boxlist_angle_unit(raw_target, "radian")
    set_boxlist_angle_unit(relation_target, "radian")

    def fake_branch(features, image_hw, device, *, use_d2):
        result = BoxList(torch.tensor([[32.0, 32.0, 16.0, 8.0, 0.0]]), (64, 64), "xywha")
        set_boxlist_angle_unit(result, "radian")
        result.add_field("predict_logits", torch.zeros((1, 49)))
        result.add_field("pred_logits", result.get_field("predict_logits"))
        result.add_field("pred_labels", torch.tensor([5]))
        result.add_field("pred_scores", torch.tensor([0.9]))
        result.add_field("scores", torch.tensor([0.9]))
        result.add_field("labels", torch.tensor([5]))
        result.add_field("boxes_per_cls", result.bbox[:, None, :].expand(-1, 49, -1).clone())
        result.add_field("pair_labels", torch.zeros((1, 1), dtype=torch.long))
        result.add_field("relation_triplets", torch.zeros((0, 3), dtype=torch.long))
        return [result]

    model._detect_sgdet_detector_branch = fake_branch
    outputs = model._detect_sgdet_raw_to_relation_view(
        torch.randn(1, 3, 64, 64),
        [raw_target],
        [relation_target],
    )
    output = outputs[0]
    assert output.size == (32, 32)
    assert torch.allclose(output.bbox[0, :4], torch.tensor([16.0, 16.0, 8.0, 4.0]))
    assert torch.allclose(
        output.get_field("boxes_per_cls")[0, 0, :4],
        torch.tensor([16.0, 16.0, 8.0, 4.0]),
    )


def test_task_configs_match_toolkit_task_protocols():
    expected = {
        "configs/star_sgcls_obb_train.py": ("sgcls", True, False),
        "configs/star_sgdet_obb_train.py": ("sgdet", False, False),
    }
    for path, values in expected.items():
        spec = importlib.util.spec_from_file_location("task_cfg", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        cfg = module.cfg
        task, use_gt_box, use_gt_label = values
        rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
        assert cfg["MODEL"]["TASK"] == task
        assert rel_cfg["USE_GT_BOX"] is use_gt_box
        assert rel_cfg["USE_GT_OBJECT_LABEL"] is use_gt_label
        assert rel_cfg["TEST_FILTER_METHOD"] in {"PPG", "PPN", "RSGP"}
        # Batch size and iteration schedule are experiment/runtime knobs, not
        # part of the SGG-Toolkit task semantics.
        assert cfg["DATALOADER"]["TRAIN_BATCH_SIZE"] > 0
        assert cfg["SOLVER"]["MAX_ITER"] > 0


def test_star_sgdet_defaults_to_original_detected_proposal_gt_protocol():
    spec = importlib.util.spec_from_file_location(
        "sgdet_protocol_cfg", "configs/star_sgdet_obb_train.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sgdet_cfg = module.cfg["MODEL"]["SGDET_COMPAT"]
    assert sgdet_cfg["ADD_GTBOX_TO_PROPOSAL_IN_TRAIN"] is False
    assert sgdet_cfg["TRAIN_LABEL_SOURCE"] == "matched_gt"
    assert sgdet_cfg["EVAL_LABEL_SOURCE"] == "matched_gt"
