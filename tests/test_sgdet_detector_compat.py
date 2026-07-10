import importlib.util

import torch

from sgg.config.defaults import get_default_cfg
from sgg.modeling.core.obb_ops import set_boxlist_angle_unit
from sgg.modeling.detectors.scene_graph_detector import SceneGraphDetector
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

    def fake_detect(image, original_size, *, use_d2=False):
        calls.append((tuple(image.shape[-2:]), original_size, use_d2))
        empty = BoxList(image.new_zeros((0, 5)), original_size, "xywha")
        set_boxlist_angle_unit(empty, "radian")
        return empty

    model._detect_single_scale_patches = fake_detect
    result = model._detect_single_image_multiscale(torch.randn(3, 64, 64))

    # Source ``batch`` omits 64x64, then runs d1 at 32x32 and d2 at 16x16.
    assert calls == [((32, 32), (64, 64), False), ((16, 16), (64, 64), True)]
    assert len(result) == 0


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


def test_task_configs_match_toolkit_task_and_optimizer_contracts():
    expected = {
        "configs/star_sgcls_obb_train.py": ("sgcls", True, False, 4, 10000, [6000, 8500]),
        "configs/star_sgdet_obb_train.py": ("sgdet", False, False, 2, 5000, [3000, 4000]),
    }
    for path, values in expected.items():
        spec = importlib.util.spec_from_file_location("task_cfg", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        cfg = module.cfg
        task, use_gt_box, use_gt_label, batch_size, max_iter, steps = values
        rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
        assert cfg["MODEL"]["TASK"] == task
        assert rel_cfg["USE_GT_BOX"] is use_gt_box
        assert rel_cfg["USE_GT_OBJECT_LABEL"] is use_gt_label
        assert rel_cfg["TEST_FILTER_METHOD"] in {"PPG", "PPN", "RSGP"}
        assert cfg["DATALOADER"]["TRAIN_BATCH_SIZE"] == batch_size
        assert cfg["SOLVER"]["MAX_ITER"] == max_iter
        assert cfg["SOLVER"]["STEPS"] == steps
