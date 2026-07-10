import torch
from torch import nn

from sgg.engine.trainer import Trainer
from sgg.modeling.detectors.class_channel_order import (
    BACKGROUND_FIRST,
    BACKGROUND_LAST,
    DETECTOR_CLASSIFIER_KEYS,
    reorder_detector_classifier_state_dict,
)
from sgg.modeling.detectors.scene_graph_detector import _to_onehot_logits
from sgg.modeling.detectors.scene_graph_detector import SceneGraphDetector
from tools.eval_once import _legacy_rpcm_key_candidates, load_model_only_checkpoint
from tools.migrate_detector_class_order import migrate_checkpoint


def _classifier_state():
    state = {}
    for index, key in enumerate(sorted(DETECTOR_CLASSIFIER_KEYS)):
        if key.endswith("weight"):
            state[key] = torch.arange(49 * 3, dtype=torch.float32).reshape(49, 3) + 1000 * index
        else:
            state[key] = torch.arange(49, dtype=torch.float32) + 1000 * index
    state["unrelated.weight"] = torch.tensor([3.0])
    return state


def test_background_last_to_first_reorders_both_detector_branches():
    state = _classifier_state()
    original = {key: value.clone() for key, value in state.items()}
    changed = reorder_detector_classifier_state_dict(
        state,
        source_order=BACKGROUND_LAST,
        target_order=BACKGROUND_FIRST,
    )
    assert set(changed) == DETECTOR_CLASSIFIER_KEYS
    for key in DETECTOR_CLASSIFIER_KEYS:
        assert torch.equal(state[key][0], original[key][48])
        assert torch.equal(state[key][1:], original[key][:-1])
    assert torch.equal(state["unrelated.weight"], original["unrelated.weight"])


def test_real_star_detector_classifier_weights_follow_expected_permutation():
    checkpoint = torch.load("pretrained/OBB_swin_L_OBD.pth", map_location="cpu")
    source = checkpoint["state_dict"]
    state = {key: source[key].clone() for key in DETECTOR_CLASSIFIER_KEYS}
    changed = reorder_detector_classifier_state_dict(state, source_order=BACKGROUND_LAST)
    assert set(changed) == DETECTOR_CLASSIFIER_KEYS
    for key in DETECTOR_CLASSIFIER_KEYS:
        assert torch.equal(state[key][0], source[key][48])
        assert torch.equal(state[key][1:], source[key][:-1])


def test_scene_detector_pretrained_loader_applies_real_star_permutation(tmp_path):
    class LoadProbe:
        def __init__(self, source_state):
            self.pretrained_detector_class_order = BACKGROUND_LAST
            self.loaded = None
            self._state = {
                key: torch.zeros_like(value)
                for key, value in source_state.items()
            }

        def state_dict(self):
            return self._state

        def load_state_dict(self, state_dict, strict=False):
            self.loaded = dict(state_dict)
            return [], []

    raw_source = torch.load("pretrained/OBB_swin_L_OBD.pth", map_location="cpu")["state_dict"]
    source = {key: raw_source[key].clone() for key in DETECTOR_CLASSIFIER_KEYS}
    detector_path = tmp_path / "real_star_classifier_rows.pth"
    torch.save({"state_dict": source}, detector_path)
    probe = LoadProbe(source)
    SceneGraphDetector.load_detector_pretrained(probe, str(detector_path))
    assert set(probe.loaded) == DETECTOR_CLASSIFIER_KEYS
    for key in DETECTOR_CLASSIFIER_KEYS:
        assert torch.equal(probe.loaded[key][0], source[key][48])
        assert torch.equal(probe.loaded[key][1:], source[key][:-1])


def test_predcls_onehot_path_is_independent_of_detector_classifier_row_order():
    labels = torch.tensor([1, 7, 48])
    before = _to_onehot_logits(labels, num_classes=49)
    state = _classifier_state()
    reorder_detector_classifier_state_dict(state, source_order=BACKGROUND_LAST)
    after = _to_onehot_logits(labels, num_classes=49)
    assert torch.equal(before, after)


def test_legacy_classifier_key_has_current_detector_targets_for_reordering():
    source_key = "roi_heads.box.predictor.cls_score.weight"
    targets = _legacy_rpcm_key_candidates(source_key)
    assert "roi_head.bbox_head.fc_cls.weight" in targets
    assert "roi_head_d2.bbox_head.fc_cls.weight" in targets


class _LegacyLoadProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.roi_head = nn.Module()
        self.roi_head.bbox_head = nn.Module()
        self.roi_head.bbox_head.fc_cls = nn.Linear(3, 49)
        self.roi_heads = nn.Module()
        self.roi_heads.relation = nn.Module()
        self.roi_heads.relation.predictor = nn.Module()
        self.roi_heads.relation.predictor.register_parameter("dummy", nn.Parameter(torch.zeros(1)))


class _TrainerProbe:
    def __init__(self, model):
        self.model = model
        self.device = "cpu"

    def _legacy_rpcm_key_candidates(self, key):
        return Trainer._legacy_rpcm_key_candidates(key)


def _legacy_source_for_probe(model):
    source = {
        "roi_head.bbox_head.fc_cls.weight": torch.arange(49 * 3, dtype=torch.float32).reshape(49, 3),
        "roi_head.bbox_head.fc_cls.bias": torch.arange(49, dtype=torch.float32),
        "roi_heads.relation.predictor.dummy": torch.ones(1),
    }
    assert set(source).issubset(model.state_dict())
    return source


def test_legacy_direct_mmrotate_classifier_key_is_not_loaded_without_reordering(tmp_path):
    model = _LegacyLoadProbe()
    source = _legacy_source_for_probe(model)
    checkpoint = tmp_path / "legacy_direct_mmrotate.pth"
    torch.save({"state_dict": source}, checkpoint)

    load_model_only_checkpoint(_TrainerProbe(model), str(checkpoint), legacy_rpcm=True)
    loaded = model.state_dict()
    assert torch.equal(loaded["roi_head.bbox_head.fc_cls.weight"][0], source["roi_head.bbox_head.fc_cls.weight"][48])
    assert torch.equal(loaded["roi_head.bbox_head.fc_cls.weight"][1:], source["roi_head.bbox_head.fc_cls.weight"][:-1])
    assert torch.equal(loaded["roi_head.bbox_head.fc_cls.bias"][0], source["roi_head.bbox_head.fc_cls.bias"][48])

    model_for_trainer = _LegacyLoadProbe()
    probe = _TrainerProbe(model_for_trainer)
    Trainer._load_legacy_rpcm_model_only(probe, source, str(checkpoint))
    loaded = model_for_trainer.state_dict()
    assert torch.equal(loaded["roi_head.bbox_head.fc_cls.weight"][0], source["roi_head.bbox_head.fc_cls.weight"][48])
    assert torch.equal(loaded["roi_head.bbox_head.fc_cls.bias"][1:], source["roi_head.bbox_head.fc_cls.bias"][:-1])


def test_explicit_checkpoint_migration_marks_output_and_drops_training_state(tmp_path):
    source_path = tmp_path / "legacy.pth"
    output_path = tmp_path / "migrated.pth"
    original_state = _classifier_state()
    torch.save(
        {
            "model": {key: value.clone() for key, value in original_state.items()},
            "optimizer": {"state": {1: {"momentum_buffer": torch.ones(1)}}, "param_groups": []},
            "scheduler": {"last_epoch": 3},
        },
        source_path,
    )

    report = migrate_checkpoint(source_path, output_path, source_order=BACKGROUND_LAST)
    migrated = torch.load(output_path, map_location="cpu")
    untouched = torch.load(source_path, map_location="cpu")
    assert report["optimizer_scheduler_dropped"] is True
    assert migrated["detector_class_channel_order"] == BACKGROUND_FIRST
    assert migrated["optimizer"] is None
    assert migrated["scheduler"] is None
    for key in DETECTOR_CLASSIFIER_KEYS:
        assert torch.equal(migrated["model"][key][0], original_state[key][48])
        assert torch.equal(untouched["model"][key], original_state[key])

    try:
        migrate_checkpoint(output_path, tmp_path / "twice.pth", source_order=BACKGROUND_LAST)
    except ValueError as error:
        assert "already declares detector_class_channel_order" in str(error)
    else:
        raise AssertionError("Expected a second migration to be rejected.")
