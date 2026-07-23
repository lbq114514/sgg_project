from __future__ import annotations

import torch
import torch.nn.functional as F

from sgg.modeling.roi_heads.roi_relation_predictors import (
    LegacyPredicatePrototypeHead,
    RPCMLegacy,
    _RPCM_6850_ANTONYM_ID_PAIRS,
)


def _make_exact_head(
    glove: torch.Tensor,
    *,
    ant_pairs=(),
    lambda_ant: float = 0.1,
    ant_margin: float = -0.2,
) -> LegacyPredicatePrototypeHead:
    return LegacyPredicatePrototypeHead(
        d_in=glove.size(1),
        d=glove.size(1),
        glove_init=glove,
        use_proj=False,
        ema_alpha=0.9,
        lambda_pull=0.2,
        lambda_sep=0.01,
        num_proto_per_cls=1,
        use_vis_proto=False,
        checkpoint_compat_2d=True,
        exact_6850=True,
        ant_pairs=ant_pairs,
        lambda_ant=lambda_ant,
        ant_margin=ant_margin,
    )


def test_exact_6850_copies_glove_without_random_perturbation():
    glove = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype=torch.float32,
    )
    torch.manual_seed(1)
    head_a = _make_exact_head(glove)
    torch.manual_seed(999)
    head_b = _make_exact_head(glove)

    torch.testing.assert_close(head_a.base_prototypes, glove, rtol=0.0, atol=0.0)
    torch.testing.assert_close(head_b.base_prototypes, glove, rtol=0.0, atol=0.0)
    assert head_a.base_prototypes.data_ptr() != glove.data_ptr()
    assert head_a.base_prototypes.shape == (3, 3)


def test_exact_6850_proto_ema_is_static_during_training():
    glove = torch.eye(3, dtype=torch.float32)
    head = _make_exact_head(glove)
    head.train()
    before = head.proto_ema.clone()

    features = F.normalize(
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]]), dim=1
    )
    labels = torch.tensor([0, 1], dtype=torch.long)
    logits, losses = head(features, labels)

    assert logits.shape == (2, 3)
    assert set(losses) == {"pull", "sep"}
    torch.testing.assert_close(head.proto_ema, before, rtol=0.0, atol=0.0)


def test_exact_6850_antonym_loss_preserves_historic_margin_expression():
    glove = torch.tensor(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]], dtype=torch.float32
    )
    head = _make_exact_head(
        glove,
        ant_pairs=[(0, 1)],
        lambda_ant=0.1,
        ant_margin=-0.2,
    )
    with torch.no_grad():
        head.mapper.weight.copy_(torch.eye(2))
        head.proto_ema.copy_(F.normalize(head.base_prototypes, dim=1))
    head.train()

    _, losses = head(torch.tensor([[1.0, 0.0]]), torch.tensor([0]))

    # cos(p_0, p_1)=-1, so the historical formula is
    # 0.1 * relu(-0.2 - (-1.0)) = 0.08.
    torch.testing.assert_close(losses["ant"], torch.tensor(0.08))


def test_exact_6850_adds_no_checkpoint_state():
    glove = torch.eye(3, dtype=torch.float32)
    exact = _make_exact_head(glove, ant_pairs=_RPCM_6850_ANTONYM_ID_PAIRS)
    compatible_later = LegacyPredicatePrototypeHead(
        d_in=3,
        d=3,
        glove_init=glove,
        use_proj=False,
        ema_alpha=0.9,
        num_proto_per_cls=1,
        use_vis_proto=False,
        checkpoint_compat_2d=True,
        exact_6850=False,
    )

    expected_keys = {"tau", "base_prototypes", "mapper.weight", "proto_ema"}
    assert set(exact.state_dict()) == expected_keys
    assert set(compatible_later.state_dict()) == expected_keys


def test_main_predcls_config_selects_exact_6850_relation_base():
    from configs.star_predcls_obb_train import cfg

    rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
    assert rel_cfg["PREDICTOR"] == "RPCM_ORIGINAL_LEGACY"
    assert rel_cfg["RPCM_RELATION_GRAPH_MODE"] == "dual_view"
    assert rel_cfg["RPCM_GLOVE_INIT_MODE"] == "rpcm"
    assert rel_cfg["RPCM_LEGACY_NUM_PROTO"] == 1
    assert rel_cfg["RPCM_LEGACY_USE_VIS_PROTO"] is False
    assert rel_cfg["RPCM_LEGACY_PROTO_2D_COMPAT"] is True
    assert rel_cfg["RPCM_LEGACY_6850_EXACT"] is True
    assert rel_cfg["RPCM_LEGACY_ANT_LOSS_WEIGHT"] == 0.1
    assert rel_cfg["RPCM_LEGACY_ANT_MARGIN"] == -0.2


def test_hprc_scratch_config_uses_corrected_detector_only_recipe(monkeypatch):
    import importlib

    # Keep this config test independent of variables exported by a running
    # experiment shell.
    monkeypatch.setenv("MAX_EPOCHS", "300")
    monkeypatch.setenv("STEPS", "10000,14000,16000")
    config_module = importlib.import_module("configs.star_predcls_obb_tail_aux_train")
    config_module = importlib.reload(config_module)
    cfg = config_module.cfg

    rel_cfg = cfg["MODEL"]["ROI_RELATION_HEAD"]
    assert cfg["MODEL"]["PRETRAINED_DETECTOR"] == "pretrained/OBB_swin_L_OBD.pth"
    assert cfg["MODEL"]["PRETRAINED_DETECTOR_CLASS_ORDER"] == "background_last"
    assert cfg["MODEL"]["FREEZE_BACKBONE"] is True
    assert cfg["MODEL"]["FREEZE_NECK"] is True
    assert cfg["MODEL"]["FREEZE_RPN_HEAD"] is True
    assert cfg["MODEL"]["FREEZE_ROI_HEAD"] is True
    assert rel_cfg["PREDICTOR"] == "RPCM_ORIGINAL_LEGACY"
    assert rel_cfg["RPCM_RELATION_GRAPH_MODE"] == "dual_view"
    assert rel_cfg["RPCM_GLOVE_INIT_MODE"] == "rpcm"
    assert rel_cfg["RPCM_LEGACY_6850_EXACT"] is True
    assert rel_cfg["RPCM_LEGACY_ANT_LOSS_WEIGHT"] == 0.1
    assert rel_cfg["PREDICATE_AUX_LOGIT_ADJUST_WEIGHT"] == 0.1
    assert rel_cfg["RPCM_TAIL_AUX_ENABLED"] is True
    assert rel_cfg["RPCM_TAIL_AUX_LOSS_WEIGHT"] == 0.2
    assert rel_cfg["OBJECT_REFINE_LOSS_WEIGHT"] == 0.0
    assert cfg["SOLVER"]["MAX_EPOCHS"] == 300
    assert cfg["SOLVER"]["STEPS"] == [10000, 14000, 16000]


def test_rpcm_legacy_constructs_the_exact_route_from_config():
    cfg = {
        "MODEL": {
            "BOX_MODE": "obb",
            "ROI_BOX_HEAD": {
                "NUM_CLASSES": 3,
                "CLASS_NAMES": ["__background__", "ship", "car"],
            },
            "ROI_RELATION_HEAD": {
                "PREDICTOR": "RPCM_ORIGINAL_LEGACY",
                "NUM_CLASSES": 3,
                "RELATION_NAMES": ["__background__", "over", "adjacent"],
                "USE_GT_BOX": True,
                "USE_GT_OBJECT_LABEL": True,
                "CONTEXT_POOLING_DIM": 8,
                "CONTEXT_HIDDEN_DIM": 8,
                "RPCM_MLP_DIM": 4,
                "RPCM_FEAT_UPDATE_STEP": 1,
                "RPCM_RELATION_GRAPH_MODE": "dual_view",
                "RPCM_LEGACY_6850_EXACT": True,
                "RPCM_LEGACY_NUM_PROTO": 1,
                "RPCM_LEGACY_USE_VIS_PROTO": False,
                "RPCM_LEGACY_PROTO_2D_COMPAT": True,
                "RPCM_GLOVE_INIT_MODE": "rpcm",
                "EMBED_DIM": 4,
                "RPCM_PROTO_EMBED_DIM": 4,
                "SEMANTIC_GLOVE_PATH": "",
                "WORD_EMBEDDING_FEATURES": True,
                "EDGE_FEATURES_REPRESENTATION": "fusion",
                "CAUSAL": {"SPATIAL_FOR_VISION": True},
                "RPCM_LEGACY_REGISTER_BIAS_MODULE": False,
            },
        },
        "TEST": {
            "GRAPH_DEBUG": {"ENABLED": False},
            "RELATION": {"LATER_NMS_PREDICTION_THRES": 0.3},
        },
    }

    model = RPCMLegacy(cfg, in_channels=8)

    assert model.exact_6850 is True
    assert model.relation_graph_mode == "dual_view"
    assert model.rel_proto.exact_6850 is True
    assert model.rel_proto.base_prototypes.shape == (3, 4)
    assert not hasattr(model.rel_proto, "proto_vis")
