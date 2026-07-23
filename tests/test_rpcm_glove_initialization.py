from __future__ import annotations

import torch
import torch.nn.functional as F

from sgg.modeling.roi_heads.roi_relation_predictors import (
    _OriginalRPCMPairwiseFeatureExtractor,
    _build_rpcm_object_glove_init,
    _build_rpcm_relation_glove_init,
)


def _write_glove_table(tmp_path, dim: int, values: dict[str, list[float]]):
    words = list(values)
    vectors = torch.tensor([values[word] for word in words], dtype=torch.float32)
    path = tmp_path / f"glove.6B.{dim}d.pt"
    torch.save(({word: idx for idx, word in enumerate(words)}, vectors, dim), path)
    return tmp_path / f"glove.6B.{dim}d.txt", vectors, {word: idx for idx, word in enumerate(words)}


def test_rpcm_object_glove_matches_raw_obj_edge_vectors_rule(tmp_path):
    glove_path, table, word_to_idx = _write_glove_table(
        tmp_path,
        4,
        {
            "ship": [2.0, 0.0, 0.0, 0.0],
            "traffic": [0.0, 4.0, 0.0, 0.0],
            "light": [0.0, 0.0, 6.0, 0.0],
        },
    )
    names = ["__background__", "ship", "traffic_light"]

    torch.manual_seed(17)
    actual, diagnostics = _build_rpcm_object_glove_init(names, str(glove_path), 4)
    torch.manual_seed(17)
    random_rows = torch.empty((len(names), 4)).normal_(0.0, 1.0)

    torch.testing.assert_close(actual[0], random_rows[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[1], table[word_to_idx["ship"]], rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        actual[2],
        (table[word_to_idx["traffic"]] + table[word_to_idx["light"]]) / 2.0,
        rtol=0.0,
        atol=0.0,
    )
    assert diagnostics == {"missing_classes": [], "missing_tokens": []}
    # RPCM does not normalize foreground object embeddings.
    assert actual[1].norm().item() == 2.0


def test_rpcm_relation_glove_uses_negation_polarity_and_pair_boost(tmp_path):
    glove_path, table, word_to_idx = _write_glove_table(
        tmp_path,
        4,
        {
            "working": [1.0, 0.2, 0.0, 0.0],
            "not": [0.0, 1.0, 0.1, 0.0],
            "driving": [0.3, 0.1, 1.0, 0.0],
            "lane": [0.1, 0.0, 0.4, 1.0],
            "same": [1.0, 0.0, 0.2, 0.0],
            "different": [0.0, 1.0, 0.0, 0.2],
        },
    )
    names = [
        "__background__",
        "working on",
        "not working on",
        "driving in the same lane with",
        "driving in the different lane with",
    ]

    torch.manual_seed(23)
    actual, diagnostics = _build_rpcm_relation_glove_init(names, str(glove_path), 4)

    normalize = lambda value: F.normalize(value, dim=0)
    working = normalize(table[word_to_idx["working"]])
    not_vec = normalize(table[word_to_idx["not"]])
    not_working = normalize(working - not_vec)
    neg_axis = normalize(not_working - working)
    expected_working = normalize(working - 0.2 * neg_axis)
    expected_not_working = normalize(not_working + 0.2 * neg_axis)

    driving = normalize(table[word_to_idx["driving"]])
    lane = normalize(table[word_to_idx["lane"]])
    sif_weight = 1e-3 / (1e-3 + 1.0)
    base = normalize(
        (torch.stack((driving, lane)) * sif_weight).sum(dim=0)
        / (torch.tensor([sif_weight, sif_weight]).sum() + 1e-9)
    )
    polar_axis = normalize(
        normalize(table[word_to_idx["different"]])
        - normalize(table[word_to_idx["same"]])
    )
    same_lane = normalize(base - polar_axis)
    different_lane = normalize(base + polar_axis)
    pair_axis = normalize(different_lane - same_lane)
    expected_same_lane = normalize(same_lane - 0.2 * pair_axis)
    expected_different_lane = normalize(different_lane + 0.2 * pair_axis)

    torch.testing.assert_close(actual[1], expected_working, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual[2], expected_not_working, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual[3], expected_same_lane, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual[4], expected_different_lane, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual.norm(dim=1), torch.ones(len(names)), rtol=1e-6, atol=1e-6)
    assert diagnostics == {"missing_predicates": []}
    # Negation is not implemented as the previous whole-phrase sign flip.
    assert not torch.allclose(actual[2], -actual[1])


def test_original_pairwise_extractor_copies_rpcm_object_glove(tmp_path):
    glove_path, _, _ = _write_glove_table(
        tmp_path,
        4,
        {
            "ship": [2.0, 0.0, 0.0, 0.0],
            "traffic": [0.0, 4.0, 0.0, 0.0],
            "light": [0.0, 0.0, 6.0, 0.0],
        },
    )
    cfg = {
        "MODEL": {
            "BOX_MODE": "obb",
            "ROI_BOX_HEAD": {
                "NUM_CLASSES": 3,
                "CLASS_NAMES": ["__background__", "ship", "traffic_light"],
            },
            "ROI_RELATION_HEAD": {
                "NUM_CLASSES": 2,
                "RELATION_NAMES": ["__background__", "over"],
                "USE_GT_BOX": True,
                "USE_GT_OBJECT_LABEL": True,
                "EMBED_DIM": 4,
                "CONTEXT_HIDDEN_DIM": 8,
                "CONTEXT_POOLING_DIM": 8,
                "EDGE_FEATURES_REPRESENTATION": "fusion",
                "WORD_EMBEDDING_FEATURES": True,
                "SEMANTIC_GLOVE_PATH": str(glove_path),
                "RPCM_GLOVE_INIT_MODE": "rpcm",
                "CAUSAL": {"SPATIAL_FOR_VISION": True},
            },
        }
    }
    torch.manual_seed(31)
    expected, _ = _build_rpcm_object_glove_init(
        cfg["MODEL"]["ROI_BOX_HEAD"]["CLASS_NAMES"], str(glove_path), 4
    )
    torch.manual_seed(31)
    extractor = _OriginalRPCMPairwiseFeatureExtractor(cfg, in_channels=8)

    torch.testing.assert_close(
        extractor.obj_embed_on_prob_dist.weight,
        expected,
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        extractor.obj_embed_on_pred_label.weight,
        expected,
        rtol=0.0,
        atol=0.0,
    )

