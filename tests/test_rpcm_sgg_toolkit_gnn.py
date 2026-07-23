import torch
from torch import nn

from sgg.modeling.roi_heads.roi_relation_predictors import (
    RPCMLegacy,
    _SGGToolkitGraphConvolutionLayerCollect,
    _SGGToolkitGraphConvolutionLayerUpdate,
)


class _Proposal:
    def __init__(self, count: int):
        self.bbox = torch.zeros((count, 5), dtype=torch.float32)

    def __len__(self):
        return self.bbox.size(0)


def _mapping_only_model():
    model = RPCMLegacy.__new__(RPCMLegacy)
    nn.Module.__init__(model)
    model.relation_graph_mode = "sgg_toolkit"
    return model


def test_sgg_toolkit_collect_matches_original_degree_average():
    collect = _SGGToolkitGraphConvolutionLayerCollect(2, 2)
    for unit in collect.collect_units:
        with torch.no_grad():
            unit.fc.weight.copy_(torch.eye(2))
            unit.fc.bias.zero_()

    source = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    attention = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    expected = torch.tensor([[2.0, 3.0], [4.0, 5.0]])

    for unit_id in range(6):
        actual = collect(torch.zeros_like(expected), source, attention, unit_id)
        torch.testing.assert_close(actual, expected)


def test_sgg_toolkit_update_is_target_plus_source():
    update = _SGGToolkitGraphConvolutionLayerUpdate()
    target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    source = torch.tensor([[0.5, 1.0], [1.5, 2.0]])
    torch.testing.assert_close(update(target, source, 0), target + source)
    torch.testing.assert_close(update(target, source, 1), target + source)


def test_sgg_toolkit_maps_match_original_block_and_endpoint_logic():
    model = _mapping_only_model()
    proposals = [_Proposal(3), _Proposal(2)]
    pairs = [
        torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        torch.tensor([[0, 1]], dtype=torch.long),
    ]

    subj, obj, rel_rel, unused, obj_obj = model._get_map_idxs(proposals, pairs)

    expected_obj_obj = torch.tensor(
        [
            [0, 1, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
        ],
        dtype=torch.float32,
    )
    expected_rel_rel = torch.tensor(
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]],
        dtype=torch.float32,
    )

    assert subj.shape == (5, 3)
    assert obj.shape == (5, 3)
    assert unused.shape == (0, 0)
    torch.testing.assert_close(obj_obj, expected_obj_obj)
    torch.testing.assert_close(rel_rel, expected_rel_rel)


def test_ablation_a_selects_full_sgg_toolkit_gnn():
    from configs.star_predcls_obb_ablation_unified_rca_train import cfg

    assert cfg["MODEL"]["ROI_RELATION_HEAD"]["RPCM_RELATION_GRAPH_MODE"] == "sgg_toolkit"

