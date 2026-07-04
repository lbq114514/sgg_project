import torch

from sgg.modeling.roi_heads.pair_graph_builder import PairGraphBuilder
from sgg.modeling.roi_heads.typed_hyper_rpcm import (
    FAMILY_PREDICATES,
    SparseHypergraphLayer,
    SparseTypedMessageLayer,
    TypedHyperRPCM,
    validate_family_mapping,
)


def test_family_mapping_is_complete_and_unique():
    mapping = validate_family_mapping(59)
    assert mapping.shape == (59,)
    assert sum(len(values) for values in FAMILY_PREDICATES.values()) == 59
    assert mapping[0].item() == 0
    assert set(mapping[1:].tolist()) == set(range(1, 7))
    assert mapping[38].item() == 3
    assert mapping[39].item() == 2


def test_hierarchy_logits_fill_global_predicate_space():
    module = TypedHyperRPCM.__new__(TypedHyperRPCM)
    torch.nn.Module.__init__(module)
    module.num_rel_classes = 59
    module.family_head = torch.nn.Linear(8, 7)
    module.fine_heads = torch.nn.ModuleDict({
        str(family): torch.nn.Linear(8, len(predicates))
        for family, predicates in FAMILY_PREDICATES.items() if family
    })
    logits = module._hierarchy_logits(torch.randn(5, 8))
    assert logits.shape == (5, 59)
    assert torch.isfinite(logits).all()
    assert torch.allclose(logits.exp().sum(1), torch.ones(5), atol=1e-5)


def test_degree_cap_partition_and_gt_schedule():
    builder = PairGraphBuilder.__new__(PairGraphBuilder)
    builder.topk = 20
    builder.max_degree = 2
    builder.relaxed_degree = 3
    builder.partition_size = 4
    builder.gt_inject_full_epochs = 5
    builder.gt_inject_end_epoch = 10
    ranked = torch.tensor([(h, t) for h in range(6) for t in range(6) if h != t])
    selected = builder._degree_constrained(ranked, 6)
    out_degree = torch.bincount(selected[:, 0], minlength=6)
    in_degree = torch.bincount(selected[:, 1], minlength=6)
    assert int(out_degree.max()) <= 3
    assert int(in_degree.max()) <= 3
    parts = builder.partition(selected)
    assert all(len(part) <= 4 for part in parts)
    assert builder._gt_injection_ratio(5) == 1.0
    assert builder._gt_injection_ratio(10) == 0.0
    assert 0 < builder._gt_injection_ratio(7) < 1


def test_sparse_layers_forward_backward_without_square_maps():
    torch.manual_seed(3)
    num_nodes, num_edges, dim = 6888, 10000, 16
    pairs = torch.stack((
        torch.randint(num_nodes, (num_edges,)),
        torch.randint(num_nodes, (num_edges,)),
    ), dim=1)
    entities = torch.randn(num_nodes, dim, requires_grad=True)
    relations = torch.randn(num_edges, dim, requires_grad=True)
    sparse = SparseTypedMessageLayer(dim, 0.0)
    next_entities, next_relations = sparse(entities, relations, pairs)
    assert next_entities.shape == entities.shape
    assert next_relations.shape == relations.shape
    assert not any(t.ndim == 2 and t.shape == (num_edges, num_edges) for t in (next_entities, next_relations))
    (next_entities.square().mean() + next_relations.square().mean()).backward()
    assert entities.grad is not None and relations.grad is not None


def test_sparse_hypergraph_forward_backward():
    entities = torch.randn(20, 12, requires_grad=True)
    factors = torch.randn(15, 12, requires_grad=True)
    incidence = torch.tensor([
        list(range(15)) + list(range(5, 20)),
        list(range(15)) + list(range(15)),
    ])
    layer = SparseHypergraphLayer(12, 0.0)
    entity_out, factor_out = layer(entities, factors, incidence)
    (entity_out.sum() + factor_out.sum()).backward()
    assert entities.grad is not None and factors.grad is not None
