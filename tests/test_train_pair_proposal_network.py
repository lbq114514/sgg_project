import random
from types import SimpleNamespace

import numpy as np
import torch

from sgg.modeling.roi_heads.pair_proposal_network import (
    PairProposalNetwork,
    load_pair_proposal_checkpoint,
)
from tools.train_pair_proposal_network import (
    _sample_unique_negative_pairs,
    load_semantic_pair_support,
    record_to_proposal,
    train_record,
)


def _record():
    return {
        "width": 1024,
        "height": 1024,
        "boxes": np.asarray([
            [10, 10, 8, 8, 0], [30, 20, 10, 6, 10],
            [55, 40, 12, 9, -15], [80, 70, 9, 9, 0],
        ], dtype=np.float32),
        "labels": np.asarray([1, 2, 1, 3], dtype=np.int64),
        "relations": np.asarray([[0, 1, 2], [0, 1, 3], [2, 3, 1]], dtype=np.int64),
    }


def test_negative_sampling_is_unique_and_excludes_positives():
    positives = {(0, 1), (2, 3)}
    sampled = _sample_unique_negative_pairs(6888, 1000, positives, random.Random(7))
    assert len(sampled) == 1000
    assert len(set(sampled)) == 1000
    assert all(head != tail for head, tail in sampled)
    assert not (set(sampled) & positives)


def test_negative_sampling_excludes_semantically_supported_label_pairs(tmp_path):
    import json

    prior = [[[0] for _ in range(4)] for _ in range(4)]
    prior[1][2][0] = 1
    support_path = tmp_path / "support.json"
    support_path.write_text(json.dumps(prior), encoding="utf-8")
    forbidden = load_semantic_pair_support(support_path, 4)
    labels = [1, 2, 3, 3]
    sampled = _sample_unique_negative_pairs(
        4, 6, set(), random.Random(9), labels=labels,
        forbidden_label_pairs=forbidden,
    )
    assert all(not (labels[head] == 1 and labels[tail] == 2) for head, tail in sampled)


def test_negative_sampling_can_request_every_eligible_pair():
    labels = [1, 1, 2, 2]
    forbidden = torch.ones((3, 3), dtype=torch.bool)
    forbidden[1, 2] = False
    sampled = _sample_unique_negative_pairs(
        4, 100, set(), random.Random(5), labels=labels,
        forbidden_label_pairs=forbidden,
    )
    assert set(sampled) == {(0, 2), (0, 3), (1, 2), (1, 3)}


def test_train_step_and_checkpoint_reload(tmp_path):
    torch.manual_seed(1)
    glove = torch.randn(4, 6)
    glove[0].zero_()
    model = PairProposalNetwork(4, label_dim=5, box_dim=8, hidden_dim=16, dropout=0, glove_init=glove)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    args = SimpleNamespace(
        negative_ratio=2, hard_negative_pool=2, block_size=16, grad_clip=5.0
    )
    stats = train_record(
        model, optimizer, _record(), torch.device("cpu"), args, random.Random(3)
    )
    assert stats["pairs"] > 0
    assert np.isfinite(stats["loss"])
    assert model.glove_embedding.weight.grad is None
    assert model.label_embedding.weight.grad is not None
    assert any(parameter.grad is not None for parameter in model.box_encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.pair_scorer.parameters())

    path = tmp_path / "checkpoint.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "num_obj_classes": 4, "label_dim": 5, "box_dim": 8,
            "hidden_dim": 16, "dropout": 0,
        },
    }, path)
    restored, _ = load_pair_proposal_checkpoint(path)
    proposal = record_to_proposal(_record(), torch.device("cpu"))
    pairs = torch.tensor([[0, 1], [2, 3]])
    model.eval()
    restored.eval()
    assert torch.allclose(model.score_pairs(proposal, pairs), restored.score_pairs(proposal, pairs))
