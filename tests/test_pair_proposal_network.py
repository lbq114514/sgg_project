import torch

from sgg.modeling.roi_heads.pair_proposal_network import (
    PairProposalNetwork,
    load_glove_class_matrix,
)
from sgg.structures.boxes import BoxList


def _proposal(n=20):
    boxes = torch.zeros((n, 5), dtype=torch.float32)
    boxes[:, 0] = torch.arange(n) * 5 + 10
    boxes[:, 1] = torch.arange(n) * 3 + 10
    boxes[:, 2:4] = 8
    proposal = BoxList(boxes, (1024, 1024), mode="xywha")
    proposal.add_field("labels", torch.arange(n) % 48 + 1)
    return proposal


def test_topk_is_bounded_and_matches_full_scoring():
    torch.manual_seed(3)
    model = PairProposalNetwork(num_obj_classes=49, dropout=0).eval()
    proposal = _proposal()
    pairs, scores = model.topk_pairs(proposal, topk=37, block_size=55)
    assert pairs.shape == (37, 2)
    assert scores.shape == (37,)
    assert torch.all(pairs[:, 0] != pairs[:, 1])

    n = len(proposal)
    full = torch.ones((n, n), dtype=torch.bool)
    full.fill_diagonal_(False)
    all_pairs = full.nonzero()
    all_scores = model.score_pairs(proposal, all_pairs)
    expected = all_scores.topk(37).values.sort(descending=True).values
    assert torch.allclose(scores, expected, atol=1e-6)


def test_semantic_mask_is_applied_during_streaming():
    model = PairProposalNetwork(num_obj_classes=49, dropout=0).eval()
    proposal = _proposal(12)
    allowed = torch.zeros((49, 49), dtype=torch.bool)
    allowed[1, 2] = True
    pairs, _ = model.topk_pairs(proposal, topk=100, block_size=20, allowed_label_pairs=allowed)
    labels = proposal.get_field("labels")
    assert pairs.numel() > 0
    assert torch.all((labels[pairs[:, 0]] == 1) & (labels[pairs[:, 1]] == 2))


def test_glove_is_frozen_and_residual_receives_gradient(tmp_path):
    vocabulary = {"ship": 0, "goods": 1, "yard": 2}
    vectors = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    glove_path = tmp_path / "glove.pt"
    torch.save((vocabulary, vectors, 4), glove_path)
    matrix, diagnostics = load_glove_class_matrix(
        glove_path, ["__background__", "ship", "goods_yard"]
    )
    assert diagnostics["covered_classes"] == 2
    assert torch.equal(matrix[0], torch.zeros(4))
    assert torch.allclose(matrix[2], 0.5 * (vectors[1] + vectors[2]))

    model = PairProposalNetwork(3, label_dim=5, glove_init=matrix, dropout=0)
    proposal = _proposal(3)
    proposal.add_field("labels", torch.tensor([1, 2, 1]))
    logits = model.score_pairs(proposal, torch.tensor([[0, 1], [1, 2]]))
    logits.sum().backward()
    assert model.glove_embedding.weight.requires_grad is False
    assert model.glove_embedding.weight.grad is None
    assert model.label_embedding.weight.grad is not None
