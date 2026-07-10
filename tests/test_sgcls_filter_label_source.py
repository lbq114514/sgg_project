import torch

from sgg.modeling.roi_heads.relation_head import ROIRelationHead, _normalize_sgcls_filter_label_source
from sgg.structures.boxes import BoxList


class _CaptureSemanticFilter:
    enabled = True

    def __init__(self):
        self.labels = None

    def filter_pairs(self, pair_idx, labels):
        self.labels = labels.clone()
        return pair_idx


class _CapturePairFilter:
    filter_method = "PPG"
    threshold = 0
    topk = 100

    def __init__(self):
        self.labels = None

    def filter_pairs(self, proposal, pair_idx):
        self.labels = proposal.get_field("labels").clone()
        return pair_idx.flip(0)


def _proposal():
    proposal = BoxList(torch.zeros((3, 5)), (256, 256), "xywha")
    proposal.add_field("labels", torch.tensor([2, 3, 4]))
    proposal.add_field("gt_labels", torch.tensor([11, 12, 13]))
    proposal.add_field("pred_labels", torch.tensor([2, 3, 4]))
    return proposal


def _head(source):
    head = ROIRelationHead.__new__(ROIRelationHead)
    torch.nn.Module.__init__(head)
    head.task = "sgcls"
    head.sgcls_filter_label_source = source
    head.sema_filter = _CaptureSemanticFilter()
    head.ppg = _CapturePairFilter()
    head.legacy_filter_flow = False
    head.filter_method = "PPG"
    return head


def test_sgcls_gt_filter_mode_uses_gt_only_during_filtering():
    proposal = _proposal()
    head = _head("gt")
    pairs = torch.tensor([[0, 1], [1, 2]])
    selected = head._filter_test_pairs_for_proposal(proposal, pairs)

    expected_gt = torch.tensor([11, 12, 13])
    assert torch.equal(head.sema_filter.labels, expected_gt)
    assert torch.equal(head.ppg.labels, expected_gt)
    assert torch.equal(proposal.get_field("labels"), torch.tensor([2, 3, 4]))
    assert proposal.get_field("filter_label_source") == "gt"
    assert torch.equal(selected, pairs.flip(0))


def test_sgcls_pred_filter_mode_preserves_current_behavior():
    proposal = _proposal()
    head = _head("pred")
    pairs = torch.tensor([[0, 1], [1, 2]])
    head._filter_test_pairs_for_proposal(proposal, pairs)

    expected_pred = torch.tensor([2, 3, 4])
    assert torch.equal(head.sema_filter.labels, expected_pred)
    assert torch.equal(head.ppg.labels, expected_pred)
    assert proposal.get_field("filter_label_source") == "pred"


def test_sgcls_gt_filter_mode_falls_back_without_gt_labels():
    proposal = _proposal()
    del proposal.extra_fields["gt_labels"]
    head = _head("gt")
    head._filter_test_pairs_for_proposal(proposal, torch.tensor([[0, 1]]))
    assert torch.equal(head.sema_filter.labels, torch.tensor([2, 3, 4]))
    assert proposal.get_field("filter_label_source") == "pred_fallback_missing_gt"


def test_sgcls_filter_label_source_validation():
    assert _normalize_sgcls_filter_label_source("GT") == "gt"
    try:
        _normalize_sgcls_filter_label_source("oracle")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid sgcls filter source must raise ValueError.")
