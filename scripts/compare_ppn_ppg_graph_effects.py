from __future__ import annotations

import argparse
import copy
import importlib.util
import statistics
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sgg.data.build import build_dataloaders, build_datasets
from sgg.engine import Trainer
from sgg.modeling.detectors.scene_graph_detector import SceneGraphDetector
from sgg.modeling.roi_heads.pair_proposal_network import PairProposalNetworkFilter
from sgg.modeling.roi_heads.ppg import PairProposalGenerator


def load_config(path: str):
    spec = importlib.util.spec_from_file_location("comparison_cfg", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.cfg if hasattr(module, "cfg") else module.get_cfg()


def make_filter(cfg: dict, method: str, model_path: str):
    local_cfg = copy.deepcopy(cfg)
    rel_cfg = local_cfg["MODEL"]["ROI_RELATION_HEAD"]
    rel_cfg["TEST_FILTER_METHOD"] = method
    if method == "PPN":
        rel_cfg["PPN_ENABLED"] = True
        rel_cfg["PPN_MODEL_PATH"] = model_path
        return PairProposalNetworkFilter(local_cfg)
    rel_cfg["PPG_ENABLED"] = True
    rel_cfg["PPG_MODEL_PATH_OBB"] = model_path
    return PairProposalGenerator(local_cfg)


def graph_stats(pair_idx: torch.Tensor, num_nodes: int):
    if num_nodes <= 0:
        return 0.0, 0
    degree = torch.zeros(num_nodes, dtype=torch.long, device=pair_idx.device)
    if pair_idx.numel():
        degree.scatter_add_(0, pair_idx[:, 0], torch.ones_like(pair_idx[:, 0]))
        degree.scatter_add_(0, pair_idx[:, 1], torch.ones_like(pair_idx[:, 1]))
    return float(degree.float().mean().item()), int(degree.max().item())


def pair_lookup(pair_idx: torch.Tensor):
    return {(int(s), int(o)): i for i, (s, o) in enumerate(pair_idx.tolist())}


def relation_forward(model, relation_head, features, proposals, pair_idxs):
    roi_feats = model.bbox_roi_extractor(features, list(proposals))
    roi_features = model.bbox_head(roi_feats)
    union_features = relation_head.union_feature_extractor(
        features, proposals, pair_idxs, OBj=model
    ) if relation_head.use_union_box else None
    output = relation_head.predictor(
        proposals, pair_idxs, None, None, roi_features, union_features, logger=None
    )
    return output[0]


def main():
    parser = argparse.ArgumentParser(
        description="Compare PPN/PPG graph density and RPCM logits on common GT pairs."
    )
    parser.add_argument("--config", default="configs/star_predcls_obb_train.py")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ppn", default="pretrained/PPN_OBB.pth")
    parser.add_argument("--ppg", default="pretrained/STAR_OBB.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-batches", type=int, default=-1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    datasets = build_datasets(cfg, splits=("test",))
    metadata = datasets["test"].metadata
    cfg["MODEL"]["ROI_BOX_HEAD"]["CLASS_NAMES"] = [
        metadata.categories[i] for i in sorted(metadata.categories)
    ]
    rel_names = [metadata.predicates[i] for i in sorted(metadata.predicates)]
    cfg["MODEL"]["ROI_RELATION_HEAD"]["RELATION_NAMES"] = rel_names
    loaders = build_dataloaders(
        cfg, splits=("test",), datasets=datasets, shuffle_map={"test": False}
    )

    model = SceneGraphDetector(cfg)
    trainer = Trainer(cfg, model, device=args.device, dataloaders=loaders)
    trainer.load_checkpoint(args.checkpoint)
    model.eval()
    relation_head = model.roi_heads.relation
    ppn = make_filter(cfg, "PPN", args.ppn)
    ppg = make_filter(cfg, "PPG", args.ppg)

    stats = {
        "images": 0,
        "ppn_edges": 0,
        "ppg_edges": 0,
        "common_edges": 0,
        "ppn_degree_mean": [],
        "ppg_degree_mean": [],
        "ppn_degree_max": [],
        "ppg_degree_max": [],
        "common_gt": 0,
        "ppn_top1": 0,
        "ppg_top1": 0,
        "ppn_ranks": [],
        "ppg_ranks": [],
        "top1_changed": 0,
    }

    with torch.no_grad():
        for batch_id, (images, targets, _) in enumerate(loaders["test"]):
            if args.max_batches > 0 and batch_id >= args.max_batches:
                break
            images = images.to(trainer.device)
            targets = trainer._move_targets(targets)
            features = model._extract_features(images)
            proposals = model._prepare_eval_proposals(features, images, targets)
            device = next(iter(features.values())).device
            base = relation_head.samp_processor.prepare_test_pairs(device, proposals)

            ppn_pairs, ppg_pairs = [], []
            for proposal, pairs in zip(proposals, base):
                if proposal.has_field("labels"):
                    pairs = relation_head.sema_filter.filter_pairs(
                        pairs, proposal.get_field("labels").long()
                    )
                ppn_pairs.append(ppn.filter_pairs(proposal, pairs))
                ppg_pairs.append(ppg.filter_pairs(proposal, pairs))

            ppn_logits = relation_forward(model, relation_head, features, proposals, ppn_pairs)
            ppg_logits = relation_forward(model, relation_head, features, proposals, ppg_pairs)

            for proposal, target, pairs_n, pairs_g, logits_n, logits_g in zip(
                proposals, targets, ppn_pairs, ppg_pairs, ppn_logits, ppg_logits
            ):
                stats["images"] += 1
                stats["ppn_edges"] += len(pairs_n)
                stats["ppg_edges"] += len(pairs_g)
                lookup_n, lookup_g = pair_lookup(pairs_n), pair_lookup(pairs_g)
                stats["common_edges"] += len(set(lookup_n) & set(lookup_g))
                mean_n, max_n = graph_stats(pairs_n, len(proposal))
                mean_g, max_g = graph_stats(pairs_g, len(proposal))
                stats["ppn_degree_mean"].append(mean_n)
                stats["ppg_degree_mean"].append(mean_g)
                stats["ppn_degree_max"].append(max_n)
                stats["ppg_degree_max"].append(max_g)

                if not target.has_field("relation_triplets"):
                    continue
                for subj, obj, pred in target.get_field("relation_triplets").long().tolist():
                    key = (int(subj), int(obj))
                    if key not in lookup_n or key not in lookup_g or pred <= 0:
                        continue
                    score_n = logits_n[lookup_n[key]]
                    score_g = logits_g[lookup_g[key]]
                    order_n = torch.argsort(score_n[1:], descending=True) + 1
                    order_g = torch.argsort(score_g[1:], descending=True) + 1
                    rank_n = int(torch.nonzero(order_n == pred)[0]) + 1
                    rank_g = int(torch.nonzero(order_g == pred)[0]) + 1
                    top_n, top_g = int(order_n[0]), int(order_g[0])
                    stats["common_gt"] += 1
                    stats["ppn_top1"] += int(top_n == pred)
                    stats["ppg_top1"] += int(top_g == pred)
                    stats["ppn_ranks"].append(rank_n)
                    stats["ppg_ranks"].append(rank_g)
                    stats["top1_changed"] += int(top_n != top_g)

            print(f"processed batch {batch_id + 1}", flush=True)

    images = max(stats["images"], 1)
    common_gt = max(stats["common_gt"], 1)
    print("\nCandidate graph")
    print(f"images={stats['images']}")
    print(f"PPN edges/image={stats['ppn_edges'] / images:.2f}")
    print(f"PPG edges/image={stats['ppg_edges'] / images:.2f}")
    union_edges = stats["ppn_edges"] + stats["ppg_edges"] - stats["common_edges"]
    print(f"edge Jaccard={stats['common_edges'] / max(union_edges, 1):.6f}")
    print(
        f"PPN mean-degree/image={statistics.mean(stats['ppn_degree_mean']):.2f}, "
        f"mean max-degree={statistics.mean(stats['ppn_degree_max']):.2f}"
    )
    print(
        f"PPG mean-degree/image={statistics.mean(stats['ppg_degree_mean']):.2f}, "
        f"mean max-degree={statistics.mean(stats['ppg_degree_max']):.2f}"
    )
    print("\nCommon retained GT pairs")
    print(f"relations={stats['common_gt']}")
    print(f"PPN predicate top1={stats['ppn_top1'] / common_gt:.6f}")
    print(f"PPG predicate top1={stats['ppg_top1'] / common_gt:.6f}")
    print(f"PPN correct-predicate mean rank={statistics.mean(stats['ppn_ranks']):.3f}")
    print(f"PPG correct-predicate mean rank={statistics.mean(stats['ppg_ranks']):.3f}")
    print(f"top1 changed by graph={stats['top1_changed'] / common_gt:.6f}")


if __name__ == "__main__":
    main()
