# STAR Typed RPCM Stage1

The implementation keeps the detector and frozen PPN independent from the new
relation predictor. The retained configuration is:

- `configs/star_predcls_obb_typed_stage1_train.py`: hierarchy, typed experts and anchor assignment.

Optionally build the fixed training topology cache first:

```bash
python tools/build_typed_pair_graph_cache.py \
  --split train \
  --checkpoint pretrained/PPN_OBB.pth \
  --output outputs/pair_graph_cache/train_ppn_top10000.pth
```

Set `GRAPH_CACHE_PATH` in the selected configuration to that file. The loader
rejects the cache if the PPN SHA-256 or top-k differs.

To initialize the compatible RPCM blocks and strictly migrate the rest of the
detector checkpoint:

```bash
python train.py \
  --config configs/star_predcls_obb_typed_stage1_train.py \
  --init-rpcm outputs/star_predcls_obb_train_large_long/model_last.pth \
  --device cuda
```

Use `--resume` instead of `--init-rpcm` after the first TypedHyperRPCM
checkpoint. The two options are intentionally mutually exclusive.

One-command background launchers are also available:

```bash
bash scripts/run_typed_rpcm_stage1.sh
```

The launcher writes `train.log`, `train.pid`, and `exit_code.txt` under the
output directory. Override runtime settings through environment variables,
for example:

```bash
RESUME=outputs/star_predcls_obb_typed_stage1/model_last.pth \
  bash scripts/run_typed_rpcm_stage1.sh
```
