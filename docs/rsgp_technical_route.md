# RSGP 技术路线：面向 STAR 遥感场景的 Pair Proposal 改进

## 1. 当前结论

当前实验表明，PPN 不能直接替换 PPG 作为测试阶段 pair filter。

在相同 relation checkpoint 下：

```text
PPG: R@2000=0.6919, mR@2000=0.4417, HMR@2000=0.5392, final coverage=0.7989
PPN: R@2000=0.6884, mR@2000=0.4350, HMR@2000=0.5331, final coverage=0.9124
```

核心判断：

- PPN 的 GT pair coverage 更高，但 downstream triplet recall 更低。
- 因此问题不是简单的“pair 是否进入候选”，而是候选图质量是否适合后续 RPCM 关系分类。
- 新方法不应只优化 pair recall，而应优化最终 `R/mR/HMR@1500/2000`。

PPG 和 PPN 在当前流程中均只作为测试阶段 pair filter，不参与 relation head 的训练。

## 2. 方法目标

新方法命名为：

```text
Remote-sensing Graph-aware Pair Proposal (RSGP)
```

目标：

```text
生成更适合遥感关系和当前 RPCM dense rel-rel GCN 的候选关系图。
```

第一版范围：

- 只做测试阶段 filter。
- 不修改训练流程。
- 不修改 relation head。
- 不重训 checkpoint。
- 默认使用当前 `tail_aux` 最优 checkpoint。

当前主模型：

```text
CONFIG=configs/star_predcls_obb_tail_aux_train.py
CHECKPOINT=outputs/star_predcls_obb_tail_aux/best_bgfirst.pth
```

## 3. RSGP 默认流程

默认推理流程：

```text
semantic filter
→ PPG protected pool
→ PPN recall completion pool
→ RS geometry/topology scoring
→ degree/label-pair quota greedy selection
→ final top-10000
→ RPCM
```

设计原则：

- 保留 PPG 的高精度候选图先验。
- 用 PPN 补充 PPG 漏掉的高召回候选。
- 用遥感几何、锚点、局部拓扑和长尾关系先验重新排序。
- 用图约束控制候选图密度，避免干扰 RPCM dense rel-rel GCN。

## 4. 遥感先验模块

### 4.1 OBB geometry expert

用于建模几何交互类关系，例如：

```text
over
adjacent
through
converge
intersect
run along
not run along
pass across
pass under
```

建议特征：

- rotated IoU
- 归一化中心距离
- 面积比
- union compactness
- 主轴夹角
- 中心连线与主轴夹角
- 投影重叠长度

### 4.2 Anchor / shared-different expert

用于建模区域锚点类关系，例如：

```text
parking in the same apron with
parking in the different apron with
docking at the same dock with
docking at the different dock with
in the same parking with
in the different parking with
```

默认 anchor 类：

```text
apron
truck_parking
car_parking
dock
runway
taxiway
breakwater
goods_yard
```

不存在的类别自动忽略。

候选特征：

- subject/object 到 anchor 的 nearest/enclosing assignment
- same-anchor confidence
- different-anchor confidence
- anchor 类型匹配度

### 4.3 Vehicle motion / lane topology expert

用于车辆运动和车道类关系，例如：

```text
driving in the same lane with
driving in the different lane with
driving in the same direction with
driving in the opposite direction with
driving alongside with
within safe distance of
within danger distance of
```

STAR 数据集没有道路/车道标注，因此不应依赖 road anchor。

建议使用车辆 OBB 本身推断 lane-like topology：

- 局部 kNN
- OBB 主轴方向聚类
- 沿主轴投影距离
- 横向 offset
- 方向差
- 局部线性排列置信度

### 4.4 Network path expert

用于网络/连通类关系，例如：

```text
directly connected to
indirectly connected to
directly transmit electricity to
indirectly transmit electricity to
within same line of
within different line of
```

建议构建局部图：

- kNN graph
- radius graph
- minimum spanning forest
- 1-hop direct candidate
- 2/3-hop indirect candidate
- connected-component same/different line signal

### 4.5 Tail relation quota / bias

用于保护低召回关系，例如：

```text
not run along
not parked alongside with
not docked alongside with
driving in the same lane with
driving in the different lane with
driving alongside with
within danger distance of
indirectly connected to
indirectly transmit electricity to
not working on
```

该模块只提供小幅 bonus 或 quota，不应覆盖主排序。

## 5. RPCM 适配原则

当前 relation base 忠实采用 `6850_4135.pth` 对应的 RPCM 版本：每个
predicate 使用一个 GloVe prototype，`proto_ema` 是静态初始化锚点；GNN
分别在 shared-subject 和 shared-object 两个 dense relation view 上传递
信息，迭代 4 次并对输入层及全部更新层取均值。RSGP 不修改这些 relation
head 规则，只改变送入 RPCM 的候选边集合。

候选 pair 数和图拓扑会直接影响：

```text
pred_pred_subj = subj_pred_map.T @ subj_pred_map
pred_pred_obj  = obj_pred_map.T  @ obj_pred_map
```

因此 RSGP 必须控制候选图密度。

必须遵守：

- 控制 dense rel-rel GCN 的候选图规模。
- 控制 max in-degree / max out-degree。
- 控制 label-pair quota。
- 控制局部 component density。
- 不以 pair coverage 作为唯一目标。

默认约束建议：

```text
RSGP_TOPK = 10000
RSGP_MAX_OUT_DEGREE = 96
RSGP_MAX_IN_DEGREE = 96
RSGP_LABEL_PAIR_QUOTA = 800
```

若候选不足，可第二轮放宽：

```text
max degree: 96 -> 128
label-pair quota: 800 -> 1200
```

## 6. 第一轮实验矩阵

统一设置：

```text
CONFIG=configs/star_predcls_obb_tail_aux_train.py
CHECKPOINT=outputs/star_predcls_obb_tail_aux/best_bgfirst.pth
TEST_BATCH_SIZE=1
VAL_BATCH_SIZE=1
```

对比方法：

```text
PPG 10000
PPN 10000
RSGP RS_ONLY
RSGP PPN_GRAPH
RSGP HYBRID 9000/1000
RSGP HYBRID 8000/2000
RSGP HYBRID 7000/3000
```

当前实现入口：

```text
TEST_FILTER_METHOD / FILTER_METHOD = RSGP
RSGP_MODE = RS_ONLY | PPN_GRAPH | HYBRID
scripts/eval_rsgp_grid.sh
```

filter 行为统一由 `TEST_FILTER_METHOD` 控制。`PPG_ENABLED`、`PPN_ENABLED`、`RSGP_ENABLED`
只作为旧配置兼容字段保留，不作为运行时主开关。

其中 HYBRID 的含义是：

```text
PPG protected pool + PPN completion pool
```

例如：

```text
RSGP HYBRID 8000/2000
```

表示优先保留最多 8000 条 PPG 高置信候选，再用 PPN/RS scoring 补充最多 2000 条候选。

## 7. 验收标准

以 PPG baseline 为目标：

```text
PPG R@2000   = 0.6919
PPG mR@2000  = 0.4417
PPG HMR@2000 = 0.5392
```

第一阶段接受标准：

```text
R@2000 >= 0.690
mR@2000 > 0.4417
HMR@2000 > 0.5392
```

必须同时报告：

- `R/mR/HMR@1500/2000`
- final GT pair coverage
- per-predicate recall
- per-predicate candidate coverage
- 平均/最大 in-degree
- 平均/最大 out-degree
- PPG/PPN/RSGP edge overlap
- hardest images

## 8. 后续方向

若 inference-only RSGP 有收益，再考虑训练新的 proposal network。

训练目标不应只是 pairness BCE，而应包含：

```text
L = GT pair BCE
  + PPG rank distillation
  + class-balanced coverage loss
  + degree regularization
  + frozen-RPCM-aware ranking loss
```

其中 PPG rank distillation 的目的不是复现 PPG，而是保留 PPG 在测试阶段体现出的候选图先验。

长期目标：

```text
让 pair proposal 直接服务 downstream R/mR/HMR，而不是单独最大化 pair recall。
```
