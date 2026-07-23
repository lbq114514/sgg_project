# 论文主要贡献、创新点与实验记录

> Status: working draft for paper writing  
> Last updated: 2026-07-22  
> Dataset/task scope: STAR, OBB, PredCls/SGCls/SGDet  
> Reference baseline: SGG-ToolKit implementation and the STAR paper  
> Result unit in all tables: percentage (%)

本文档固定当前工作的论文边界、方法叙述、实验协议和结果来源。标记为 **TBD** 的位置需要后续实验填写；标记为 **existing** 的结果已能从当前日志或 JSON 中复现。最终投稿前，应统一 checkpoint 选择方式和评估脚本后再冻结数字。

---

## 1. 一句话定位

针对大幅遥感图像中实体数量多、候选关系图稠密、关系类别分布不均且不同实体角色具有非对称语义的问题，本文在 STAR 的 OBB 场景图生成框架上提出一种由 **遥感图感知候选关系筛选（RSGP）**、**角色感知边视角上下文传播（Role-aware RCA）** 和 **困难谓词残差校准（Hard-Predicate Residual Calibration, HPRC）** 组成的方法，在控制候选图规模的同时提升整体召回率与类别均衡召回率。

可用英文表述：

> We present a remote-sensing-oriented scene graph generation framework that jointly improves candidate graph construction and predicate reasoning through graph-aware pair proposal, role-aware relation context propagation, and hard-predicate residual calibration.

---

## 2. 与 STAR/SGG-ToolKit 的边界

### 2.1 STAR 原工作已经具备的内容

以下内容来自 STAR 原工作，不能作为本文首创：

- STAR 数据集、48 个前景实体类别和 58 个前景关系类别；
- HBB/OBB 三项标准任务：PredCls、SGCls 和 SGDet；
- HOD-Net/OBB detector 以及大图多尺度 patch 检测流程；
- PPG（Pair Proposal Generator）及其 top-10,000 pair filtering；
- RPCM 中的对象上下文增强、关系上下文增强和 prototype matching；
- RPCM 中已有的 relation-to-relation 信息传播。

因此，本文不能使用“首次在 STAR 中引入关系边之间的 GNN”或“首次进行 relation-to-relation message passing”之类的表述。

### 2.2 本文真正改变的部分

本文相对 SGG-ToolKit/STAR 的变化集中在：

1. 不再把 pair proposal 仅视为 GT pair recall 最大化问题，而是将其视为面向下游关系推理的候选图构建问题；
2. 将原本合并的关系邻接拆分为 shared-subject 和 shared-object 两个角色视角，防止跨角色共享实体引入不加区分的上下文；
3. 提出 HPRC：保留主 CE 分类目标，以小权重 logit-adjust 辅助项校正类别先验，并用轻量残差校准头修正持续低召回或高混淆的谓词，避免完全改成多标签或强重加权后导致主召回率下降；
4. 在统一 OBB detector 语义、角度、NMS 和评估协议后，扩展并验证 PredCls、SGCls、SGDet 三项任务。

### 2.3 不建议作为算法贡献的内容

- detector background 通道重排、OBB angle/offset 修复属于复现和兼容性修正；
- 忠实移植 RPCM predictor 属于 baseline/reproduction infrastructure；
- PPN 可以作为 RSGP 的 recall-completion 分支，但单独 PPN 是否构成贡献需要视论文叙述和训练方法而定。

---

## 3. 论文主要贡献（可直接用于 Introduction）

### Contribution 1: Remote-sensing Graph-aware Pair Proposal

提出 **Remote-sensing Graph-aware Pair Proposal（RSGP）**。不同于只按独立 pair score 排序的 PPG/PPN，RSGP 同时考虑：

- PPG 的高精度候选；
- PPN 的高覆盖候选；
- OBB 距离、交叠、紧致度和主轴方向等遥感几何先验；
- apron、dock、parking、runway/taxiway 等共享/不同锚点结构；
- vehicle motion/lane 和 functional network topology；
- hard-predicate-supported label pairs；
- 节点入度、出度和 label-pair quota。

RSGP 在固定 top-10,000 预算下生成更适合 RPCM 上下文传播的候选关系图。现有结果显示，PPN 虽然获得更高的 GT pair coverage，但其下游 R/mR/HMR 低于 PPG 和 RSGP，说明单纯优化 pair coverage 并不能保证 scene graph recall。

可用英文贡献句：

> We propose RSGP, a remote-sensing graph-aware pair proposal strategy that combines high-precision and high-recall proposal sources with OBB geometry and graph-level degree/label-pair constraints, optimizing the candidate graph for downstream predicate reasoning rather than pair recall alone.

### Contribution 2: Role-aware Relation Context Aggregation

提出 **role-aware dual-view relation context aggregation**。对于候选关系边

\[
e_i=(s_i,o_i),
\]

分别构造 shared-subject 和 shared-object 邻接：

\[
A^{s}_{ij}=\mathbb{1}[s_i=s_j], \qquad
A^{o}_{ij}=\mathbb{1}[o_i=o_j].
\]

两个视角共享 GNN 参数，但分别进行消息传播，随后与 subject-to-relation 和 object-to-relation 消息融合：

\[
h_i^{(l+1)}=\frac{1}{4}\left(
m_{s\rightarrow r,i}^{(l)}+
m_{o\rightarrow r,i}^{(l)}+
m_{r_s\rightarrow r,i}^{(l)}+
m_{r_o\rightarrow r,i}^{(l)}
\right).
\]

STAR/SGG-ToolKit 的统一邻接将任意共享端点及 subject-object 跨角色匹配合并到同一 relation graph。本文的创新不在于“有 relation GNN”，而在于显式保留实体在关系中的 subject/object 角色，减少跨角色上下文混叠。

可用英文贡献句：

> We introduce a role-aware relation context module that decouples shared-subject and shared-object relation graphs, preserving endpoint semantics that are discarded by unified relation adjacency.

### Contribution 3: Hard-Predicate Residual Calibration

提出 **Hard-Predicate Residual Calibration（HPRC）**。针对 STAR 显著的类别不均衡以及近义、反义关系混淆，采用“稳定主分类器 + 弱频率先验校准 + 困难谓词残差修正”的训练方式：

\[
\mathcal{L}_{rel}
=\operatorname{CE}(z,y)
+\lambda_{LA}\operatorname{CE}(z+\tau\log \pi,y),
\]

其中主 CE 保持原关系分类器的判别能力，logit-adjust auxiliary loss 以较小权重改善类别先验偏置。HPRC 的残差校准头对根据训练频率、历史验证召回和系统性混淆确定的困难谓词进行轻量修正：

\[
z'_c=z_c+\alpha z^{HPRC}_c,
\qquad
\alpha=\tanh(a)\alpha_{\max},
\]

并用带 `pos_weight` 的 BCE 训练校准 logits。由于融合 scale 从 0 初始化，模型初始行为与原分类器一致，降低对总体 R 的破坏风险。这里使用“困难谓词”而不是简单的“尾类”，因为被校准集合既包含低频关系，也包含 `same/different lane` 等样本量不低但长期低召回的关系。

可用英文贡献句：

> We introduce Hard-Predicate Residual Calibration (HPRC), which combines weak prior-aware logit adjustment with a zero-initialized residual calibration head to correct systematically under-recognized predicates without replacing the stable relation classifier.

### Contribution 4（可选）: Complete OBB SGG Evaluation Pipeline

在统一的 OBB detector、类别通道、旋转角度、NMS 和指标实现下完成 PredCls、SGCls、SGDet 三项任务。

该项更适合作为“完整实验与可复现性贡献”，不应替代前三项算法贡献。

### 3.5 Contribution-to-evidence matrix

| Claim | Required evidence | Current evidence boundary |
|---|---|---|
| Role-aware relation GNN is a distinct RPCM base relative to SGG-ToolKit | Code-level topology/update comparison and the reproduced 6850 checkpoint | Method distinction is established; an isolated causal gain over the paper GNN is **not** claimed without a controlled retraining |
| HPRC improves class-balanced predicate recognition | Compare the 6850 PPG checkpoint with its warm-started HPRC descendant | Existing comparison supports the **combined HPRC stage**, not separate LA-only or residual-head-only causal effects |
| RSGP improves downstream graph reasoning | Same relation checkpoint, only inference filter differs | Existing PPG vs RSGP evaluations form a strict inference-time comparison |
| Higher pair coverage is not sufficient | PPN coverage is higher but triplet metrics are lower | PPG/PPN/RSGP graph-quality table |
| RSGP components are necessary | Remove one component at a time | RSGP component ablation |
| Improvements generalize beyond PredCls | Same task checkpoint evaluated with PPG and RSGP | SGCls/SGDet cross-task test |

---

## 4. 方法总体结构

```text
OBB entities / detections
        │
        ▼
Semantic Filter
        │
        ▼
RSGP candidate graph
  ├─ PPG protected pool
  ├─ PPN completion pool
  ├─ OBB geometry / anchor / topology priors
  └─ degree and label-pair constrained selection
        │
        ▼
Pairwise and union visual features
        │
        ▼
Role-aware dual-view relation GNN
  ├─ entity → relation
  ├─ shared-subject relation → relation
  └─ shared-object relation → relation
        │
        ▼
Prototype predicate classifier
        │
        ├─ CE + weak logit-adjust auxiliary loss
        └─ zero-initialized HPRC head
        │
        ▼
Scene graph triplets
```

### 4.1 Problem formulation

给定一幅遥感图像中的实体集合

\[
\mathcal V=\{v_i=(b_i,l_i,f_i)\}_{i=1}^{N},
\]

其中，\(b_i=(x_i,y_i,w_i,h_i,\theta_i)\) 是 OBB，\(l_i\) 是实体类别，\(f_i\) 是 RoI visual feature。场景图生成的目标是在有向实体对 \((v_i,v_j),i\neq j\) 上预测 predicate \(r_{ij}\in\{0,\ldots,C_r-1\}\)，其中 0 表示 background，STAR 中 \(C_r=59\)。

全部有向 pair 的数量为 \(N(N-1)\)。定义初始候选集：

\[
\mathcal E_0=\{(i,j)\mid i\neq j,\ M^{sem}_{l_i,l_j}=1\},
\]

其中，\(M^{sem}\) 是由 `SF_list_support.json` 给出的 label-pair semantic support。由于大幅 STAR 图像可能包含数千个实体，直接在 \(\mathcal E_0\) 上进行关系推理会带来不可接受的计算开销。RSGP 的目标是在固定预算 \(K=10,000\) 下构建候选关系图：

\[
\mathcal E^*=\operatorname{RSGP}(\mathcal V,\mathcal E_0),
\qquad |\mathcal E^*|\le K.
\]

当 \(|\mathcal E_0|\le K\) 时不触发 proposal pruning，直接令 \(\mathcal E^*=\mathcal E_0\)；只有候选数超过阈值时才执行以下多源排序和约束选择。

PredCls 使用 GT boxes/labels；SGCls 使用 GT boxes 和预测 object logits；SGDet 使用 detector boxes/logits。为了与 SGG-ToolKit 的 STAR 协议对齐，当前 SGCls/SGDet 的 pair filtering labels 分别采用 GT 和 matched-GT labels，该差异只改变候选图构建时使用的 \(l_i\)。

本文后续统一使用以下符号：

| Symbol | Meaning |
|---|---|
| \(N\) | 当前图像中的实体数 |
| \(E\) | 筛选后的有向候选边数 |
| \(K\) | relation candidate budget，默认 10,000 |
| \(C_o,C_r\) | object/predicate 类别总数，含 background |
| \(M_s,M_o\) | subject/object incidence matrix |
| \(H^e,H^r\) | entity/relation hidden representation |
| \(S_{ij}\) | pair \((i,j)\) 的 proposal score |

### 4.2 Remote-sensing Graph-aware Pair Proposal

#### 4.2.1 Multi-source candidate pools

RSGP 使用三个互补的候选来源。

**PPG precision pool.** 对 pair \((i,j)\) 构造：

\[
x_{ij}=[\operatorname{onehot}(l_i),
        \operatorname{onehot}(l_j),q^{spatial}_{ij}],
\]

其中 \(q^{spatial}_{ij}\in\mathbb R^7\) 包括 rotated IoU、对角线比例、中心距离归一化、面积比例和 union-area 比例。两个级联 autoencoders 给出 reconstruction anomaly：

\[
\ell^{PPG}_{ij}=\frac{1}{2}\left(
\|AE_1(x_{ij})-x_{ij}\|_2^2+
\|AE_2(AE_1(x_{ij}))-AE_1(x_{ij})\|_2^2
\right).
\]

PPG 保留 anomaly 最低的 pair。RSGP 使用其排序分数

\[
S^{PPG}_{ij}=1-\frac{\operatorname{rank}_{PPG}(i,j)}{|\mathcal E_{PPG}|-1}.
\]

当前默认将 PPG top-8,000 作为优先选择池，并把 PPG top-10,000 加入总候选池。

**PPN recall-completion pool.** 独立 PPN 不读取 detector/RoI feature，仅使用 label 和 OBB：

\[
u_i=[g(l_i),e(l_i),\phi_b(b_i),\phi_a(v_i)],
\]

其中 \(g\) 是冻结 200-D GloVe embedding，\(e\) 是可学习 label residual embedding，\(\phi_b\) 编码归一化中心、宽高、面积、长宽比和 \((\sin\theta,\cos\theta)\)，\(\phi_a\) 编码邻近基础设施锚点。pair feature 和 pairness logit 为：

\[
p_{ij}=[u_i,u_j,q^{pair}_{ij}],\qquad
S^{PPN}_{ij}=\operatorname{MLP}_{pair}(p_{ij}),
\]

其中 \(q^{pair}_{ij}\in\mathbb R^{14}\) 包括相对中心偏移、对两端宽高归一化的偏移、距离、尺度比、面积比和 overlap proxy。默认保留 PPN top-12,000 作为 recall-completion pool。

**Remote-sensing prior pool.** RSGP 再从遥感几何、锚点和拓扑先验分数中选取 top-12,000。三个来源的并集为：

\[
\mathcal E_{pool}=\operatorname{Unique}\left(
\mathcal E_{PPG}\cup\mathcal E_{PPN}\cup\mathcal E_{RS}
\right).
\]

#### 4.2.2 OBB geometry prior

将中心和宽高按图像尺寸归一化。对 pair \((i,j)\)，定义：

\[
\Delta c_{ij}=c_j-c_i,\quad
d_{ij}=\|\Delta c_{ij}\|_2,\quad
\bar d_{ij}=\frac{d_{ij}}{(\delta_i+\delta_j)/2},
\]

其中 \(\delta_i=\sqrt{w_i^2+h_i^2}\)。方向差和方向一致性为：

\[
\Delta\theta_{ij}=\left|\operatorname{atan2}
(\sin(\theta_i-\theta_j),\cos(\theta_i-\theta_j))\right|,
\quad p_{ij}^{\theta}=|\cos\Delta\theta_{ij}|.
\]

实现中使用由归一化中心和宽高形成的 axis-aligned envelope 计算快速 overlap proxy \(IoU^{env}_{ij}\)，并定义：

\[
c_{ij}^{close}=\exp(-\operatorname{clip}(\bar d_{ij},0,20)),
\]

\[
c_{ij}^{compact}=\min\left(
\frac{w_ih_i+w_jh_j}{A_{bbox-union}(i,j)},2
\right).
\]

最终几何分数为：

\[
S^{geom}_{ij}=0.35IoU^{env}_{ij}
+0.30c_{ij}^{close}
+0.20c_{ij}^{compact}
+0.15p_{ij}^{\theta}.
\]

这里需要在论文中写成 axis-aligned envelope geometry proxy，而不能写成 exact rotated IoU。

#### 4.2.3 Anchor prior

锚点类别包括 apron、dock、runway、taxiway、breakwater、car/truck parking 和 goods yard。对每个实体 \(v_i\)，找到最近的 anchor instance \(a_i\)，并计算：

\[
\gamma_i=\max\left(
\exp\left[-\operatorname{clip}
\left(\frac{\|c_i-c_{a_i}\|_2}{\delta_{a_i}},0,20\right)\right],
\mathbb 1[v_i\text{ lies inside }a_i]
\right).
\]

pair anchor score 为：

\[
S^{anchor}_{ij}=\max\left\{
\mathbb 1[a_i=a_j]\gamma_i\gamma_j,
0.6\mathbb 1[a_i\neq a_j]\gamma_i\gamma_j,
0.4\mathbb 1[v_i\text{ or }v_j\text{ is an anchor}]
\right\}.
\]

该分数同时允许 shared-anchor 和 different-anchor candidate，不在 proposal 阶段强制决定最终 predicate。

#### 4.2.4 Motion and network topology prior

对 vehicle pair，以 subject 主轴方向：

\[
d_i=(\cos\theta_i,\sin\theta_i)
\]

将相对位移分解为 along-axis 和 lateral 分量：

\[
a_{ij}=|\Delta c_{ij}^{\top}d_i|,
\qquad
q_{ij}=|\Delta x_{ij}\sin\theta_i-\Delta y_{ij}\cos\theta_i|.
\]

车辆拓扑分数为：

\[
S^{veh}_{ij}=0.55|\cos\Delta\theta_{ij}|
+0.30\exp\left(-\frac{q_{ij}}{(w_i+h_i)/2}\right)
+0.15\exp\left(-0.25\frac{a_{ij}}{\delta_i}\right).
\]

对 lattice tower、substation、genset、transmission line 等 network entity pair：

\[
S^{net}_{ij}=\exp(-0.5\bar d_{ij}).
\]

最终：

\[
S^{topo}_{ij}=\max(S^{veh}_{ij},S^{net}_{ij}),
\]

不属于对应实体类型的 pair 其分数为 0。

#### 4.2.5 Hard-predicate support and degree-balance prior

设 \(\mathcal H\) 为预定义 hard predicates，训练数据统计的 label-predicate support 为 \(F(l_i,l_j,r)\)，则：

\[
S^{hard}_{ij}=\mathbb 1\left[
\sum_{r\in\mathcal H}F(l_i,l_j,r)>0
\right].
\]

对于候选池中的初始入度和出度，degree-balance score 定义为：

\[
S^{degree}_{ij}=-\log\left(1+d_{out}(i)+d_{in}(j)\right).
\]

它降低高拥塞端点 pair 的优先级，避免少量高 degree entity 主导后续 relation context。

#### 4.2.6 Hybrid scoring and constrained graph selection

对连续分数进行 pool-wise z-score 标准化：

\[
\hat S=\frac{S-\mu(S)}{\sigma(S)+\epsilon}.
\]

RSGP 的组合分数为：

\[
S^{hyb}_{ij}=w_p\hat S^{PPG}_{ij}
+w_n\hat S^{PPN}_{ij}
+w_g\hat S^{geom}_{ij}
+w_a\hat S^{anchor}_{ij}
+w_t\hat S^{topo}_{ij}
+w_l S^{hard}_{ij}
+w_d\hat S^{degree}_{ij},
\]

默认权重为：

\[
(w_p,w_n,w_g,w_a,w_t,w_l,w_d)
=(1.0,0.35,0.35,0.25,0.20,0.15,0.15).
\]

RS pool 内部用于预筛选的分数为：

\[
S^{RS}_{ij}=S^{geom}_{ij}+0.7S^{anchor}_{ij}
+0.6S^{topo}_{ij}+0.35S^{hard}_{ij}.
\]

按照 \(S^{hyb}\) 降序 greedy 选边。对已选集合 \(\mathcal E\)，严格阶段要求：

\[
d^{\mathcal E}_{out}(i)<D_{out},\qquad
d^{\mathcal E}_{in}(j)<D_{in},\qquad
n^{\mathcal E}_{l_i,l_j}<Q,
\]

默认 \(D_{out}=D_{in}=96,Q=800\)。算法依次处理 PPG top-8,000 优先池和其余排序候选；不足 \(K\) 时放宽到 \(D=128,Q=1200\)，最后进行无约束补齐，直至达到 top-10,000 或候选耗尽。

完整 inference-time selection 可写为：

```text
Algorithm 1: RSGP Hybrid Pair Selection
Input : entities V, semantic-valid pairs E0, budget K
Output: selected directed relation graph E

1  if |E0| <= K: return E0
2  Eppg <- TopK(PPG(E0), 10000);  P <- first 8000 edges of Eppg
3  Eppn <- TopK(PPN(E0), 12000)
4  Ers  <- TopK(S_RS(E0), 12000)
5  Epool <- Unique(Eppg union Eppn union Ers)
6  compute normalized hybrid score S_hyb for Epool
7  E <- GreedySelect(P, degree=96, label_pair_quota=800)
8  E <- GreedySelect(Epool sorted by S_hyb, degree=96, quota=800, seed=E)
9  if |E| < K: repeat with degree=128 and quota=1200
10 if |E| < K: append remaining pairs by S_hyb without structural caps
11 return first min(K, |Epool|) pairs in E
```

默认实现超参数汇总如下：

| Item | Default |
|---|---:|
| Final pair budget \(K\) | 10,000 |
| PPG protected pool | 8,000 |
| PPG union pool | 10,000 |
| PPN completion pool | 12,000 |
| RS-prior pool | 12,000 |
| Strict in/out degree cap | 96 / 96 |
| Relaxed in/out degree cap | 128 / 128 |
| Strict/relaxed label-pair quota | 800 / 1,200 |
| Pair scoring block size | 200,000 |

RSGP 目前仅用于 inference-time filtering，不参与 relation-head 反向传播，因此 D/E 消融能够在同一 checkpoint 上隔离 candidate graph 的影响。

### 4.3 Pair and union representation

对每个候选 pair \((i,j)\)，原 RPCM pair extractor 融合 subject/object RoI feature、word embedding、相对 OBB position encoding 和 union visual feature，得到实体表示 \(h_i^e\) 及初始关系表示 \(h_{ij}^{r,0}\)：

\[
h_{ij}^{r,0}=\Phi_{pair}
(f_i,f_j,g(l_i),g(l_j),\phi_{pos}(b_i,b_j),f_{ij}^{union}).
\]

该模块沿用 RPCM，不作为本文新贡献；本文主要改变其后的 relation graph construction 和 message passing。

### 4.4 Role-aware dual-view relation GNN

设候选图含 \(N\) 个实体和 \(E\) 条关系边。定义 subject/object incidence matrices：

\[
M_s\in\{0,1\}^{N\times E},\quad
(M_s)_{v,e}=\mathbb 1[v=s_e],
\]

\[
M_o\in\{0,1\}^{N\times E},\quad
(M_o)_{v,e}=\mathbb 1[v=o_e].
\]

SGG-ToolKit 使用统一关系邻接：

\[
A_u=\mathbb 1\left[(M_s+M_o)^\top(M_s+M_o)>0\right]-I,
\]

它将 shared-subject、shared-object 以及 subject-object cross-role sharing 合并。其实体邻接是每张图内部除自身外的完全图：

\[
A_e^{base}=\operatorname{blockdiag}
\left(\mathbf 1_{N_i\times N_i}-I_{N_i}\right).
\]

原版六路 collection 的统一形式为：

\[
\operatorname{Collect}_q(T,S,A)
=\frac{A\operatorname{ReLU}(SW_q+b_q)}{A\mathbf 1+\epsilon},
\]

其中 (q\in\{0,\ldots,5\}) 分别对应 relation→subject entity、relation→object entity、subject entity→relation、object entity→relation、entity→entity 和 relation→relation。原版 update 不含额外投影或激活：

\[
\operatorname{Update}(T,C)=T+C.
\]

因此 SGG-ToolKit baseline 的每轮更新为：

\[
H^{e,l+1}=H^{e,l}+\frac{1}{3}\left[
\operatorname{Collect}_4(H^{e,l},H^{e,l},A_e^{base})
+\operatorname{Collect}_0(H^{e,l},H^{r,l},M_s)
+\operatorname{Collect}_1(H^{e,l},H^{r,l},M_o)
\right],
\]

\[
H_{base}^{r,l+1}=H^{r,l}+\frac{1}{3}\left[
\operatorname{Collect}_2(H^{r,l},H^{e,l},M_s^\top)
+\operatorname{Collect}_3(H^{r,l},H^{e,l},M_o^\top)
+\operatorname{Collect}_5(H^{r,l},H^{r,l},A_u)
\right].
\]

六个 collection unit 和 update unit 在所有传播轮次间共享，分类器使用最后一轮 (H_{base}^{r,L})。本文的 dual-view GNN 则分别构造：

\[
A_s=\mathbb 1[M_s^\top M_s>0]-I,
\qquad
A_o=\mathbb 1[M_o^\top M_o>0]-I.
\]

实体图由当前 relation endpoints 构造：

\[
A_e=\mathbb 1[M_oM_s^\top+(M_oM_s^\top)^\top>0]-I.
\]

对任意邻接 \(A\)，dense residual GCN 为：

\[
\tilde A=A+I,\qquad
\hat A=D^{-1/2}\tilde A D^{-1/2},
\]

\[
X=\operatorname{Drop}(H),
\qquad
\operatorname{GCN}(H,A)=\sigma
\left(\hat AXW+b+X\right).
\]

实体到关系的 role-specific collection 使用 incidence attention：

\[
\operatorname{Collect}_{u}(H^e,M_u)
=\frac{M_u^\top\operatorname{ReLU}(H^eW_u+b_u)}
{M_u^\top\mathbf 1+\epsilon},\qquad u\in\{s,o\}.
\]

第 \(l\) 层更新为：

\[
H^{e,l+1}=\operatorname{GCN}_e(H^{e,l},A_e),
\]

\[
H^{r,l+1}=\frac{1}{4}\left[
\operatorname{Collect}_{s}(H^{e,l},M_s)
+\operatorname{Collect}_{o}(H^{e,l},M_o)
+\operatorname{GCN}_r(H^{r,l},A_s)
+\operatorname{GCN}_r(H^{r,l},A_o)
\right].
\]

shared-subject 和 shared-object 视角共享同一组 \(\operatorname{GCN}_r\)
参数。与 SGG-ToolKit baseline 相比，本模块同时改变 relation graph 的
角色分解和 GNN 更新方式。现有 6850 结果验证了该模块作为完整基础模型的
可用性，但由于没有完成只替换 GNN block 的同初始化重训练，本文不把它与
paper baseline 的数值差直接解释为该模块的独立因果增益。经过 \(L\) 轮传播后，聚合所有层：

\[
\bar H^r=\frac{1}{L+1}\sum_{l=0}^{L}H^{r,l},
\]

并对 entity states 使用相同的 all-layer aggregation：

\[
\bar H^e=\frac{1}{L+1}\sum_{l=0}^{L}H^{e,l}.
\]

PredCls 的 object output 由 GT labels 直接给出，因此 \(\bar H^e\) 不改变
PredCls 指标；SGCls/SGDet 的 object refinement 使用该聚合表示。

先将 relation context 下采样到分类器维度：

\[
\bar H^r_{down}=\operatorname{DownSamp}(\bar H^r),
\]

\[
H^{rel}=\operatorname{LayerNorm}
\left(\bar H^r_{down}+\operatorname{MLP}_{res}(\bar H^r_{down})\right).
\]

PredCls 使用 \(L=4\)，SGCls/SGDet 使用 \(L=3\)。

### 4.5 Semantic prototype classifier

设 predicate 类别文本 embedding 为 \(t_c\)，映射并归一化得到 prototype：

\[
p_c=\operatorname{Norm}(W_pt_c).
\]

本文使用的 `6850_4135.pth` 版本为每类一个 300-D GloVe
prototype（\(K=1\)），没有 multi-prototype 聚类，也没有 batch-wise visual
prototype 更新。初始化时保存一个静态映射锚点

\[
p_c^{0}=\operatorname{Norm}(W_p^{0}t_c),
\]

训练期间 \(W_p\) 和 \(t_c\) 对应的 `base_prototypes` 可学习，但
\(p_c^0\) 不更新。当前 prototype 为：

\[
\bar p_c=\operatorname{Norm}
\left(\rho p_c+(1-\rho)p_c^{0}\right),
\qquad \rho=0.9.
\]

因此，代码中沿用的 `proto_ema` 名称只表示**初始化锚点**，并不是训练中
持续更新的 visual EMA。该细节与 6850 checkpoint 的参数和训练行为保持一致。
历史 HPRC 运行发生在恢复该细节之前，其保存的 `proto_ema`
已经几乎对齐到当时的 mapped prototype；加载 checkpoint 后该 buffer 是固定
模型状态，所以现有 PredCls 推理数字仍然有效。论文中应把“静态初始化锚点”
明确归属于 6850 base，而不要声称后续 HPRC checkpoint 从头到尾采用了完全相同的
buffer 更新过程。

relation feature 投影为：

\[
z_{ij}=\operatorname{Norm}(\Phi_{proj}(h_{ij}^{rel})),
\]

predicate logit 为 cosine similarity：

\[
z^{proto}_{ij,c}=\tau z_{ij}^{\top}\bar p_c,
\]

其中 \(\tau\) 为可学习 temperature scale。prototype pull loss 为：

\[
\mathcal L_{pull}=\lambda_{pull}
\frac{1}{|\mathcal E_{train}|}
\sum_{(i,j)}\left(1-z_{ij}^{\top}\bar p_{y_{ij}}\right).
\]

使用 ETF separation，\(C=C_r\)，prototype Gram matrix \(G_{cd}=\bar p_c^\top\bar p_d\)，则：

\[
\mathcal L_{sep}=\lambda_{sep}
\frac{1}{C(C-1)}
\sum_{c\ne d}\left(G_{cd}+\frac{1}{C-1}\right)^2.
\]

6850 还保留一组固定语义配对 \(\mathcal A\) 的历史 margin 正则：

\[
\mathcal L_{ant}=\frac{\lambda_{ant}}{|\mathcal A|}
\sum_{(a,b)\in\mathcal A}
\max\left(0,m_{ant}-\bar p_a^\top\bar p_b\right),
\]

其中 \(\lambda_{ant}=0.1\)，\(m_{ant}=-0.2\)。这里忠实保留 6850
实现中的符号；尽管变量名为 `ant`，该表达式实际约束 cosine similarity
不要低于 \(-0.2\)，不应在论文中解释为普通的“把反义类无限推远”。

当前参数为 \(\lambda_{pull}=0.2,\lambda_{sep}=0.01\)。该 prototype
classifier 和三项 prototype regularization 源自既有 RPCM-6850 基础模型；
本文保留它们作为稳定分类基础，而不将其列为新贡献。

### 4.6 Hard-Predicate Residual Calibration

由训练集 predicate counts \(n_c\) 定义类别先验：

\[
\pi_c=\frac{\max(n_c,1)}{\sum_k\max(n_k,1)}.
\]

记 \(\tilde z\) 为最终 predicate logits：未启用 HPRC 时 \(\tilde z=z^{proto}\)，启用时使用下文定义的融合 logits \(z'\)。主分类损失保持 CE：

\[
\mathcal L_{CE}=\operatorname{CE}(\tilde z,y).
\]

仅增加小权重 logit-adjust auxiliary：

\[
\mathcal L_{LA}=\operatorname{CE}
\left(\tilde z+\tau_{LA}\log\pi,y\right),
\]

\[
\mathcal L_{cls}=\mathcal L_{CE}
+\lambda_{LA}\mathcal L_{LA},
\qquad \lambda_{LA}=0.1,\quad\tau_{LA}=0.5.
\]

定义困难谓词集合 \(\mathcal H\)。该集合综合训练频率、历史验证召回和稳定混淆模式确定，因而不等同于简单的低频集合。轻量残差校准头输出 \(a_{ij}\in\mathbb R^{|\mathcal H|}\)，并融合到对应类别：

\[
\alpha=\alpha_{max}\tanh(s),\qquad
z'_{ij,c}=z^{proto}_{ij,c}+\alpha a_{ij,c},\quad c\in\mathcal H,
\]

其中 \(s\) 初始化为 0，\(\alpha_{max}=0.3\)，因此初始模型严格退化为原 prototype logits，并令 \(\tilde z=z'\)。HPRC target 为当前单标签 predicate 在 \(\mathcal H\) 上的 binary vector（当前协议下至多一个正类），损失为：

\[
\mathcal L_{HPRC}=-\frac{1}{|\mathcal E||\mathcal H|}
\sum_{(i,j),c\in\mathcal H}
\left[w_cy_{ij,c}\log\sigma(a_{ij,c})
+(1-y_{ij,c})\log(1-\sigma(a_{ij,c}))\right],
\]

\[
w_c=\operatorname{clip}
\left(\frac{\sum_{k>0}n_k}{n_c},1,20\right).
\]

6850 base 的预训练目标为：

\[
\mathcal L_{6850}=\mathcal L_{CE}
+\mathcal L_{pull}
+\mathcal L_{sep}
+\mathcal L_{ant}.
\]

现有 `best_bgfirst.pth` 对应的历史 HPRC 训练日志只包含 `pull`、`sep`
和新增的 LA/HPRC 项，没有再次记录 `ant`，因此用于本文结果的 calibration
目标应写为：

\[
\mathcal L_{HPRC\text{-}stage}=\mathcal L_{CE}
+0.1\mathcal L_{LA}
+0.2\mathcal L_{HPRC}
+\mathcal L_{pull}
+\mathcal L_{sep}.
\]

换言之，`ant` 的影响已经包含在 warm-start 的 6850 权重中，但不应伪装成
现有 HPRC 阶段额外优化过的 loss。

SGCls/SGDet 额外加入 object refinement CE：

\[
\mathcal L=\mathcal L_{pred}
+\mathbb 1[task\ne PredCls]\mathcal L_{obj}.
\]

冻结 detector 的 RPN/box losses 不参与当前 relation-stack 优化。

### 4.7 Task-specific object refinement

对于 SGCls/SGDet，object refinement feature 为：

\[
q_i^{obj}=W_o[f_i,g(l_i\text{ or }P_i),\phi_{pos}(b_i)],
\qquad
z_i^{obj}=W_{cls}q_i^{obj},
\]

其中，SGCls/SGDet 使用 detector distribution \(P_i\) 与 object embedding matrix 的加权和；PredCls 直接返回 GT label one-hot logits，不计算 object classification loss。

图约束推理时，每个候选 pair 只保留分数最高的前景 predicate：

\[
\hat r_{ij}=\arg\max_{c\in\{1,\ldots,C_r-1\}}p(r=c\mid i,j).
\]

用于全图 top-\(K\) 排序的 triplet score 为：

\[
S^{triplet}_{ij}
=p(r=\hat r_{ij}\mid i,j)\,
p(l_i\mid b_i)\,p(l_j\mid b_j).
\]

PredCls 中 object scores 为 1；SGCls/SGDet 中保留 object confidence 的乘积。因此，RSGP 改变的是进入关系头的候选图，而最终排序仍遵循与 STAR evaluator 一致的 graph-constrained triplet scoring。

### 4.8 RSGP 默认流程摘要

```text
semantic filter
→ PPG protected top-8000
→ PPN top-12000 recall pool
→ RS-prior top-12000 pool
→ hybrid ranking
→ max in/out degree 96
→ label-pair quota 800
→ relaxed second pass
→ final top-10000
→ RPCM relation reasoning
```

---

## 5. 实验设置

### 5.1 Dataset

| Item | Setting |
|---|---:|
| Dataset | STAR |
| Box representation | OBB |
| Foreground object classes | 48 |
| Foreground predicate classes | 58 |
| Train images | 771 |
| Validation images | 245 |
| Test images | 264 |
| Split | Fixed split |
| Train duplicate relations | Filtered for current single-label GC protocol |
| Pair budget | top-10,000 |

### 5.2 Tasks

| Task | Boxes | Object labels | Predicate |
|---|---|---|---|
| PredCls | GT | GT | Predicted |
| SGCls | GT | Predicted | Predicted |
| SGDet | Predicted | Predicted | Predicted |

为对齐 SGG-ToolKit/STAR，当前 SGCls pair filter 使用 GT object labels；SGDet pair filter/训练匹配使用 `matched_gt` labels，但最终 object/predicate 输出仍由模型产生。论文中必须明确写出这一 legacy protocol，并建议附加 `pred` label-source 的严格协议作为补充实验。

### 5.3 Metrics

主指标为：

- Recall: `R@K`；STAR 原论文中记作 `MR@K`；
- Mean Recall: `mR@K`；STAR 原论文中记作 `mMR@K`；
- Harmonic Mean Recall:

设测试集中有效图像数为 \(T\)，第 \(n\) 张图的 GT 有向 pair 集为 \(G_n\)，top-\(K\) 预测匹配到的 GT relation index 集为 \(M_n^K\)。当前 evaluator 先计算每图 recall，再对图像取平均：

\[
R@K=\frac{1}{T}\sum_{n=1}^{T}
\frac{|M_n^K|}{\max(|G_n|,1)}.
\]

对前景 predicate \(c\in\{1,\ldots,C_r-1\}\)，令 \(T_c\) 为含该类 GT 的图像集合，\(G_{n,c}\) 和 \(M_{n,c}^K\) 分别为该图像中类别 \(c\) 的 GT 与命中集合，则：

\[
R_c@K=\frac{1}{|T_c|}\sum_{n\in T_c}
\frac{|M_{n,c}^K|}{|G_{n,c}|},
\]

\[
mR@K=\frac{1}{C_r-1}\sum_{c=1}^{C_r-1}R_c@K.
\]

没有在测试集中出现的 predicate 在当前实现中贡献 0；因此主表必须保持相同的 58 类词表和 fixed test split。综合指标定义为：

\[
HMR@K=\frac{2\cdot R@K\cdot mR@K}{R@K+mR@K}.
\]

主表报告 `K=1500,2000`，补充材料可报告 `K=1000`。所有方法必须使用同一 evaluator、同一 graph-constrained protocol 和同一 filter label source。

### 5.4 Detector and relation model

| Component | Setting |
|---|---|
| OBB detector | Swin-L OBB detector from `pretrained/OBB_swin_L_OBD.pth` |
| Detector status | Frozen |
| Relation predictor | `RPCM_ORIGINAL_LEGACY` |
| Pair feature dimension | 4096 |
| RPCM MLP dimension | 2048 |
| PredCls propagation steps | 4 |
| SGCls/SGDet propagation steps | 3 |
| Relation graph | 6850 dual view: shared-subject and shared-object graphs |
| Graph output | Mean of the input state and all propagation states |
| Predicate classifier | One GloVe prototype per class + static initialization anchor |
| Prototype losses | pull 0.2, ETF separation 0.01, historical pair-margin 0.1 |
| Default filter | RSGP Hybrid 8000/2000 for proposed method |

### 5.5 Optimization

| Task | Batch | Optimizer | Configured base LR | Effective stop | LR milestones |
|---|---:|---|---:|---:|---|
| RPCM-6850 base | 16 | SGD | 0.016 effective | 20,000 iters; selected at 17,600 | 13,000, 18,000 iters |
| Historical PredCls HPRC checkpoint (reported results) | 16 | SGD | 0.016 | 200 epochs | 6000, 8500, 10000 iters |
| Clean PredCls HPRC scratch config (future controlled run) | 16 | SGD | 0.016 | 300 epochs | 10,000, 14,000, 16,000 iters |
| SGCls | 16 | SGD | 0.001 with batch scaling | 15000 iters | 8000, 13000 |
| SGDet | 8 | SGD | 0.001 with batch scaling | 12000 iters | 8000, 10000 |

Common settings: momentum 0.9, weight decay `1e-4`, gradient clipping 5.0. The
RPCM-6850 run used 500 warmup iterations. The historical PredCls HPRC
checkpoint reported in the existing tables starts from `6850_4135.pth`; its
residual-calibration loss weight is 0.2 and its logit-adjust auxiliary weight
is 0.1 with `tau=0.5`. For future controlled training,
`bash scripts/run_star_tail_aux.sh` now initializes only
`pretrained/OBB_swin_L_OBD.pth`; the corrected exact-6850 relation stack and
HPRC parameters start from their configured initializers and use the longer
scratch schedule shown above. This change does not alter the provenance of the
already reported checkpoint. The implementation keeps the historical
`tail_aux` configuration/checkpoint names for compatibility; HPRC is the
paper-facing name of the combined calibration stage.

### 5.6 Reproducibility fields to complete before submission

| Field | Value |
|---|---|
| GPU model | **TBD** |
| CUDA version | **TBD** |
| PyTorch version | **TBD** |
| Number of runs/seeds | Current results are single runs; **TBD** |
| Random seed | Current training entry does not explicitly set one; **TBD/fix required** |
| Checkpoint selection split | Existing runs used test-time periodic evaluation; see protocol warning below |
| Inference batch size | PredCls 2; SGCls 1; SGDet 1 unless overridden |

---

## 6. Main results

### 6.1 PredCls main comparison

The STAR baseline row is copied from Table IV of the STAR paper. Project rows use the current standardized evaluator and `best_bgfirst.pth`.

| Method | Pair filter | R@1500 | R@2000 | mR@1500 | mR@2000 | HMR@1500 | HMR@2000 | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| STAR RPCM (SGG-ToolKit) | PPG | 64.23 | 65.86 | 41.24 | 42.30 | 50.23 | 51.51 | Paper reported |
| Role-aware RCA base (`6850_4135.pth`) | PPG | 68.12 | 69.68 | 41.02 | 42.18 | 51.21 | 52.55 | Existing standardized test |
| Ours: Role-aware RCA + HPRC | PPG | 67.61 | 69.17 | 42.91 | 44.14 | 52.50 | 53.89 | Existing standardized test |
| Ours: Role-aware RCA + HPRC | PPN | 67.34 | 68.84 | 42.33 | 43.50 | 51.98 | 53.31 | Existing standardized test |
| **Ours: Role-aware RCA + HPRC + RSGP** | **RSGP Hybrid 8000/2000** | **69.71** | **71.02** | **44.81** | **45.93** | **54.55** | **55.78** | Existing standardized test |

Compared with the paper-reported STAR RPCM baseline, the current full PredCls result improves:

| Metric | @1500 | @2000 |
|---|---:|---:|
| R | +5.48 | +5.16 |
| mR | +3.57 | +3.63 |
| HMR | +4.32 | +4.27 |

Suggested result paragraph:

> On PredCls, the proposed full model reaches 71.02% R@2000, 45.93% mR@2000 and 55.78% HMR@2000, outperforming the STAR RPCM baseline by 5.16, 3.63 and 4.27 percentage points, respectively. The simultaneous improvements in R and mR indicate that the method benefits both frequent and systematically under-recognized predicates rather than trading one metric for the other.

### 6.2 SGCls main comparison

Current project values correspond to the training-time test evaluation that produced `outputs/star_sgcls_obb_train/model_best_HR.pth` at epoch 237. A final one-shot evaluation should be run before submission.

| Method | Pair filter | R@1500 | R@2000 | mR@1500 | mR@2000 | HMR@1500 | HMR@2000 | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| STAR RPCM (SGG-ToolKit) | PPG | 51.29 | 52.72 | 30.04 | 30.85 | 37.89 | 38.92 | Paper reported |
| Ours | PPG | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | Run cross-task eval |
| **Ours** | **RSGP Hybrid 8000/2000** | **56.06** | **57.12** | **32.22** | **32.99** | **40.92** | **41.82** | Existing; one-shot replay pending |

The current best-HMR SGCls checkpoint is higher than the paper-reported STAR RPCM by 4.77/4.40 points in R, 2.18/2.14 points in mR, and 3.03/2.90 points in HMR at K=1500/2000. These differences remain preliminary until the standalone one-shot evaluation is completed.

### 6.3 SGDet main comparison

Current project values correspond to the training-time test evaluation that produced `outputs/star_sgdet_obb_train/model_best_HR.pth` at epoch 93. A final standalone one-shot evaluation should be run before submission.

| Method | Pair filter | R@1500 | R@2000 | mR@1500 | mR@2000 | HMR@1500 | HMR@2000 | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| STAR RPCM (SGG-ToolKit) | PPG | 27.23 | 28.50 | 11.53 | 12.07 | 16.20 | 16.96 | Paper reported |
| Ours | PPG | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | **TBD** | Run cross-task eval |
| **Ours** | **RSGP Hybrid 8000/2000** | **35.68** | **36.42** | **19.07** | **19.77** | **24.85** | **25.63** | Existing; one-shot replay pending |

The current best-HMR SGDet checkpoint improves over the paper-reported STAR RPCM by 8.45/7.92 points in R, 7.54/7.70 points in mR, and 8.65/8.67 points in HMR at K=1500/2000. Because SGDet is sensitive to detector/NMS/label-source details, these values should be claimed only after a standalone one-shot evaluation confirms the same protocol.

### 6.4 Supplementary @1000 PredCls results

| Method | R@1000 | mR@1000 | HMR@1000 |
|---|---:|---:|---:|
| Role-aware RCA + HPRC + PPG | 65.22 | 40.90 | 50.28 |
| Role-aware RCA + HPRC + PPN | 65.19 | 40.53 | 49.98 |
| Role-aware RCA + HPRC + RSGP | **67.44** | **42.78** | **52.35** |

---

## 7. Ablation studies

### 7.1 Existing-checkpoint ablation（primary retrospective ablation）

Most completed experiments follow the same RPCM-6850 lineage. The base row is
the original `6850_4135.pth`; the HPRC model was initialized from that
checkpoint, and PPG/RSGP evaluate exactly the same HPRC checkpoint. This
allows the existing PredCls results to be used without retraining after the
6850 compatibility restoration: the restoration adds no state-dict tensors,
and PredCls relation logits loaded from these checkpoints are unchanged.

| ID | Model/checkpoint lineage | Dual-view RCA | HPRC | Filter | R@1500 | R@2000 | mR@1500 | mR@2000 | HMR@1500 | HMR@2000 |
|---|---|:---:|:---:|---|---:|---:|---:|---:|---:|---:|
| A | STAR RPCM, paper reported |  |  | PPG | 64.23 | 65.86 | 41.24 | 42.30 | 50.23 | 51.51 |
| B | RPCM `6850_4135.pth` | ✓ |  | PPG | 68.12 | 69.68 | 41.02 | 42.18 | 51.21 | 52.55 |
| C | `best_bgfirst.pth`, warm-started from B | ✓ | ✓ | PPG | 67.61 | 69.17 | 42.91 | 44.14 | 52.50 | 53.89 |
| D | same checkpoint as C | ✓ | ✓ | RSGP Hybrid 8000/2000 | **69.71** | **71.02** | **44.81** | **45.93** | **54.55** | **55.78** |

The valid interpretations are deliberately limited:

- `C − B` measures the **combined warm-started HPRC stage**. It
  increases mR by 1.89/1.96 points and HMR by 1.29/1.34 points at
  K=1500/2000, while R decreases by 0.51/0.51 points.
- `D − C` is a strict causal filter comparison because only the inference-time
  pair graph changes. R/mR/HMR all improve.
- `B − A` is useful as a comparison with the paper-reported baseline, but it
  is not an isolated GNN ablation: implementation history, training run and
  evaluator are not fully controlled. The paper may describe the role-aware
  module and report B, but must not claim that this row alone proves its
  independent gain.
- Separate LA-only and residual-head-only improvements are not claimed from this table.
  The unfinished scratch A/B/C runs remain optional supplementary experiments,
  not required inputs to the main retrospective table.

### 7.2 Pair proposal comparison and graph-quality evidence

This table uses the standardized `best_bgfirst.pth` results.

| Filter | Final GT pair coverage | R@2000 | mR@2000 | HMR@2000 |
|---|---:|---:|---:|---:|
| PPG | 79.89 | 69.17 | 44.14 | 53.89 |
| PPN | **91.24** | 68.84 | 43.50 | 53.31 |
| RSGP Hybrid | 75.98 | **71.02** | **45.93** | **55.78** |

RSGP 的 PPN source pool coverage 为 95.83%，strict degree-cap stage coverage 为 61.76%，最终 coverage 为 75.98%。虽然最终 GT pair coverage 不是最高，RSGP 的三项 scene graph 指标均最高。这一结果直接支持核心动机：

> A candidate graph with higher pair recall is not necessarily more effective for downstream relational reasoning; graph structure and contextual compatibility are equally important.

### 7.3 Existing RSGP mode/quota search（exploratory）

The following grid used the historical `best.pth`/older evaluation run. It is useful for selecting Hybrid 8000/2000, but should not be mixed with the standardized main table without rerunning.

| Method | R@1500 | R@2000 | mR@1500 | mR@2000 | HMR@1500 | HMR@2000 | Final pair coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPG 10000 | 67.63 | 69.19 | 42.89 | 44.17 | 52.49 | 53.92 | 79.89 |
| PPN 10000 | 67.37 | 68.88 | 42.15 | 43.43 | 51.86 | 53.27 | **91.24** |
| RS only | 66.07 | 66.88 | 41.63 | 42.26 | 51.08 | 51.79 | 65.88 |
| PPN graph | **70.01** | **71.33** | 44.78 | 45.82 | **54.63** | 55.80 | 81.11 |
| Hybrid 9000/1000 | 69.78 | 71.10 | 44.69 | 45.92 | 54.49 | 55.80 | 76.01 |
| **Hybrid 8000/2000** | 69.76 | 71.08 | **44.80** | **45.96** | 54.56 | **55.82** | 75.98 |
| Hybrid 7000/3000 | 69.74 | 71.06 | 44.68 | 45.90 | 54.47 | 55.77 | 75.93 |

### 7.4 RSGP component ablation（reserved）

| Variant | PPN completion | RS priors | Degree control | Label-pair quota | Hard-predicate prior | R@2000 | mR@2000 | HMR@2000 |
|---|:---:|:---:|:---:|:---:|:---:|---:|---:|---:|
| Full RSGP | ✓ | ✓ | ✓ | ✓ | ✓ | 71.02 | 45.93 | 55.78 |
| w/o PPN completion |  | ✓ | ✓ | ✓ | ✓ | **TBD** | **TBD** | **TBD** |
| w/o RS priors | ✓ |  | ✓ | ✓ |  | **TBD** | **TBD** | **TBD** |
| w/o degree control | ✓ | ✓ |  | ✓ | ✓ | **TBD** | **TBD** | **TBD** |
| w/o label-pair quota | ✓ | ✓ | ✓ |  | ✓ | **TBD** | **TBD** | **TBD** |
| w/o hard-predicate prior | ✓ | geometry/anchor/topology | ✓ | ✓ |  | **TBD** | **TBD** | **TBD** |

### 7.5 Hard-predicate calibration analysis（existing 6850 → HPRC）

由于不同 predicate 的 GT 数量相差较大，主文采用带 `GT count` 的数值表，而不使用对所有类别赋予相同视觉权重的柱状图或哑铃图。这样可以同时呈现类别基数、绝对召回率和百分点变化。完整 58 类结果可放入补充材料；下表保留 HPRC 所关注的 15 个困难谓词，并同时报告提升和退化项，避免选择性展示。

| Predicate | GT count | RPCM-6850 R@2000 | RPCM-6850 + HPRC R@2000 | Delta (pp) |
|---|---:|---:|---:|---:|
| randomly docked at | 562 | 0.00 | 0.53 | +0.53 |
| randomly parked on | 375 | 2.40 | 3.35 | +0.95 |
| not run along | 36 | 0.00 | 5.00 | +5.00 |
| not parked alongside with | 655 | 7.45 | 17.71 | +10.26 |
| running along the different taxiway with | 466 | 7.15 | 19.07 | +11.92 |
| running along the same taxiway with | 116 | 11.36 | 13.79 | +2.43 |
| within danger distance of | 666 | 22.46 | 23.05 | +0.59 |
| incorrectly parked on | 61 | 20.28 | 26.94 | +6.66 |
| not docked alongside with | 528 | 0.66 | 0.93 | +0.27 |
| driving in the different lane with | 1737 | 13.47 | 13.35 | -0.12 |
| driving in the same lane with | 1499 | 5.20 | 4.67 | -0.53 |
| driving alongside with | 307 | 18.74 | 11.59 | -7.15 |
| indirectly connected to | 34 | 15.15 | 23.11 | +7.96 |
| indirectly transmit electricity to | 27 | 9.09 | 22.73 | +13.64 |
| not working on | 19 | 6.25 | 12.50 | +6.25 |

Values are percentages and come from two PPG evaluations with the same fixed
test split. HPRC strongly improves several systematically under-recognized
predicates, but the effect is not uniform: the two lane predicates and
`driving alongside with` do not improve. The result therefore supports HPRC
as a calibration mechanism that improves aggregate class balance; it does not
claim that every calibrated predicate necessarily increases. `Delta` is
reported in percentage points rather than relative percentage, which avoids
inflating changes for classes whose baseline recall is close to zero.

At the aggregate level, HPRC changes the PPG result from
`R/mR/HMR@2000 = 69.68/42.18/52.55` to `69.17/44.14/53.89`. Thus mR and HMR
increase by 1.96 and 1.34 points, respectively, while R decreases by 0.51
points. This trade-off motivates coupling HPRC with RSGP in the full model,
which raises all three metrics to `71.02/45.93/55.78`.

---

## 8. Qualitative/diagnostic experiments to include

### Figure A: PPG vs PPN vs RSGP candidate graph

预留图：对同一 STAR 图像可视化三种 filter 的 top-10,000 graph。

- blue: correctly retained GT pairs；
- red/purple: missed GT pairs；
- gray: non-GT candidate pairs；
- node size: degree；
- annotate max/mean degree and label-pair concentration.

预期展示：PPN 保留更多 GT pairs，但产生的局部结构不一定最适合 RPCM；RSGP 通过 degree/quota 重构图后获得更高 triplet recall。

### Figure B: Role-aware relation context

预留图：展示共享同一实体但角色不同的 relations。例如：

```text
(airplane_A, parked_on, apron)
(airplane_B, parked_on, apron)
(apron, adjacent_to, terminal)
```

统一图会把三条边直接混合；dual-view 图将前两条 shared-object 上下文与第三条跨角色上下文分开。

### Figure C: HPRC hard-predicate examples

建议选择：

- driving in the same/different lane with；
- within safe/danger distance of；
- parked/not parked alongside with；
- directly/indirectly connected to；
- working/not working on。

---

## 9. Minimal experiment execution checklist

### 9.1 PredCls existing-checkpoint table

No new relation-head training is required to reproduce the primary
retrospective table. Its three project rows are obtained with:

```bash
# RPCM-6850 dual-view base + PPG
bash scripts/eval_predcls_minimal_ablation.sh 6850

# Warm-started HPRC checkpoint + PPG
bash scripts/eval_predcls_minimal_ablation.sh D

# The exact same HPRC checkpoint + RSGP
bash scripts/eval_predcls_minimal_ablation.sh E
```

The STAR/SGG-ToolKit row is copied from the paper and is not retrained. The
scratch launchers `run_predcls_minimal_ablation.sh A|B|C` remain available for
future controlled supplementary analysis, but they are not prerequisites for
the numbers currently reported in Section 7.1.

For a new controlled HPRC run that does not inherit the historical 6850
checkpoint, use:

```bash
bash scripts/run_star_tail_aux.sh
```

This launcher forces `INIT_RPCM=''`. The frozen detector is loaded from
`pretrained/OBB_swin_L_OBD.pth` by the model config, while the corrected
dual-view RPCM, GloVe prototypes and HPRC head are initialized from scratch.
Its default output is `outputs/star_predcls_obb_hprc_scratch`, so it does not
overwrite the historical `outputs/star_predcls_obb_tail_aux` results used in
the current tables.

### 9.2 RSGP component ablation

```bash
bash scripts/eval_predcls_rsgp_ablation.sh FULL
bash scripts/eval_predcls_rsgp_ablation.sh NO_PPN
bash scripts/eval_predcls_rsgp_ablation.sh NO_RS
bash scripts/eval_predcls_rsgp_ablation.sh NO_DEGREE
bash scripts/eval_predcls_rsgp_ablation.sh NO_QUOTA
bash scripts/eval_predcls_rsgp_ablation.sh NO_TAIL
```

### 9.3 Cross-task PPG/RSGP evaluation

```bash
bash scripts/eval_cross_task_minimal.sh SGCLS_PPG
bash scripts/eval_cross_task_minimal.sh SGCLS_RSGP
bash scripts/eval_cross_task_minimal.sh SGDET_PPG
bash scripts/eval_cross_task_minimal.sh SGDET_RSGP
```

---

## 10. Protocol warnings before publication

### 10.1 Test-set checkpoint selection

Current historical runs use periodic test evaluation and save `model_best_HR.pth` according to test HMR. This is acceptable for internal exploration and reproduces the current workflow, but it is not a clean validation protocol.

Formal options:

1. Preferred: use STAR validation split for checkpoint selection and evaluate test exactly once；
2. Minimal retrospective: use a fixed epoch selected before inspecting test results；
3. If retaining current results, explicitly disclose the model-selection protocol and avoid claims based on tiny metric differences.

### 10.2 Initialization/training-budget fairness

The completed project experiments use the following checkpoint lineage:

```text
pretrained/OBB_swin_L_OBD.pth
  -> RPCM/weights/6850_4135.pth
  -> outputs/star_predcls_obb_tail_aux/best_bgfirst.pth
  -> PPG / PPN / RSGP evaluations of the same HPRC checkpoint
```

The restored 6850 compatibility path changes training behavior only when
initializing a new model: exact GloVe copying, static `proto_ema`, the historic
pair-margin loss and layer averaging. It adds no parameters or buffers and
checkpoint loading overwrites all affected learned tensors. Therefore the
existing 6850-derived **PredCls** inference results do not need to be repeated
solely because of this restoration.

This statement does not automatically cover SGCls/SGDet. The exact 6850 graph
path averages all entity states before object refinement, whereas an earlier
project revision used only the last entity state. PredCls returns GT one-hot
object logits and is unaffected; SGCls/SGDet use `out_obj` and should retain
their planned standalone one-shot evaluation under the frozen final code.

Fairness must nevertheless be described precisely. The 6850 → HPRC comparison
is warm-started calibration, not two independent from-scratch runs. It supports
the effect of the combined HPRC stage, but cannot isolate LA from the residual
calibration head. Only PPG → RSGP, using one identical checkpoint, is a strictly
controlled one-variable comparison. Optional scratch A/B/C runs may be used in
supplementary material if a reviewer requires an isolated GNN or LA ablation.

### 10.3 SGCls/SGDet label-source protocol

Current legacy-compatible settings are:

```text
SGCLS_FILTER_LABEL_SOURCE=gt
SGDET_FILTER_LABEL_SOURCE=matched_gt
```

These settings reproduce SGG-ToolKit's STAR-specific pair-filter behavior. They should be used for direct comparison, but strict predicted-label results should be provided in supplementary material if possible.

### 10.4 Single-run uncertainty

Current results are single runs without an explicitly fixed global random seed. Before making claims about improvements smaller than approximately 0.3 percentage points, run at least three seeds or report that the experiment is deterministic and verify it empirically.

---

## 11. TBD 获取与回填索引

### 11.1 通用回填规则

所有 `eval_*.sh` wrapper 都在后台启动任务。命令返回只表示进程已经启动，必须等待对应目录中的 `test_metrics.json` 写完，并确认 `test.log` 无 traceback 后再回填。

标准 JSON 中的指标单位为 `[0,1]`，论文表格使用百分数，因此按下列字段乘以 100：

```text
R@1500   <- metrics.R["1500"]  * 100
R@2000   <- metrics.R["2000"]  * 100
mR@1500  <- metrics.mR["1500"] * 100
mR@2000  <- metrics.mR["2000"] * 100
HMR@1500 <- metrics.HR["1500"] * 100
HMR@2000 <- metrics.HR["2000"] * 100
```

`candidate-stage-coverage.final` 等 coverage 同样乘以 100。当前 JSON 没有序列化逐 predicate recall；这部分应从同次运行的 `test.log` 中 `Per-Relation Recall` 表读取，不能混用另一个 checkpoint/filter 的日志。

### 11.2 环境和可复现性字段（Section 5.6）

运行：

```bash
mkdir -p outputs/paper_repro
python tools/check_environment.py \
  --require-cuda \
  --output outputs/paper_repro/environment.json
nvidia-smi --query-gpu=name,driver_version \
  --format=csv,noheader > outputs/paper_repro/gpu.csv
```

| TBD field | Source |
|---|---|
| GPU model | `outputs/paper_repro/gpu.csv` 第 1 列 |
| CUDA version | `environment.json -> mmcv_ops.torch_cuda`；同时保留 `compiled_cuda` 用于说明 mmcv 编译版本 |
| PyTorch version | `environment.json -> packages.torch.installed` |
| Number of runs/seeds | 当前结果应填 `1`；完成多 seed 后按实际成功运行数更新 |
| Random seed | 当前应填 `not explicitly fixed`，而不是虚构 seed；增加全局 seed 配置并重跑后再替换 |

### 11.3 SGCls/SGDet 主表（Sections 6.2–6.3）

使用完全相同的 task checkpoint，只改变 inference filter：

| Table row | Command | Result JSON |
|---|---|---|
| SGCls / PPG | `bash scripts/eval_cross_task_minimal.sh SGCLS_PPG` | `outputs/paper_cross_task/sgcls_ppg/test_metrics.json` |
| SGCls / RSGP one-shot replay | `bash scripts/eval_cross_task_minimal.sh SGCLS_RSGP` | `outputs/paper_cross_task/sgcls_rsgp/test_metrics.json` |
| SGDet / PPG | `bash scripts/eval_cross_task_minimal.sh SGDET_PPG` | `outputs/paper_cross_task/sgdet_ppg/test_metrics.json` |
| SGDet / RSGP one-shot replay | `bash scripts/eval_cross_task_minimal.sh SGDET_RSGP` | `outputs/paper_cross_task/sgdet_rsgp/test_metrics.json` |

SGDet 默认读取 `outputs/star_sgdet_detection_cache`。若实际使用其他 cache，命令必须显式加：

```bash
SGDET_DETECTION_CACHE_DIR=outputs/<the_exact_v5_cache> \
bash scripts/eval_cross_task_minimal.sh SGDET_PPG
```

PPG 和 RSGP 必须使用同一个 cache hash、同一个 checkpoint 和 `SGDET_FILTER_LABEL_SOURCE=matched_gt`，否则不能作为 filter 消融。

### 11.4 Existing-checkpoint PredCls ablation（Section 7.1）

| ID | Checkpoint/result role | Evaluation command | Existing result JSON/log |
|---|---|---|---|
| A | STAR paper-reported RPCM | none; copy Table IV | source PDF, Table IV |
| B | `/home/ubuntu/research/ssd/RPCM/weights/6850_4135.pth` | `bash scripts/eval_predcls_minimal_ablation.sh 6850` | `outputs/paper_ablation_predcls/B_dual_rca_6850_ppg/test_metrics.json` / `test.log` |
| C | `outputs/star_predcls_obb_tail_aux/best_bgfirst.pth` | `bash scripts/eval_predcls_minimal_ablation.sh D` | `outputs/star_predcls_obb_tail_aux_eval_ppg/test_metrics.json` / `test.log` (or standardized D output) |
| D | same checkpoint as C | `bash scripts/eval_predcls_minimal_ablation.sh E` | `outputs/star_predcls_obb_tail_aux_eval_RSGP/test_metrics.json` / `test.log` (or standardized E output) |

Rows B–D already exist and are filled in Section 7.1. Re-running these commands
is only a reproducibility check, not a new training requirement. Do not replace
row B with an unfinished `_scratch` or `_ft80` model. The optional scratch
launchers cannot be mixed into this table unless all compared variants are
retrained and selected with a common protocol.

### 11.5 RSGP component ablation（Section 7.4）

这些实验无需重新训练 relation head：

| Variant | Command | Result JSON |
|---|---|---|
| Full RSGP | `bash scripts/eval_predcls_rsgp_ablation.sh FULL` | `outputs/paper_ablation_rsgp/rsgp_full/test_metrics.json` |
| w/o PPN completion | `bash scripts/eval_predcls_rsgp_ablation.sh NO_PPN` | `outputs/paper_ablation_rsgp/rsgp_no_ppn_completion/test_metrics.json` |
| w/o RS priors | `bash scripts/eval_predcls_rsgp_ablation.sh NO_RS` | `outputs/paper_ablation_rsgp/rsgp_no_rs_priors/test_metrics.json` |
| w/o degree control | `bash scripts/eval_predcls_rsgp_ablation.sh NO_DEGREE` | `outputs/paper_ablation_rsgp/rsgp_no_degree_control/test_metrics.json` |
| w/o label-pair quota | `bash scripts/eval_predcls_rsgp_ablation.sh NO_QUOTA` | `outputs/paper_ablation_rsgp/rsgp_no_label_pair_quota/test_metrics.json` |
| w/o hard-predicate prior | `bash scripts/eval_predcls_rsgp_ablation.sh NO_TAIL` | `outputs/paper_ablation_rsgp/rsgp_no_tail_prior/test_metrics.json` |

Section 7.4 只取上述 JSON 的 `@2000` 三项指标。`FULL` 应先与 Section 6.1 的 RSGP 行数值对齐；若不一致，先检查 checkpoint 和环境变量，不应直接填表。

### 11.6 HPRC hard-predicate table（Section 7.5）

该表已经由 6850 base 和 HPRC checkpoint 的两次 PPG evaluation
回填，不需要增加实验分支：

| Column | Source |
|---|---|
| Count | `outputs/paper_ablation_predcls/B_dual_rca_6850_ppg/test.log` 的对应 predicate 行 `count` |
| Before HPRC R@2000 | `outputs/paper_ablation_predcls/B_dual_rca_6850_ppg/test.log` |
| After HPRC R@2000 | `outputs/star_predcls_obb_tail_aux_eval_ppg/test.log` |
| Delta | `After - Before`，统一按百分点报告 |

两次日志都使用 PPG 和相同 fixed test split。由于 after 模型同时包含 LA
和困难谓词残差校准头，该表应命名为“6850 → HPRC calibration”，不能写成
LA 或残差校准头的独立因果消融。论文表格中的 `tail_aux` 路径只表示历史
代码/文件名，不再作为方法名称。

### 11.7 当前 TBD 状态速查

| TBD group | Current state |
|---|---|
| Environment | 可立即运行 11.2 获取 |
| SGCls/SGDet PPG rows | 尚无标准 one-shot JSON；运行 11.3 |
| PredCls existing-checkpoint table | 已由 STAR paper、6850、HPRC+PPG、HPRC+RSGP 回填；无需新训练 |
| RSGP component rows | 尚未生成 `outputs/paper_ablation_rsgp/*` |
| HPRC per-predicate before/after | 已由 6850 PPG 与 HPRC+PPG 日志回填 |

---

## 12. Result provenance

### Paper baseline

- Source PDF: `Star A first-ever dataset and a large-scale benchmark for scene graph generation in large-size satellite imagery.pdf`
- Main SGG results: Table IV；
- PPG results: Table V；
- RPCM iteration/component ablations: Tables VI–VII。

### Current standardized PredCls results

```text
outputs/star_predcls_obb_tail_aux_eval_ppg/test_metrics.json
outputs/star_predcls_obb_tail_aux_eval_ppn/test_metrics.json
outputs/star_predcls_obb_tail_aux_eval_RSGP/test_metrics.json
```

### Reconstructed RPCM-6850 base

```text
checkpoint: /home/ubuntu/research/ssd/RPCM/weights/6850_4135.pth
evaluation: outputs/paper_ablation_predcls/B_dual_rca_6850_ppg/test_metrics.json
log:        outputs/paper_ablation_predcls/B_dual_rca_6850_ppg/test.log
historical training log: /home/ubuntu/research/ssd/RPCM/nohup.out
```

The checkpoint name records the historical R@1500/mR@1500 values at iteration
17,600 (`0.6850/0.4135`). The current project evaluation is separately reported
as `0.6812/0.4102`; these should not be silently interchanged.

### Historical RSGP search

```text
outputs/rsgp_grid/*/test_metrics.json
```

### Current SGCls/SGDet best-HMR results

```text
outputs/star_sgcls_obb_train/train.log
outputs/star_sgcls_obb_train/model_best_HR.pth
outputs/star_sgdet_obb_train/train.log
outputs/star_sgdet_obb_train/model_best_HR.pth
```

---

## 13. Final contribution paragraph template

> 本文面向大幅遥感图像中候选实体对数量庞大、关系上下文角色混叠以及困难谓词长期低召回等问题，提出一种图感知的 OBB 场景图生成方法。首先，RSGP 将 pair proposal 从独立 pair 排序重新表述为受遥感几何和图结构约束的候选图构建问题，在固定计算预算下联合利用 PPG 的精度、PPN 的覆盖率以及 OBB 几何、锚点和拓扑先验。其次，角色感知关系上下文模块分别在 shared-subject 与 shared-object 图上传播信息，避免统一关系邻接造成的跨角色语义混合。最后，HPRC 将弱 logit-adjust 辅助监督与零初始化残差校准头结合，在保留主 CE 分类器稳定性的同时修正系统性低召回谓词。STAR OBB 上的 PredCls、SGCls 和 SGDet 实验表明，该方法能够同时提高 R、mR 和 HMR；同一 checkpoint 上的 filter 消融验证了 RSGP 的直接贡献，而 6850 到 HPRC 模型的回溯比较表明困难谓词残差校准能够提高类别均衡召回率。
