---
title: "Kimi K3 学习笔记（四）：Muon 优化器"
date: 2026-09-21
description: "围绕 Kimi K3 报告，学习和整理 Muon 优化器及其相关工作。"
tags: [llm, learning-log, kimi, optimizer]
references:
  - key: adam
    title: "Adam: A Method for Stochastic Optimization"
    authors: "Kingma & Ba · ICLR 2015"
    url: https://arxiv.org/abs/1412.6980
  - key: adamw
    title: "Decoupled Weight Decay Regularization"
    authors: "Loshchilov & Hutter · ICLR 2019"
    url: https://arxiv.org/abs/1711.05101
  - key: adagrad
    title: "Adaptive Subgradient Methods for Online Learning and Stochastic Optimization"
    authors: "Duchi, Hazan & Singer · JMLR 2011"
    url: https://jmlr.org/papers/v12/duchi11a.html
  - key: shampoo
    title: "Shampoo: Preconditioned Stochastic Tensor Optimization"
    authors: "Gupta, Koren & Singer · ICML 2018"
    url: https://arxiv.org/abs/1802.09568
  - key: orthogonalising-gradients
    title: "Orthogonalising Gradients to Speed Up Neural Network Optimisation"
    authors: "2022"
    url: https://arxiv.org/abs/2202.07052
  - key: prodigy
    title: "Prodigy: An Expeditiously Adaptive Parameter-Free Learner"
    authors: "Mishchenko & Defazio · ICML 2024"
    url: https://arxiv.org/abs/2306.06101
  - key: old-optimizer-new-norm
    title: "Old Optimizer, New Norm: An Anthology"
    authors: "Bernstein & Newhouse · 2024"
    url: https://arxiv.org/abs/2409.20325
  - key: muon
    title: "Muon: An Optimizer for Hidden Layers in Neural Networks"
    authors: "Jordan et al. · 2024"
    url: https://kellerjordan.github.io/posts/muon/
  - key: muon-code
    title: "Muon Official Code"
    authors: "KellerJordan"
    url: https://github.com/KellerJordan/Muon
  - key: kimi-k2
    title: "Kimi K2: Open Agentic Intelligence"
    authors: "Kimi Team · 2025"
    url: https://arxiv.org/abs/2507.20534
  - key: kimi-k3
    title: "Kimi K3: Open Frontier Intelligence"
    authors: "Kimi Team · 2026"
    url: https://arxiv.org/abs/2607.24653
---

这篇笔记围绕 Kimi K3 报告中的 Muon 优化器展开，阅读和整理 Muon 及其相关工作。

## 1. Adam：动量与自适应尺度

*Adam: A Method for Stochastic Optimization* {% include cite.html key="adam" %} 不同于普通的梯度更新。Adam 同时维护了梯度的动量估计和逐元素的梯度尺度估计，并用它们进行参数更新。这样一方面可以让更新方向更加稳定，另一方面也可以根据近期的梯度情况自动调整不同参数的更新步长。

但是这里有一个问题：对于普通的梯度优化，只要每一步足够小，整个优化过程就可以稳定下降；Adam 却对梯度进行了非常大幅度的修改，那么它如何保证优化仍然稳定下降呢？

我们可以先观察逐元素的尺度调整。尺度估计的除数始终大于零，因此它不会改变每个维度上的更新方向。换句话说，如果暂时不考虑动量，这种变化不会把梯度更新推到相反的一侧；在步长足够小时，它仍然是一个下降方向。

更主要的问题来自历史动量的平均。动量显然有可能让某一步的 loss 上升：我们可以构造一个变化比较狭窄的 landscape，Adam 可能因为延续之前的更新方向而错过这一小段下降区域，使某一步的 loss 反而上升。但从另一个角度想，我们不见得希望优化器敏感地捕捉这种狭窄的局部变化，因为这反而可能让优化过早陷入局部最优。因此，Adam 的优化过程本身就带有一种平滑和长期考虑。

所以我们也可以从平滑的角度理解 Adam。首先，一阶动量使用历史信息对更新方向进行平滑，这一步已经让它的优化行为不同于普通的梯度下降：它丢失了每一步严格下降的保证，而更加在意长期方向的稳定性。另一方面，二阶矩又在高维空间的不同维度上进行了一次平滑。某个维度上的梯度如果一直很大，对应的步长就会被压小，因此它也可以被理解为一种控制不同方向更新尺度的稳定项。

不过，原版 Adam 仍然不够完善，在一些简单的凸优化问题上也可能优化失败，因此后续还有很多针对 Adam 的修正以及新的更新方法。

## 2. AdamW：解耦 Weight Decay

*Decoupled Weight Decay Regularization* {% include cite.html key="adamw" %} 主要讨论的是 Adam 与 weight decay 的关系。

首先，weight decay 非常重要，因为我们需要对权重进行正则化。这本身是一种 inductive bias：我们希望权重尽量小，并认为这会带来各种好处。这里先把它看成一个经典且合理的结论。

对于普通的梯度下降，在 loss 中加入 L2 正则项，与直接对权重进行 weight decay 是等价的。但是 Adam 的更新已经不同于普通的梯度下降。如果仍然把正则项直接加入 loss，那么正则项产生的梯度也会经过 Adam 的动量和二阶矩缩放，L2 正则便不再等价于统一地衰减权重。

从直觉上看，正则项本来是一种抑制权重的效果，我们并不希望这种抑制本身又被二阶矩抑制，导致它无法发挥预期的正则作用。简单想象一种情况：当前权重非常大，正则项希望缩小它，但二阶矩同时压小了这个方向上的更新，于是权重衰减的实际强度也发生了变化。这显然不是我们原本想要的结果。

这里是否真的会导致优化无法收敛到原本的解，需要严格分析；相关工作也确实给出了 Adam 可能失效的简单例子。因此，如果我们希望在使用 Adam 的同时进行 weight decay，就需要一种更合适的处理方式。AdamW 的做法正是把 weight decay 从 Adam 的梯度更新中解耦出来，直接对权重进行衰减，让这一项不再受到 Adam 自适应缩放的影响。

简单来说，复杂的优化 landscape 仍然交给 Adam 去平滑；而权重正则这一项本身比较简单，就单独用类似梯度下降的方式直接收缩权重。这样既保留了 Adam 对原始优化问题的处理能力，也保证了 weight decay 可以按照预期发挥作用。

## 3. Full-matrix AdaGrad：在空间方向上进行平滑

在 Full-matrix AdaGrad {% include cite.html key="adagrad" %} 中，先把矩阵参数的梯度拉平成 $g_t\in\mathbb{R}^{mn}$，然后累计梯度的外积：

$$
H_t=\epsilon I_{mn}+\sum_{s=1}^{t}g_sg_s^\top.
$$

这里的 $g_sg_s^\top$ 表示不同梯度维度之间共同变化的大小。把它沿时间累计，就相当于维护了各个相关方向上的历史变化统计。对 $H_t$ 进行特征值分解后，可以看到有些方向的历史变化很大，有些方向的历史变化很小。最值得记住的是：

$$
g_t=\sum_k a_kq_k
\quad\longrightarrow\quad
H_t^{-1/2}g_t
=
\sum_k\frac{a_k}{\sqrt{\lambda_k}}q_k.
$$

这个做法实际上是另一种空间平滑：历史上经常出现、累计变化较大的方向会被抑制，而累计变化较小的方向会被相对放大。它和 Adam 的思路类似，只是 Adam 直接在每个坐标维度上进行缩放，而 Full-matrix AdaGrad 会先找出空间中的相关方向，再沿这些方向缩放。从空间角度理解，这似乎是一种更有几何意义的做法，但也带来了非常大的计算和存储复杂度。而且它仍然只是一种预条件方法，并不是真正掌握了 loss 的二阶 Hessian 信息。

## 4. Shampoo：按行列结构近似空间预条件

Shampoo {% include cite.html key="shampoo" %} 不再把矩阵梯度完全拉平。对于矩阵梯度 $G_t$，它分别累计左右两个统计量：

$$
L_t=\epsilon I+\sum_{s=1}^{t}G_sG_s^\top,
$$

$$
R_t=\epsilon I+\sum_{s=1}^{t}G_s^\top G_s,
$$

并使用下面的方式更新参数：

$$
W_t
=
W_{t-1}-\eta L_t^{-1/4}G_tR_t^{-1/4}.
$$

Shampoo 直接对矩阵做乘积，分别得到行方向和列方向之间的整体相关变化。它不再维护所有元素维度之间完整的相关关系，而是累计行、列两个方向的统计，再通过左右预条件抑制累计变化大的行列方向，并相对放大累计变化小的方向。左右两侧各使用 $-1/4$ 次方，合起来对应 Full-matrix AdaGrad 中整体的 $-1/2$ 次预条件。

这样做的计算量和存储量都比 Full-matrix AdaGrad 更小，但条件也更弱：它利用了参数原本的矩阵结构，却无法表达任意元素维度之间完整的相关关系。

## 5. Orthogonalising Gradients：归一化不同方向的更新

*Orthogonalising Gradients to Speed Up Neural Network Optimisation* {% include cite.html key="orthogonalising-gradients" %} 的思路很直接。神经网络中的很多梯度对应一个权重矩阵，而我们可以通过限制矩阵梯度的性质，间接给权重矩阵的更新带来一些希望得到的性质。这篇文章选择对矩阵梯度进行正交化，让不同方向的信息得到更充分的利用。

具体来说，对梯度矩阵进行 SVD：

$$
G=U\Sigma V^\top,
$$

然后把非零奇异值全部改成 $1$，得到正交化后的更新：

$$
G_{\mathrm{orth}}=UV^\top.
$$

这样相当于强制归一化各个奇异方向上的更新幅度，使原本幅度较小、容易被主方向遮盖的信息也能够得到充分利用。如果梯度中存在一个绝对占优的方向，其他方向携带的信息可能几乎无法影响更新，整体的信息利用效率就会很低；正交化试图避免这种情况。

但它也有明显的问题。第一，强制把所有非零奇异值归一化，可能会放大原本只是噪声的弱方向。第二，直接计算 SVD 本身比较昂贵，需要衡量额外计算是否值得。尽管如此，它的核心思路仍然和前面的工作一致：通过归一化不同方向的尺度，更充分地利用梯度中携带的信息。

## 6. Prodigy：自适应估计学习率

*Prodigy: An Expeditiously Adaptive Parameter-Free Learner* {% include cite.html key="prodigy" %} 主要调整的是整体步长，也就是学习率。它根据当前解相对初始解发生了多大变化，以及这个变化与历史梯度累积之间的关系，估计当前位置到最优解的距离尺度，并据此调整当前步长。

因此，Prodigy 改变的不是每个参数方向上的预条件方式，而是优化器整体使用的学习率尺度。它可以和 Adam 结合：Adam 继续负责动量与逐元素的自适应缩放，Prodigy 则在外层估计一个更合适的整体学习率。

## 7. Old Optimizer, New Norm：从范数理解优化器

*Old Optimizer, New Norm: An Anthology* {% include cite.html key="old-optimizer-new-norm" %} 是一篇机制解释工作。它认为前面的很多方法都可以从一阶梯度的角度理解：我们仍然希望沿着能够让 loss 下降最多的方向更新，只是需要在某种范数下限制更新的大小。一个统一的写法是：

$$
\min_{\Delta}\;\langle g,\Delta\rangle
\qquad
\text{s.t.}\quad
\|\Delta\|\leq \rho.
$$

梯度的反方向能够带来最大的局部下降，而不同优化器的 inductive bias，就体现在如何定义更新的范数以及约束半径上。

在这个框架中，如果暂时不考虑 EMA，Adam 可以理解为 $\ell_\infty$ 范数约束下的最速下降。每个维度都直接取到允许的最大幅度，因此梯度原本的绝对大小基本被消除，更新主要保留每个维度上的方向符号。论文把它写进统一形式的思路可以理解，不过我觉得这里的表达并不算特别漂亮：开头看起来像是在联合优化梯度范数，实际更直接的理解仍然是一个 $\ell_\infty$ 球上的约束问题。但这个区别并不是特别重要。

Shampoo 则可以理解为谱范数约束下的梯度优化：在谱范数不超过给定半径的条件下，寻找最接近原始下降目标的更新。前面通过 SVD 把奇异值变成 $1$ 的做法，也正好对应谱范数单位球上的最速下降方向。Prodigy 处理的则主要是约束半径，也就是整体应该走多远。

所以，这篇文章想表达的是：这些方法的核心仍然是在梯度范数上引入不同约束，把它们作为一种 inductive bias；它们不一定真的使用了 Hessian，因此可以继续放在一阶优化的框架里理解。这个视角对于认识不同方法的实质确实很有启发。

但这件事其实没有那么必然。第一，约束半径与步长有关，而步长本身完全可以引入二阶相关的信息。第二，范数也可以被定义得更加复杂，例如涉及 $U$ 和 $V$ 的结构时，实际上已经利用了不同元素方向之间的关联信息。因此，能够把方法写成范数约束形式，并不能说明我们不需要从二阶角度继续思考。

另外，上面的统一解释主要不考虑历史累计。一旦加入 EMA 或其他历史状态，当前更新就不再只由当前梯度决定，优化行为已经与一般的单步梯度最速下降产生了偏差。

## 8. Muon：动量之后的奇异值归一化

Muon {% include cite.html key="muon" %} 的目标可以简单理解为：和前面的 Shampoo 及梯度正交化类似，最终都希望归一化矩阵更新的奇异值，但具体实现有几点不同。具体实现可以参考官方代码 {% include cite.html key="muon-code" %}。

第一，Muon 不像 Shampoo 那样累计左右两个二阶统计矩阵，而是维护一个动量来累计历史梯度。它先把当前梯度融合进历史动量，再对融合后的动量矩阵进行奇异值归一化。因此，Muon 中参与正交化的并不是单步梯度，而是经过历史平滑的更新方向。

第二，Muon 不直接进行 SVD 分解，而是使用 Newton–Schulz 迭代近似矩阵的正交化。这个迭代可以让非零奇异值快速移动到 $1$ 附近，而原本为 $0$ 的奇异值仍然保持为 $0$，因此不会凭空增加矩阵的 rank。这样可以在保持矩阵秩结构的同时，用更适合加速器的矩阵乘法近似完成奇异值归一化。

## 9. Kimi K2：MuonClip 与 QK-Clip

Kimi K2 {% include cite.html key="kimi-k2" %} 在 Muon 的基础上又加入了几项面向大规模训练的处理。

第一，对 Muon 的更新幅度进行 RMS 校准。对于形状为 $A\times B$ 的正交化更新 $O_t$，K2 使用：

$$
\widetilde O_t
=
0.2\sqrt{\max(A,B)}\,O_t.
$$

正交化之后，矩阵元素的 RMS 会随着矩阵形状变化。这个缩放把不同尺寸权重矩阵的更新校准到更一致的 RMS 尺度，避免同一个学习率在不同矩阵形状上产生差异过大的实际更新。

第二，在 Muon 更新中加入 weight decay，对权重进行独立衰减。

第三，引入 QK-Clip。训练过程中统计每个 attention head 由 QK 计算得到的 score；如果某个 head 的 score 超过设定阈值，就按需缩放这个 head 对应的 Q、K 权重。它不是统一压低所有 attention head，而是只处理出现异常放大的 head，从而控制 attention logits 的数值增长。

## 10. Kimi K3：Per-Head Muon 与分布式 Newton–Schulz

Kimi K3 {% include cite.html key="kimi-k3" %} 进一步把 Muon 应用到每个 attention head。Q、K、V 不再作为完整的大矩阵统一进行 Muon 更新，而是按照 head 拆开，让每个 head 的 Q、K、V 分别进行奇异值归一化。这样可以在更细的粒度上控制不同 head 的更新尺度和方向。

另一方面，K3 将 Newton–Schulz 迭代改成分片的分布式计算。矩阵被切分并保存在不同设备上，计算过程中通过 P2P 通信获取当前矩阵乘法所需的对应分片，而不需要让每张卡都保存完整矩阵。这样可以把 Muon 的矩阵计算和存储开销分散到多个设备上，使它能够扩展到更大的模型。

## 总结与问题

读到这里有一个很重要的发现：无论是 Adam 还是 Muon，这些优化器似乎都不太关心梯度长度或者 scale 本身携带的信息，而更加关心方向信息。比如 Adam 或 Shampoo，如果去掉历史累计，实际上主要保留的是每个维度上的 sign，或者矩阵在空间中的单位方向。

那么我们自然会提出两个问题。第一，为什么神经网络优化会表现出这样的性质，梯度的方向信息为什么比原始幅度更加重要？这个现象背后是否还存在更深刻的内在联系？

第二，除了 SVD 或 Newton–Schulz 所实现的奇异值归一化，是否还存在其他更合适的更新方式？Muon 是否已经触及了这个问题比较本质的形式，以后的工作只需要继续在 Muon 上改进就够了？这些问题值得进一步思考，也需要通过后续工作和实验继续观察。
