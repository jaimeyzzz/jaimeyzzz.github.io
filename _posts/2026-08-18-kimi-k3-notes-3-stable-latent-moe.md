---
title: "Kimi K3 学习笔记（三）：Stable LatentMoE"
date: 2026-08-18
description: "学习和整理 Kimi K3 中 Stable LatentMoE 的设计。"
tags: [llm, learning-log, kimi, moe]
references:
  - key: sparsely-gated-moe
    title: "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"
    authors: "Shazeer et al. · 2017"
    url: https://arxiv.org/abs/1701.06538
  - key: switch-transformer
    title: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"
    authors: "Fedus, Zoph & Shazeer · 2021"
    url: https://arxiv.org/abs/2101.03961
  - key: deepseek-moe
    title: "DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models"
    authors: "Dai et al. · 2024"
    url: https://arxiv.org/abs/2401.06066
  - key: loss-free-balancing
    title: "Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts"
    authors: "Wang et al. · 2024"
    url: https://arxiv.org/abs/2408.15664
  - key: kimi-k2
    title: "Kimi K2: Open Agentic Intelligence"
    authors: "Kimi Team · 2025"
    url: https://arxiv.org/abs/2507.20534
  - key: latent-moe
    title: "LatentMoE: Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts"
    authors: "Elango et al. · 2026"
    url: https://arxiv.org/abs/2601.18089
  - key: kimi-k3
    title: "Kimi K3: Open Frontier Intelligence"
    authors: "Kimi Team · 2026"
    url: https://arxiv.org/abs/2607.24653
  - key: gshard
    title: "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding"
    authors: "Lepikhin et al. · 2020"
    url: https://arxiv.org/abs/2006.16668
  - key: moonep
    title: "MoonEP: A Perfectly Balanced Expert Parallelism Library via Dynamic Redundant Experts"
    authors: "Chen et al. · MoonshotAI · 2026"
    url: https://github.com/MoonshotAI/MoonEP
---

这一篇继续学习 Kimi K3，主要整理 Stable LatentMoE 的设计与思路。

## 1. Sparsely-Gated MoE

*Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer* {% include cite.html key="sparsely-gated-moe" %} 的核心思路是，模型的参数容量可以随着专家数量扩展，而每个 token 的计算量则通过 top-$K$ 选择来控制：

$$
y=\sum_{i=1}^{n}G_i(x)E_i(x).
$$

但是 gating 本身也需要保持平衡。选择 gate 时，文章在 logits 中加入高斯噪声，避免训练中的正反馈让负载逐渐集中到少数专家上。

同时，为了平衡负载，loss 也有特殊的设计。首先定义每个专家的 importance：

$$
\operatorname{Importance}(X)=\sum_{x\in X}G(x).
$$

然后通过下面的 loss，让不同专家的重要性尽量均衡：

$$
L_{\text{importance}}
=
w_{\text{importance}}
\frac{
\operatorname{Var}\left(\operatorname{Importance}(X)\right)
}{
\operatorname{Mean}\left(\operatorname{Importance}(X)\right)^2
}.
$$

但是 importance 统计的是 gating 权重之和，并没有直接考虑每个专家实际处理的 token 数量，所以还需要加入对 token 数量的平衡。对于每个 token，计算当前专家被 top-$K$ 选中的概率，也就是它的 noisy logit 大于第 $K$ 个 logit 的概率；再把不同 token 的概率加起来，得到专家的期望负载：

$$
\widehat{\operatorname{Load}}_i(X)
=
\sum_{x\in X}P(x,i).
$$

最后再对这个期望负载做均衡：

$$
L_{\text{load}}
=
w_{\text{load}}
\frac{
\operatorname{Var}\left(\widehat{\operatorname{Load}}(X)\right)
}{
\operatorname{Mean}\left(\widehat{\operatorname{Load}}(X)\right)^2
}.
$$

这里其实会产生一个很自然的疑问：既然 importance 不足以表达真实的 token 负载，所以还需要额外设计 load loss，那么是不是实际上只使用 load loss 就可以了？作者也做了对应的消融实验，结果还真是如此。虽然从直觉上看，不加入 importance loss 可能会出现 token 数量平均、但某些专家只是在滥竽充数的情况，不过实验中，单独使用 load loss 和同时使用 importance loss 与 load loss 的表现差异非常小。

这个结果其实也很好理解。这里仍然涉及网络 scaling 的问题：如果我们认为网络所做的事情，是压缩和归纳大量数据之间的关系模式，那么在学习足够充分的情况下，我们可能并不需要加入特殊机制来阻止空转。因为空转意味着网络使用了大量计算，却没有形成有效的信息压缩和归纳，这本身就是一种非常低效的状态。

类似的逻辑也可以转移到 attention 的 QK score 和 V 输出上。理论上，某个关系的 QK score 很高，但对应的 V 输出仍然可能接近于零；但是我们应该倾向于认为，score 较高的关系，其对应的 V 输出通常也是有效的。否则，这种压缩和归纳同样非常低效，网络应该把较高的 score 转移到更加关键的关系上。

## 2. Switch Transformer

*Switch Transformer* {% include cite.html key="switch-transformer" %} 使用 MoE 替换 Transformer 中的 FFN，每个 token 只激活少量专家。Switch 在这里进一步简化为每个 token 只选择一个专家，不过现代大语言模型通常仍然会同时激活多个专家。

设一个 batch 中有 $T$ 个 token、$N$ 个专家。首先定义专家 $i$ 实际收到的 token 比例：

$$
f_i
=
\frac{1}{T}
\sum_{x\in B}
\mathbf{1}\left[i^*(x)=i\right].
$$

这是 hard routing 得到的实际负载，不能求导。再定义 router 分配给专家 $i$ 的平均软概率：

$$
P_i
=
\frac{1}{T}
\sum_{x\in B}p_i(x).
$$

这里的 softmax 概率可以求导。Switch 的负载均衡 loss 是：

$$
L_{\mathrm{balance}}
=
\alpha N
\sum_{i=1}^{N}f_iP_i,
$$

其中论文使用：

$$
\alpha=10^{-2}.
$$

首先，这种计算方式比原版 Sparsely-Gated MoE 更简单。其次，这个版本会按照实际的 hard routing 选取结果来调整软概率。对于一组稳定但彼此接近的概率，如果实际 routing 一直选择同一个专家，原版方法的调整可能不会很大，而 Switch 的这个 loss 会带来更强的调整。

简单来说，Switch 优化的是可微分的平均概率，但是会使用实际负载对这个概率进行放大。原因是，平均概率虽然可以求导，却不能准确反映实际负载：两个专家的概率可能只相差很小，但 token 仍然会稳定地落入概率更大的专家，最终使两者的实际负载相差很大。加入实际负载之后，就可以在一定程度上放大这种差异。

不过这里也有一个问题。这个设计虽然绕开了离散选择本身不可微分的问题，但实际优化仍然可能在离散边界附近反复横跳，不见得能够稳定地收敛下来。

通信方面，Switch 使用 all-to-all：每台机器先把发往相同设备的 token 打包发送，接收端完成专家计算之后，再通过 all-to-all 把结果发回原来的设备。

## 3. DeepSeekMoE

*DeepSeekMoE* {% include cite.html key="deepseek-moe" %} 的核心改动，一是把专家划分得更加细粒度，让不同 token 可以组合出更灵活的专家集合；二是额外设置始终参与计算的共享专家，用来承载不同 token 之间较为通用的知识。

它使用了类似 Switch Transformer 的专家级负载均衡 loss。这里的主要调整是对应系数：Switch 每个 token 只选择一个专家，而 DeepSeekMoE 会选择多个专家，因此需要针对 top-$K$ 的多专家路由调整系数。同时，它又加入了设备级别的负载均衡。因此，它在训练阶段就已经结合 expert parallel 的设备分布，对负载均衡进行了考虑。

不过，更细粒度的专家也意味着一次计算会激活和访问更多专家，因此通信开销会显著增大。

## 4. Auxiliary-Loss-Free Load Balancing

*Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts* {% include cite.html key="loss-free-balancing" %} 的出发点是，负载均衡 loss 可能会干扰模型原本的学习目标，从而导致效果下降，所以希望在不添加额外 loss 的情况下实现专家负载均衡。

既然不再通过 loss 进行优化，就需要在选取专家时加入一种能够自动平衡的策略。文章的做法是给每个专家的 routing logits 添加一个 bias。首先根据平均负载得到目标值，再统计当前 batch 结束后的实际路由情况，计算每个专家相对于目标负载的差值，并用这个差值更新 bias：负载过多的专家减小 bias，负载不足的专家增大 bias。这样就可以通过 bias 调整下一个 batch 的专家选择，因此本质上是当前 batch 的路由结果影响下一个 batch。

这个策略一定会与 batch 的顺序和历史有关，也需要大量且充分混合的 batch。文章从全局统计来看能够得到很好的累积平衡效果，例如不平衡程度可以降到 0.04；但这是大量样本累积后的全局结果，如果只观察少量 batch，负载平衡可能并没有这么好。

另外，推理时应该也需要保留训练最终得到的 bias，继续用它修正专家选择。

## 5. Kimi K2 的 MoE

Kimi K2 {% include cite.html key="kimi-k2" %} 实际上沿用了 DeepSeekMoE 的设计，只是进一步增加了专家数量，使 MoE 的稀疏度更高。

## 6. LatentMoE

*LatentMoE* {% include cite.html key="latent-moe" %} 把每个专家做得更窄，再用节省下来的参数和通信预算容纳更多专家。但是不能只做低维压缩，还必须通过增加专家数量来补偿模型能力。

在具体的信息流中，完整的 hidden state 仍然负责决定路由；完成路由之后，压缩得到的 latent state 才会真正通过 all-to-all 发送，并由对应的 routed experts 进行处理。因此，LatentMoE 压缩的主要是专家计算和跨设备通信，而不是路由时用于判断 token 应该进入哪些专家的信息。

## 7. Kimi K3：Stable LatentMoE

### 7.1 专家输出的数值稳定性

Kimi K3 {% include cite.html key="kimi-k3" %} 的 Stable LatentMoE 首先对专家输出加入 RMSNorm。专家数量增加之后，如果不做 normalization，多个专家输出的累积可能造成数值爆炸。这里我的猜测是，因为模型多经过并组合了很多矩阵变换，所以需要在中间增加 RMSNorm 来控制数值；从实验结果来看，至少在专家输出累积之后加入 RMSNorm 是比较有效的。

另外，Stable LatentMoE 还使用了 $\tanh$。双曲正切的输出始终位于 $[-1,1]$，因此也可以限制相关数值的范围。

### 7.2 Quantile Balancing

对于一个 batch 中的每个 token，可以根据它对所有专家的 routing score，通过 top-$(K+1)$ 找到对应的选择阈值。然后针对每个专家寻找一个 bias，使它在加入这个 bias 后能够满足合适的期望负载，再把得到的 bias 提供给下一个 batch 使用。

直方图的作用是简化这个过程中的计算和通信。原本需要保存每个 token 对每个专家所对应的预期 bias 数值，再据此决定下一轮使用的 bias；这个过程还需要在所有机器和加速卡之间进行通信。使用直方图之后，可以把这些预期 bias 数值归入一段一段的区间，相当于对它们进行了压缩，因此所需的计算和通信都会更小。

### 7.3 Expert Parallel Infrastructure

GShard {% include cite.html key="gshard" %} 通过为每个专家设置固定容量，并让超出容量的 token 跳过专家变换，来保证最终的负载上限，避免少数过载专家拖慢整体计算。

不过 GShard 的做法仍然存在缺陷。第一，它实际上丢失了一部分专家计算，这会对模型精度产生影响；第二，对于负载很低的专家，预留的计算容量仍然会空闲下来。因此，它解决的是最慢设备拖累整体速度的问题，但不能真正把所有设备上的计算铺平。

MoonEP {% include cite.html key="moonep" %} 则使用动态均衡策略：根据当前 routing 结果在线复制热点专家，再把原本集中在这些专家上的 token 分配到不同副本进行计算，从而把计算负载铺平。

## 8. 总结

总结来说，MoE 首先通过每个 token 只激活少量专家来节省计算，接着需要寻找更合适的负载均衡策略。更多、更细粒度的专家提高了模型的稀疏度，但也带来了更大的通信开销，因此还需要更有效的负载均衡与动态均衡策略，让不同专家和设备上的计算更加均匀，从而提高训练效率。

沿着这条路线，我们还可以提出一个新的问题：虽然均衡策略不再直接体现在 loss 中，但这并不意味着它不会影响 loss。无论是调整 routing bias，还是复制热点专家，本质上都可以看成一种带约束优化；这种均衡约束势必会影响原始 loss 的下降过程。

因此，可以进一步思考：在 MoE 中是否存在更好的带约束优化策略？同时也需要重新斟酌，负载均衡所引入的 inductive bias 是否必要，以及它实际上会给优化过程带来什么影响。

更具体来说，平衡每个专家接收到的 token 数量，会不会实际上并不是最优的归纳偏置？我们真正需要保证的，也许只是所有专家都得到了充分学习；但是“充分学习”和“处理相同数量的 token”并不是同一个要求。不同专家所学习的模式、样本难度和收敛速度都可能不同，因此它们需要的 token 数量也不一定相同。

那么，是否有可能不再要求严格的 token 数量均衡，而只要求每个专家都得到充分训练，再结合额外的负载均衡策略解决系统效率问题？这样的设计也许能够在模型效果和计算效率之间达到更好的 Pareto frontier。我觉得这也是一个值得继续思考的问题。
