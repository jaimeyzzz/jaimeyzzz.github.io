---
title: "Looped Transformer 学习笔记"
date: 2026-10-22
description: "学习 Looped Transformer、循环深度、KV 复用与隐空间推理。"
tags: [llm, learning-log, transformer, latent-reasoning]
references:
  - key: universal-transformers
    title: "Universal Transformers"
    authors: "Dehghani et al. · 2018"
    url: https://arxiv.org/abs/1807.03819
  - key: yoco
    title: "You Only Cache Once: Decoder-Decoder Architectures for Language Models"
    authors: "Sun et al. · 2024"
    url: https://arxiv.org/abs/2405.05254
  - key: cla
    title: "Reducing Transformer Key-Value Cache Size with Cross-Layer Attention"
    authors: "Brandon et al. · 2024"
    url: https://arxiv.org/abs/2405.12981
  - key: coconut
    title: "Training Large Language Models to Reason in a Continuous Latent Space"
    authors: "Hao et al. · 2024"
    url: https://arxiv.org/abs/2412.06769
  - key: huginn
    title: "Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach"
    authors: "2025"
    url: https://arxiv.org/abs/2502.05171
  - key: latent-thoughts
    title: "Reasoning with Latent Thoughts: On the Power of Looped Transformers"
    authors: "2025"
    url: https://arxiv.org/abs/2502.17416
  - key: mixture-of-recursions
    title: "Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation"
    authors: "2025"
    url: https://arxiv.org/abs/2507.10524
  - key: ouro
    title: "Scaling Latent Reasoning via Looped Language Models"
    authors: "2025"
    url: https://arxiv.org/abs/2510.25741
  - key: loopformer
    title: "LoopFormer: Elastic-Depth Looped Transformers for Latent Reasoning via Shortcut Modulation"
    authors: "2026"
    url: https://arxiv.org/abs/2602.11451
  - key: deepseek-v41
    title: "DeepSeek-V4.1 Technical Report"
    authors: "DeepSeek-AI · 2026"
    url: https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf

---

这篇笔记按时间顺序，阅读 Looped Transformer 及相关工作。

## 1. Universal Transformers：深度方向参数共享

*Universal Transformers* {% include cite.html key="universal-transformers" %} 的 encoder 使用一个共享 block，decoder 也使用一个共享 block，然后动态决定这个 block 需要运行多少次。我理解这种设计的影响范围有点太大了：整个结构实际上只有两个反复使用的 block，但数据分布的复杂性显然需要更复杂的结构分工来处理。因此，这种方法的效果很难足够好。参数量过少，只通过不断增加深度方向的迭代次数，能够存储的信息仍然有限。

所以这篇文章更多是在提出一种 idea，很难真的仅靠重复同一个 block 的计算进行 scaling。

不过，也需要把它放回当时的研究背景中理解。Universal Transformers 属于较早期的工作，当时业界还没有形成后来关于 scaling 的经验公式和系统认知。因此，它更像是一篇奠基性的早期论文：重点不是证明仅靠重复同一个 block 就足以持续 scaling，而是展示深度方向参数共享与动态计算步数这种结构本身的价值。

## 2. YOCO：全局 KV 复用

简单来说，*YOCO* {% include cite.html key="yoco" %} 把 decoder 分成两部分。前半部分会生成并维护自己的 KV，在读取历史、整合上下文之后，构建并输出一份公共的中间 KV 表示。后半部分则完全基于这份中间 KV 继续逐层 decode，相当于对 decoder 进行了分工：前半部分负责历史信息的汇聚，后半部分只负责进一步加工，不再把每层的加工结果分别写成独立的历史 KV，供未来 token 使用。

这种分工思路后来也被 DeepSeek 的 CED {% include cite.html key="deepseek-v41" %} 采用。可以把它理解为一种关于网络行为的归纳偏置：我们假设网络的功能更倾向于分成“先收集信息、再整理加工”两个阶段，因此也在结构上把信息汇聚与后续计算拆开处理。

## 3. Cross-Layer Attention：跨层共享 KV

*Cross-Layer Attention* {% include cite.html key="cla" %} 让相邻两层复用同一份 KV，因此 KV cache 可以直接减半，而模型效果只出现很小的差异。

这是一种比较直接的 KV 节省思路：不从 sequence 维度压缩历史信息，而是从深度维度减少重复存储。它背后的归纳偏置是，相邻层所需的 KV 可能具有很大的重复性，或者不同层能够共享某些共同的处理策略与信息内容。因此，可以让多个层读取同一份 KV，在尽量保留模型能力的同时降低 KV cache 的开销。

## 4. Coconut：连续隐空间推理

*Coconut* {% include cite.html key="coconut" %} 不会在每一步推理时都输出 token，而是把 hidden state 作为下一步的输入，让模型在连续隐空间中重复计算。文章将它理解为一种隐式的思维链：相比于生成明确的离散 token，hidden state 可以保留更多信息。直觉上，可以把它类比为一种尚未经过采样的“叠加态”，其中可能同时包含多种潜在的推理方向。

这个思路听起来是合理的，但训练会成为一个关键问题。隐空间中的这种“叠加态”没有直接的语义标签，我们也很难像监督显式 token 一样，对其中每一步进行有效训练；它可能更适合依赖最终结果反馈的 RL 学习方式。因此，这种范式提供了一条可能提升推理能力的路线，但在获得潜在收益的同时，也带来了更困难的训练与信用分配问题。

## 5. Huginn：Recurrent Depth

*Huginn* {% include cite.html key="huginn" %} 把网络分成三部分：前置 block、中间的 recurrent block 和输出 block。其中，中间的 recurrent block 会被重复执行多次，具体次数由用户指定的参数决定。调整循环次数，就可以为模型分配不同的推理计算量。

## 6. Latent Thoughts：循环深度与推理

*Reasoning with Latent Thoughts* {% include cite.html key="latent-thoughts" %} 中的 Looped Transformer 是 k 层的模型，循环 L 次。但和 Coconut 不同，它不会作为一个新位置，不保存对应的 KV，这样前 L-1 次的中间结果并不会被后续的 token 再看到。

## 7. Mixture-of-Recursions：Token 级动态深度

*Mixture-of-Recursions* {% include cite.html key="mixture-of-recursions" %}（MoR）实现了 token 级别的自适应计算，包含专家选择路由和 token 选择路由。所谓专家选择路由，是在序列范围内判断哪些 token 需要多跑几轮，这其实是看到了未来的信息。

这里存在一个比较大的问题：利用未来 token 决定当前 token 的计算深度，在某种意义上类似“作弊”。真实的自回归推理无法看到尚未生成的内容，因此训练或并行计算时使用的路由依据，在实际生成时并不完整。这种训练与推理之间的不一致，可能使路由决策变差，并导致效果损失。

KV 策略有两种：一种是保存所有活跃的 KV，一种是只保存最开始的一次的 KV。

## 8. Ouro：可学习的深度分配

*Ouro* {% include cite.html key="ouro" %} 与 Looped Transformer 类似，会让一个由 $L$ 层组成的 block 循环执行 $K$ 次；不同之处在于，这里的循环次数 $K$ 不是固定的。Ouro 设计了一个累计概率退出门，当退出概率累积到给定阈值后，就停止当前 token 的 loop。因此，它实际上是一种根据不同 token 自适应分配计算深度的策略。

## 9. LoopFormer：弹性循环深度

*LoopFormer* {% include cite.html key="loopformer" %} 希望模型能够在任意循环步数下稳定推理。它把一次循环中的表示演化看成一条轨迹。这里会有一种似曾相识的感觉：这其实与 diffusion 的思路很相似——把逐步演变的过程视为一条轨迹，再让模型学习如何沿着这条轨迹推进。

### 9.1 轨迹条件化

所谓轨迹条件化，我理解是把不同循环步骤统一映射到 $[0,1]$ 的时间坐标，并使用 $\Delta t$ 表示每次演化的步长。模型不仅要学习表示如何变化，还要在当前时间与步长的条件下，学习应该完成多大幅度的表示更新。这样，同一个模型就可以处理不同的循环次数与步长。

### 9.2 Shortcut-Consistency Training

这里采用的不是分阶段训练，而是在训练中采样不同长度的轨迹。较短的轨迹可以把多个细粒度更新合并成一次更大的更新，相当于直接走一条“捷径”。训练目标同时约束长轨迹与 ground truth 的差异、短轨迹与 ground truth 的差异，以及长短两条轨迹之间的差异。通过共同训练这些目标，模型学习让不同步数的计算尽量到达一致的结果，从而能够在任意步数下保持较稳定的推理表现。

不过，LoopFormer 的灵活性仍然停留在轨迹级别。每次推理时，用户需要指定一条固定的轨迹，或者等价地指定统一的 loop 次数；同一次推理中的所有 token 都按照这套设置执行。因此，它允许用户在不同推理任务中选择不同的计算深度，但还不能根据每个 token 的实际需要，自适应地分配不同的 loop 次数。

## 10. DeepSeek-V4.1：CED

*DeepSeek-V4.1 Technical Report* {% include cite.html key="deepseek-v41" %} 中的 CED 与 YOCO 有一个共同的核心结构：encoder 各层输出的 hidden state 会经过投影，形成一份全局 KV，再由后续的 decoder 层共享使用。不过，两者的具体实现存在几个关键区别。

第一，CED 的 encoder 不像 YOCO 那样使用 sliding-window attention（SWA），而是仍然维护全局注意力。也就是说，负责汇聚信息的 encoder 本身就可以直接建模全局上下文。

第二，CED 的 decoder 不只读取 encoder 产生的全局 KV，同时还维护自己的 SWA。前者提供经过 encoder 汇聚的全局信息，后者保留 decoder 局部窗口内的细粒度信息。

第三，decoder 并不会以 dense attention 的方式读取全部全局 KV，而是通过 sparse attention 进行选择性访问。可以把 CED 理解为在 YOCO 的 encoder–decoder 分工基础上，进一步组合了 encoder 全局注意力、decoder 局部 SWA，以及对共享全局 KV 的稀疏读取。

这三项针对 YOCO 的调整看起来都比较合理。encoder 的全局注意力增强了全局信息整合能力，也尽量延续了以往架构中成熟的全局建模方式；decoder 自身的 SWA 让它在读取共享全局信息之外，仍然能够利用最近的局部上下文；而对全局 KV 使用 sparse attention，则试图在保留全局信息收益的同时节省计算量，在效率与模型性能之间取得平衡。

## 总结

首先，循环确实是一种值得考虑的计算方式。从这些工作的演进中可以看到，增加推理计算本身能够带来收益，而循环结构又有两个直接的优点：相比于简单增加网络层数，它可以通过参数共享减少额外参数；同时，结合跨层 KV 共享等设计，它也有机会降低 KV cache 的开销。这意味着，参数量、KV cache 与重复计算之间可能存在一种更优的压缩和分配策略。在合适的结构下，我们也许可以用更少的参数和缓存，换取更多有效的循环计算。

其次，这些工作也提供了另一种理解 Transformer 内部计算的角度。Diffusion 和 flow matching 试图通过连续的流变换完成分布匹配；类似地，Transformer 内部复杂的表示演化轨迹，也可能被拆解成一系列逐步的分布变换。沿着这个视角，我们在某种程度上把序列建模与分布匹配联系了起来：模型不只是逐层处理 token，也是在学习如何让内部表示沿一条轨迹逐渐演化到目标状态。

因此，这次阅读留下了两个值得继续思考的方向。一方面，我们需要从 Looped Transformer 的视角出发，寻找参数、缓存与循环计算之间更优的组合策略；另一方面，也需要继续研究分布匹配的思想能否用于理解和改进 LLM 内部的表示演化过程。
