---
title: "Residual 学习笔记（三）：Hyper-Connections 的后续工作"
description: "阅读 Hyper-Connections 路线的后续研究。"
tags: [llm, learning-log, residual]
references:
  - key: stream-collapse
    title: "Analyzing Stream Collapse in Hyper-Connections: From Diagnosis to Mitigation"
    authors: "Alimaskina, Molodtsov & Beznosikov · 2026"
    url: https://arxiv.org/abs/2606.03483
  - key: ablate-rescue
    title: "Ablate and Rescue: A Causal Analysis of Residual Stream Hyper-Connections"
    authors: "Peng et al. · 2026"
    url: https://arxiv.org/abs/2603.14833
  - key: xhc
    title: "xHC: Expanded Hyper-Connections"
    authors: "Zhang et al. · 2026"
    url: https://arxiv.org/abs/2607.14530
  - key: shc
    title: "Beyond the Birkhoff Polytope: Spectral-Sphere-Constrained Hyper-Connections"
    authors: "Liu, Zhang & Li · 2026"
    url: https://arxiv.org/abs/2603.20896
---

上一篇读到最后，我们发现 HC 这条路线也出现了一些后续进展。这一篇继续阅读这些工作，看看它们如何理解和处理多流 residual 的问题。

## Stream Collapse 与 LSS

*Analyzing Stream Collapse in Hyper-Connections* {% include cite.html key="stream-collapse" %} 提出的 LSS，相当于在原本一致的多流初始化中引入一些接近单位映射的扰动。这些扰动会使最终的优化落到不同的结果上：当各条流完全一致时，多流会倾向于集中成一个主流；引入这种扰动之后，多流则会表现出一些更好的混合特性。

从这个角度理解，我觉得这反而印证了多流设计本身存在缺陷。多流之间需要充分混合时表现才会更好，说明模型确实需要混合更多不同的历史分支信息；但是通过矩阵逐层传递这些信息，又会使它们在深层网络中产生不正确的变化，从而难以实现更好的混合。

## Ablate and Rescue

*Ablate and Rescue* {% include cite.html key="ablate-rescue" %} 提供了一种分析训练后模型中多条 residual stream 各自作用的方法。简单来说，它得到的判断是：不同流之间存在冗余和不对称，但是没有表现出明显的互补关系。

这个结论倒是很符合之前的分析。我们对 $H_{res}$ 的矩阵性质进行了限制，使流之间本来就不应该存在复杂的信息交换；那么，冗余和不对称也就可以理解为训练多个分支之后自然产生的结果。

## xHC：扩大流的数量

*xHC* {% include cite.html key="xhc" %} 进一步扩大了 residual stream 的数量。首先，它扩展了写入的内容：除了当前 token 的当前层输出，还会读取一些历史 token 在同一层的 hidden feature。其次，为了控制开销，它只选择部分 stream 进行写回；这种选择部分 stream 的逻辑有些类似 MoE 中固定路径与动态路由的结合。

这里为了效率引入了很多平衡，但从本质上看，有一点很值得注意：增加同层历史 token 的信息，对 residual 同样有帮助。这也是一个值得继续思考的问题。我们发现，沿 residual 深度维度和沿序列维度引入额外连接，看起来都能够带来增益；这些设计之所以能够奏效，是不是存在某些更深层的原因，而不只是表面的结构差异？

## sHC：改变矩阵约束

*sHC* {% include cite.html key="shc" %} 设计了一种新的 $H_{res}$ 矩阵处理方式。它的性质更接近正交矩阵，能够支持旋转和置换，而不再局限于原本的双随机矩阵约束。

## 总结

总的来说，从个人角度，我始终认为使用多流传递历史信息加入了太多限制。一方面要限制流的数量，一方面又希望信息不损失，同时还希望不同流之间能够充分混合。如果背后真正起作用的是历史信息的混合，那么这种多流的归纳偏置并不适合信息的高效传递。

或者从网络的角度看，即使 $H_{res}$ 能够让多流信息发生旋转和置换，我也不认为应该让网络真的去学习这些所谓的旋转和置换。想象一下，真正奏效的可能是高维空间中 layer 之间或者序列之间的连接；而多流结构及其传递限制，是把这些连接转换成一些满足特定条件的投影。这种投影传递可以在一定程度上压缩和利用信息，但显然也会带来损失和混淆，本身限制甚至改变了连接的形式。

但是，我们也需要认识到一个更重要的前提：现代 LLM 的尺度太大，计算开销的影响已经难以忽视。因此，我们考虑任何改进方案时，都不能只从它是否符合底层原理来思考，还需要看它能否在有限的计算 budget 下获得足够的收益。在这种限制下能够达到的最优，也可能是另一种层面上对本质的符合。

所以，除了继续思考这些工作真正奏效的原因之外，这些方案中关于计算效率和开销节省的思考也同样非常重要。
