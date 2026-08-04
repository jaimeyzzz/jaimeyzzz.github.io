---
title: "Residual 学习笔记（二）：跨层信息聚合方法"
description: "阅读其他 residual 相关工作。"
tags: [llm, learning-log, residual]
references:
  - key: dca
    title: "DeepCrossAttention: Supercharging Transformer Residual Connections"
    authors: "Heddes et al. · ICML 2025"
    url: https://arxiv.org/abs/2502.06785
  - key: muddformer
    title: "MUDDFormer: Breaking Residual Bottlenecks in Transformers via Multiway Dynamic Dense Connections"
    authors: "Xiao et al. · ICML 2025"
    url: https://arxiv.org/abs/2502.12170
  - key: delta-attnres
    title: "Delta Attention Residuals"
    authors: "Luo, Cai & Hu · 2026"
    url: https://arxiv.org/abs/2605.18855
  - key: depth-attention
    title: "Depth-Attention: Cross-Layer Value Mixing for Language Models"
    authors: "Zeng et al. · 2026"
    url: https://arxiv.org/abs/2606.05014
  - key: low-rank-attnres
    title: "Low-Rank Attention Residuals"
    authors: "Jonathan Su · 2026"
    url: https://arxiv.org/abs/2607.09694
  - key: moda
    title: "Mixture-of-Depths Attention"
    authors: "Zhu et al. · 2026"
    url: https://arxiv.org/abs/2603.15619
---

上一篇读完 HC 和 mHC 之后，最开始提出的问题仍然没有得到解决。这一篇继续阅读其他 residual 相关工作，看看能否从不同的结构和视角中找到答案。

## DeepCrossAttention：静态到动态的直接过渡

*DeepCrossAttention* {% include cite.html key="dca" %} 的 v1 就是 DenseFormer，v2 给每个 feature 增加了独立的权重，v3 则使用当前层的固定参数对历史动态输出进行打分。可以看到，v3 带来的提升很少，最终的 DCA 则进一步加入了 QKV，效果才比较明显。

我个人来说其实并不是很认同这篇文章的演变思路。但是从实验结果来看，v3 相比 v2 多出来的部分，正是使用固定的层参数结合历史动态输出进行打分，而它带来的收益非常低。这一点与 AttnRes 的结果完全相悖，这里先留待获得更多线索之后再讨论。

最终版本直接引入了对 QKV 的修改，这个改动略微有些大。QKV 的动态变化会不会带来训练和功能上的不稳定性，目前还不清楚。但是从某种程度上看，它又有些类似 HC：两者都引入了多流的额外信息，只是这里额外引入的 Q、K、V 并不是均等的。

考虑到这目前只是一个单独工作的结果，这里也先不做更多判断，留待获得更多线索之后再展开讨论。

## MUDDFormer：动态组合与 QKVR 多流

*MUDDFormer* {% include cite.html key="muddformer" %} 根据当前 token 的动态输出，为所有历史层的 value 生成权重，并对所有层进行组合。这本质上就是上面提到的 dynamic query、static key 的组合；同时，Q、K、V、R 分别具有不同的固定 key。

从论文的消融实验来看，引入 dynamic 带来的好处最大，引入 QKVR 多流带来的好处也相当明显，而只引入 static dense connection 带来的好处最小。

这同样是一个在小网络上验证的单一工作。虽然它支持了之前关于 dynamic query、static key 的假设，但是类似 DCA，我们又得到了一个新的、值得思考的问题：QKV 多流会对 attention 带来什么样的影响？这个问题也留到最后再综合讨论。

## Delta Attention Residuals

*Delta Attention Residuals* {% include cite.html key="delta-attnres" %} 保留原本的 residual stream，同时额外使用 attention 选择各层产生的增量，再把结果加回 residual stream。

但是我看 AttnRes 似乎也是直接对各个 block 的输出做选择，并不存在论文所说的 cumulative query 问题。这里暂时还没有想清楚，留待后面再看。

## Depth-Attention

*Depth-Attention* {% include cite.html key="depth-attention" %} 的设计确实非常不同。每个 query 会先与历史层的 key 做 attention，得到由历史层累积而成的 value，再由同一层的序列 attention 使用这个 value。相当于把 residual 的跨层累积放进了 value 里面。

但这里有一个很大的疑问：这种累积显然会改变 feature dimension 的性质。每一维 feature 不仅需要处理当前层的信息，还需要同时处理所有历史层的信息；这里的 feature 也就同时表达了序列相关性和功能相关性。

但是仔细思考一下这个设计，会发现它反而是一个更加保守的设计。使用 AttnRes 时，每个 layer 的 feature 都会被重新组合，其中也包括 QKV；而 Depth-Attention 其实只想修改 V，其他部分仍然使用传统 residual。因此，它的增益并不是来自更强的自由度，而是在尽量少改动原始结构的情况下复用 KV cache，从而节省存储。

但这同时也带来了一个问题：QK 的传统 residual 信息流与 V 的信息流并不匹配。这也可能是为什么同时修改 K 和 V 的效果反而不好，因为 Q 没有跟着改变信息流，会影响 QK 原本的功能。相比之下，只修改 V 是一种尽量小的改动。

另外，它额外带来的 feature 聚合效果其实与 AttnRes 类似。Depth-Attention 并没有引入更强的归纳偏置，而是在更少改变原有结构的前提下实现类似的跨层聚合。

## Low-Rank Attention Residuals

*Low-Rank Attention Residuals* {% include cite.html key="low-rank-attnres" %} 可以说越来越接近我们关注的问题了。它的观点与我们有些相似，主要关注 AttnRes 中 layer output 同时作为 key 和 value 的问题，因此设计了一个单独的 low-rank key，将路由和内容表达解耦，不过 query 仍然保留固定的设计。文章的结论是，这样的效果比原始设计更好。

如果引入 low-rank key 不仅不会让效果变差，反而会更好，恰恰说明路由并不需要太多信息就能够完成。那么，我们自然也可以继续提出之前的疑问：路由最少需要什么信息？是不是只需要能够表示层功能的指纹即可？

不过这篇文章毕竟还比较新，我们先继续观察。大规模模型上的表现也可能会有所不同。

## Mixture-of-Depths Attention

*Mixture-of-Depths Attention* {% include cite.html key="moda" %} 把历史深度上的 KV 与当前层序列中的 KV 放在一起，让 query 在同一次 attention 中共同选择。

## 总结和延伸

首先，关于我们之前提出的疑问，这些新的阅读并没有在小规模实验上推翻它，反而有一些实验结果对它形成了支持。这个问题可以留待后续的发展继续观察。

另外，这些工作也引出了其他维度的思考：第一，Q、K、V 分别聚合历史信息会带来什么效应？第二，深度层信息与历史序列信息有什么区别，各自又应当如何使用？

这也引出了一个新的问题：我们讨论 residual 时，实际究竟在讨论什么？它带来的收益本质上又是什么？我打算再开一篇文章做一些延伸讨论，希望能够把这些问题统一起来，用一个更底层的框架来思考。

写到这里，我们会发现这些内容已经不只与 Kimi 的报告有关，而是延伸出了更多额外的问题。另外，我发现 HC 这条路线也出现了一些后续进展。一个合理的分析框架，理论上也应该能够把 HC 统一进来一起研究。
