---
title: "Kimi K3 学习笔记（二）：Attention Residuals"
description: ""
tags: [llm, learning-log, kimi]
references:
  - key: resnet
    title: "Deep Residual Learning for Image Recognition"
    authors: "He, Zhang, Ren & Sun · CVPR 2016"
    url: https://arxiv.org/abs/1512.03385
  - key: transformer
    title: "Attention Is All You Need"
    authors: "Vaswani et al. · NeurIPS 2017"
    url: https://arxiv.org/abs/1706.03762
  - key: xiong2020
    title: "On Layer Normalization in the Transformer Architecture"
    authors: "Xiong et al. · ICML 2020"
    url: https://arxiv.org/abs/2002.04745
  - key: denseformer
    title: "DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging"
    authors: "Pagliardini et al. · 2024"
    url: https://arxiv.org/abs/2402.02622
  - key: attnres
    title: "Attention Residuals"
    authors: "Kimi Team · 2026"
    url: https://arxiv.org/abs/2603.15031
---

## 残差连接

要谈 attention residuals，得先回到残差连接的源头：*Deep Residual Learning for Image Recognition* {% include cite.html key="resnet" %}。

它的想法是：原本我们想让网络学习的是目标映射 $H(x)$，残差连接把它改写成——

$$
H(x) = x + F(x),
$$

于是网络真正需要学习的，其实是残差——

$$
F(x) = H(x) - x.
$$

注意这里用的是恒等映射（把 $x$ 原样加回去）。但"恒等映射是不是最好的"其实有待讨论——后面要读的文章正是对这个映射做了不同的操作。我的认知是：**"有这条连接"这件事本身非常重要**，重要到它成了后续模型的标准模块。

它的重要性，我理解有两个方面。

一是**优化的稳定性**。笔者曾在可微分模拟领域，对长序列的可优化性做过一些分析，其中一个突出问题就是梯度爆炸：信号经过一条长序列（类比到深度网络，就是很多层）之后，梯度可能因为反复放大而增大数值误差和优化难度；反过来，深层网络中也可能出现梯度消失。而残差连接提供了一条长程的“短路”，让信息和梯度可以绕过中间变换直接传播，从而让优化更稳定。

二是从**网络本身**的角度。我的理解是，模型往往会自然地在不同层分化出不同的功能——有些是人为设定的结构差异（比如把 full attention 层和 linear attention 层混在一起），有些则是网络自己分化的结果。要更好地利用这些不同的功能，就需要视情况去调用不同的层，而这也需要残差连接来支撑。

## Transformer 中的残差连接

接下来是 *Attention Is All You Need* {% include cite.html key="transformer" %}。论文中的每个 Transformer block 包含两个子层：一个 Multi-Head Attention 子层和一个 MLP/FFN 子层。每个子层外面各有一次残差连接，因此每个 block 通常会进行两次残差更新。

原始论文采用的是 Post-LN，也就是先把子层输出和输入相加，再进行 LayerNorm：

$$
\tilde{x}_l = \operatorname{LN}\!\left(x_l + \operatorname{Attention}(x_l)\right),
$$

$$
x_{l+1} = \operatorname{LN}\!\left(\tilde{x}_l + \operatorname{MLP}(\tilde{x}_l)\right).
$$

这里先只需要注意一件事：原始 Transformer 的 LayerNorm 位于残差相加之后。这个看似局部的位置差异，会直接影响残差连接在深层网络中的性质。

## Post-LN 与 Pre-LN

*On Layer Normalization in the Transformer Architecture* {% include cite.html key="xiong2020" %} 系统分析了 LayerNorm 位置对 Transformer 训练的影响。论文指出，Post-LN 在初始化时靠近输出层的梯度会比较大，因此往往需要 learning-rate warm-up 来避免训练不稳定；把 LayerNorm 移到子层内部，也就是改成 Pre-LN，则能让梯度在初始化时表现得更稳定。

从残差连接的角度看，Post-LN 可以写成：

$$
x_{l+1} = \operatorname{LN}\!\left(x_l + F_l(x_l)\right).
$$

虽然这里存在从 $x_l$ 到 $x_{l+1}$ 的残差连接，但相加后的结果还要经过一次非线性的 LayerNorm。因此，上层传回来的梯度和本层分支的梯度都会经过 LayerNorm 的 Jacobian，残差路径不再是一条数值上的恒等映射。换句话说，Post-LN 虽然在结构上有 residual connection，却削弱了我们通常期待的那种跨越很多层的直接“短路”。每一层都重新归一化累计状态，历史信息无法简单地沿深度线性展开。这里我进一步猜测，这也会让靠后的层更容易对最终表示产生显著影响；不过这是我的理解，并不是论文直接给出的结论。

Pre-LN 则是先对输入归一化，再计算子层，最后直接写回 residual stream：

$$
\tilde{x}_l = x_l + \operatorname{Attention}\!\left(\operatorname{LN}(x_l)\right),
$$

$$
x_{l+1} = \tilde{x}_l + \operatorname{MLP}\!\left(\operatorname{LN}(\tilde{x}_l)\right).
$$

因为相加之后不再经过 LayerNorm，Pre-LN 保留了一条真正直接的加法路径。沿着网络深度，可以把它理解成一个持续累加的过程：

$$
x_L = x_0 + \sum_{i=0}^{L-1}\Delta_i,
$$

其中每个 $\Delta_i$ 都是某个 Attention 或 MLP 子层写入 residual stream 的增量。这条恒等路径让早期表示和梯度能够更直接地到达后面的层，也是 Pre-LN 训练更稳定的重要原因。

但这种稳定性也带来了另一种结构倾向。随着前面各层的输出不断累积，residual stream 的尺度会逐渐增大；而当前子层接收到的是归一化后的输入，它每次写入的增量尺度相对受限。于是越靠后的层，单次更新相对于已有状态的比例可能越小，更像是在已有表示上做相对较小的局部修正，而不是对整体表示做大幅改变。网络由此容易形成一种层级化结构：前面的层搭出主体，后面的层持续细化。

因此，Post-LN 的问题是长程残差路径不够直接，优化相对不稳定；Pre-LN 修复了这条长程路径，却让所有历史更新以固定权重不断累加，并可能削弱深层对既有表示的改写能力。

## DenseFormer：学习如何组合历史层

*DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging* {% include cite.html key="denseformer" %}。它在 Transformer 层之间加入了 Depth Weighted Average（DWA），不再让信息只能沿着相邻 block 一层一层地向后传递。

DenseFormer 会保留 embedding 和每个完整 Transformer block 的直接输出。记第 $j$ 个历史表示为 $X_j$，那么在第 $i$ 层之后，DWA 使用一组可学习的静态标量重新组合截至当前层的全部表示：

$$
Y_i = \sum_{j=0}^{i}\alpha_{i,j}X_j.
$$

组合后的 $Y_i$ 会作为下一个 Transformer block 的输入。模型仍然按顺序计算所有 block，但历史信息不再只能通过相邻层反复传递，而是可以借助 DWA 建立直接的跨层连接。

我的理解是：既然我们既想保留长程连接，又认为 Pre-LN 中逐层等权累积的方式不够好，那么一个自然的做法，就是引入可学习参数，让后面的层自己决定该怎样使用前面的层。DenseFormer 提供的正是这样的能力——后续层可以对所有历史 block 输出做线性组合，学习应该从哪些深度取出多少信息，而不是被固定为一路累加到当前层。

其实讲到这里，单从残差连接的角度看，这套机制似乎已经足够完整了：不同的 layer 或 block 可以在训练过程中，根据各自的功能和实际需要选择前面不同深度的表示；网络结构对后层输出的抑制效应，似乎也可以借此绕开。那么，后续的 Attention Residuals 又进一步做了什么？

## Attention Residuals：动态选择历史层

到了 *Attention Residuals* {% include cite.html key="attnres" %}，讨论的维度其实已经和前面所说的长程连接稳定性有所不同了。

先简单描述它做了什么。AttnRes 同样加权组合前面各层的表示：

$$
h_l = \sum_{i=0}^{l-1}\alpha_{i\to l}v_i,
$$

但与 DenseFormer 不同，这里的组合权重不是训练完成后固定不变的标量，而是通过一个沿深度方向的注意力计算得到：

$$
\alpha_{i\to l}
=
\operatorname{softmax}_i\!\left(
w_l^\top\operatorname{RMSNorm}(v_i)
\right).
$$

其中，$w_l$ 是第 $l$ 个子层学习到的 pseudo-query，可以理解成这一层用来查询历史表示的“层查询向量”；$v_i$ 则是第 $i$ 个历史子层对当前 token 产生的表示。计算分成两步。第一步，对当前 token 在每个历史深度上的表示 $v_i$ 打分：

$$
s_i = w_l^\top\operatorname{RMSNorm}(v_i).
$$

第二步，沿深度维度做 softmax，把这些分数变成归一化权重：

$$
\alpha_i = \frac{e^{s_i}}{\sum_j e^{s_j}}.
$$

这里开启了一个新的自由度。DenseFormer 学到的是“第 $l$ 层通常应该怎样组合历史层”，同一个目标层对所有输入使用同一组深度权重；AttnRes 学到的则更像是“第 $l$ 层应该如何判断当前 token 需要哪些历史层”。因为分数依赖当前 token 在各历史层的表示 $v_i$，权重通常会随 token 和输入内容动态变化。$w_l=0$ 时，它会退化为对所有历史表示做均匀平均；即使 $w_l\ne0$，特殊情况下也可能得到相同权重，但它已经具备了内容相关的动态选择能力。

换一个角度理解，AttnRes 的假设是：每一层要处理的 feature 与历史层信息之间的关系并不是固定的。同一个后续层面对不同 token 时，可能需要调用完全不同深度的表示，因此历史信息不仅应该可以跨层组合，还应该根据当前内容动态组合。

## 一个还没有想明白的问题

但这里似乎又引入了一个“鸡生蛋、蛋生鸡”的问题。

我们一开始认为，不同层之间存在某种功能上的分工合作，因此希望让各层的 feature 得到更稳定的表达，不要因为固定累积带来的归纳偏置而限制学习能力。DenseFormer 允许后续层自由组合不同深度的输出，看起来已经能避免结构天然抑制后层作用的问题。

但 AttnRes 又引入了另一种归纳偏置：它认为层与层之间的组合关系不应该是固定的，而应该根据当前 token 动态变化。这里新增的动态性主要来自跨层路由权重——各层产生的 feature 在 DenseFormer 中本来也会随输入变化。于是问题变成了：稳定的层级分工与动态的跨层组合会形成怎样的功能组织？不同层是否还会产生稳定、可解释的分工，还是每层的角色也会随上下文一起漂移？目前我还没有找到一种很好的方式来理解。

甚至从论文的消融实验看，引入更多动态性可能还会更好。论文尝试用当前 hidden state 投影出 input-dependent query，而不是使用每层固定学习的 pseudo-query，验证损失进一步降低；代价是每层增加一个 $d\times d$ 投影，并在解码时引入串行的内存访问，因此最终方案仍选择了固定的 learned query。

如果把问题抽象成 query 与 depth key 是否依赖当前输入，就会得到一个 $2\times2$ 的组合：

| | 静态 depth key | 动态 depth key |
|---|---|---|
| **固定 query** | 静态跨层混合，可近似理解为 DWA 一类方法 | 论文默认的 AttnRes |
| **动态 query** | 尚未在文中单独验证 | input-dependent query 消融 |

这里的对应关系只是一种抽象。DWA 并不会显式地用固定 query 查询固定 key，而是直接学习静态混合系数；把它放进左上角，是因为两者最终都产生不依赖当前输入的跨层路由权重。

这样一来，自然还剩下一个没有回答的象限：用动态 query 查询静态的层级 key。它的计算性能显然仍会比完全静态的混合更差，因为 query 依赖当前 hidden state；但模型效果会比默认 AttnRes 更好还是更差呢？

带着这些问题，我决定另开一篇文章，继续阅读和学习其他 residual 方向的工作，看看能否找到更好的理解框架。
