---
title: "Kimi K3 学习笔记（一）：Kimi Delta Attention"
description: ""
tags: [llm, learning-log, kimi]
references:
  - key: katharopoulos2020
    title: "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention"
    authors: "Katharopoulos, Vyas, Pappas & Fleuret · ICML 2020"
    url: https://arxiv.org/abs/2006.16236
  - key: performer
    title: "Rethinking Attention with Performers"
    authors: "Choromanski et al. · ICLR 2021"
    url: https://arxiv.org/abs/2009.14794
  - key: cosformer
    title: "cosFormer: Rethinking Softmax in Attention"
    authors: "Qin et al. · ICLR 2022"
    url: https://arxiv.org/abs/2202.08791
  - key: schlag2021
    title: "Linear Transformers Are Secretly Fast Weight Programmers"
    authors: "Schlag, Irie & Schmidhuber · ICML 2021"
    url: https://arxiv.org/abs/2102.11174
  - key: gateddeltanet
    title: "Gated Delta Networks: Improving Mamba2 with Delta Rule"
    authors: "Yang, Kautz & Hatamizadeh · ICLR 2025"
    url: https://arxiv.org/abs/2412.06464
  - key: kimilinear
    title: "Kimi Linear: An Expressive, Efficient Attention Architecture"
    authors: "Kimi Team · 2025"
    url: https://arxiv.org/abs/2510.26692
---

## Linear Attention

线性注意力最早的工作之一，是 *Transformers are RNNs* {% include cite.html key="katharopoulos2020" %}。它的出发点是对标准注意力做变换以节省计算。

标准的 softmax 注意力，对每个位置都要和所有位置算一遍相似度——

$$
A_l(x) = V' = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{D}}\right)V \tag{2}
$$

其中 $Q=xW_Q,\ K=xW_K,\ V=xW_V$。这带来 $O(N^2)$ 的计算与显存开销。

第一步，把 softmax 换成一个一般的相似度函数 $\mathrm{sim}(\cdot)$，写出广义的注意力：

$$
V_i' = \frac{\sum_{j=1}^{N}\mathrm{sim}(Q_i,K_j)\,V_j}{\sum_{j=1}^{N}\mathrm{sim}(Q_i,K_j)} \tag{3}
$$

只要 $\mathrm{sim}$ 非负，它就仍是一个合法的注意力。接着取 $\mathrm{sim}(q,k)=\phi(q)^\top\phi(k)$——用一个特征映射 $\phi$ 的内积来当相似度，这里其实是最关键的，这个变换是一个非平凡的变换，极大地改变了attention的性质：

$$
V_i' = \frac{\sum_{j=1}^{N}\phi(Q_i)^\top\phi(K_j)\,V_j}{\sum_{j=1}^{N}\phi(Q_i)^\top\phi(K_j)} \tag{4}
$$

变换之后，因为 $\phi(Q_i)$ 与内层求和无关，利用矩阵乘法的结合律，可以把它提到求和外面——

$$
V_i' = \frac{\phi(Q_i)^\top\sum_{j=1}^{N}\phi(K_j)\,V_j^\top}{\phi(Q_i)^\top\sum_{j=1}^{N}\phi(K_j)} \tag{5}
$$

于是 $\sum_{j}\phi(K_j)V_j^\top$ 和 $\sum_{j}\phi(K_j)$ 只需算一次、被所有 query 复用，复杂度从 $O(N^2)$ 降到 $O(N)$。这就是 linear attention 的核心。

换个视角，把分子那块累加写成一个随时间更新的状态 $S_t$——每来一个 token，就往状态里加一份 key–value 外积，读取时再用 query 去取：

$$
S_t = S_{t-1} + k_tv_t^\top, \qquad o_t = S_t^\top q_t.
$$

（为简洁，这里及以后都用 $k_t$ 代表 $\phi(k_t)$。）这也正是"Transformers are RNNs"这个名字的由来：一个固定大小的状态 $S$，被一路累加更新。而后面所有的改进，改的都是"这一步状态该怎么更新"。

这样做的好处很直接：它把 $QK^\top$ 的相乘做了分解——如 (5) 式，转而只结合 K 与 V，得到一个小得多的矩阵。于是推理时需要存储的 kv cache 也就省下来了。

但坏处也随之而来。第一，也最显而易见：这是一个近似，而近似必然带来误差；更麻烦的是，这种误差会不断地把历史里的一些信息丢掉——对于过去某些高频、精确的信息，这种方式没办法准确地定位、再把它恢复出来。

第二，我认为它会影响到对 feature dimension 的学习。简单说：原本的 feature 空间是比较自由的——假定存在一个较好的 feature 空间，注意力会倾向于去学到它，让相关联的 token 之间的信息变得更尖锐（选取更集中）；而 linear attention 想达到同样尖锐的选取效果，就会对 feature dimension 提出额外要求。

原本的 softmax，即使面对低维的特征，也能做出高维的分区。举个例子——在二维空间里，feature 维度很低，但我们仍然可以把它切成很多细小的区间。

这些区间都能变成一些稀疏的分支，每一个都可以成为一个 one-hot，作为后续一个清晰的选择。而 linear attention 能表达多少这样清晰的分支，取决于特征映射的输出维度 $d_\phi$：为了逼近 one-hot 那样的策略，它只能采用类似 codebook 的方式，而这个 codebook 的容量正由 $d_\phi$ 决定，所以清晰分支的数目是有限的。

<figure class="fig">
<svg viewBox="0 0 780 330" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="softmax 与 linear attention 在二维特征空间里的对比"><text x="200" y="30" font-size="15" font-weight="600" fill="var(--text)" text-anchor="middle" font-family="sans-serif">softmax</text><g stroke="var(--text-faint)" stroke-width="1" opacity="0.5"><line x1="82" y1="175" x2="318" y2="175"/><line x1="200" y1="57" x2="200" y2="293"/></g><text x="321" y="179" font-size="11" fill="var(--text-faint)" font-style="italic">f₁</text><text x="195" y="54" font-size="11" fill="var(--text-faint)" font-style="italic" text-anchor="end">f₂</text><path d="M200,175 L300.0,175.0 A100,100 0 0 1 286.6,225.0 Z" fill="#d05739" stroke="var(--bg)" stroke-width="1.8" opacity="0.92"/><path d="M200,175 L286.6,225.0 A100,100 0 0 1 250.0,261.6 Z" fill="#d05f39" stroke="var(--bg)" stroke-width="1.8" opacity="0.92"/><path d="M200,175 L250.0,261.6 A100,100 0 0 1 200.0,275.0 Z" fill="#d06739" stroke="var(--bg)" stroke-width="1.8" opacity="0.92"/><path d="M200,175 L200.0,275.0 A100,100 0 0 1 150.0,261.6 Z" fill="#d06f39" stroke="var(--bg)" stroke-width="1.8" opacity="0.92"/><path d="M200,175 L150.0,261.6 A100,100 0 0 1 113.4,225.0 Z" fill="#d07639" stroke="var(--bg)" stroke-width="1.8" opacity="0.92"/><path d="M200,175 L113.4,225.0 A100,100 0 0 1 100.0,175.0 Z" fill="#d07e39" stroke="var(--bg)" stroke-width="1.8" opacity="0.92"/><path d="M200,175 L100.0,175.0 A100,100 0 0 1 113.4,125.0 Z" fill="#d08639" stroke="var(--bg)" stroke-width="1.8" opacity="0.92"/><path d="M200,175 L113.4,125.0 A100,100 0 0 1 150.0,88.4 Z" fill="#d08e39" stroke="var(--bg)" stroke-width="1.8" opacity="0.92"/><path d="M200,175 L150.0,88.4 A100,100 0 0 1 200.0,75.0 Z" fill="#d09639" stroke="var(--bg)" stroke-width="1.8" opacity="0.92"/><path d="M200,175 L200.0,75.0 A100,100 0 0 1 250.0,88.4 Z" fill="#d09d39" stroke="var(--bg)" stroke-width="1.8" opacity="0.92"/><path d="M200,175 L250.0,88.4 A100,100 0 0 1 286.6,125.0 Z" fill="#d0a539" stroke="var(--bg)" stroke-width="1.8" opacity="0.92"/><path d="M200,175 L286.6,125.0 A100,100 0 0 1 300.0,175.0 Z" fill="#d0ad39" stroke="var(--text)" stroke-width="2.2"/><circle cx="258.0" cy="159.5" r="4" fill="var(--text)"/><text x="200" y="309" font-size="12.5" fill="var(--text-mute)" text-anchor="middle">许多尖锐分区 → one-hot</text><text x="585" y="30" font-size="15" font-weight="600" fill="var(--text)" text-anchor="middle" font-family="sans-serif">linear attention</text><circle cx="585" cy="175" r="100" fill="var(--bg-tint)" stroke="var(--border-strong)" stroke-width="1.4"/><g stroke="var(--text-faint)" stroke-width="1" opacity="0.5"><line x1="467" y1="175" x2="703" y2="175"/><line x1="585" y1="57" x2="585" y2="293"/></g><text x="706" y="179" font-size="11" fill="var(--text-faint)" font-style="italic">f₁</text><text x="580" y="54" font-size="11" fill="var(--text-faint)" font-style="italic" text-anchor="end">f₂</text><path d="M585,175 L684.0,161.1 A100,100 0 0 1 684.0,188.9 Z" fill="#d0ad39"/><path d="M585,175 L571.1,76.0 A100,100 0 0 1 598.9,76.0 Z" fill="#d05739"/><line x1="585" y1="175" x2="685.0" y2="175.0" stroke="#d0ad39" stroke-width="3"/><path d="M685.0,175.0 L675.3,180.2 L675.3,169.8 Z" fill="#d0ad39"/><line x1="585" y1="175" x2="585.0" y2="75.0" stroke="#d05739" stroke-width="3"/><path d="M585.0,75.0 L590.2,84.7 L579.8,84.7 Z" fill="#d05739"/><line x1="585" y1="175" x2="641.6" y2="118.4" stroke="var(--text-faint)" stroke-width="2" stroke-dasharray="5 4"/><path d="M641.6,118.4 L638.4,129.0 L631.0,121.6 Z" fill="var(--text-faint)"/><text x="691" y="169" font-size="12" fill="#d0ad39" font-weight="600">[1,0]</text><text x="592" y="77" font-size="12" fill="#d05739" font-weight="600">[0,1]</text><text x="649" y="115" font-size="11.5" fill="var(--text-mute)" font-style="italic">非正交</text><text x="585" y="309" font-size="12.5" fill="var(--text-mute)" text-anchor="middle">只有两个正交 one-hot</text></svg>
<figcaption>同样的二维（低维）特征空间：softmax 能切出许多尖锐的 one-hot 分区（左）；linear attention 受特征维度限制，只有两个正交方向能成为干净的 one-hot，其余方向都是非正交的叠加（右）。</figcaption>
</figure>

如果 $\phi$ 只是逐元素地作用在每个分量上，那么 $d_\phi=d$——二维就是 $d_\phi=2$，只能容纳两个正交的 one-hot（正如上图）；但 $\phi$ 也可以设计得更复杂，把 $d_\phi$ 抬上去，从而容纳更多清晰的分支。而"把 $\phi$ 设计得更复杂"这条思路，也正是 Performer {% include cite.html key="performer" %}、cosFormer {% include cite.html key="cosformer" %} 这些后续改进的出发点，想办法把这个 rank 重新做大一些来补偿一些表达能力。所以这条认知，一定程度上可以被印证。

## Delta Rule

*Linear Transformers Are Secretly Fast Weight Programmers* {% include cite.html key="schlag2021" %}。

我理解，delta rule 这件事，同样是 rank 缩减带来的效应。原本rank 足够的方式里，k 有足够的空间去选择，我们并不需要考虑"相同 k"的历史关联性。但在 linear attention 这种设计下，由于 rank 的损失、S 被压缩，k 被迫和历史关联了起来——在这种计算下，不仅有之前所说的k的特征表达受限，我们还通过结合的方式直接把k和v乘了起来，这个过程其实可以这么理解，首先我们仍然可以认为这里是qkv的三者相乘，我们认为qk的发生的作用仍然是得到一个注意力矩阵，这个矩阵再去对v进行信息提取。但是由于这里的rank的损失的设计，对于相同的k，我们对kv的更新总会以一种方式修改过去的k的信息，而这些k的信息也会在q关注到相同的k的时候被一起提取出来，我们无法区分他们，所以这里看作一种"key 对应 value 的信息提取"。实际上还是由于前面的低秩变换带来的不良影响。

于是，为了减少这种影响，我们需要补充一些设定，让这种关联性按我们期望的形式去发展。比如这里就认为：这种关联应该用 delta 的方式——始终更关心最新的情况，或者采用某种策略去平衡这种历史关联性。

具体到 DeltaNet，它把"直接累加"换成了"按预测误差来修正"：先用当前的 key 从状态里读出一个"旧值" $S_{t-1}^\top k_t$，再朝真实的 $v_t$ 纠一步——

$$
S_t = S_{t-1} + \beta_tk_t\,(v_t - S_{t-1}^\top k_t)^\top .
$$

$\beta_t$ 是这一步修正的强度（写入的学习率）。

但这种假定也只是一种在实践序列上的补偿，相比原本的attention并不能真的补全所有的能力，毕竟天下没有免费的午餐。首先，这个假定带来了对时间轴上相同key不同value的的平衡。比如对相同的 key，远近并不总意味着重要性的高低——这种关联性很可能是视情况而定，那么这里的这种策略能够去自动平衡这种权重。但是呢，其实每个不同的时间轴上的q，对这里的key的关注可能也是不一样的，比如有的q可能更关注近的，有的q更关注远的，这种补偿也并不能解决这个问题，还是产生了表达能力的限制。那么这里没有学好的这种知识，会在别的地方发生代偿，或者说需要更大的网络或者其他的结构来补充这里丢失的关联性。

## Gated DeltaNet

*Gated Delta Networks: Improving Mamba2 with Delta Rule* {% include cite.html key="gateddeltanet" %} 的核心，是在 DeltaNet 上增加一个 forget gate：由于固定大小的状态矩阵 $S$ 会不断压入历史信息，旧知识容易相互混杂、产生干扰，并最终造成状态饱和，因此模型根据当前输入动态预测 $\alpha_t$，对历史状态进行衰减，再通过 $\beta_t$ 和当前的 $k_t$ 对相应的映射进行局部修正。

写成状态更新，就是先对旧状态整体乘一个标量遗忘门 $\alpha_t$，再做一次 delta 修正——

$$
\widetilde S_{t-1} = \alpha_t S_{t-1}, \qquad
S_t = \widetilde S_{t-1} + \beta_tk_t\,(v_t - \widetilde S_{t-1}^\top k_t)^\top .
$$

它本质上还是是对"固定状态压缩"这一缺陷的一种"缝缝补补"：提供一个输入相关的遗忘策略，帮助释放容量、减少旧信息的干扰。但这种选择性，主要是"什么时候遗忘、遗忘多少"——它并不能精确地选中状态里的某一条历史记忆，也没有解决"无法重新访问原始历史 token"这个根本限制，相反地，相比于原本的delta rule，由于这个门显然是不对称的，它总是倾向于忘记历史而不会选择忘记最近的事情，所以它也会倾向于记住更近处的情况，忘掉远处的情况，这也是人为引入的一种归纳偏置。

## KDA

*Kimi Linear: An Expressive, Efficient Attention Architecture* {% include cite.html key="kimilinear" %} 提出了 Kimi Delta Attention（KDA）。它从基础 linear attention 的直接累加 $S_t=S_{t-1}+k_tv_t^\top$ 出发，先用 delta rule 改为按预测误差修正当前的 key–value 映射，再在 Gated DeltaNet「全局标量遗忘门」的基础上，把 $\alpha_t$ 从一个标量扩展成逐 key-feature 维度的向量门——

$$
S_t = (I-\beta_tk_tk_t^\top)\,\operatorname{Diag}(\alpha_t)\,S_{t-1} + \beta_tk_tv_t^\top .
$$

这里的 $q_t,k_t$ 都是经过可学习特征变换、再做 L2 归一化后的表示。

KDA 仍然是在固定大小的状态 $S$ 上做改进。历史信息被持续压缩之后，会发生混杂、碰撞和状态饱和；Gated DeltaNet 只能对整个 head 一起遗忘，而 KDA 允许不同的 feature channel 分别决定保留多少历史，从而实现更细粒度的容量管理与遗忘策略。

但说到底，它仍然是一种更细粒度的缝补与信息补充：仍不能像标准 attention 那样重新访问任意的原始历史 token，也无法真正按照未来的 query，精确地选中某一条历史记忆。前述的两个问题，一个是q的关注问题，一个是遗忘的归纳偏置问题也都是存在的。

## 总结

起点是一个简化：把注意力里的相似度换成特征映射的内积 $\mathrm{sim}(q,k)=\phi(q)^\top\phi(k)$。用结合律改用一个固定大小的状态 $S=\sum_{j}\phi(k_j)v_j^\top$ 去取代 $O(N^2)$ 的显式注意力。

而在这个固定状态之上，"每一步该怎么更新 $S$"不断演化：

$$
\boxed{
\begin{aligned}
\text{Linear Attention:}\quad &S_t=S_{t-1}+k_tv_t^\top \\[2mm]
\text{DeltaNet:}\quad &S_t=S_{t-1} +\beta_tk_t(v_t-S_{t-1}^\top k_t)^\top \\[2mm]
\text{Gated DeltaNet:}\quad &\widetilde S_{t-1}=\alpha_tS_{t-1} \\
&S_t=\widetilde S_{t-1} +\beta_tk_t(v_t-\widetilde S_{t-1}^\top k_t)^\top \\[2mm]
\text{KDA:}\quad &\widetilde S_{t-1} =\operatorname{Diag}(\alpha_t)S_{t-1} \\
&S_t=\widetilde S_{t-1} +\beta_tk_t(v_t-\widetilde S_{t-1}^\top k_t)^\top
\end{aligned}
}
$$

一句话概括：

$$
\text{直接累加}\ \rightarrow\ \text{误差修正}\ \rightarrow\ \text{整体遗忘后修正}\ \rightarrow\ \text{逐维遗忘后修正}
$$

有几个问题仍未解决：

- feature 特性被改变。
- 不同 query 的历史关注缺失。
- 历史遗忘的偏置。