---
title: "Training Dynamics 学习笔记（一）"
date: 2026-11-04
description: "从训练过程理解模型能力的形成，并持续整理 Training Dynamics 相关论文。"
tags: [llm, learning-log, training-dynamics, optimization]
topic: "模型训练与优化"
topic_order: 3
references:
  - key: training-dynamics-position
    title: "Position: Don't Just \"Fix it in Post\": A Science of AI Must Study Training Dynamics"
    authors: "Biderman et al. · 2026"
    url: https://arxiv.org/abs/2606.06533
  - key: saxe-deep-linear
    title: "Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Neural Networks"
    authors: "Saxe, McClelland, and Ganguli · 2014"
    url: https://www.saxelab.org/assets/papers/Saxe2014.pdf
  - key: ntk
    title: "Neural Tangent Kernel: Convergence and Generalization in Neural Networks"
    authors: "Jacot, Gabriel, and Hongler · 2018"
    url: https://arxiv.org/abs/1806.07572
  - key: semantic-development
    title: "A Mathematical Theory of Semantic Development in Deep Neural Networks"
    authors: "Saxe, McClelland, and Ganguli · 2019"
    url: https://arxiv.org/abs/1810.10531
  - key: lazy-training
    title: "On Lazy Training in Differentiable Programming"
    authors: "Chizat, Oyallon, and Bach · 2019"
    url: https://arxiv.org/abs/1812.07956
  - key: edge-of-stability
    title: "Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability"
    authors: "Cohen et al. · 2022"
    url: https://arxiv.org/abs/2103.00065
  - key: self-stabilization
    title: "Self-Stabilization: The Implicit Bias of Gradient Descent at the Edge of Stability"
    authors: "Damian, Nichani, and Lee · 2023"
    url: https://arxiv.org/abs/2209.15594
---

Training dynamics 也是 dynamics。*Position: Don't Just "Fix it in Post"* {% include cite.html key="training-dynamics-position" %} 可以作为这个方向的入口。相比人脑，神经网络对我们来说已经足够“白盒”，因此也是研究学习行为非常好的对象。只研究训练完成的模型，远远不足以理解大模型的内在规律；模型能力如何在训练过程中逐渐形成，可能是更加关键的问题。

Training dynamics 与架构设计、优化器和数据都有直接关系，可以说非常重要，但也比研究模型的静态计算过程更加抽象：它描述的是一个持续演化的过程，其中包含大量尚未学习完成的中间状态。

## 基础分析问题

给定网络 $f(x;\theta)$ 和训练集 $\{(x_i,y_i)\}_{i=1}^n$，可以用凸函数 $\Phi$ 生成的 Bregman 散度统一表示一类损失：

$$
D_\Phi(y,z)=\Phi(y)-\Phi(z)-\langle\nabla\Phi(z),y-z\rangle,
\qquad
\mathcal L(\theta)=\sum_{i=1}^n D_\Phi\bigl(y_i,f(x_i;\theta)\bigr).
$$

平方损失是 $\Phi(z)=\frac12\lVert z\rVert^2$ 的特例；在概率空间取负熵作为 $\Phi$ 时得到 KL 散度，对 one-hot 标签等价于交叉熵。令 $f(\theta)$ 表示所有训练样本上的输出，连续时间下的一般梯度流为

$$
\dot\theta=-\nabla_\theta\mathcal L(\theta)
=-J(\theta)^\top\nabla_f\mathcal L,
$$

$J(\theta)$ 是模型输出对参数的雅可比矩阵。平方损失下 $\nabla_f\mathcal L=f-y$；softmax 交叉熵对 logits 的梯度则是 $p-y$。

Training dynamics 要研究的是：$\theta(t)$ 和 loss 如何随时间演化，训练是平滑收敛还是会经历平台期与多阶段变化；不同方向或模态以什么顺序被学到；以及初始化、网络深度和其他训练设定如何改变这条轨迹及其收敛速度。

## 深度线性网络中的非线性学习过程

第一篇是 *Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Neural Networks* {% include cite.html key="saxe-deep-linear" %}。

这篇论文主要针对一个简化的深度线性模型进行理想化分析。虽然网络没有非线性激活，权重也只是线性矩阵，但优化过程并不是线性的。多个矩阵相乘后，参数之间会形成高阶耦合，每一层的更新都会受到其他层当前状态的影响。

文章认为，这种相互作用会带来训练中的平台期，以及平台期之后的快速变化。它进一步从不同模态方向分析学习过程：不同方向的强度和初始尺度不同，因此被学到的速度也不同。通过更合适的初始化，可以缩短平台期并加速深层网络的学习；对于矩阵参数，也需要综合考虑不同模态方向以及多层之间的初始化量级。

我觉得这一点对理解后续的 Muon optimizer 也有一定启发。两者讨论的对象并不完全相同，但都提示我们，矩阵优化不能只关注整体尺度，不同谱方向上的更新和演化同样重要。

## Neural Tangent Kernel

*Neural Tangent Kernel: Convergence and Generalization in Neural Networks* {% include cite.html key="ntk" %} 定义了由参数梯度构成的核函数：

$$
K_\theta(x,x')=\nabla_\theta f(x;\theta)^\top\nabla_\theta f(x';\theta).
$$

在无限宽极限下，这个核会收敛并在训练过程中保持不变。于是，原本复杂的参数演化可以转化为输出函数在固定核下的演化；对于平方损失，它进一步变成一个线性微分方程，因此可以直接分析不同核模态的学习速度和收敛过程。

简单来说，这篇文章给出了一个参考锚点：在无限宽、核基本不变的情况下，training dynamics 会变成一个容易求解的问题。不过我觉得它也有两个明显限制。

第一，无限宽假设比较理想化。它依赖宽度足够大时的集中规律，而真实网络是否接近这个极限、需要多宽才能接近，都并不简单。尤其当有限宽网络在训练中发生明显的特征学习时，固定核近似的参考意义会下降。

第二，把函数演化线性化是让问题可分析的一种充分条件，但不是理解 training dynamics 的必要条件。NTK 给出了一个很有意思的可解极限，却没有覆盖网络表示本身持续变化的情况，因此能够直接迁移到真实大模型训练中的结论仍然有限。

## Semantic Development

*A Mathematical Theory of Semantic Development in Deep Neural Networks* {% include cite.html key="semantic-development" %} 使用一个三层深线性网络，在由进化树生成的人工层次化语义数据上进行梯度下降。通过对数据相关矩阵做 SVD，文章把训练动力学分解成一组近似独立的模态，每个模态都呈现 S 形学习曲线。奇异值越大的模态越早被学会，因此网络会表现出从粗粒度类别到细粒度概念的“先粗后细”学习顺序，并可以用来解释一些语义发展的阶段性现象。

不过批判性地看，首先，这个网络仍然过于简单，与真实神经网络的行为可能存在较大差距。其次，数据本身就是按照分叉层次人工构造的，网络从中拟合出相应的层次规律也比较自然。这些结论未必能够直接 scaling 到真实数据和大模型，但仍然可以引发一些关于能力学习顺序的定性思考。

## Lazy Training

*On Lazy Training in Differentiable Programming* {% include cite.html key="lazy-training" %} 可以看作对 NTK 图景的一次批判和补充。文章指出，lazy training 并不只是由网络无限宽或参数过多导致的，更关键的是模型参数化的尺度。当尺度超过一定范围后，参数相对初始化只发生很小变化，模型就可以由初始化处的一阶线性化描述，训练也退化为固定核下的学习。

但是，这种可分析性不一定是我们真正希望得到的训练方式。文章的实验显示，把网络推入 lazy regime 反而可能损害性能，说明真实神经网络的成功很可能依赖参数和表示在训练中发生实质变化。这与前面对 NTK 的思考类似：它给出了一个理想、容易分析的极限，但未必对应实际问题中最有效的状态。

放到 LLM 中看，参数量、训练数据和计算量都需要仔细配比，实际训练也并不是单纯依靠远多于训练要求的参数进入固定核状态。token/parameter 比例不能直接构成对 NTK 的数学反证，但至少说明真实大模型追求的是有效的特征学习与资源平衡，而不是把模型推到极端的 lazy regime。

## Edge of Stability

*Gradient Descent on Neural Networks Typically Occurs at the Edge of Stability* {% include cite.html key="edge-of-stability" %} 总结了一种与经典优化图景很不一样的现象：训练开始后，loss landscape 会持续锐化，最大 Hessian 特征值逐渐接近稳定性边界 $2/\eta$。到达边界后，训练并没有像局部二次模型预测的那样直接发散，而是在边缘附近形成某种自我调节；loss 在短时间尺度上可能震荡甚至上升，但在长时间尺度上仍然持续下降。

这意味着，把训练看成在一个近似固定的局部二次曲面上稳定下降，并不能描述深度网络实际的优化过程。网络本身在更新，所处曲率也在变化，训练会长期运行在经典稳定条件刚刚失效的位置附近。

这篇论文主要是在提出和确认一种现象，但我觉得这个现象非常有意思，也进一步说明了神经网络训练过程的复杂性。为什么网络会锐化到稳定性边缘，又如何在边缘附近继续优化收敛，是后续 training dynamics 中很重要的一类问题。

## Self-Stabilization

*Self-Stabilization: The Implicit Bias of Gradient Descent at the Edge of Stability* {% include cite.html key="self-stabilization" %} 从观察现象进一步推进到解释机制。按照 Edge of Stability 的图景，训练会先经历 progressive sharpening，快速进入曲率较高的区域。文章随后分析一种特殊情形：只有最大特征值对应的方向越过稳定边界，迭代开始沿这个方向来回振荡。

在这种情况下，二次近似本身无法解释为什么训练不会发散，而三次项会让参数偏离时的局部曲率下降，使稳定区域变宽，最终把优化重新推回稳定边界附近。直观上，振荡本身产生了负反馈，使梯度下降不断在边缘试探，而不是真的进入持续发散的区域。

这个解释有明确条件：局部只有一个方向失稳，并且三阶展开能够描述附近的几何；纯二次模型或者多个方向同时失稳时，分析并不直接成立。不过，在已经观察到训练会迅速进入尖锐区域的前提下，这种单一主导方向的分析仍然很有启发性。至于高维空间中多个尖锐方向是否确实更少见，可以作为进一步的直觉，但不是这篇论文已经证明的结论。

<figure class="fig">
<svg viewBox="0 0 900 590" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="valley-title valley-desc">
  <title id="valley-title">One valley, two regimes: oscillation at the narrow end and convergence at the wide end</title>
  <desc id="valley-desc">A perspective view of a curved loss valley that is narrow and highly curved near the starting point, then widens and becomes less curved. Red points show gradient descent iterates oscillating across the valley while moving toward the wide end.</desc>
  <defs>
    <linearGradient id="left-wall" x1="0.82" y1="0.05" x2="0.16" y2="0.92">
      <stop offset="0" stop-color="#3c8d86" stop-opacity="0.92"/>
      <stop offset="0.62" stop-color="#b8ddd7" stop-opacity="0.88"/>
      <stop offset="1" stop-color="#e2f1ee" stop-opacity="0.72"/>
    </linearGradient>
    <linearGradient id="right-wall" x1="0.52" y1="0.02" x2="0.48" y2="1">
      <stop offset="0" stop-color="#27736d" stop-opacity="0.94"/>
      <stop offset="0.55" stop-color="#78b8b0" stop-opacity="0.88"/>
      <stop offset="1" stop-color="#c9e5e0" stop-opacity="0.74"/>
    </linearGradient>
    <linearGradient id="valley-depth" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#2f7772" stop-opacity="0.18"/>
      <stop offset="1" stop-color="#ffffff" stop-opacity="0.04"/>
    </linearGradient>
    <clipPath id="valley-clip">
      <path d="M548 96 C458 194 289 309 82 354 C238 487 477 548 782 407 C679 338 658 208 613 112 C590 99 569 94 548 96 Z"/>
    </clipPath>
    <marker id="trajectory-arrow" viewBox="0 0 10 10" refX="8.2" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0L10 5L0 10Z" fill="#c94f32"/>
    </marker>
    <marker id="label-arrow" viewBox="0 0 10 10" refX="8.2" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0 0L10 5L0 10Z" fill="var(--text-mute)"/>
    </marker>
  </defs>

  <rect x="1" y="1" width="898" height="588" rx="16" fill="var(--bg-tint)" stroke="var(--border)"/>
  <text x="42" y="43" fill="var(--text)" font-family="sans-serif" font-size="23" font-weight="650">One valley, two regimes</text>
  <text x="42" y="70" fill="var(--text-mute)" font-family="sans-serif" font-size="15.5">Oscillation at the narrow end · convergence at the wide end</text>

  <!-- Two steep walls meet at a recessed trough. The tight V at the far end
       is intentionally much sharper than the broad near end. -->
  <path d="M548 96 C458 194 289 309 82 354 C238 487 365 522 458 505 C493 421 536 310 581 224 C571 165 558 112 548 96 Z"
        fill="url(#left-wall)" stroke="#2b6f69" stroke-width="2.4" stroke-linejoin="round"/>
  <path d="M613 112 C658 208 679 338 782 407 C669 474 550 512 458 505 C493 421 536 310 581 224 C596 166 606 121 613 112 Z"
        fill="url(#right-wall)" stroke="#245f5b" stroke-width="2.4" stroke-linejoin="round"/>
  <path d="M548 96 Q581 224 613 112 Q596 166 581 224 Q571 165 548 96 Z"
        fill="#174f4b" opacity="0.32"/>

  <g clip-path="url(#valley-clip)" fill="none" stroke="#397b76" stroke-width="0.9" opacity="0.34">
    <path d="M558 102 C497 207 345 374 175 468"/>
    <path d="M568 103 C527 222 414 405 283 510"/>
    <path d="M579 104 C558 237 488 432 405 536"/>
    <path d="M590 106 C591 239 558 429 534 521"/>
    <path d="M601 108 C624 235 631 393 655 467"/>
    <path d="M612 112 C661 224 700 341 754 401"/>

    <path d="M548 96 Q581 224 613 112" stroke-width="2" opacity="0.95"/>
    <path d="M517 137 Q573 270 629 145" stroke-width="1.7" opacity="0.86"/>
    <path d="M477 181 Q558 316 650 190" stroke-width="1.5" opacity="0.76"/>
    <path d="M426 229 Q539 365 674 239" stroke-width="1.3" opacity="0.66"/>
    <path d="M362 282 Q512 424 704 294" stroke-width="1.15" opacity="0.58"/>
    <path d="M280 334 Q480 474 739 350" stroke-width="1" opacity="0.5"/>
    <path d="M178 379 Q458 530 775 405" stroke-width="0.9" opacity="0.44"/>
  </g>

  <path d="M548 96 C458 194 289 309 82 354" fill="none" stroke="#236a65" stroke-width="4" stroke-linecap="round"/>
  <path d="M613 112 C658 208 679 338 782 407" fill="none" stroke="#236a65" stroke-width="4" stroke-linecap="round"/>
  <path d="M581 224 C550 310 503 421 458 505" fill="none" stroke="#123f3c" stroke-width="3.8" stroke-linecap="round" opacity="0.95"/>

  <path d="M623 182 L566 194 L632 220 L538 248 L654 282 L431 318 L650 373 L242 433 L458 505"
        fill="none" stroke="#c94f32" stroke-width="2.6" stroke-linejoin="round" stroke-linecap="round" marker-end="url(#trajectory-arrow)"/>
  <g fill="#c94f32" stroke="var(--bg)" stroke-width="1.8">
    <circle cx="623" cy="182" r="5.2"/>
    <circle cx="566" cy="194" r="4.7"/>
    <circle cx="632" cy="220" r="4.7"/>
    <circle cx="538" cy="248" r="4.7"/>
    <circle cx="654" cy="282" r="4.7"/>
    <circle cx="431" cy="318" r="4.7"/>
    <circle cx="650" cy="373" r="4.7"/>
    <circle cx="242" cy="433" r="4.7"/>
    <circle cx="458" cy="505" r="5.2"/>
  </g>

  <path d="M665 172 C650 176 638 179 627 182" fill="none" stroke="var(--text-mute)" stroke-width="1.4" marker-end="url(#label-arrow)"/>
  <text x="672" y="166" fill="#c94f32" font-family="sans-serif" font-size="14" font-weight="600">Start</text>

  <rect x="680" y="93" width="176" height="70" rx="10" fill="var(--bg)" stroke="var(--border)"/>
  <text x="698" y="119" fill="#236a65" font-family="sans-serif" font-size="17" font-weight="650">Narrow end</text>
  <text x="698" y="143" fill="var(--text-mute)" font-family="sans-serif" font-size="14">Transverse curvature ≈ 2.70</text>
  <path d="M680 145 C645 151 612 178 584 217" fill="none" stroke="var(--text-mute)" stroke-width="1.3" marker-end="url(#label-arrow)"/>

  <rect x="43" y="444" width="204" height="70" rx="10" fill="var(--bg)" stroke="var(--border)"/>
  <text x="61" y="470" fill="#236a65" font-family="sans-serif" font-size="17" font-weight="650">Wide end</text>
  <text x="61" y="494" fill="var(--text-mute)" font-family="sans-serif" font-size="14">Transverse curvature ≈ 0.14</text>
  <path d="M248 471 C280 480 315 490 349 497" fill="none" stroke="var(--text-mute)" stroke-width="1.3" marker-end="url(#label-arrow)"/>

  <rect x="572" y="498" width="284" height="49" rx="10" fill="var(--bg)" stroke="var(--border)"/>
  <line x1="592" y1="522" x2="628" y2="522" stroke="#c94f32" stroke-width="2.6" marker-end="url(#trajectory-arrow)"/>
  <circle cx="592" cy="522" r="4.3" fill="#c94f32"/>
  <text x="641" y="527" fill="var(--text-mute)" font-family="sans-serif" font-size="13.5">Gradient-descent iterates</text>

  <line x1="331" y1="91" x2="381" y2="91" stroke="#236a65" stroke-width="4" stroke-linecap="round"/>
  <text x="394" y="96" fill="var(--text-mute)" font-family="sans-serif" font-size="13.5">Equal-loss side contours (L = 2)</text>
</svg>
<figcaption>同一条 loss valley 可以在起点附近很窄、横向曲率很高，并沿下降方向逐渐变宽。红色轨迹在窄端反复跨越谷底，进入宽端后震荡减弱。图根据上传的参考图重新绘制，图内标注均为英文。</figcaption>
</figure>

## 小结

从这些工作中，可以看到我们对 training dynamics 的认识经历了一个明显的转变。早期研究希望通过构造理想化模型，精确分析网络内部发生的事情；但这些假设往往很难直接用于现实网络。后来的研究甚至发现，深度神经网络恰好长期工作在传统稳定性理论的边界附近。这意味着，我们可能需要新的学习理论和描述方式，而不只是继续套用经典优化的局部图景。

不过我认为，对于 Edge of Stability 和 Self-Stabilization，仅仅解释网络如何在边界附近形成稳定约束还不够。稳定收敛并不是深度网络天然具备的性质，而是架构、初始化、归一化和优化方法不断演进后共同塑造的结果；研究这种约束，某种程度上是在研究这些设计如何防止训练失控。相比之下，更核心的问题可能是 progressive sharpening 本身：网络为什么会快速进入尖锐区域，并主动逼近稳定边界？

从某种角度看，这又像是一种合理的学习行为：学习并不是停留在已经掌握的区域，而是在理解与不理解的边界上持续推进，从而扩展能力范围。人类学习也有类似的直觉。当然，这还只是一个类比，并不能直接作为神经网络训练机制的解释，但它提示我们，接近边界可能不只是训练中的异常现象，而可能正是学习过程本身的重要组成部分。
