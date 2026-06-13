---
layout: homepage
---

<section class="section section--showcase" id="industry">
<h2 class="section__title">Industry · miHoYo</h2>

<div class="showcase">
  <figure class="work-figure showcase__main">
    <div class="fig-num">Fig. 01</div>
    <picture>
      <source media="(max-width: 720px)" srcset="./assets/img/shenhe-960.jpg">
      <img src="./assets/img/shenhe.jpg" alt="Shenhe — new outfit promotional art" loading="lazy" decoding="async">
    </picture>
    <figcaption>
      <span><em>Shenhe</em> — 冷花幽露 · character animation</span>
      <span>Genshin Impact · miHoYo</span>
    </figcaption>
  </figure>

  <div class="showcase__side">
    <figure class="work-figure">
      <div class="fig-num">Fig. 02</div>
      <picture>
        <source srcset="./assets/img/lumi-moon.avif" type="image/avif">
        <img src="./assets/img/lumi-moon.jpg" alt="Lumi — official Mid-Autumn artwork, miHoYo digital human" loading="lazy" decoding="async">
      </picture>
      <figcaption>
        <span><em>Lumi</em> — 千里共明月 · CFX</span>
        <span>official PV · miHoYo</span>
      </figcaption>
    </figure>

    <div class="showcase__links">
      <a class="lumi__card" href="https://www.bilibili.com/video/BV1GH4y1Z7yS" target="_blank" rel="noopener">
        <span class="lumi__icon"><svg><use href="#i-play"/></svg></span>
        <span>
          <span class="lumi__title">yoyo_Lumi · CFX</span>
          <span class="lumi__sub">bilibili · breakdown</span>
        </span>
      </a>
      <a class="lumi__card" href="https://www.bilibili.com/video/BV1LV4y1b7ba" target="_blank" rel="noopener">
        <span class="lumi__icon"><svg><use href="#i-play"/></svg></span>
        <span>
          <span class="lumi__title">yoyo_Lumi · Live Show</span>
          <span class="lumi__sub">bilibili · performance</span>
        </span>
      </a>
    </div>
  </div>
</div>
</section>


<section class="section" id="about">
<h2 class="section__title">About</h2>

<div class="about">
<p>I am <strong>Jia-Ming Lu</strong>, a researcher working at the intersection of <strong>physically-based simulation</strong>, <strong>animation</strong>, and <strong>AI for 3D content creation</strong>. I&rsquo;m deeply dedicated to unlocking the full potential of human creativity, driven by a passion for delving into the underlying, fundamental relationships between things.</p>
</div>

<div class="timeline">
  <div class="timeline__row">
    <div class="timeline__when">2024 — now</div>
    <div class="timeline__what"><strong>ByteDance</strong> — Researcher, AI3D</div>
  </div>
  <div class="timeline__row">
    <div class="timeline__when">2021 — 2024</div>
    <div class="timeline__what"><strong>miHoYo</strong> — CFX Tech Lead &amp; Cloth Simulation Topic Owner. Worked on the digital human <em>Lumi</em> and <em>Genshin Impact</em>.</div>
  </div>
  <div class="timeline__row">
    <div class="timeline__when">2016 — 2021</div>
    <div class="timeline__what"><strong>Ph.D., Tsinghua University</strong> — Computer Graphics, advised by Prof. Shi-Min Hu.</div>
  </div>
  <div class="timeline__row">
    <div class="timeline__when">2012 — 2016</div>
    <div class="timeline__what"><strong>B.S., Tsinghua University</strong></div>
  </div>
</div>
</section>


<section class="section" id="interests">
<h2 class="section__title">Research Interests</h2>

<div class="tags">
  <span class="tag"><strong>CG</strong> physically-based simulation</span>
  <span class="tag"><strong>CG</strong> differentiable simulation</span>
  <span class="tag"><strong>ML</strong> animation generation</span>
  <span class="tag"><strong>ML</strong> world model</span>
  <span class="tag"><strong>3D</strong> neural fields &amp; representations</span>
</div>
</section>


{% include_relative _includes/highlights.md %}


{% include_relative _includes/publications.md %}


{% include_relative _includes/services.md %}


<section class="section" id="games">
<h2 class="section__title">Game Achievements</h2>

<p style="color:var(--text-mute); margin: 0 0 18px; font-size: 15px;">Outside of work, I&rsquo;m a long-time action-game enthusiast. A few personal milestones I&rsquo;m proud of.</p>

<div class="games">
  <div class="game">
    <span class="game__icon">🐺</span>
    <span class="game__body">
      <span class="game__title">Monster Hunter</span>
      <span class="game__detail">Zinogre (雷狼龙) Arena · <strong>6:30</strong></span>
    </span>
  </div>
  <div class="game">
    <span class="game__icon">🔥</span>
    <span class="game__body">
      <span class="game__title">Dark Souls II</span>
      <span class="game__detail"><strong>100%</strong> achievements</span>
    </span>
  </div>
  <div class="game">
    <span class="game__icon">🗡️</span>
    <span class="game__body">
      <span class="game__title">Sekiro: Shadows Die Twice</span>
      <span class="game__detail"><strong>100%</strong> achievements</span>
    </span>
  </div>
  <div class="game">
    <span class="game__icon">👑</span>
    <span class="game__body">
      <span class="game__title">Elden Ring</span>
      <span class="game__detail"><strong>100%</strong> achievements</span>
    </span>
  </div>
  <div class="game">
    <span class="game__icon">
      <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true" style="display:block">
        <defs>
          <linearGradient id="hs-orange" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stop-color="#fdba74"/>
            <stop offset="1" stop-color="#ea580c"/>
          </linearGradient>
        </defs>
        <polygon points="12,2 20.66,7 20.66,17 12,22 3.34,17 3.34,7" fill="url(#hs-orange)"/>
      </svg>
    </span>
    <span class="game__body">
      <span class="game__title">Hearthstone</span>
      <span class="game__detail">Top <strong>2000 Legend</strong></span>
    </span>
  </div>
</div>
</section>
