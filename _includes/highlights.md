<section class="section" id="selected">
<h2 class="section__title">Selected Works</h2>

<div class="highlights">
{% for p in site.data.publications.main %}{% if p.featured %}
  <article class="highlight">
    <div class="highlight__media">
      {% if p.image %}<img src="{{ p.image }}" alt="{{ p.title }}" loading="lazy" decoding="async">{% endif %}
      {% if p.conference_short %}<span class="highlight__badge">{{ p.conference_short }}</span>{% endif %}
    </div>
    <div class="highlight__body">
      <h3 class="highlight__title">
        {% if p.pdf %}<a href="{{ p.pdf }}" target="_blank" rel="noopener">{{ p.title }}</a>{% else %}{{ p.title }}{% endif %}
      </h3>
      <div class="highlight__authors">{{ p.authors }}</div>
      <div class="highlight__venue">{{ p.conference }}</div>
      {% if p.blurb %}<p class="highlight__blurb">{{ p.blurb }}</p>{% endif %}
      <div class="highlight__links">
        {% if p.pdf %}<a class="btn" href="{{ p.pdf }}" target="_blank" rel="noopener">PDF</a>{% endif %}
        {% if p.code %}<a class="btn" href="{{ p.code }}" target="_blank" rel="noopener">Code</a>{% endif %}
        {% if p.page %}<a class="btn" href="{{ p.page }}" target="_blank" rel="noopener">Project</a>{% endif %}
        {% if p.bibtex %}<a class="btn" href="{{ p.bibtex }}" target="_blank" rel="noopener">BibTeX</a>{% endif %}
      </div>
    </div>
  </article>
{% endif %}{% endfor %}
</div>
</section>
