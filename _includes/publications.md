<section class="section" id="publications">
<h2 class="section__title"><span class="section__sigil">§ 05</span>Publications</h2>

<ol class="pubs">
{% for p in site.data.publications.main %}
  <li class="pub">
    <div class="pub__media">
      {% if p.image %}<img src="{{ p.image }}" alt="{{ p.title }}" loading="lazy" decoding="async">{% endif %}
      {% if p.conference_short %}<span class="pub__badge">{{ p.conference_short }}</span>{% endif %}
    </div>
    <div class="pub__body">
      <h3 class="pub__title">
        {% if p.pdf %}<a href="{{ p.pdf }}" target="_blank" rel="noopener">{{ p.title }}</a>{% else %}{{ p.title }}{% endif %}
      </h3>
      <div class="pub__authors">{{ p.authors }}</div>
      <div class="pub__venue">{{ p.conference }}</div>
      {% if p.notes %}<div class="pub__note">{{ p.notes }}</div>{% endif %}
      <div class="pub__links">
        {% if p.pdf %}<a class="btn" href="{{ p.pdf }}" target="_blank" rel="noopener">PDF</a>{% endif %}
        {% if p.code %}<a class="btn" href="{{ p.code }}" target="_blank" rel="noopener">Code</a>{% endif %}
        {% if p.page %}<a class="btn" href="{{ p.page }}" target="_blank" rel="noopener">Project</a>{% endif %}
        {% if p.bibtex %}<a class="btn" href="{{ p.bibtex }}" target="_blank" rel="noopener">BibTeX</a>{% endif %}
      </div>
    </div>
  </li>
{% endfor %}
</ol>
</section>
