#!/usr/bin/env python3
"""
blog_preview.py — preview blog drafts & posts locally, without Jekyll.

Renders every file in _drafts/ and _posts/ into standalone HTML under
.blog_preview/ using the real blog CSS (assets/css/blog.scss), so you can
see almost exactly what a post will look like once published — including
drafts that are NOT committed and that nobody else can see.

Usage:
    python3 tools/preview/render.py            # regenerate .blog_preview/*.html
    python3 tools/preview/render.py --shots    # also save PNG screenshots (needs playwright)

Then open .blog_preview/index.html in a browser.
"""
import os
import re
import sys
import glob
import html
import datetime

import sass
import yaml
import markdown

# Repo root is two levels up from tools/preview/render.py
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, ".blog_preview")

# Markdown feature set shared by the live server and the screenshot renderer.
MD_EXTS = [
    "fenced_code",           # ``` fenced code blocks
    "codehilite",            # pygments highlighting -> .highlight token spans
    "tables",
    "footnotes",
    "sane_lists",
    "pymdownx.arithmatex",   # $...$ / $$...$$  ->  \(..\) / \[..\]  for MathJax
]
MD_CFG = {
    "codehilite": {"css_class": "highlight", "guess_lang": False},
    "pymdownx.arithmatex": {"generic": True},
}


def render_markdown(text):
    """Markdown -> HTML with code highlighting + math, used by all previews."""
    return markdown.markdown(text, extensions=MD_EXTS, extension_configs=MD_CFG)


# Top-of-post disclaimer, mirrored from _layouts/post.html (keep the text in sync).
_DISCLAIMER = ('<aside class="disclaimer">我不是这个领域的专家，也谈不上真懂，'
               '这里写的不一定正确。我只是在用费曼学习法：一边读 paper 学习，'
               '一边把思路完整地展开讲清楚，借此搭建自己的直觉，'
               '并随着新的进展不断修正它。</aside>')


def disclaimer_html(meta):
    return "" if (meta or {}).get("disclaimer") is False else _DISCLAIMER


def blog_config():
    """(blog_title, blog_tagline) from _config.yml, so previews match the live site."""
    cfg = {}
    cfg_path = os.path.join(ROOT, "_config.yml")
    if os.path.exists(cfg_path):
        cfg = yaml.safe_load(open(cfg_path, encoding="utf-8")) or {}
    return cfg.get("blog_title", "Notes"), cfg.get("blog_tagline", "")


# ---- Citations: mirror the Jekyll {% include cite.html key="..." %} + References ----
CITE_RE = re.compile(
    r"""\{%\s*include\s+cite\.html\s+key=["']?([A-Za-z0-9_:.\-]+)["']?\s*%\}""")


def process_citations(text, references):
    """Replace {% cite key %} with a superscript [n] linking to the reference."""
    keys = [str(r.get("key", "")) for r in (references or [])]

    def repl(m):
        k = m.group(1)
        if k in keys:
            return f'<sup class="cite"><a href="#ref-{k}">{keys.index(k) + 1}</a></sup>'
        return '<sup class="cite" title="unknown citation">[?]</sup>'

    return CITE_RE.sub(repl, text)


def render_references(references):
    """Build the auto References section from a post's `references:` front matter."""
    if not references:
        return ""
    items = []
    for r in references:
        key = html.escape(str(r.get("key", "")))
        title = html.escape(str(r.get("title", "(untitled)")))
        url = r.get("url")
        link = (f'<a href="{html.escape(str(url))}" target="_blank" rel="noopener">{title}</a>'
                if url else title)
        extra = ""
        if r.get("authors"):
            extra += f' · {html.escape(str(r.get("authors")))}'
        if r.get("year"):
            extra += f' ({html.escape(str(r.get("year")))})'
        items.append(f'<li id="ref-{key}">{link}{extra}</li>')
    return ('<section class="references"><h2>References</h2><ol>'
            + "".join(items) + "</ol></section>")


# MathJax loader (CDN). arithmatex emits \(..\) / \[..\]; match those delimiters.
# Kept out of the .format templates (it contains { } ) — passed in as {mathjax}.
MATHJAX = r'''<script>
window.MathJax={tex:{inlineMath:[['\\(','\\)']],displayMath:[['\\[','\\]']]},options:{skipHtmlTags:['script','noscript','style','textarea','pre','code']},startup:{pageReady:function(){return MathJax.startup.defaultPageReady().then(function(){window.fitDisplayMath();setTimeout(window.fitDisplayMath,300);});}}};
window.fitDisplayMath=function(){document.querySelectorAll('mjx-container[display="true"]').forEach(function(c){c.style.fontSize='';var m=c.firstElementChild;if(!m)return;var pw=(c.parentNode&&c.parentNode.clientWidth)||9999;var avail=Math.min(c.clientWidth||9999,pw);if(avail>=9999)return;var nat=m.getBoundingClientRect().width;if(nat>avail+1){c.style.fontSize=Math.max(45,((avail-2)/nat)*96)+'%';}});};
var _fitT;window.addEventListener('resize',function(){clearTimeout(_fitT);_fitT=setTimeout(window.fitDisplayMath,120);});
window.addEventListener('load',function(){setTimeout(window.fitDisplayMath,200);});
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>'''

FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


def compile_css():
    """Compile blog.scss (minus its Jekyll front matter) to CSS."""
    scss_path = os.path.join(ROOT, "assets/css/blog.scss")
    raw = open(scss_path, encoding="utf-8").read()
    m = FRONT_MATTER_RE.match(raw)
    scss_src = m.group(2) if m else raw
    return sass.compile(string=scss_src, output_style="expanded")


def font_face_css():
    """Load font.css and rewrite /assets paths to file:// so fonts load locally."""
    path = os.path.join(ROOT, "assets/css/font.css")
    if not os.path.exists(path):
        return ""
    css = open(path, encoding="utf-8").read()
    css = css.replace("url(/assets", f"url(file://{ROOT}/assets")
    # font.css ends with a body{} rule meant for the old homepage — drop it.
    return re.sub(r"\nbody\s*\{.*?\}\s*$", "", css, flags=re.DOTALL)


def parse_doc(path):
    raw = open(path, encoding="utf-8").read()
    m = FRONT_MATTER_RE.match(raw)
    if m:
        meta = yaml.safe_load(m.group(1)) or {}
        body = m.group(2)
    else:
        meta, body = {}, raw
    return meta, body


def date_from_filename(name):
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})-", name)
    if m:
        return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def slugify(name):
    base = os.path.splitext(os.path.basename(name))[0]
    base = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", base)
    return base


PAGE_TMPL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · preview</title>
<style>{fonts}</style>
<style>{css}</style>
</head>
<body>
{banner}
<div class="page">
  <header class="masthead">
    <h1 class="masthead__title" style="font-size:26px"><a href="index.html">{blog_title}</a></h1>
  </header>
  <article class="article">
    <header class="article__header">
      <div class="article__meta">{meta}</div>
      <h1 class="article__title">{title}</h1>
      {lead}
    </header>
    {disclaimer}
    <div class="article__body">
{body}
    </div>
    {references}
    {tags}
    <footer class="article__footer">
      <a class="back-link" href="index.html">Back to all posts</a>
    </footer>
  </article>
</div>
{mathjax}
</body>
</html>
"""

INDEX_TMPL = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blog preview</title>
<style>{fonts}</style>
<style>{css}</style>
</head>
<body>
<div class="page">
  <header class="masthead">
    <p class="masthead__kicker">Local preview</p>
    <h1 class="masthead__title">{blog_title}</h1>
    <p class="masthead__tagline">{blog_tagline}</p>
  </header>
  <main>
    {draft_section}
    <h2 style="font-family:serif;margin:8px 0 4px">Published</h2>
    <ul class="post-list">
      {published}
    </ul>
  </main>
</div>
</body>
</html>
"""


def render_item(meta, body, css, fonts, blog_title, is_draft, date):
    refs = meta.get("references")
    body_html = render_markdown(process_citations(body, refs))
    references_html = render_references(refs)
    title = html.escape(str(meta.get("title", "(untitled)")))
    lead = meta.get("lead") or meta.get("description")
    lead_html = f'<p class="article__lead">{html.escape(str(lead))}</p>' if lead else ""
    meta_line = date.strftime("%B %-d, %Y") if date else "Draft — unpublished"
    tags = meta.get("tags") or []
    tags_html = ""
    if tags:
        chips = "".join(f'<span class="tag">{html.escape(str(t))}</span>' for t in tags)
        tags_html = f'<div class="article__tags">{chips}</div>'
    banner = ('<div class="draft-banner">Draft preview — this is git-ignored and not published</div>'
              if is_draft else "")
    return PAGE_TMPL.format(
        title=title, fonts=fonts, css=css, banner=banner, blog_title=html.escape(blog_title),
        meta=meta_line, lead=lead_html, body=body_html, tags=tags_html, mathjax=MATHJAX,
        references=references_html, disclaimer=disclaimer_html(meta),
    )


def collect_docs():
    """Return (drafts, posts) as lists of (slug, meta, body, date) tuples."""
    drafts, posts = [], []
    for path in sorted(glob.glob(os.path.join(ROOT, "_drafts", "*.md"))):
        meta, body = parse_doc(path)
        drafts.append((slugify(path), meta, body, None))
    for path in sorted(glob.glob(os.path.join(ROOT, "_posts", "*.md")), reverse=True):
        meta, body = parse_doc(path)
        date = meta.get("date") or date_from_filename(os.path.basename(path))
        if isinstance(date, datetime.datetime):
            date = date.date()
        posts.append((slugify(path), meta, body, date))
    return drafts, posts


def main():
    shots = "--shots" in sys.argv
    os.makedirs(OUT, exist_ok=True)

    config = {}
    cfg_path = os.path.join(ROOT, "_config.yml")
    if os.path.exists(cfg_path):
        config = yaml.safe_load(open(cfg_path, encoding="utf-8")) or {}
    blog_title = config.get("blog_title", "Notes")
    blog_tagline = config.get("blog_tagline", "")

    css = compile_css()
    fonts = font_face_css()

    drafts, published = [], []

    for path in sorted(glob.glob(os.path.join(ROOT, "_drafts", "*.md"))):
        meta, body = parse_doc(path)
        slug = "draft-" + slugify(path)
        out = os.path.join(OUT, slug + ".html")
        open(out, "w", encoding="utf-8").write(
            render_item(meta, body, css, fonts, blog_title, True, None))
        drafts.append((meta, slug + ".html", None))

    for path in sorted(glob.glob(os.path.join(ROOT, "_posts", "*.md")), reverse=True):
        meta, body = parse_doc(path)
        date = meta.get("date") or date_from_filename(os.path.basename(path))
        if isinstance(date, datetime.datetime):
            date = date.date()
        slug = slugify(path)
        out = os.path.join(OUT, slug + ".html")
        open(out, "w", encoding="utf-8").write(
            render_item(meta, body, css, fonts, blog_title, False, date))
        published.append((meta, slug + ".html", date))

    def li(meta, href, date, draft=False):
        title = html.escape(str(meta.get("title", "(untitled)")))
        when = date.strftime("%Y · %b %-d") if date else "draft"
        summary = meta.get("description") or meta.get("lead") or ""
        summary = html.escape(str(summary))
        return (f'<li class="post-list__item"><span class="post-list__date">{when}</span>'
                f'<h2 class="post-list__title"><a href="{href}">{title}</a></h2>'
                f'<p class="post-list__excerpt">{summary}</p></li>')

    draft_section = ""
    if drafts:
        items = "".join(li(m, h, d, True) for m, h, d in drafts)
        draft_section = (
            '<h2 style="font-family:serif;margin:8px 0 4px">Drafts '
            '<span style="font-size:13px;color:var(--text-faint)">(only you can see these)</span></h2>'
            f'<ul class="post-list">{items}</ul>')

    index_html = INDEX_TMPL.format(
        fonts=fonts, css=css, blog_title=html.escape(blog_title),
        blog_tagline=html.escape(blog_tagline),
        draft_section=draft_section,
        published="".join(li(m, h, d) for m, h, d in published) or
                  '<li class="empty-note">Nothing published yet.</li>',
    )
    open(os.path.join(OUT, "index.html"), "w", encoding="utf-8").write(index_html)

    print(f"Preview written to {OUT}/index.html")
    print(f"  {len(drafts)} draft(s), {len(published)} published post(s)")

    if shots:
        take_shots()


def take_shots():
    from playwright.sync_api import sync_playwright
    files = sorted(glob.glob(os.path.join(OUT, "*.html")))
    with sync_playwright() as p:
        for scheme in ("light", "dark"):
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 820, "height": 1000},
                                    color_scheme=scheme, device_scale_factor=2)
            for f in files:
                page.goto("file://" + f)
                png = f.replace(".html", f".{scheme}.png")
                page.screenshot(path=png, full_page=True)
            browser.close()
    print("Screenshots saved next to each .html")


if __name__ == "__main__":
    main()
