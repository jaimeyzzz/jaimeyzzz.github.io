#!/usr/bin/env python3
"""
server.py — a live local preview server for the blog.

Renders _posts/ and _drafts/ on every request (so edits show on refresh),
serves the real fonts + CSS over HTTP, and needs NO Jekyll and NO push.

    python3 tools/preview/server.py         # serve on 127.0.0.1:8811 (loopback only)
    python3 tools/preview/server.py 9000    # pick a different port

Bound to 127.0.0.1 — never public. Reached in the browser through the
auth-gated nginx location  https://ec2.jaimeyzzz.com/blog-preview/  (see
tools/preview/README.md). Nothing is published until you move a draft into
_posts/ and push.
"""
import os
import re
import sys
import glob
import html
import datetime
import mimetypes
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import markdown
from render import (
    ROOT, compile_css, parse_doc, date_from_filename, slugify,
    render_markdown, MATHJAX, blog_config,
    process_citations, render_references, disclaimer_html,
)

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8811

SHELL = """<!DOCTYPE html><html lang="zh"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<base href="{base}">
<title>{title}</title>
<link rel="stylesheet" href="assets/css/font.css">
<style>{css}</style>
<style>body::after{{content:"LOCAL PREVIEW";position:fixed;bottom:10px;right:12px;
font-family:sans-serif;font-size:10px;letter-spacing:.15em;color:var(--text-faint);
border:1px solid var(--border);border-radius:999px;padding:3px 9px;opacity:.7}}</style>
</head><body>{banner}<div class="page">{body}</div><script defer src="/assets/js/blog-toc.js"></script>{mathjax}</body></html>"""


def collect():
    drafts, posts = [], []
    for p in sorted(glob.glob(os.path.join(ROOT, "_drafts", "*.md"))):
        meta, body = parse_doc(p)
        drafts.append((slugify(p), meta, body, None))
    for p in sorted(glob.glob(os.path.join(ROOT, "_posts", "*.md")), reverse=True):
        meta, body = parse_doc(p)
        d = meta.get("date") or date_from_filename(os.path.basename(p))
        if isinstance(d, datetime.datetime):
            d = d.date()
        posts.append((slugify(p), meta, body, d))
    return drafts, posts


def render_index(css, base="/"):
    drafts, posts = collect()

    def row(slug, meta, date, draft):
        title = html.escape(str(meta.get("title", "(untitled)")))
        when = date.strftime("%Y · %b %-d") if date else "draft"
        summ = html.escape(str(meta.get("description") or meta.get("lead") or ""))
        zhihu = ('' if draft else
                 f'<a href="zhihu/{slug}" style="display:inline-block;margin-top:8px;font-size:13px">知乎复制页 →</a>')
        return (f'<li class="post-list__item"><span class="post-list__date">{when}</span>'
                f'<h2 class="post-list__title"><a href="p/{slug}">{title}</a></h2>'
                f'<p class="post-list__excerpt">{summ}</p>{zhihu}</li>')

    b_title, b_tagline = blog_config()
    parts = ['<header class="masthead"><p class="masthead__kicker">Local preview</p>'
             f'<h1 class="masthead__title">{html.escape(b_title)}</h1>'
             f'<p class="masthead__tagline">{html.escape(b_tagline)}</p></header>']
    if drafts:
        parts.append('<h2 style="font-family:serif;margin:8px 0 4px">Drafts '
                     '<span style="font-size:13px;color:var(--text-faint)">(只有你能看到)</span></h2>'
                     '<ul class="post-list">'
                     + "".join(row(s, m, None, True) for s, m, b, d in drafts) + "</ul>")
    parts.append('<h2 style="font-family:serif;margin:24px 0 4px">Published</h2><ul class="post-list">'
                 + ("".join(row(s, m, d, False) for s, m, b, d in posts)
                    or '<li class="empty-note">还没有已发布的文章。</li>') + "</ul>")
    return SHELL.format(base=base, title="Notes · preview", css=css, banner="", body="".join(parts), mathjax=MATHJAX)


def render_post(css, slug, base="/"):
    drafts, posts = collect()
    for s, meta, body, date in drafts:
        if s == slug:
            return _article(css, meta, body, None, True, base, s)
    for s, meta, body, date in posts:
        if s == slug:
            return _article(css, meta, body, date, False, base, s)
    return None


def published_source(slug):
    for path in glob.glob(os.path.join(ROOT, "_posts", "*.md")):
        if slugify(path) == slug:
            return path
    return None


def generate_zhihu_page(slug):
    source = published_source(slug)
    if not source:
        return None
    subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "zhihu_export.py"), "--post", source],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    output = os.path.join(ROOT, ".preview", "zhihu", f"{slug}.html")
    return output if os.path.isfile(output) else None


def _article(css, meta, body, date, is_draft, base="/", slug=""):
    refs = meta.get("references")
    body_html = render_markdown(process_citations(body, refs))
    body_html = re.sub(r'(?P<attr>src|href)="/assets/', r'\g<attr>="assets/', body_html)
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
    banner = ('<div class="draft-banner">Draft preview — 本地草稿，未发布</div>'
              if is_draft else "")
    zhihu = ('' if is_draft else
             '<p style="margin:12px 0 20px"><a href="zhihu/'
             + slug
             + '" id="zhihu-copy-link">知乎富文本复制页 →</a></p>')
    body = (f'<header class="masthead"><h1 class="masthead__title" style="font-size:26px">'
            f'<a href="./">Notes</a></h1></header>'
            f'<article class="article"><header class="article__header">'
            f'<div class="article__meta">{meta_line}</div>'
            f'<h1 class="article__title">{title}</h1>{lead_html}{zhihu}</header>'
            f'{disclaimer_html(meta)}'
            f'<details class="article-toc" id="article-toc" hidden><summary><span>文章目录</span>'
            f'<span class="article-toc__count"></span></summary><nav aria-label="文章目录">'
            f'<ol class="article-toc__list"></ol></nav></details>'
            f'<div class="article__body">{body_html}</div>{references_html}{tags_html}'
            f'<footer class="article__footer"><a class="back-link" href="./">Back to all posts</a></footer>'
            f'</article><button class="back-to-toc" id="back-to-toc" type="button" hidden '
            f'aria-label="返回文章目录"><span aria-hidden="true">↑</span><span>目录</span></button>')
    return SHELL.format(base=base, title=title + " · preview", css=css, banner=banner, body=body, mathjax=MATHJAX)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="text/html; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        # Honor nginx's X-Forwarded-Prefix so links work under /blog-preview/.
        prefix = self.headers.get("X-Forwarded-Prefix", "").rstrip("/")
        base = (prefix + "/") if prefix else "/"
        css = compile_css()
        if path == "/":
            return self._send(200, render_index(css, base))
        m = re.match(r"^/p/([^/]+)$", path)
        if m:
            page = render_post(css, m.group(1), base)
            return self._send(200, page) if page else self._send(404, "Not found")
        m = re.match(r"^/zhihu/([^/.]+)$", path)
        if m:
            try:
                fp = generate_zhihu_page(m.group(1))
            except subprocess.CalledProcessError:
                return self._send(500, "Zhihu export failed")
            if fp:
                with open(fp, encoding="utf-8") as f:
                    return self._send(200, f.read())
            return self._send(404, "Post not found")
        if path.startswith("/assets/"):
            fp = os.path.normpath(os.path.join(ROOT, path.lstrip("/")))
            if fp.startswith(ROOT) and os.path.isfile(fp):
                ctype = mimetypes.guess_type(fp)[0] or "application/octet-stream"
                with open(fp, "rb") as f:
                    return self._send(200, f.read(), ctype)
        if path.startswith("/zhihu/"):
            relative = path.removeprefix("/zhihu/")
            root = os.path.join(ROOT, ".preview", "zhihu")
            fp = os.path.normpath(os.path.join(root, relative))
            if fp.startswith(root) and os.path.isfile(fp):
                ctype = mimetypes.guess_type(fp)[0] or "application/octet-stream"
                with open(fp, "rb") as f:
                    return self._send(200, f.read(), ctype)
        return self._send(404, "Not found")


if __name__ == "__main__":
    # Bind to loopback only — never public. Reachable solely via the
    # auth-gated nginx /blog-preview/ location, or an SSH tunnel.
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Blog preview live at http://127.0.0.1:{PORT}/  (private; Ctrl-C to stop)")
    srv.serve_forever()
