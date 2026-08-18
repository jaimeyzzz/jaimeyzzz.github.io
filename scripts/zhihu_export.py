#!/usr/bin/env python3

import argparse
import base64
import html
import mimetypes
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import urljoin

import yaml
from bs4 import BeautifulSoup

import wechat_sync


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SITE_URL = "https://www.jaimeyzzz.com/"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a Zhihu-friendly rich-text copy page.")
    parser.add_argument("--post", type=Path, help="Post or draft Markdown file; defaults to latest post.")
    parser.add_argument("--site-url", default=DEFAULT_SITE_URL, help="Public blog origin used for images.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".preview/zhihu")
    return parser.parse_args()


def source_file(path):
    if path:
        resolved = path.resolve()
        if not resolved.is_file():
            raise RuntimeError(f"Article not found: {resolved}")
        return resolved
    return wechat_sync.latest_post().resolve()


def front_matter(path):
    text = path.read_text(encoding="utf-8")
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not match:
        raise RuntimeError(f"Missing YAML front matter: {path}")
    return yaml.safe_load(match.group(1)) or {}


def built_page(path):
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})-(.+)\.md$", path.name)
    if not match:
        raise RuntimeError("Zhihu export currently expects a dated file under _posts/.")
    year, month, day, slug = match.groups()
    return ROOT / "_site" / year / month / day / f"{slug}.html", slug


def data_uri(path):
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def clean_content(page_path, source_url, site_url, asset_dir):
    soup = BeautifulSoup(page_path.read_text(encoding="utf-8"), "lxml")
    article_body = soup.select_one(".article__body")
    if article_body is None:
        raise RuntimeError("Generated page has no .article__body element.")

    content = BeautifulSoup("<article></article>", "lxml").article
    disclaimer = soup.select_one(".disclaimer")
    if disclaimer:
        content.append(BeautifulSoup(str(disclaimer), "lxml").body.contents[0])
    for child in list(article_body.contents):
        content.append(child)
    references = soup.select_one(".references")
    if references:
        content.append(BeautifulSoup(str(references), "lxml").body.contents[0])

    for citation in content.select("sup.cite"):
        number = citation.get_text(strip=True)
        replacement = soup.new_tag("sup")
        replacement.string = f"[{number}]"
        citation.replace_with(replacement)

    reference_heading = content.select_one(".references h2")
    if reference_heading:
        reference_heading.string = "参考文献"

    formula_count = 0
    for formula in content.select(".kdmath"):
        formula_count += 1
        formula_path = asset_dir / f"formula-{formula_count}.png"
        wechat_sync.render_formula(formula.get_text(), formula_path)
        image = soup.new_tag("img")
        image["src"] = data_uri(formula_path)
        image["alt"] = formula.get_text(" ", strip=True)
        image["data-embedded"] = "formula"
        formula.replace_with(image)

    for image in content.select("img[src]"):
        source = image.get("src", "")
        if source.startswith("/assets/"):
            image["src"] = urljoin(site_url, source.lstrip("/"))
        elif source and not source.startswith(("http://", "https://", "data:")):
            image["src"] = urljoin(source_url, source)

    for anchor in content.select("a[href]"):
        href = anchor.get("href", "")
        if href.startswith("#"):
            anchor["href"] = f"{source_url}{href}"
        elif href.startswith("/"):
            anchor["href"] = urljoin(site_url, href.lstrip("/"))
        anchor.attrs.pop("target", None)
        anchor.attrs.pop("rel", None)

    styles = {
        "article": "font-size:16px;line-height:1.8;color:#242424;",
        "h2": "margin:30px 0 14px;font-size:22px;line-height:1.45;font-weight:600;",
        "h3": "margin:24px 0 10px;font-size:19px;line-height:1.45;font-weight:600;",
        "p": "margin:12px 0;line-height:1.8;",
        "aside": "margin:16px 0;padding:12px 14px;background:#f6f6f6;color:#646464;border-left:3px solid #8590a6;",
        "blockquote": "margin:16px 0;padding:8px 14px;background:#f6f6f6;border-left:3px solid #8590a6;",
        "ul": "margin:12px 0;padding-left:1.5em;list-style-type:disc;",
        "ol": "margin:12px 0;padding-left:1.5em;list-style-type:decimal;",
        "li": "margin:6px 0;line-height:1.75;",
        "table": "width:100%;border-collapse:collapse;margin:16px 0;font-size:14px;",
        "th": "padding:8px;border:1px solid #d3d3d3;background:#f6f6f6;",
        "td": "padding:8px;border:1px solid #d3d3d3;",
        "pre": "overflow-x:auto;margin:14px 0;padding:12px;background:#f6f6f6;font-size:13px;",
        "code": "font-family:monospace;background:#f6f6f6;padding:1px 3px;",
        "a": "color:#175199;text-decoration:none;",
        "img": "display:block;max-width:100%;height:auto;margin:20px auto;",
        "sup": "font-size:11px;vertical-align:super;",
    }
    for tag_name, style in styles.items():
        for tag in content.select(tag_name):
            tag["style"] = style

    for tag in content.find_all(True):
        allowed = {"href", "src", "alt", "style", "data-embedded"}
        tag.attrs = {key: value for key, value in tag.attrs.items() if key in allowed}
    return content


def export_page(title, content, source_url):
    escaped_title = html.escape(title)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escaped_title} · 知乎导出</title>
<style>
body{{margin:0;background:#f5f5f5;color:#242424;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif}}
.toolbar{{position:sticky;top:0;z-index:2;padding:14px;background:#fff;border-bottom:1px solid #e8e8e8}}
.toolbar__inner,.sheet{{max-width:760px;margin:auto}}button{{margin:4px 8px 4px 0;padding:9px 15px;border:0;border-radius:6px;background:#1772f6;color:#fff;cursor:pointer;font-size:14px}}button.secondary{{background:#8590a6}}
.status{{display:inline-block;color:#646464;font-size:13px}}.note{{margin:12px 0 0;color:#646464;font-size:13px;line-height:1.6}}
.sheet{{margin-top:24px;margin-bottom:40px;padding:38px 46px;background:#fff;box-sizing:border-box}}.title{{font-size:30px;line-height:1.35;margin:0 0 30px}}@media(max-width:700px){{.sheet{{margin-top:0;padding:24px 18px}}}}
</style>
</head>
<body>
<div class="toolbar"><div class="toolbar__inner">
  <button onclick="copyTitle()">复制标题</button>
  <button onclick="copyBody()">复制正文（含图片）</button>
  <button class="secondary" onclick="selectBody()">仅选中正文</button>
  <span id="status" class="status">建议粘贴后逐张确认图片。</span>
  <p class="note">操作顺序：复制标题到知乎标题栏，再复制正文到编辑器。普通图片使用已发布博客的公网地址；公式图已内嵌。若知乎未自动转存某张图，可从原文下载后手动插入。原文：<a href="{html.escape(source_url)}">{html.escape(source_url)}</a></p>
</div></div>
<main class="sheet"><h1 class="title">{escaped_title}</h1><div id="zhihu-content">{content}</div></main>
<script>
const statusNode=document.getElementById('status');
function setStatus(text){{statusNode.textContent=text}}
async function writeClipboard(htmlText,plainText){{
  if(navigator.clipboard&&window.ClipboardItem){{
    const item=new ClipboardItem({{'text/html':new Blob([htmlText],{{type:'text/html'}}),'text/plain':new Blob([plainText],{{type:'text/plain'}})}});
    await navigator.clipboard.write([item]);return;
  }}
  throw new Error('rich clipboard unavailable');
}}
function selectBody(){{
  const selection=window.getSelection(),range=document.createRange();
  range.selectNodeContents(document.getElementById('zhihu-content'));selection.removeAllRanges();selection.addRange(range);setStatus('正文已选中，请按 Ctrl/Cmd+C。');
}}
async function copyTitle(){{
  try{{await navigator.clipboard.writeText({title!r});setStatus('标题已复制。')}}catch(error){{setStatus('标题复制失败，请手动选择标题。')}}
}}
async function copyBody(){{
  const body=document.getElementById('zhihu-content');
  try{{await writeClipboard(body.innerHTML,body.innerText);setStatus('正文已复制，请粘贴到知乎并检查图片。')}}
  catch(error){{selectBody();try{{document.execCommand('copy');setStatus('正文已复制，请粘贴到知乎并检查图片。')}}catch(copyError){{setStatus('浏览器禁止自动复制，请按 Ctrl/Cmd+C。')}}}}
}}
</script>
</body></html>"""


def main():
    args = parse_args()
    post = source_file(args.post)
    metadata = front_matter(post)
    page_path, slug = built_page(post)
    wechat_sync.run_jekyll()

    relative_page = page_path.relative_to(ROOT / "_site").as_posix()
    source_url = urljoin(args.site_url, relative_page)
    output_dir = args.output_dir.resolve()
    asset_dir = output_dir / f"{slug}-assets"
    shutil.rmtree(asset_dir, ignore_errors=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    content = clean_content(page_path, source_url, args.site_url, asset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{slug}.html"
    output.write_text(export_page(str(metadata.get("title", slug)), content, source_url), encoding="utf-8")
    print(f"Zhihu copy page written: {output}")
    print(f"Private preview URL: /blog-preview/zhihu/{output.name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
