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

from bs4 import BeautifulSoup

import wechat_sync


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SITE_URL = "https://www.jaimeyzzz.com/"


def parse_args():
    parser = argparse.ArgumentParser(description="Generate a Zhihu rich-text copy page.")
    parser.add_argument("--post", type=Path, help="Published Markdown file; defaults to latest post.")
    parser.add_argument("--site-url", default=DEFAULT_SITE_URL, help="Public blog origin used for images.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".preview/zhihu")
    return parser.parse_args()


def source_file(path):
    source = path.resolve() if path else wechat_sync.latest_post().resolve()
    if not source.is_file():
        raise RuntimeError(f"Article not found: {source}")
    return source


def built_page(path):
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})-(.+)\.md$", path.name)
    if not match:
        raise RuntimeError("Zhihu export expects a dated article under _posts/.")
    year, month, day, slug = match.groups()
    return ROOT / "_site" / year / month / day / f"{slug}.html", slug


def data_uri(path):
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def prepare_content(page_path, source_url, site_url, asset_dir):
    content, _ = wechat_sync.extract_article(page_path, source_url, asset_dir)
    counts = {"formula": 0, "figure": 0, "image": 0}

    for image in content.select("img"):
        local = image.attrs.pop("data-local-image", None)
        if local:
            path = Path(local)
            image["src"] = data_uri(path)
            image["data-embedded"] = "true"
            if path.name.startswith(("formula-", "inline-formula-")):
                counts["formula"] += 1
            elif path.name.startswith("figure-"):
                counts["figure"] += 1
            continue

        source = image.get("src", "")
        if source.startswith("/"):
            image["src"] = urljoin(site_url, source.lstrip("/"))
        elif source and not source.startswith(("http://", "https://", "data:")):
            image["src"] = urljoin(source_url, source)
        if image.get("src"):
            counts["image"] += 1

    # Keep the clipboard payload small and limited to attributes Zhihu can use.
    for tag in content.find_all(True):
        allowed = {"href", "src", "alt", "style", "width", "height", "data-embedded"}
        tag.attrs = {key: value for key, value in tag.attrs.items() if key in allowed}
    return content, counts


def export_page(title, content, source_url, counts):
    escaped_title = html.escape(title)
    count_text = (
        f"已准备 {counts['image']} 张正文图片、{counts['formula']} 个公式图片"
        f"和 {counts['figure']} 张矢量配图。"
    )
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
.status{{display:inline-block;color:#646464;font-size:13px}}.note{{margin:10px 0 0;color:#646464;font-size:13px;line-height:1.65}}
.sheet{{margin-top:24px;margin-bottom:40px;padding:38px 46px;background:#fff;box-sizing:border-box}}.title{{font-size:30px;line-height:1.35;margin:0 0 30px}}@media(max-width:700px){{.sheet{{margin-top:0;padding:24px 18px}}}}
</style>
</head>
<body>
<div class="toolbar"><div class="toolbar__inner">
  <button onclick="copyTitle()">复制标题</button>
  <button onclick="copyBody()">复制正文（含图片与公式）</button>
  <button class="secondary" onclick="selectBody()">兼容模式：选中正文</button>
  <span id="status" class="status">{count_text}</span>
  <p class="note">先复制标题，再复制正文并粘贴到知乎编辑器。粘贴后请等待知乎完成图片上传，再保存草稿；若浏览器阻止一键复制，使用兼容模式后按 Ctrl/Cmd+C。原文：<a href="{html.escape(source_url)}">{html.escape(source_url)}</a></p>
</div></div>
<main class="sheet"><h1 class="title">{escaped_title}</h1><div id="zhihu-content">{content}</div></main>
<script>
const statusNode=document.getElementById('status');
function setStatus(text){{statusNode.textContent=text}}
function selectBody(){{
  const selection=window.getSelection(),range=document.createRange();
  range.selectNodeContents(document.getElementById('zhihu-content'));
  selection.removeAllRanges();selection.addRange(range);
  setStatus('正文已选中，请按 Ctrl/Cmd+C，然后粘贴到知乎。');
}}
async function copyTitle(){{
  try{{await navigator.clipboard.writeText({title!r});setStatus('标题已复制。')}}
  catch(error){{setStatus('标题复制失败，请手动选择标题。')}}
}}
function copyBody(){{
  const body=document.getElementById('zhihu-content');
  const images=[...body.querySelectorAll('img')];
  const clone=body.cloneNode(true);
  [...clone.querySelectorAll('img')].forEach((image,index)=>image.src=images[index].src);
  const onCopy=event=>{{
    event.clipboardData.setData('text/html',clone.innerHTML);
    event.clipboardData.setData('text/plain',body.innerText);
    event.preventDefault();
  }};
  document.addEventListener('copy',onCopy,{{once:true}});
  try{{
    if(!document.execCommand('copy'))throw new Error('Copy command unavailable');
    setStatus('正文已复制。粘贴后请等待知乎完成图片上传。');
  }}catch(error){{
    document.removeEventListener('copy',onCopy);
    selectBody();
    setStatus('浏览器禁止自动复制，请保持选中并按 Ctrl/Cmd+C。');
  }}
}}
</script>
</body></html>"""


def main():
    args = parse_args()
    post = source_file(args.post)
    metadata = wechat_sync.read_front_matter(post)
    page_path, slug = built_page(post)
    wechat_sync.run_jekyll()

    source_url = urljoin(args.site_url, page_path.relative_to(ROOT / "_site").as_posix())
    output_dir = args.output_dir.resolve()
    asset_dir = output_dir / f"{slug}-assets"
    shutil.rmtree(asset_dir, ignore_errors=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    content, counts = prepare_content(page_path, source_url, args.site_url, asset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{slug}.html"
    output.write_text(export_page(str(metadata.get("title", slug)), content, source_url, counts), encoding="utf-8")
    print(f"Zhihu copy page written: {output}")
    print(f"Images: {counts['image']}; formulas: {counts['formula']}; figures: {counts['figure']}")
    print(f"Private preview URL: /blog-preview/zhihu/{output.name}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
