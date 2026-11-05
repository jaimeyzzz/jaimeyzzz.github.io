#!/usr/bin/env python3

import argparse
import html
import re
import sys
from pathlib import Path
from urllib.parse import quote, urljoin

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


def formula_source(value):
    formula = value.strip()
    formula = re.sub(r"^\$\$|\$\$$", "", formula).strip()
    return formula


def equation_image(soup, formula, block=False):
    image = soup.new_tag("img")
    image["eeimg"] = "2" if block else "1"
    image["class"] = ["ee_img", "tr_noresize"]
    image["src"] = f"https://www.zhihu.com/equation?tex={quote(formula, safe='')}"
    image["alt"] = formula.replace("\n", " ")
    if block:
        image["style"] = "display:block;max-width:100%;height:auto;margin:18px auto;"
    else:
        image["style"] = "display:inline;max-width:100%;height:auto;margin:0 0.08em;vertical-align:middle;"
    return image


def prepare_content(page_path, source_url, site_url, public_asset_dir, public_asset_url):
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

    counts = {"formula": 0, "figure": 0, "image": 0}

    # Zhihu recognizes semantic blockquotes, but not the blog's custom aside.
    for aside in list(content.find_all("aside")):
        blockquote = soup.new_tag("blockquote")
        if aside.find(["p", "ul", "ol"], recursive=False):
            for child in list(aside.contents):
                blockquote.append(child.extract())
        else:
            paragraph = soup.new_tag("p")
            for child in list(aside.contents):
                paragraph.append(child.extract())
            blockquote.append(paragraph)
        aside.replace_with(blockquote)

    for blockquote in content.find_all("blockquote"):
        if not blockquote.find(["p", "ul", "ol"], recursive=False):
            paragraph = soup.new_tag("p")
            for child in list(blockquote.contents):
                paragraph.append(child.extract())
            blockquote.append(paragraph)

    for citation in content.select("sup.cite"):
        number = citation.get_text(strip=True)
        replacement = soup.new_tag("sup")
        replacement.string = f"[{number}]"
        citation.replace_with(replacement)

    reference_heading = content.select_one(".references h2")
    if reference_heading:
        reference_heading.string = "参考文献"

    for formula in list(content.select(".kdmath")):
        source = formula_source(formula.get_text())
        paragraph = soup.new_tag("p")
        paragraph.append(equation_image(soup, source, block=True))
        formula.replace_with(paragraph)
        counts["formula"] += 1

    inline_pattern = re.compile(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$")
    text_nodes = list(content.find_all(string=lambda value: "$" in value))
    for text_node in text_nodes:
        if text_node.parent and text_node.parent.name in {"code", "pre", "script", "style"}:
            continue
        matches = list(inline_pattern.finditer(str(text_node)))
        if not matches:
            continue
        replacements = []
        cursor = 0
        for match in matches:
            if match.start() > cursor:
                replacements.append(str(text_node)[cursor:match.start()])
            replacements.append(equation_image(soup, match.group(1).strip(), block=False))
            counts["formula"] += 1
            cursor = match.end()
        if cursor < len(str(text_node)):
            replacements.append(str(text_node)[cursor:])
        for replacement in reversed(replacements):
            text_node.insert_after(replacement)
        text_node.extract()

    svg_colors = {
        "var(--text)": "#2b2926",
        "var(--text-faint)": "#9a9289",
        "var(--text-mute)": "#6b6258",
        "var(--bg)": "#ffffff",
        "var(--bg-tint)": "#f7f2ea",
        "var(--border)": "#d8d2c8",
        "var(--border-strong)": "#b8aea2",
    }
    public_asset_dir.mkdir(parents=True, exist_ok=True)
    for index, svg in enumerate(list(content.find_all("svg")), start=1):
        if "viewbox" in svg.attrs:
            svg["viewBox"] = svg.attrs.pop("viewbox")
        svg_source = str(svg)
        for variable, color in svg_colors.items():
            svg_source = svg_source.replace(variable, color)
        filename = f"figure-{index}.png"
        output = public_asset_dir / filename
        wechat_sync.cairosvg.svg2png(
            bytestring=svg_source.encode("utf-8"),
            write_to=str(output),
            output_width=1560,
            background_color="#ffffff",
        )
        image = soup.new_tag("img")
        image["src"] = urljoin(public_asset_url, filename)
        image["alt"] = svg.get("aria-label", "article figure")
        image["style"] = "display:block;max-width:100%;height:auto;margin:20px auto;"
        image["data-generated-figure"] = "true"
        svg.replace_with(image)
        counts["figure"] += 1

    # Figure/figcaption are not stable paste targets in Zhihu. Keep the image
    # and turn the caption into an ordinary emphasized paragraph.
    for caption in list(content.find_all("figcaption")):
        paragraph = soup.new_tag("p")
        emphasis = soup.new_tag("em")
        for child in list(caption.contents):
            emphasis.append(child.extract())
        paragraph.append(emphasis)
        caption.replace_with(paragraph)
    for figure in list(content.find_all("figure")):
        figure.unwrap()

    for image in content.select("img"):
        if image.get("eeimg"):
            continue
        if image.attrs.pop("data-generated-figure", None):
            continue
        source = image.get("src", "")
        if source.startswith("/"):
            image["src"] = urljoin(site_url, source.lstrip("/"))
        elif source and not source.startswith(("http://", "https://", "data:")):
            image["src"] = urljoin(source_url, source)
        if image.get("src"):
            counts["image"] += 1

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
        "figure": "margin:20px 0;",
        "figcaption": "margin:8px 0 0;text-align:center;font-size:13px;line-height:1.55;color:#646464;",
        "sup": "font-size:11px;vertical-align:super;",
    }
    for tag_name, style in styles.items():
        for tag in content.select(tag_name):
            if tag_name != "img" or not tag.get("eeimg"):
                tag["style"] = style

    for tag in content.find_all(True):
        allowed = {"href", "src", "alt", "style", "class", "eeimg"}
        tag.attrs = {key: value for key, value in tag.attrs.items() if key in allowed}
    return content, counts


def export_page(title, content, source_url, counts):
    escaped_title = html.escape(title)
    count_text = (
        f"已准备 {counts['formula']} 个知乎原生公式、{counts['image']} 张正文图片"
        f"和 {counts['figure']} 张公网配图。"
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
  <p class="note">先复制标题，再复制正文并粘贴到知乎编辑器。公式会按知乎原生格式写入；图片使用已发布博客的 HTTPS 地址。粘贴后请等待知乎完成图片转存，再保存草稿。若浏览器阻止一键复制，使用兼容模式后按 Ctrl/Cmd+C。原文：<a href="{html.escape(source_url)}">{html.escape(source_url)}</a></p>
</div></div>
<main class="sheet"><h1 class="title">{escaped_title}</h1><div id="zhihu-content">{content}</div></main>
<script>
const statusNode=document.getElementById('status');
function setStatus(text){{statusNode.textContent=text}}
function articleBody(){{return document.querySelector('#zhihu-content > article')||document.getElementById('zhihu-content')}}
function selectBody(){{
  const selection=window.getSelection(),range=document.createRange();
  range.selectNodeContents(articleBody());
  selection.removeAllRanges();selection.addRange(range);
  setStatus('正文已选中，请按 Ctrl/Cmd+C，然后粘贴到知乎。');
}}
async function copyTitle(){{
  try{{await navigator.clipboard.writeText({title!r});setStatus('标题已复制。')}}
  catch(error){{setStatus('标题复制失败，请手动选择标题。')}}
}}
function copyBody(){{
  const body=articleBody();
  const images=[...body.querySelectorAll('img')];
  const clone=body.cloneNode(true);
  [...clone.querySelectorAll('img')].forEach((image,index)=>image.src=images[index].src);
  [...clone.querySelectorAll('[style]')].forEach(element=>element.removeAttribute('style'));
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
    public_asset_dir = ROOT / "assets" / "zhihu" / slug
    public_asset_url = urljoin(args.site_url, f"assets/zhihu/{slug}/")

    content, counts = prepare_content(
        page_path, source_url, args.site_url, public_asset_dir, public_asset_url
    )
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
