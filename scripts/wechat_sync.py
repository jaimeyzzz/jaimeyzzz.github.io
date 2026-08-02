#!/usr/bin/env python3

import argparse
import html
import json
import mimetypes
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import cairosvg
import yaml
from bs4 import BeautifulSoup
from PIL import Image, ImageChops, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
API_ROOT = "https://api.weixin.qq.com"


def parse_args():
    parser = argparse.ArgumentParser(description="Convert a Jekyll post to a WeChat draft.")
    parser.add_argument("--post", type=Path, help="Source file under _posts; defaults to latest post.")
    parser.add_argument("--secrets", type=Path, default=ROOT / ".secrets/wechat.env")
    parser.add_argument("--cover", type=Path, help="Optional custom cover; defaults to a generated title cover.")
    parser.add_argument("--preview", type=Path, default=ROOT / ".preview/wechat-draft.html")
    parser.add_argument("--submit", action="store_true", help="Create a draft through the WeChat API.")
    parser.add_argument("--update-media-id", help="Update an existing WeChat draft instead of creating one.")
    return parser.parse_args()


def latest_post():
    posts = sorted((ROOT / "_posts").glob("*.md"))
    if not posts:
        raise RuntimeError("No Markdown posts found under _posts.")
    return posts[-1]


def read_front_matter(path):
    text = path.read_text(encoding="utf-8")
    match = re.match(r"\A---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    if not match:
        raise RuntimeError(f"Missing YAML front matter: {path}")
    return yaml.safe_load(match.group(1)) or {}


def parse_credentials(path):
    values = {}
    raw_values = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = re.match(r"([^:=\s]+)\s*[:=\s]\s*(\S+)$", line)
        if match:
            key = re.sub(r"[^a-z]", "", match.group(1).lower())
            values[key] = match.group(2).strip()
            raw_values.append(match.group(2).strip())
        else:
            raw_values.append(line)

    app_id = next((values[key] for key in values if key in {"appid", "wechatappid"}), None)
    app_secret = next(
        (values[key] for key in values if key in {"secret", "appsecret", "wechatappsecret"}),
        None,
    )
    if not app_id and raw_values:
        app_id = raw_values[0]
    if not app_secret and len(raw_values) > 1:
        app_secret = raw_values[1]
    if not app_id or not app_secret:
        raise RuntimeError("Secret file must contain AppID and AppSecret on separate lines.")
    return app_id, app_secret


def run_jekyll():
    environment = {"JEKYLL_NO_BUNDLER_REQUIRE": "true"}
    subprocess.run(
        ["jekyll", "build"],
        cwd=ROOT,
        env={**__import__("os").environ, **environment},
        check=True,
        stdout=subprocess.DEVNULL,
    )


def generated_path(post):
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})-(.+)\.md$", post.name)
    if not match:
        raise RuntimeError(f"Post filename must start with YYYY-MM-DD: {post.name}")
    year, month, day, slug = match.groups()
    return ROOT / "_site" / year / month / day / f"{slug}.html"


def trim_image(path):
    image = Image.open(path).convert("RGB")
    background = Image.new("RGB", image.size, "white")
    difference = ImageChops.difference(image, background)
    box = difference.getbbox()
    if box:
        image = image.crop(box)
    padding = 24
    canvas = Image.new("RGB", (image.width + padding * 2, image.height + padding * 2), "white")
    canvas.paste(image, (padding, padding))
    canvas.save(path, optimize=True)


def mixed_fonts(cjk_path, latin_path, size):
    return ImageFont.truetype(str(cjk_path), size), ImageFont.truetype(str(latin_path), size)


def character_font(character, fonts):
    return fonts[1] if ord(character) < 0x2E80 else fonts[0]


def mixed_text_width(draw, text, fonts):
    return sum(draw.textlength(character, font=character_font(character, fonts)) for character in text)


def fitting_mixed_text(draw, text, cjk_path, latin_path, max_size, min_size, max_width, max_lines):
    tokens = re.findall(r"[A-Za-z0-9]+|\s+|.", text)
    for size in range(max_size, min_size - 1, -2):
        fonts = mixed_fonts(cjk_path, latin_path, size)
        lines = []
        current = ""
        for token in tokens:
            candidate = current + token
            width = mixed_text_width(draw, candidate, fonts)
            if current and width > max_width:
                lines.append(current.rstrip())
                current = token.lstrip()
            else:
                current = candidate
        if current:
            lines.append(current)
        if len(lines) <= max_lines:
            return size, fonts, lines
    return min_size, mixed_fonts(cjk_path, latin_path, min_size), lines[:max_lines]


def draw_mixed_text(draw, position, text, fonts, fill):
    x, y = position
    for character in text:
        font = character_font(character, fonts)
        draw.text((x, y), character, fill=fill, font=font)
        x += draw.textlength(character, font=font)


def generate_cover(title, output):
    width, height = 1200, 630
    image = Image.new("RGB", (width, height), "#f7f2ea")
    draw = ImageDraw.Draw(image)
    cjk_font = Path("/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf")
    if not cjk_font.exists():
        cjk_font = Path("/usr/share/fonts/google-noto-vf/NotoSans[wght].ttf")
    latin_font = Path("/usr/share/fonts/google-noto-vf/NotoSans[wght].ttf")

    slogan = "γνῶθι σεαυτόν · Know thyself."
    slogan_font = ImageFont.truetype(str(latin_font), 30)
    slogan_box = draw.textbbox((0, 0), slogan, font=slogan_font)
    slogan_width = slogan_box[2] - slogan_box[0]
    draw.text(((width - slogan_width) / 2, 105), slogan, fill="#8a8178", font=slogan_font)

    title_size, title_fonts, lines = fitting_mixed_text(draw, title, cjk_font, latin_font, 76, 48, 980, 3)
    line_height = int(title_size * 1.35)
    block_height = line_height * len(lines)
    top = 255 - max(0, block_height - line_height) / 2
    for index, line in enumerate(lines):
        line_width = mixed_text_width(draw, line, title_fonts)
        draw_mixed_text(
            draw,
            ((width - line_width) / 2, top + index * line_height),
            line,
            title_fonts,
            "#2b2926",
        )

    accent_width = 80
    draw.rounded_rectangle(
        ((width - accent_width) / 2, 520, (width + accent_width) / 2, 526),
        radius=3,
        fill="#d96f32",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, optimize=True)


def render_formula(source, output):
    formula = source.strip()
    formula = re.sub(r"^\$\$|\$\$$", "", formula).strip()
    has_unicode = any(ord(character) > 127 for character in formula)
    font_setup = "\\usepackage{fontspec}\n\\setmainfont{Droid Sans Fallback}" if has_unicode else "\\usepackage[utf8]{inputenc}"
    document = rf"""\documentclass[12pt]{{article}}
{font_setup}
\usepackage{{amsmath,amssymb}}
\pagestyle{{empty}}
\begin{{document}}
\[
{formula}
\]
\end{{document}}
"""
    with tempfile.TemporaryDirectory(prefix="wechat-formula-") as temporary:
        temporary_path = Path(temporary)
        tex_path = temporary_path / "formula.tex"
        tex_path.write_text(document, encoding="utf-8")
        subprocess.run(
            ["xelatex" if has_unicode else "pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=temporary_path,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            ["pdftoppm", "-png", "-r", "180", "-singlefile", "formula.pdf", "formula"],
            cwd=temporary_path,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes((temporary_path / "formula.png").read_bytes())
    trim_image(output)


def extract_article(page_path, source_url, formula_directory):
    soup = BeautifulSoup(page_path.read_text(encoding="utf-8"), "lxml")
    body = soup.select_one(".article__body")
    if body is None:
        raise RuntimeError("Generated page has no .article__body element.")

    content = BeautifulSoup("<section></section>", "lxml").section
    disclaimer = soup.select_one(".disclaimer")
    if disclaimer:
        content.append(BeautifulSoup(str(disclaimer), "lxml").body.contents[0])
    for child in list(body.contents):
        content.append(child)
    references = soup.select_one(".references")
    if references:
        content.append(BeautifulSoup(str(references), "lxml").body.contents[0])

    for citation in content.select("sup.cite"):
        number = citation.get_text(strip=True)
        replacement = soup.new_tag("sup")
        replacement.string = f"[{number}]"
        citation.replace_with(replacement)

    reference_section = content.select_one("section.references")
    if reference_section:
        heading = reference_section.find("h2")
        if heading:
            heading.string = "参考文献"
        reference_list = reference_section.find("ol")
        if reference_list:
            reference_items = soup.new_tag("section")
            reference_items["class"] = "wechat-reference-list"
            for index, item in enumerate(reference_list.find_all("li", recursive=False), start=1):
                paragraph = soup.new_tag("p")
                paragraph["class"] = "wechat-reference-item"
                reference_text = item.get_text(" ", strip=True)
                reference_text = re.sub(r"\s+", " ", reference_text)
                reference_text = re.sub(r"\s*·\s*", " · ", reference_text).strip()
                paragraph.string = f"[{index}] {reference_text}"
                reference_items.append(paragraph)
            reference_list.replace_with(reference_items)

    for unordered_list in reversed(content.find_all("ul")):
        list_section = soup.new_tag("section")
        list_section["class"] = "wechat-bullet-list"
        for item in unordered_list.find_all("li", recursive=False):
            paragraph = soup.new_tag("p")
            paragraph["class"] = "wechat-bullet-item"
            paragraph.append("● ")
            while item.contents:
                paragraph.append(item.contents[0])
            list_section.append(paragraph)
        unordered_list.replace_with(list_section)

    for ordered_list in reversed(content.find_all("ol")):
        list_section = soup.new_tag("section")
        list_section["class"] = "wechat-numbered-list"
        for index, item in enumerate(ordered_list.find_all("li", recursive=False), start=1):
            paragraph = soup.new_tag("p")
            paragraph["class"] = "wechat-numbered-item"
            marker = soup.new_tag("span")
            marker["class"] = "wechat-numbered-marker"
            marker.string = f"{index}."
            paragraph.append(marker)
            while item.contents:
                paragraph.append(item.contents[0])
            list_section.append(paragraph)
        ordered_list.replace_with(list_section)

    formulas = []
    for index, formula in enumerate(content.select(".kdmath"), start=1):
        output = formula_directory / f"formula-{index}.png"
        render_formula(formula.get_text(), output)
        placeholder = soup.new_tag("img")
        placeholder["data-local-image"] = str(output)
        placeholder["alt"] = formula.get_text(" ", strip=True)
        formula.replace_with(placeholder)
        formulas.append(output)

    svg_colors = {
        "var(--text)": "#2b2926",
        "var(--text-faint)": "#9a9289",
        "var(--text-mute)": "#6b6258",
        "var(--bg)": "#ffffff",
        "var(--bg-tint)": "#f7f2ea",
        "var(--border-strong)": "#b8aea2",
    }
    for index, svg in enumerate(content.find_all("svg"), start=1):
        if "viewbox" in svg.attrs:
            svg["viewBox"] = svg.attrs.pop("viewbox")
        svg_source = str(svg)
        for variable, color in svg_colors.items():
            svg_source = svg_source.replace(variable, color)
        output = formula_directory / f"figure-{index}.png"
        cairosvg.svg2png(
            bytestring=svg_source.encode("utf-8"),
            write_to=str(output),
            output_width=1560,
            output_height=660,
            background_color="#ffffff",
        )
        placeholder = soup.new_tag("img")
        placeholder["data-local-image"] = str(output)
        placeholder["alt"] = svg.get("aria-label", "article figure")
        svg.replace_with(placeholder)

    for anchor in content.select("a[href]"):
        href = anchor.get("href", "")
        if href.startswith("#"):
            anchor["href"] = f"{source_url}{href}"
        elif href.startswith("/"):
            anchor["href"] = urllib.parse.urljoin(source_url, href)
        anchor.attrs.pop("target", None)
        anchor.attrs.pop("rel", None)

    styles = {
        "section": "font-size:16px;line-height:1.8;color:#2b2926;",
        "h2": "margin:30px 0 14px;font-size:22px;line-height:1.4;color:#171512;",
        "p": "margin:12px 0;line-height:1.8;",
        "aside": "margin:16px 0;padding:12px 14px;background:#f7f2ea;color:#6b6258;border-left:3px solid #b59a76;",
        "blockquote": "margin:16px 0;padding:8px 14px;background:#f7f7f7;border-left:3px solid #999;",
        "ul": "margin:12px 0;padding-left:1.4em;",
        "ol": "margin:12px 0;padding-left:1.4em;",
        "li": "margin:6px 0;word-break:break-word;",
        "table": "width:100%;border-collapse:collapse;margin:16px 0;font-size:14px;",
        "th": "padding:8px;border:1px solid #d8d2c8;background:#f3eee6;",
        "td": "padding:8px;border:1px solid #d8d2c8;",
        "pre": "overflow-x:auto;margin:14px 0;padding:12px;background:#f5f5f5;font-size:13px;",
        "code": "font-family:monospace;background:#f5f5f5;padding:1px 3px;",
        "a": "color:#8b5e34;text-decoration:none;",
        "img": "display:block;max-width:100%;height:auto;margin:18px auto;",
        "figure": "margin:20px 0;",
        "figcaption": "margin:8px 0 0;text-align:center;font-size:13px;line-height:1.55;color:#6b6258;",
        "sup": "font-size:11px;vertical-align:super;",
    }
    for tag_name, style in styles.items():
        for tag in content.select(tag_name):
            tag["style"] = style

    for item in content.select("p.wechat-bullet-item"):
        item["style"] = "margin:4px 0;line-height:1.8;"
    for item in content.select("p.wechat-numbered-item"):
        item["style"] = "margin:4px 0;line-height:1.8;padding-left:1.5em;text-indent:-1.5em;"
    for marker in content.select("span.wechat-numbered-marker"):
        marker["style"] = "display:inline-block;width:1.5em;color:#8b5e34;"
    for item in content.select("p.wechat-reference-item"):
        item["style"] = "margin:2px 0;line-height:1.55;font-size:14px;word-break:break-word;"

    for tag in content.find_all(True):
        allowed = {"href", "src", "alt", "style", "data-local-image"}
        tag.attrs = {key: value for key, value in tag.attrs.items() if key in allowed}
    return content, formulas


def api_json(path, payload=None, query=None):
    url = f"{API_ROOT}{path}"
    if query:
        url = f"{url}?{urllib.parse.urlencode(query)}"
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"WeChat HTTP error: {error.code}") from error
    if result.get("errcode", 0) != 0:
        raise RuntimeError(f"WeChat API error {result.get('errcode')}: {result.get('errmsg')}")
    return result


def access_token(app_id, app_secret):
    result = api_json(
        "/cgi-bin/stable_token",
        {"grant_type": "client_credential", "appid": app_id, "secret": app_secret, "force_refresh": False},
    )
    return result["access_token"]


def multipart_upload(path, token, file_path, field="media"):
    boundary = f"----codex-{uuid.uuid4().hex}"
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        f'Content-Disposition: form-data; name="{field}"; filename="{file_path.name}"\r\n'.encode()
    )
    body.extend(f"Content-Type: {mime_type}\r\n\r\n".encode())
    body.extend(file_path.read_bytes())
    body.extend(f"\r\n--{boundary}--\r\n".encode())
    url = f"{API_ROOT}{path}?{urllib.parse.urlencode({'access_token': token, 'type': 'image'})}"
    request = urllib.request.Request(
        url,
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    result = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
            break
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    if result is None:
        raise RuntimeError("WeChat upload returned no response.")
    if result.get("errcode", 0) != 0:
        raise RuntimeError(f"WeChat upload error {result.get('errcode')}: {result.get('errmsg')}")
    return result


def upload_content_image(token, path):
    return multipart_upload("/cgi-bin/media/uploadimg", token, path)["url"]


def upload_cover(token, path):
    return multipart_upload("/cgi-bin/material/add_material", token, path)["media_id"]


def find_draft_media_id(token, title):
    offset = 0
    matches = []
    while True:
        result = api_json(
            "/cgi-bin/draft/batchget",
            {"offset": offset, "count": 20, "no_content": 1},
            {"access_token": token},
        )
        items = result.get("item", [])
        for item in items:
            articles = item.get("content", {}).get("news_item", [])
            if articles and articles[0].get("title") == title:
                matches.append(item["media_id"])
        offset += len(items)
        if not items or offset >= result.get("total_count", 0):
            break
    if len(matches) > 1:
        raise RuntimeError(f"Multiple WeChat drafts share the title: {title}")
    return matches[0] if matches else None


def resolve_local_images(content, token):
    for image in content.select("img"):
        local_path = image.attrs.pop("data-local-image", None)
        if local_path:
            image["src"] = upload_content_image(token, Path(local_path))
            continue
        source = image.get("src", "")
        if source.startswith("/"):
            path = ROOT / source.lstrip("/")
            image["src"] = upload_content_image(token, path)


def preview_document(title, content):
    return f"<!doctype html><meta charset='utf-8'><title>{html.escape(title)}</title>{content}"


def main():
    args = parse_args()
    post = (args.post or latest_post()).resolve()
    metadata = read_front_matter(post)
    run_jekyll()
    page_path = generated_path(post)
    source_url = f"https://www.jaimeyzzz.com/{page_path.relative_to(ROOT / '_site').as_posix()}"

    formula_directory = args.preview.parent / "wechat-assets"
    shutil.rmtree(formula_directory, ignore_errors=True)
    content, _ = extract_article(page_path, source_url, formula_directory)
    cover_path = args.cover.resolve() if args.cover else args.preview.parent / "wechat-cover.png"
    if not args.cover:
        generate_cover(metadata["title"], cover_path)
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    for image in content.select("img[data-local-image]"):
        local_path = Path(image["data-local-image"])
        image["src"] = local_path.relative_to(args.preview.parent).as_posix()
    args.preview.write_text(preview_document(metadata["title"], content), encoding="utf-8")
    print(f"Preview written: {args.preview}")

    if not args.submit:
        print("Dry run complete. Add --submit to create a WeChat draft.")
        return

    app_id, app_secret = parse_credentials(args.secrets)
    token = access_token(app_id, app_secret)
    target_media_id = args.update_media_id or find_draft_media_id(token, metadata["title"])
    cover_media_id = upload_cover(token, cover_path)
    resolve_local_images(content, token)
    digest = (metadata.get("description") or BeautifulSoup(str(content), "lxml").get_text(" ", strip=True))[:120]
    payload = {
        "articles": [
            {
                "title": metadata["title"],
                "author": "Jia-Ming Lu",
                "digest": digest,
                "content": str(content),
                "content_source_url": source_url,
                "thumb_media_id": cover_media_id,
                "need_open_comment": 0,
                "only_fans_can_comment": 0,
            }
        ]
    }
    if target_media_id:
        update_payload = {
            "media_id": target_media_id,
            "index": 0,
            "articles": payload["articles"][0],
        }
        api_json("/cgi-bin/draft/update", update_payload, {"access_token": token})
        print(f"WeChat draft updated: {target_media_id}")
    else:
        result = api_json("/cgi-bin/draft/add", payload, {"access_token": token})
        print(f"WeChat draft created: {result['media_id']}")


if __name__ == "__main__":
    main()
