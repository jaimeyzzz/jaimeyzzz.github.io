#!/usr/bin/env python3

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "_site"


def fail(message):
    print(f"Build validation failed: {message}", file=sys.stderr)
    raise SystemExit(1)


def main():
    index = SITE / "index.html"
    stylesheet = SITE / "assets/css/style.css"
    if not index.is_file():
        fail("_site/index.html is missing")
    if not stylesheet.is_file():
        fail("homepage stylesheet is missing")

    html = index.read_text(encoding="utf-8")
    css = stylesheet.read_text(encoding="utf-8")
    if len(css) < 10_000:
        fail(f"homepage stylesheet is unexpectedly small ({len(css)} bytes)")
    for selector in (".wrap", ".hero", ".showcase", ".pubs"):
        if selector not in css:
            fail(f"required selector {selector} is missing")
    if 'href="./assets/css/style.css"' not in html:
        fail("homepage does not reference its stylesheet")
    if re.search(r'<meta (?:property="og:image"|name="twitter:image") content="http:', html):
        fail("social preview image is not HTTPS")

    print(f"Build validation passed: {len(css)} CSS bytes and required selectors found.")


if __name__ == "__main__":
    main()
