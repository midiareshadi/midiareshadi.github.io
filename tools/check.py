#!/usr/bin/env python3
"""
check.py — verify the published site is consistent with its source.

    python3 tools/check.py

Checks:
  1. every post in posts.json has a content/<slug>.md and a built page
  2. every built page is listed in posts.json
  3. every internal link resolves on disk
  4. every referenced figure and asset exists
  5. sitemap.xml and feed.xml list every post
  6. each page declares the current asset version
  7. series definitions reference only known posts

Exits non-zero on the first category that fails, so it can gate a commit or
run in CI. It does not rewrite anything.
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))


def rel(p):
    return os.path.relpath(p, ROOT)


def strip_query(href):
    return href.split("?")[0].split("#")[0]


def main():
    with open(os.path.join(HERE, "posts.json")) as f:
        data = json.load(f)

    site = data["site"]
    posts = data["posts"]
    slugs = [p["slug"] for p in posts]
    problems = []

    # 1 + 2 — sources, pages, and posts.json agree
    for p in posts:
        for path in (os.path.join(ROOT, "content", f"{p['slug']}.md"),
                     os.path.join(ROOT, "blog", p["slug"], "index.html")):
            if not os.path.exists(path):
                problems.append(f"{p['slug']}: missing {rel(path)}")
    blog_dir = os.path.join(ROOT, "blog")
    for name in sorted(os.listdir(blog_dir)):
        if os.path.isdir(os.path.join(blog_dir, name)) and name not in slugs:
            problems.append(f"blog/{name}/ is published but not in posts.json")

    # 3 + 4 — links and assets resolve
    pages = [os.path.join(ROOT, "index.html")]
    pages += [os.path.join(ROOT, "blog", s, "index.html") for s in slugs]
    n_links = 0
    for page in pages:
        if not os.path.exists(page):
            continue
        text = open(page).read()
        for href in re.findall(r'(?:href|src)="([^"]+)"', text):
            if href.startswith(("http://", "https://", "mailto:", "data:", "#")):
                continue
            h = strip_query(href)
            if not h:
                continue
            n_links += 1
            target = (os.path.join(ROOT, h.lstrip("/")) if h.startswith("/")
                      else os.path.join(os.path.dirname(page), h))
            target = os.path.normpath(target)
            if os.path.isdir(target):
                target = os.path.join(target, "index.html")
            if not os.path.exists(target):
                problems.append(f"{rel(page)}: broken link {href}")

    # 5 — sitemap and feed cover every post
    for name in ("sitemap.xml", "feed.xml"):
        path = os.path.join(ROOT, name)
        if not os.path.exists(path):
            problems.append(f"missing {name}")
            continue
        text = open(path).read()
        for s in slugs:
            if f"/blog/{s}/" not in text:
                problems.append(f"{name}: missing {s}")

    # 6 — every page is on the current asset version
    want = f"?v={site['asset_version']}"
    for page in pages:
        if not os.path.exists(page):
            continue
        for href in re.findall(r'(?:href|src)="((?:\.\./)*(?:css|js)/[^"]+)"', open(page).read()):
            if want not in href:
                problems.append(f"{rel(page)}: stale asset version in {href}"
                                f" (expected {want}) — run build.py")

    # 7 — series reference known posts
    for key, s in data.get("series", {}).items():
        for slug in s.get("posts", []):
            if slug not in slugs:
                problems.append(f"series '{key}': unknown post {slug}")

    print(f"checked {len(pages)} pages, {n_links} internal links, {len(slugs)} posts")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in sorted(set(problems)):
            print("  -", p)
        sys.exit(1)
    print("no problems found")


if __name__ == "__main__":
    main()
