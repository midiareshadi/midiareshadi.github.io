#!/usr/bin/env python3
"""
build.py — regenerate every shared part of the site from tools/posts.json.

    python3 tools/build.py            # rewrite the shared blocks in place
    python3 tools/build.py --bump     # same, and bump the stylesheet version
    python3 tools/build.py --check    # report what would change, write nothing

It rewrites, on the homepage and on every post page:
  * the whole <head>
  * the floating toggle + <aside class="sidebar">
  * the homepage post list, and the series navigation on each post
and regenerates sitemap.xml and feed.xml.

It never touches the body of a post, so posts written by hand and posts built
by tools/build_*_post.py are both safe to run this over.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sitegen as S

ROOT = S.ROOT

HEAD_RE = re.compile(r"<head>.*?</head>", re.S)
SIDEBAR_RE = re.compile(r'[ \t]*<button id="floating-toggle".*?</aside>', re.S)
LIST_RE = re.compile(r'[ \t]*<ul class="post-list">.*?</ul>', re.S)
NAV_RE = re.compile(r"[ \t]*<!-- series-nav -->.*?<!-- /series-nav -->\n?", re.S)
HEADER_END_RE = re.compile(r"(</header>\s*\n)")


def apply(path, new_text, changes, check):
    with open(path) as f:
        old = f.read()
    if old == new_text:
        return
    changes.append(os.path.relpath(path, ROOT))
    if not check:
        with open(path, "w") as f:
            f.write(new_text)


def build_homepage(data, changes, check):
    path = os.path.join(ROOT, "index.html")
    with open(path) as f:
        t = f.read()
    t = HEAD_RE.sub(lambda m: S.render_head(data, None, ""), t, count=1)
    t = SIDEBAR_RE.sub(lambda m: S.render_sidebar(data, ""), t, count=1)
    t = LIST_RE.sub(lambda m: S.render_post_list(data), t, count=1)
    apply(path, t, changes, check)


def build_post(data, post, changes, check):
    path = os.path.join(ROOT, "blog", post["slug"], "index.html")
    if not os.path.exists(path):
        print(f"  !! missing {os.path.relpath(path, ROOT)}")
        return
    with open(path) as f:
        t = f.read()

    t = HEAD_RE.sub(lambda m: S.render_head(data, post, "../../"), t, count=1)
    t = SIDEBAR_RE.sub(lambda m: S.render_sidebar(data, "../../"), t, count=1)

    nav = S.render_series_nav(data, post, "../../")
    t = NAV_RE.sub("", t)                                  # drop any previous nav
    if nav:
        m = HEADER_END_RE.search(t)
        if m:
            t = t[:m.end()] + nav + "\n" + t[m.end():]
        else:
            print(f"  !! no </header> in {post['slug']}, series nav not inserted")

    apply(path, t, changes, check)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bump", action="store_true", help="increment the stylesheet version")
    ap.add_argument("--check", action="store_true", help="report changes, write nothing")
    args = ap.parse_args()

    data = S.load()

    if args.bump and not args.check:
        data["site"]["asset_version"] += 1
        with open(S.DATA, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        print(f"asset version -> {data['site']['asset_version']}")

    changes = []
    build_homepage(data, changes, args.check)
    for p in data["posts"]:
        build_post(data, p, changes, args.check)

    for name, text in (("sitemap.xml", S.render_sitemap(data)),
                       ("feed.xml", S.render_feed(data))):
        path = os.path.join(ROOT, name)
        old = open(path).read() if os.path.exists(path) else None
        # the feed's build date changes every run; ignore that line when comparing
        norm = lambda s: re.sub(r"<lastBuildDate>.*?</lastBuildDate>", "", s or "")
        if norm(old) != norm(text):
            changes.append(name)
            if not args.check:
                with open(path, "w") as f:
                    f.write(text)

    verb = "would change" if args.check else "wrote"
    if changes:
        print(f"{verb} {len(changes)} file(s):")
        for c in changes:
            print("   ", c)
    else:
        print("everything already up to date")


if __name__ == "__main__":
    main()
