#!/usr/bin/env python3
"""
build.py — generate the whole site from content/ and tools/posts.json.

    python3 tools/build.py            # regenerate everything
    python3 tools/build.py --check    # report what would change, write nothing
    python3 tools/build.py --bump     # also bump the asset cache version
    python3 tools/build.py <slug>     # just one post

Every page is generated in full: there is no hand-maintained HTML and nothing
is patched in place. The source of a post is content/<slug>.md; everything
else about it lives in tools/posts.json.
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sitegen as S

ROOT = S.ROOT
CONTENT = os.path.join(ROOT, "content")


def validate(data):
    """Fail before writing anything, not halfway through."""
    errs = []
    seen = set()
    for p in data.get("posts", []):
        for key in ("slug", "title", "date_label", "published", "description", "tags"):
            if not p.get(key):
                errs.append(f"post {p.get('slug', '?')}: missing '{key}'")
        slug = p.get("slug")
        if slug in seen:
            errs.append(f"duplicate slug: {slug}")
        seen.add(slug)
        if slug and not os.path.exists(os.path.join(CONTENT, f"{slug}.md")):
            errs.append(f"post {slug}: content/{slug}.md not found")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.get("published", "")):
            errs.append(f"post {slug}: 'published' must be YYYY-MM-DD")
        for f in p.get("figures", []):
            for key in ("placeholder", "src", "caption"):
                if not f.get(key):
                    errs.append(f"post {slug}: figure missing '{key}'")
    for key, s in data.get("series", {}).items():
        for slug in s.get("posts", []):
            if slug not in seen:
                errs.append(f"series '{key}' lists unknown post: {slug}")
    if errs:
        print("posts.json is not valid:")
        for e in errs:
            print("   ", e)
        sys.exit(1)


def write(path, text, changes, check):
    rel = os.path.relpath(path, ROOT)
    old = open(path).read() if os.path.exists(path) else None
    if old == text:
        return
    changes.append(rel)
    if not check:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug", nargs="?", help="build only this post")
    ap.add_argument("--bump", action="store_true", help="bump the asset version")
    ap.add_argument("--check", action="store_true", help="report changes, write nothing")
    args = ap.parse_args()

    data = S.load()
    validate(data)

    if args.bump and not args.check:
        data["site"]["asset_version"] += 1
        with open(S.DATA, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        print(f"asset version -> {data['site']['asset_version']}")

    posts = data["posts"]
    if args.slug:
        posts = [p for p in posts if p["slug"] == args.slug]
        if not posts:
            sys.exit(f"no such post: {args.slug}")

    changes = []
    for p in posts:
        md = open(os.path.join(CONTENT, f"{p['slug']}.md")).read()
        page = S.render_post_page(data, p, md)
        write(os.path.join(ROOT, "blog", p["slug"], "index.html"), page, changes, args.check)

    if not args.slug:
        write(os.path.join(ROOT, "index.html"), S.render_homepage(data), changes, args.check)
        write(os.path.join(ROOT, "sitemap.xml"), S.render_sitemap(data), changes, args.check)

        feed_path = os.path.join(ROOT, "feed.xml")
        feed = S.render_feed(data)
        strip = lambda s: re.sub(r"<lastBuildDate>.*?</lastBuildDate>", "", s or "")
        old = open(feed_path).read() if os.path.exists(feed_path) else None
        if strip(old) != strip(feed):
            changes.append("feed.xml")
            if not args.check:
                with open(feed_path, "w") as f:
                    f.write(feed)

    verb = "would change" if args.check else "wrote"
    if changes:
        print(f"{verb} {len(changes)} file(s):")
        for c in changes:
            print("   ", c)
    else:
        print("everything already up to date")


if __name__ == "__main__":
    main()
