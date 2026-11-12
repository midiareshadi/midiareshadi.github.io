"""
site.py — one source of truth for every shared piece of the site.

Everything that appears on more than one page (the <head>, the sidebar, the
series navigation, the homepage post list, the sitemap, the feed) is rendered
from tools/posts.json by the functions here. build.py writes them to disk.

Nothing in here reads an existing page to copy markup out of it.
"""
import json
import os
from email.utils import format_datetime
from datetime import datetime, timezone
from html import escape

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, ".."))
DATA = os.path.join(HERE, "posts.json")


def load():
    with open(DATA) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# head
# --------------------------------------------------------------------------

def render_head(data, post=None, prefix=""):
    """`post` is None for the homepage. `prefix` is the path back to the root."""
    site = data["site"]
    v = site["asset_version"]

    if post is None:
        title = f'{site["title"]} — {site["tagline"]}'
        desc = site["description"]
        url = site["url"] + "/"
    else:
        title = f'{post["title"]} — {site["title"]}'
        desc = post["description"]
        url = f'{site["url"]}/blog/{post["slug"]}/'

    lines = [
        "<head>",
        '    <meta charset="UTF-8">',
        '    <meta name="viewport" content="width=device-width, initial-scale=1.0">',
        f"    <title>{escape(title)}</title>",
        f'    <meta name="description" content="{escape(desc, quote=True)}">',
        f'    <link rel="canonical" href="{url}">',
        '    <meta property="og:type" content="' + ("website" if post is None else "article") + '">',
        f'    <meta property="og:url" content="{url}">',
        f'    <meta property="og:title" content="{escape(title, quote=True)}">',
        f'    <meta property="og:description" content="{escape(desc, quote=True)}">',
        '    <meta name="twitter:card" content="summary_large_image">',
        f'    <link rel="alternate" type="application/rss+xml" title="{escape(site["title"])}"'
        f' href="{site["url"]}/feed.xml">',
        '    <link rel="preconnect" href="https://fonts.googleapis.com">',
        '    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>',
        f'    <link rel="stylesheet" href="{prefix}css/style.css?v={v}">',
        '    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700'
        '&family=Comic+Neue:wght@700&display=swap" rel="stylesheet">',
        "</head>",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# sidebar
# --------------------------------------------------------------------------

ICONS = {
    "github": '<path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>',
    "linkedin": '<path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 0 1-2.063-2.065 2.064 2.064 0 1 1 2.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>',
    "scholar": '<path d="M12 24a7 7 0 1 1 0-14 7 7 0 0 1 0 14zm0-24L0 9.5l4.838 3.94A8 8 0 0 1 12 10a8 8 0 0 1 7.162 3.44L24 9.5z"/>',
    "zulip": '<path d="M2 5a2 2 0 0 1 2-2h16a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H8l-6 4V5z"/>',
}

LINKS = [
    ("github", "GitHub", "https://github.com/midiareshadi"),
    ("linkedin", "LinkedIn", "https://www.linkedin.com/in/midia-reshadi/"),
    ("scholar", "Google Scholar", "https://scholar.google.com/citations?user=gVWj19kAAAAJ&hl=en"),
    ("zulip", "Zulip (rust-lang)", "https://rust-lang.zulipchat.com/#user/818806"),
]

TOGGLE_SVG = ('<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
              'stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/>'
              '<line x1="9" y1="3" x2="9" y2="21"/></svg>')


def render_sidebar(data, prefix=""):
    """Renders the floating toggle button and the sidebar as one block."""
    site = data["site"]
    out = ['    <button id="floating-toggle" onclick="toggleSidebar()" title="Show sidebar">',
           f"        {TOGGLE_SVG}",
           "    </button>",
           '    <aside class="sidebar">', '        <div class="sidebar-head">',
           '            <button class="sidebar-toggle" onclick="toggleSidebar()" '
           'title="Toggle sidebar">', f"                {TOGGLE_SVG}", "            </button>",
           f'            <p class="sidebar-brand">&#9998; {escape(site["author"])}\'s Blog</p>',
           "        </div>", '        <ul class="sidebar-nav">',
           f'            <li><a href="{prefix}index.html">&#8962; Home</a></li>',
           "        </ul>", '        <div class="sidebar-links">']
    for key, label, href in LINKS:
        out.append(f'            <a href="{href}" target="_blank" rel="noopener">')
        out.append('                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" '
                   'class="sidebar-icon">' + ICONS[key] + "</svg>")
        out.append(f"                {escape(label)}")
        out.append("            </a>")
    out += ["        </div>", "    </aside>"]
    return "\n".join(out)


# --------------------------------------------------------------------------
# series navigation
# --------------------------------------------------------------------------

def series_of(data, slug):
    for key, s in data["series"].items():
        if slug in s["posts"]:
            return key, s
    return None, None


def render_series_nav(data, post, prefix="../../"):
    key, s = series_of(data, post["slug"])
    if not s:
        return ""
    by_slug = {p["slug"]: p for p in data["posts"]}
    i = s["posts"].index(post["slug"])
    n = len(s["posts"])

    rows = []
    for j, slug in enumerate(s["posts"]):
        p = by_slug[slug]
        label = f"Part {j + 1}"
        if slug == post["slug"]:
            rows.append(f'                    <li class="is-current">'
                        f'<span class="series-part">{label}</span>'
                        f'<span>{escape(p["title"])}</span></li>')
        else:
            rows.append(f'                    <li><span class="series-part">{label}</span>'
                        f'<a href="{prefix}blog/{slug}/">{escape(p["title"])}</a></li>')

    return "\n".join([
        "            <!-- series-nav -->",
        '            <nav class="series-nav" aria-label="Series navigation">',
        f'                <p class="series-title">{escape(s["name"])} '
        f'<span class="series-count">part {i + 1} of {n}</span></p>',
        "                <ul>",
        "\n".join(rows),
        "                </ul>",
        "            </nav>",
        "            <!-- /series-nav -->",
    ])


# --------------------------------------------------------------------------
# homepage list
# --------------------------------------------------------------------------

def render_post_list(data):
    out = ['        <ul class="post-list">']
    for p in reversed(data["posts"]):          # newest first
        out.append('            <li class="post-item">')
        out.append(f'                <p class="post-date">{escape(p["date_label"])}</p>')
        out.append('                <p class="post-title">')
        out.append(f'                    <a href="/blog/{p["slug"]}/">')
        out.append(f"                        {escape(p['title'])}")
        out.append("                    </a>")
        out.append("                </p>")
        out.append(f'                <p class="post-desc">{escape(p["description"])}</p>')
        out.append('                <div class="post-tags">')
        for t in p["tags"]:
            out.append(f'                    <span class="tag">{escape(t)}</span>')
        out.append("                </div>")
        out.append("            </li>")
    out.append("        </ul>")
    return "\n".join(out)


# --------------------------------------------------------------------------
# sitemap + feed
# --------------------------------------------------------------------------

def render_sitemap(data):
    site = data["site"]
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
           "  <url>", f"    <loc>{site['url']}/</loc>",
           "    <changefreq>weekly</changefreq>", "    <priority>1.0</priority>", "  </url>"]
    for p in reversed(data["posts"]):
        out += ["  <url>", f"    <loc>{site['url']}/blog/{p['slug']}/</loc>",
                f"    <lastmod>{p['published']}</lastmod>",
                "    <changefreq>monthly</changefreq>", "    <priority>0.8</priority>", "  </url>"]
    out.append("</urlset>")
    return "\n".join(out) + "\n"


def render_feed(data):
    site = data["site"]
    now = format_datetime(datetime.now(timezone.utc))
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<rss version="2.0" xmlns:atom="http://www.w3.org/2005/Atom">', "  <channel>",
           f"    <title>{escape(site['title'])} — {escape(site['tagline'])}</title>",
           f"    <link>{site['url']}/</link>",
           f"    <description>{escape(site['description'])}</description>",
           "    <language>en</language>", f"    <lastBuildDate>{now}</lastBuildDate>",
           f'    <atom:link href="{site["url"]}/feed.xml" rel="self" type="application/rss+xml"/>']
    for p in reversed(data["posts"]):
        dt = datetime.strptime(p["published"], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        url = f"{site['url']}/blog/{p['slug']}/"
        out += ["    <item>", f"      <title>{escape(p['title'])}</title>",
                f"      <link>{url}</link>", f"      <guid isPermaLink=\"true\">{url}</guid>",
                f"      <pubDate>{format_datetime(dt)}</pubDate>",
                f"      <description>{escape(p['description'])}</description>"]
        for t in p["tags"]:
            out.append(f"      <category>{escape(t)}</category>")
        out.append("    </item>")
    out += ["  </channel>", "</rss>"]
    return "\n".join(out) + "\n"
