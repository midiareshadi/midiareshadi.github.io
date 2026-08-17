"""
build_cache_cliff_post.py — turn the cache-cliff draft into a styled blog page.

Reads the markdown draft, swaps the FIGURE placeholders for real <img> blocks,
and wraps it in the site skeleton. Re-run after editing the draft.
"""
import os
import markdown

DRAFT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "_drafts", "cache-cliff-draft.md")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "blog", "cache-cliff", "index.html")

FIGURES = {
    "FIGURE_CONCEPT": ("cc-concept.svg",
                       "The cache as a fixed box. Read-write must hold input and "
                       "output together, so input can only reach about half the "
                       "cache before it overflows. Read-only holds input alone, so "
                       "input can reach the whole cache — and the cliff moves to "
                       "twice the size."),
    "FIGURE_DATA": ("cc-rw-vs-ro.svg",
                    "Input bandwidth as a percentage of measured peak, against "
                    "input size. The read-write kernel cliffs at about 27 MB; the "
                    "read-only kernel stays fast until about 55 MB — almost exactly "
                    "double, because removing the output frees half the cache."),
}

TITLE = "A cliff below the cache: why a softmax slows down before it should"


def skeleton():
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "..", "blog", "oxide-moving-target", "index.html")
    html = open(src).read()
    head_end = html.find('<article>')
    assert head_end > 0, "could not find <article> in the reference post"
    head = html[:head_end]
    head = head.replace(
        "Benchmarking a moving target: my cuda-oxide results changed in weeks &mdash; Midia Reshadi",
        f"{TITLE} &mdash; Midia Reshadi")
    head = head.replace(
        "<title>Benchmarking a moving target: my cuda-oxide results changed in weeks — Midia Reshadi</title>",
        f"<title>{TITLE} — Midia Reshadi</title>")
    return head


def figure_html(key):
    src, caption = FIGURES[key]
    return (f'<div class="figure">\n'
            f'    <img src="../../assets/figures/{src}" alt="{caption}">\n'
            f'    <p class="figure-caption">{caption}</p>\n'
            f'</div>')


def body_html():
    md = open(DRAFT).read()
    lines = md.splitlines()
    assert lines[0].startswith("# "), "draft should start with an H1"
    title = lines[0][2:].strip()
    md = "\n".join(lines[1:]).lstrip("\n")
    html = markdown.markdown(md, extensions=["tables"])
    for key in FIGURES:
        html = html.replace(f"<p>{key}</p>", figure_html(key))
        html = html.replace(key, figure_html(key))
    return title, html


HEADER = '''<article>
            <header class="post-header">
                <h1>{title}</h1>
                <p class="post-meta">August 2026 &middot; ~8 min read &middot;
                    <a href="https://github.com/midiareshadi/kernel-curiosity">code</a>
                </p>
            </header>

            <div class="post-content">
'''

FOOTER = '''
            </div>
        </article>
    </main>
</body>
</html>
'''


def main():
    title, body = body_html()
    page = skeleton() + HEADER.format(title=title) + body + FOOTER
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(page)
    print(f"wrote {os.path.normpath(OUT)}")
    print(f"  title: {title}")
    print(f"  {len(page.splitlines())} lines")


if __name__ == "__main__":
    main()
