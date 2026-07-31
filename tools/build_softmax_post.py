"""
build_softmax_post.py — turn the softmax draft into a styled blog page.

Reads the markdown draft from the kernel-curiosity repo, converts it, swaps the
FIGURE_n placeholders for real <img> blocks, and wraps it in the site skeleton.
Re-run it after editing the draft; it overwrites the page.
"""
import os
import markdown

DRAFT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "_drafts", "softmax-post-draft.md")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "blog", "softmax-cache-mirage", "index.html")

FIGURES = {
    "FIGURE_1": ("sm-spec-vs-measured.svg",
                 "Spec-sheet bandwidth against what a plain copy actually reaches, "
                 "on both GPUs."),
    "FIGURE_2": ("sm-cache-mirage.svg",
                 "Percentage of measured peak bandwidth against tensor size. In the "
                 "shaded region the data fits in cache, so the numbers describe the "
                 "cache and can pass 100%. Once the data spills cache, the two "
                 "kernels agree."),
    "FIGURE_3": ("sm-two-gpus.svg",
                 "The same simple kernel on both GPUs. Past each dotted line the "
                 "data no longer fits that GPU's cache. The L4 reaches its limit; "
                 "the MI300X flattens at about half of its own."),
}


def skeleton():
    """Take the head+sidebar from an existing post so the site stays consistent."""
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(here, "..", "blog", "oxide-moving-target", "index.html")
    html = open(src).read()
    head_end = html.find('<article>')
    assert head_end > 0, "could not find <article> in the reference post"
    head = html[:head_end]
    # swap the title
    head = head.replace(
        "Benchmarking a moving target: my cuda-oxide results changed in weeks &mdash; Midia Reshadi",
        "How close to peak is a simple softmax? Two GPUs, one measurement trap &mdash; Midia Reshadi")
    head = head.replace(
        "<title>Benchmarking a moving target: my cuda-oxide results changed in weeks — Midia Reshadi</title>",
        "<title>How close to peak is a simple softmax? Two GPUs, one measurement trap — Midia Reshadi</title>")
    return head


def figure_html(key):
    src, caption = FIGURES[key]
    return (f'<div class="figure">\n'
            f'    <img src="../../assets/figures/{src}" alt="{caption}">\n'
            f'    <p class="figure-caption">{caption}</p>\n'
            f'</div>')


def body_html():
    md = open(DRAFT).read()

    # drop the H1 (the page header carries the title) and keep the rest
    lines = md.splitlines()
    assert lines[0].startswith("# "), "draft should start with an H1"
    title = lines[0][2:].strip()
    md = "\n".join(lines[1:]).lstrip("\n")

    html = markdown.markdown(md, extensions=["tables"])

    # placeholders come through as their own paragraphs
    for key in FIGURES:
        html = html.replace(f"<p>{key}</p>", figure_html(key))
        html = html.replace(key, figure_html(key))  # safety net

    return title, html


HEADER = '''<article>
            <header class="post-header">
                <h1>{title}</h1>
                <p class="post-meta">July 2026 &middot; ~10 min read &middot;
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
