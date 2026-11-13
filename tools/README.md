# tools/

## How the site is built

Every published page is generated. Nothing under `blog/` or `index.html` is
edited by hand.

| Source | Becomes |
|---|---|
| `content/<slug>.md` | the body of that post |
| `tools/posts.json` | titles, dates, tags, figures, series, the homepage intro |
| `css/style.css`, `js/site.js` | shared across every page |

```bash
python3 -m pip install -r requirements.txt   # once
python3 tools/build.py                       # regenerate everything
python3 tools/check.py                       # verify the result
```

`build.py` writes every page in full: the head, the sidebar, the post body,
the series navigation, `index.html`, `sitemap.xml` and `feed.xml`. Running it
twice in a row makes no changes the second time.

```bash
python3 tools/build.py --check     # show what would change, write nothing
python3 tools/build.py --bump      # also bump the asset cache version
python3 tools/build.py <slug>      # rebuild just one post
```

## Publishing a post

1. Write `content/<slug>.md`, starting with an `# H1` title.
   Put `FIGURE_SOMETHING` on its own line wherever a figure belongs.
2. Add the post to `tools/posts.json`: slug, title, date_label, published,
   description, tags, read_minutes, links, and one entry under `figures` for
   each placeholder you used.
   If it continues a series, add the slug to that series' `posts` list.
3. `python3 tools/build.py`
4. `python3 tools/check.py`
5. Look at it: `python3 -m http.server 8000`
6. Commit and push.

## After changing css/style.css or js/site.js

Run `python3 tools/build.py --bump`. GitHub Pages caches both for ten minutes,
so without a version bump your change looks like it did not deploy. `check.py`
fails if any page is left on an old version.

## Notes

- `published` dates default to the first of the month, because the posts only
  ever recorded a month. The RSS feed uses them for `pubDate`.
- Markdown cannot express a cell class or a link `target`, so a table or link
  that needs one is kept as raw HTML in the `.md` file. That is deliberate —
  raw HTML passes through markdown untouched.
- `_drafts/` is for work in progress and is not tracked. Published sources
  live in `content/`, which is.
