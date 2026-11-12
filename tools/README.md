# tools/

## What is generated

`tools/posts.json` is the single source of truth for post metadata. Running

```bash
python3 tools/build.py
```

regenerates, from that file:

| Where | What |
|---|---|
| `index.html` | `<head>`, sidebar, the post list |
| `blog/*/index.html` | `<head>`, sidebar, series navigation |
| `sitemap.xml` | every page |
| `feed.xml` | RSS feed |

It never touches the body of a post, so it is safe to run at any time.
Running it twice in a row makes no changes the second time.

```bash
python3 tools/build.py --check   # show what would change, write nothing
python3 tools/build.py --bump    # also bump the stylesheet cache version
```

## Publishing a post

1. Write the draft in `_drafts/<slug>-draft.md`.
2. Build the body: `python3 tools/build_<name>_post.py`
   (these scripts convert one draft to one post page).
3. Add an entry to `tools/posts.json` — slug, title, date_label, published,
   description, tags, read_minutes, repo_url, repo_label. If it belongs to a
   series, add the slug to that series' `posts` list, in reading order.
4. `python3 tools/build.py`
5. Check locally: `python3 -m http.server 8000`, then open
   <http://localhost:8000>.
6. Commit and push.

## After changing css/style.css

Run `python3 tools/build.py --bump`. GitHub Pages caches the stylesheet for
ten minutes, so without a version bump your change looks like it did not
deploy. The bump rewrites `?v=` on every page at once.

## Notes

- `published` dates default to the first of the month, because the posts only
  ever recorded a month. The feed uses them for `pubDate`; correct any that
  matter.
- `build_*_post.py` still builds each post body. They no longer need to copy a
  `<head>` from a sibling post — `build.py` overwrites the head afterwards.
