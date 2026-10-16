#!/usr/bin/env python3
"""Generate the GitHub Pages site into docs/.

    python3 docs/build.py

How the site is put together (see docs/pages/contributing.html for the guide):

- docs/pages/*.html   one file per page: a small front-matter block, then HTML.
- docs/syllabus.py    the whole book (chapters, sections) + which source files
                      each language tab shows. Roadmap and chapter pages are
                      generated from it, so planned parts appear automatically.
- Code blocks are read from the real source files at build time, so the site
  never drifts from the code. Commit the generated HTML.
"""

import html
import re
from pathlib import Path

import syllabus as S

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
PAGES_DIR = HERE / "pages"
HLJS = "https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0"
LANG_LABEL = {k: n for k, n, _ in S.LANGS}
LANG_DIR = {k: d for k, _, d in S.LANGS}


# --------------------------------------------------------------------- pages
def parse_page(path):
    text = path.read_text(encoding="utf-8")
    m = re.match(r"---\n(.*?)\n---\n(.*)", text, re.S)
    meta = {}
    for line in m.group(1).splitlines():
        k, v = line.split(":", 1)
        meta[k.strip()] = v.strip()
    meta["slug"] = path.stem
    meta["body"] = m.group(2)
    meta["order"] = int(meta.get("order", 100))
    meta["sections"] = [s.strip() for s in meta.get("section", "").split(",") if s.strip()]
    meta["extra"] = meta.get("extra") == "true"
    return meta


def chapter_key(v):
    return str(v)


def chapter_slug(num):
    return f"chapter-{num}"


PAGES = sorted((parse_page(p) for p in PAGES_DIR.glob("*.html")), key=lambda p: p["order"])
BY_SLUG = {p["slug"]: p for p in PAGES}


def pages_of(chapter, extra=False):
    return [p for p in PAGES if p.get("chapter") == chapter_key(chapter) and p["extra"] == extra]


def ready_sections(chapter):
    out = set()
    for p in pages_of(chapter.get("num")):
        out.update(p["sections"])
    return out


def chapter_status(ch):
    total = len(ch["sections"])
    ready = len(ready_sections(ch) & {s for s, _ in ch["sections"]})
    if total and ready == total:
        return "ready"
    if ready:
        return "in progress"
    return "planned"


# ------------------------------------------------------------- generated bits
def chip(status):
    cls = status.replace(" ", "-")
    return f'<span class="chip {cls}">{status}</span>'


def lang_badges(langs):
    if not langs:
        return '<span class="muted">none yet</span>'
    return " ".join(f'<span class="lang">{html.escape(LANG_LABEL[k])}</span>' for k in langs)


def book_link(ch):
    n = ch["num"]
    folder = f"ch{int(n):02d}" if str(n).isdigit() else "appendix-A"
    return f"{S.BOOK_REPO}/{folder}"


def render_chapter_page(ch):
    num = ch["num"]
    status = chapter_status(ch)
    ready = ready_sections(ch)
    page_of = {}
    for p in pages_of(num):
        for s in p["sections"]:
            page_of[s] = p
    rows = []
    for sid, title in ch["sections"]:
        p = page_of.get(sid)
        if p:
            state = f'{chip("ready")} <a href="{p["slug"]}.html">open</a>'
        else:
            state = chip("planned")
        rows.append(f"<tr><td>{sid}</td><td>{html.escape(title)}</td><td>{state}</td></tr>")
    table = ""
    if rows:
        table = "<table><tr><th>Book</th><th>Section</th><th>Status</th></tr>" + "".join(rows) + "</table>"
    else:
        table = '<p class="muted">Section list not mapped yet. See the book for the outline.</p>'
    extras = pages_of(num, extra=True)
    extras_html = ""
    if extras:
        items = "".join(f'<li><a href="{p["slug"]}.html">{html.escape(p["nav"])}</a></li>' for p in extras)
        extras_html = f"<h2>Extras</h2><ul>{items}</ul>"
    code_line = ""
    if ch["code"]:
        code_line = f'<p><b>Code:</b> {lang_badges(ch.get("langs", []))}</p>'
    planned_note = ""
    if status == "planned":
        planned_note = (
            '<div class="tip">Not written yet. The outline below follows the book. '
            'To add it, follow <a href="contributing.html">Contributing</a>; the roadmap and navigation update by themselves.</div>'
        )
    body = f"""
<p class="lede">{html.escape(ch["summary"])}</p>
<p>{chip(status)} &nbsp; <a href="{book_link(ch)}">Companion code in Raschka's repo</a></p>
{code_line}
{planned_note}
<h2>Sections</h2>
{table}
{extras_html}
"""
    label = f"Chapter {num}" if str(num).isdigit() else "Appendices"
    title = f"{label}: {ch['title']}" if str(num).isdigit() else ch["title"]
    return dict(slug=chapter_slug(num), title=title, nav=label, chapter=chapter_key(num), body=body, generated=True)


def progress_table():
    rows = []
    for ch in S.CHAPTERS:
        total = len(ch["sections"])
        ready = len(ready_sections(ch) & {s for s, _ in ch["sections"]})
        pct = round(100 * ready / total) if total else 0
        label = f"Chapter {ch['num']}" if str(ch["num"]).isdigit() else "Appendices"
        bar = f'<div class="bar"><i style="width:{pct}%"></i></div>'
        count = f"{ready}/{total}" if total else "&ndash;"
        rows.append(
            f'<tr><td><a href="{chapter_slug(ch["num"])}.html">{label}</a></td><td>{html.escape(ch["title"])}</td>'
            f"<td>{bar}<small>{count} sections</small></td><td>{chip(chapter_status(ch))}</td></tr>"
        )
    return "<table><tr><th></th><th>Topic</th><th>Written</th><th>Status</th></tr>" + "".join(rows) + "</table>"


def language_matrix():
    head = "".join(f"<th>{html.escape(n)}</th>" for _, n, _ in S.LANGS)
    rows = []
    for ch in S.CHAPTERS:
        if not ch["code"]:
            continue
        cells = "".join(
            f'<td class="{"yes" if k in ch.get("langs", []) else "no"}">{"&#10003;" if k in ch.get("langs", []) else "&ndash;"}</td>'
            for k, _, _ in S.LANGS
        )
        rows.append(f"<tr><td>Chapter {ch['num']}</td><td>{html.escape(ch['title'])}</td>{cells}</tr>")
    return f"<table class='matrix'><tr><th></th><th>Topic</th>{head}</tr>{''.join(rows)}</table>"


def render_roadmap():
    body = f"""
<p class="lede">Where the project is, and what is next. This page is generated from <code>docs/syllabus.py</code> and the page files, so it is always current.</p>
<h2>The book, chapter by chapter</h2>
{progress_table()}
<h2>Code by language</h2>
<p>A tick means that language implements the chapter's code, tested to give the same results as the others.</p>
{language_matrix()}
<h2>How the project grows</h2>
<ol>
<li><b>Cover the book.</b> One chapter at a time, in the book's order. Concepts first, then code in every language.</li>
<li><b>Extras.</b> Deeper dives that the book only hints at: worked examples, comparisons, bonus topics. They live next to their chapter.</li>
<li><b>More languages.</b> Adding a language means filling its folder with the same modules and one column in the tables.</li>
</ol>
<p>Want to help or extend it? <a href="contributing.html">Contributing</a> has the exact steps.</p>
"""
    return dict(slug="roadmap", title="Roadmap", nav="Roadmap", chapter="start", body=body, generated=True)


def cards():
    out = []
    for ch in S.CHAPTERS:
        label = f"Chapter {ch['num']}" if str(ch["num"]).isdigit() else "Appendices"
        out.append(
            f'<a class="card" href="{chapter_slug(ch["num"])}.html"><b>{label}</b>'
            f'<span>{html.escape(ch["title"])}</span>{chip(chapter_status(ch))}</a>'
        )
    return f'<div class="cards">{"".join(out)}</div>'


# ------------------------------------------------------------------ code tabs
def code_block(key):
    spec = S.CODE[key]
    bar = "".join(f'<button data-lang="{k}">{n}</button>' for k, n, _ in S.LANGS)
    panes = []
    for k, _, d in S.LANGS:
        parts = []
        for rel in spec.get(k, []):
            path = f"{d}/{rel}"
            text = (ROOT / path).read_text(encoding="utf-8")
            parts.append(
                f'<div class="file">{html.escape(path)}</div>'
                f'<pre><code class="language-{S.HLJS_LANG[k]}">{html.escape(text)}</code></pre>'
            )
        if not parts:
            parts.append('<p class="muted pad">Not implemented in this language yet.</p>')
        panes.append(f'<div class="pane" data-lang="{k}">{"".join(parts)}</div>')
    return f'<div class="tabs"><div class="tabbar">{bar}</div>{"".join(panes)}</div>'


# ------------------------------------------------------------------ assembling
def all_pages():
    """Every page in reading order: start, chapters (+ extras), reference."""
    out = []
    generated = {chapter_slug(ch["num"]): render_chapter_page(ch) for ch in S.CHAPTERS}
    roadmap = render_roadmap()
    for p in pages_of("start"):
        out.append(p)
        if p["slug"] == "index":
            out.append(roadmap)
    for ch in S.CHAPTERS:
        out.append(generated[chapter_slug(ch["num"])])
        out.extend(pages_of(ch["num"]))
        out.extend(pages_of(ch["num"], extra=True))
    out.extend(pages_of("reference"))
    return out


def nav_html(pages, current):
    def link(p, cls=""):
        on = " on" if p["slug"] == current else ""
        return f'<a class="{cls}{on}" href="{p["slug"]}.html">{html.escape(p["nav"])}</a>'

    parts = ["<h4>Start</h4>"]
    for p in pages:
        if p.get("chapter") == "start":
            parts.append(link(p))
    for ch in S.CHAPTERS:
        num = ch["num"]
        status = chapter_status(ch)
        head = next(p for p in pages if p["slug"] == chapter_slug(num))
        label = f"Chapter {num}" if str(num).isdigit() else "Appendices"
        parts.append(f'<h4><a class="h4link{" on" if head["slug"] == current else ""}" href="{head["slug"]}.html">{label}</a>'
                     f'{"" if status == "ready" else f" <em>{status}</em>"}</h4>')
        if status == "planned":
            continue
        for p in pages_of(num):
            parts.append(link(p))
        for p in pages_of(num, extra=True):
            parts.append(link(p, "extra"))
    parts.append("<h4>Reference</h4>")
    for p in pages:
        if p.get("chapter") == "reference":
            parts.append(link(p))
    parts.append(f'<h4>Source</h4><a href="{S.THIS_REPO}">This repo</a><a href="{S.BOOK_REPO[:-len("/tree/main")]}">Raschka\'s repo</a>')
    return "\n".join(parts)


def meta_line(p):
    bits = []
    if p.get("sections") and not p.get("generated"):
        bits.append(f"Book section {', '.join(p['sections'])}")
    if p.get("code"):
        bits.append("code in " + ", ".join(LANG_LABEL[k] for k, _, _ in S.LANGS if S.CODE[p["code"]].get(k)))
    if p.get("extra"):
        bits.append("extra")
    return f'<p class="meta">{" · ".join(bits)}</p>' if bits else ""


def render(p, pages, i):
    body = p["body"]
    if p.get("code") and "{{code}}" in body:
        body = body.replace("{{code}}", code_block(p["code"]))
    body = body.replace("{{progress}}", progress_table()).replace("{{cards}}", cards()).replace("{{matrix}}", language_matrix())
    prev = f'<a href="{pages[i-1]["slug"]}.html">&larr; {html.escape(pages[i-1]["nav"])}</a>' if i > 0 else "<span></span>"
    nxt = f'<a href="{pages[i+1]["slug"]}.html">{html.escape(pages[i+1]["nav"])} &rarr;</a>' if i < len(pages) - 1 else "<span></span>"
    edit = ""
    if not p.get("generated"):
        edit = f'<p class="edit"><a href="{S.THIS_REPO}/blob/main/docs/pages/{p["slug"]}.html">Edit this page on GitHub</a></p>'
    title = html.escape(p["title"])
    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="style.css">
<link rel="stylesheet" href="{HLJS}/styles/github.min.css" media="(prefers-color-scheme: light)">
<link rel="stylesheet" href="{HLJS}/styles/github-dark.min.css" media="(prefers-color-scheme: dark)">
</head><body><div class="layout">
<nav class="side"><a class="brand" href="index.html">LLM from scratch</a>
{nav_html(pages, p["slug"])}</nav>
<main><h1>{title}</h1>
{meta_line(p)}
{body}
{edit}
<div class="pager">{prev}{nxt}</div>
</main></div>
<script src="{HLJS}/highlight.min.js"></script>
<script src="{HLJS}/languages/go.min.js"></script>
<script src="{HLJS}/languages/rust.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/highlightjs-zig@1.0.0/dist/zig.min.js"></script>
<script src="site.js"></script>
</body></html>"""


def main():
    for old in HERE.glob("*.html"):
        old.unlink()  # drop pages that no longer exist
    pages = all_pages()
    for i, p in enumerate(pages):
        (HERE / f"{p['slug']}.html").write_text(render(p, pages, i), encoding="utf-8")
    print(f"wrote {len(pages)} pages to {HERE}")


main()
