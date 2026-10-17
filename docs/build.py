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
import i18n

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
PAGES_DIR = HERE / "pages"
LANG = "en"  # language being rendered; set by main()
T = i18n.UI["en"]

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


EN_PAGES = sorted((parse_page(p) for p in PAGES_DIR.glob("*.html")), key=lambda p: p["order"])


def load_pages(lang):
    if lang == "en":
        return EN_PAGES
    out = []
    for en in EN_PAGES:
        f = PAGES_DIR / lang / f"{en['slug']}.html"
        if f.exists():
            p = parse_page(f)
            for k in ("chapter", "section", "order", "code", "extra"):  # structure comes from the English page
                p[k] = en[k] if k in en else p.get(k)
            p["sections"], p["extra"] = en["sections"], en["extra"]
            out.append(p)
        else:
            out.append(dict(en, fallback=True))
    return out


PAGES = EN_PAGES


def pages_of(chapter, extra=False):
    return [p for p in PAGES if p.get("chapter") == chapter_key(chapter) and p["extra"] == extra]


# ---------------------------------------------------------------- localised syllabus
def ch_title(ch):
    if LANG == "fa":
        return i18n.FA_CHAPTERS.get(ch["num"], {}).get("title", ch["title"])
    return ch["title"]


def ch_summary(ch):
    if LANG == "fa":
        return i18n.FA_CHAPTERS.get(ch["num"], {}).get("summary", ch["summary"])
    return ch["summary"]


def sec_title(ch, sid, default):
    if LANG == "fa":
        return i18n.FA_CHAPTERS.get(ch["num"], {}).get("sections", {}).get(sid, default)
    return default


def ch_label(ch):
    return f"{T['chapter']} {ch['num']}" if str(ch["num"]).isdigit() else T["appendices"]


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
    label = {"ready": T["s_ready"], "in progress": T["s_progress"], "planned": T["s_planned"]}[status]
    return f'<span class="chip {cls}">{label}</span>'


def lang_badges(langs):
    if not langs:
        return f'<span class="muted">{T["none_yet"]}</span>'
    return " ".join(f'<span class="lang">{html.escape(LANG_LABEL[k])}</span>' for k in langs)


def book_link(ch):
    n = ch["num"]
    folder = f"ch{int(n):02d}" if str(n).isdigit() else "appendix-A"
    return f"{S.BOOK_REPO}/{folder}"


def render_chapter_page(ch):
    num = ch["num"]
    status = chapter_status(ch)
    page_of = {}
    for p in pages_of(num):
        for s in p["sections"]:
            page_of[s] = p
    rows = []
    for sid, title in ch["sections"]:
        p = page_of.get(sid)
        state = f'{chip("ready")} <a href="{p["slug"]}.html">{T["open"]}</a>' if p else chip("planned")
        rows.append(f'<tr><td dir="ltr">{sid}</td><td>{html.escape(sec_title(ch, sid, title))}</td><td>{state}</td></tr>')
    if rows:
        table = f'<table><tr><th>{T["th_book"]}</th><th>{T["th_section"]}</th><th>{T["th_status"]}</th></tr>' + "".join(rows) + "</table>"
    else:
        table = f'<p class="muted">{T["no_sections"]}</p>'
    extras = pages_of(num, extra=True)
    extras_html = ""
    if extras:
        items = "".join(f'<li><a href="{p["slug"]}.html">{html.escape(p["nav"])}</a></li>' for p in extras)
        extras_html = f'<h2>{T["extras_h"]}</h2><ul>{items}</ul>'
    code_line = f'<p><b>{T["code_label"]}</b> {lang_badges(ch.get("langs", []))}</p>' if ch["code"] else ""
    planned_note = f'<div class="tip">{T["planned_note"]}</div>' if status == "planned" else ""
    body = f"""
<p class="lede">{html.escape(ch_summary(ch))}</p>
<p>{chip(status)} &nbsp; <a href="{book_link(ch)}">{T["companion"]}</a></p>
{code_line}
{planned_note}
<h2>{T["sections_h"]}</h2>
{table}
{extras_html}
"""
    label = ch_label(ch)
    title = f"{label}: {ch_title(ch)}" if str(num).isdigit() else ch_title(ch)
    return dict(slug=chapter_slug(num), title=title, nav=label, chapter=chapter_key(num), body=body, generated=True)


def progress_table():
    rows = []
    for ch in S.CHAPTERS:
        total = len(ch["sections"])
        ready = len(ready_sections(ch) & {s for s, _ in ch["sections"]})
        pct = round(100 * ready / total) if total else 0
        bar = f'<div class="bar" role="progressbar" aria-valuenow="{pct}" aria-valuemin="0" aria-valuemax="100"><i style="width:{pct}%"></i></div>'
        count = f"{ready}/{total} {T['sections_word']}" if total else "&ndash;"
        rows.append(
            f'<tr><td><a href="{chapter_slug(ch["num"])}.html">{ch_label(ch)}</a></td><td>{html.escape(ch_title(ch))}</td>'
            f'<td>{bar}<small>{count}</small></td><td>{chip(chapter_status(ch))}</td></tr>'
        )
    return f'<table><tr><th></th><th>{T["th_topic"]}</th><th>{T["th_written"]}</th><th>{T["th_status"]}</th></tr>' + "".join(rows) + "</table>"


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
        rows.append(f"<tr><td>{ch_label(ch)}</td><td>{html.escape(ch_title(ch))}</td>{cells}</tr>")
    return f"<table class='matrix'><tr><th></th><th>{T['th_topic']}</th>{head}</tr>{''.join(rows)}</table>"


def render_roadmap():
    body = T["roadmap_body"].replace("{progress}", progress_table()).replace("{matrix}", language_matrix())
    return dict(slug="roadmap", title=T["roadmap_title"], nav=T["roadmap_title"], chapter="start", body=body, generated=True)


def cards():
    out = []
    for ch in S.CHAPTERS:
        out.append(
            f'<a class="card" href="{chapter_slug(ch["num"])}.html"><b>{ch_label(ch)}</b>'
            f'<span>{html.escape(ch_title(ch))}</span>{chip(chapter_status(ch))}</a>'
        )
    return f'<div class="cards">{"".join(out)}</div>'


# ------------------------------------------------------------------ code tabs
def code_block(key, uid):
    spec = S.CODE[key]
    bar = "".join(
        f'<button role="tab" type="button" id="t-{uid}-{k}" aria-controls="p-{uid}-{k}" data-lang="{k}">{n}</button>'
        for k, n, _ in S.LANGS
    )
    panes = []
    for k, _, d in S.LANGS:
        parts = []
        for rel in spec.get(k, []):
            path = f"{d}/{rel}"
            text = (ROOT / path).read_text(encoding="utf-8")
            parts.append(
                f'<div class="file" dir="ltr">{html.escape(path)}</div>'
                f'<pre dir="ltr"><code class="language-{S.HLJS_LANG[k]}">{html.escape(text)}</code></pre>'
            )
        if not parts:
            parts.append(f'<p class="muted pad">{T["not_impl"]}</p>')
        panes.append(f'<div class="pane" role="tabpanel" id="p-{uid}-{k}" aria-labelledby="t-{uid}-{k}" data-lang="{k}">{"".join(parts)}</div>')
    return f'<div class="tabs"><div class="tabbar" role="tablist" dir="ltr">{bar}</div>{"".join(panes)}</div>'


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
        cur = ' aria-current="page"' if on else ""
        mark = f' <span class="en-mark" title="English">{T["untranslated"]}</span>' if p.get("fallback") else ""
        return f'<a class="{cls}{on}" href="{p["slug"]}.html"{cur}>{html.escape(p["nav"])}{mark}</a>'

    parts = [f'<h4>{T["h_start"]}</h4>']
    for p in pages:
        if p.get("chapter") == "start":
            parts.append(link(p))
    for ch in S.CHAPTERS:
        num = ch["num"]
        status = chapter_status(ch)
        head = next(p for p in pages if p["slug"] == chapter_slug(num))
        state = "" if status == "ready" else f' <em>{ {"in progress": T["s_progress"], "planned": T["s_planned"]}[status] }</em>'
        parts.append(f'<h4><a class="h4link{" on" if head["slug"] == current else ""}" href="{head["slug"]}.html">{ch_label(ch)}</a>{state}</h4>')
        if status == "planned":
            continue
        for p in pages_of(num):
            parts.append(link(p))
        for p in pages_of(num, extra=True):
            parts.append(link(p, "extra"))
    parts.append(f'<h4>{T["h_reference"]}</h4>')
    for p in pages:
        if p.get("chapter") == "reference":
            parts.append(link(p))
    parts.append(f'<h4>{T["h_source"]}</h4><a href="{S.THIS_REPO}">{T["this_repo"]}</a><a href="{S.BOOK_REPO[:-len("/tree/main")]}">{T["book_repo"]}</a>')
    return "\n".join(parts)


def meta_line(p):
    bits = []
    if p.get("sections") and not p.get("generated"):
        bits.append(f'{T["book_section"]} <span dir="ltr">{", ".join(p["sections"])}</span>')
    if p.get("code"):
        bits.append(T["code_in"] + " " + ", ".join(LANG_LABEL[k] for k, _, _ in S.LANGS if S.CODE[p["code"]].get(k)))
    if p.get("extra"):
        bits.append(T["extra"])
    return f'<p class="meta">{" · ".join(bits)}</p>' if bits else ""


def add_toc(body):
    """Give every h2 an id and return (body, toc_html)."""
    items = []

    def sub(m):
        n = len(items) + 1
        items.append((n, m.group(2)))
        return f'<h2 id="s{n}"{m.group(1)}>{m.group(2)}</h2>'

    body = re.sub(r"<h2([^>]*)>(.*?)</h2>", sub, body, flags=re.S)
    if len(items) < 3:
        return body, ""
    links = "".join(f'<a href="#s{n}">{re.sub(r"<[^>]+>", "", t)}</a>' for n, t in items)
    return body, f'<aside class="toc" aria-label="{T["toc"]}"><h4>{T["toc"]}</h4>{links}</aside>'


def wrap_tables(body):
    return re.sub(r"(<table.*?</table>)", r'<div class="tbl">\1</div>', body, flags=re.S)


def render(p, pages, i):
    body = p["body"]
    if p.get("code") and "{{code}}" in body:
        body = body.replace("{{code}}", code_block(p["code"], p["slug"]))
    body = body.replace("{{progress}}", progress_table()).replace("{{cards}}", cards()).replace("{{matrix}}", language_matrix())
    body, toc = add_toc(body)
    body = wrap_tables(body)
    rtl = LANG == "fa"
    la, ra = ("&rarr;", "&larr;") if rtl else ("&larr;", "&rarr;")

    def pg(q, kind):
        arrow = la if kind == "prev" else ra
        label = T["prev"] if kind == "prev" else T["next"]
        inner = f'<small>{arrow if kind == "prev" else ""} {label} {arrow if kind == "next" else ""}</small><span>{html.escape(q["nav"])}</span>'
        return f'<a class="pg {kind}" href="{q["slug"]}.html" rel="{kind}">{inner}</a>'

    prev = pg(pages[i - 1], "prev") if i > 0 else "<span></span>"
    nxt = pg(pages[i + 1], "next") if i < len(pages) - 1 else "<span></span>"
    edit = ""
    if not p.get("generated"):
        sub = f"pages/{LANG}/" if LANG != "en" and not p.get("fallback") else "pages/"
        edit = f'<p class="edit"><a href="{S.THIS_REPO}/blob/main/docs/{sub}{p["slug"]}.html">{T["edit"]}</a></p>'
    notice = ""
    content_dir = ""
    if p.get("fallback"):
        notice = f'<div class="tip notice" dir="rtl" lang="fa">{T["fallback"]}</div>'
        content_dir = ' dir="ltr" lang="en" class="en-content"'
    title = html.escape(p["title"])
    h1 = title
    if rtl:  # keep a leading section number ("2.7-2.8:") in reading order
        h1 = re.sub(r"^([\d.,\-\s\u0648]+:)", r'<bdi dir="ltr">\1</bdi>', title)
    root = "../" if LANG != "en" else ""
    other = "en" if rtl else "fa"
    other_href = ("../" if rtl else "fa/") + f'{p["slug"]}.html'
    L = i18n.LANGS_UI[LANG]
    O = i18n.LANGS_UI[other]
    return f"""<!doctype html>
<html lang="{L['code']}" dir="{L['dir']}"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} · {T['site_title']}</title>
<meta name="color-scheme" content="light dark">
<script>try{{var t=localStorage.getItem("theme");if(t)document.documentElement.dataset.theme=t}}catch(e){{}}</script>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;600;700&display=swap">
<link rel="stylesheet" href="{root}style.css">
<link rel="stylesheet" href="{HLJS}/styles/github.min.css" id="hl-light">
<link rel="stylesheet" href="{HLJS}/styles/github-dark.min.css" id="hl-dark" disabled>
<link rel="alternate" hreflang="{O['code']}" href="{other_href}">
</head><body>
<a class="skip" href="#main">{T['skip']}</a>
<header class="top">
<button class="icon-btn menu-btn" type="button" aria-label="{T['menu']}" aria-expanded="false" aria-controls="side"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><path d="M4 7h16M4 12h16M4 17h16"/></svg></button>
<a class="brand" href="index.html">{T['brand']}</a>
<span class="grow"></span>
<a class="lang-btn" href="{other_href}" hreflang="{O['code']}" lang="{O['code']}" aria-label="{T['lang_switch']}: {O['name']}">{O['name']}</a>
<button class="icon-btn theme-btn" type="button" aria-label="{T['theme']}"><svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg></button>
</header>
<div class="layout">
<nav class="side" id="side" aria-label="{T['brand']}">
<input class="filter" type="search" placeholder="{T['filter_ph']}" aria-label="{T['filter']}" data-nomatch="{T['no_match']}">
<div class="navlinks">{nav_html(pages, p["slug"])}</div>
</nav>
<div class="scrim" hidden></div>
<main id="main" tabindex="-1"><h1>{h1}</h1>
{meta_line(p)}
{notice}
<div{content_dir or ''}>{body}</div>
{edit}
<div class="pager">{prev}{nxt}</div>
</main>
{toc}
</div>
<script src="{HLJS}/highlight.min.js"></script>
<script src="{HLJS}/languages/go.min.js"></script>
<script src="{HLJS}/languages/rust.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/highlightjs-zig@1.0.0/dist/zig.min.js"></script>
<script>window.I18N={{copy:"{T['copy']}",copied:"{T['copied']}"}}</script>
<script src="{root}site.js"></script>
</body></html>"""


def build(lang):
    global LANG, T, PAGES
    LANG, T = lang, i18n.UI[lang]
    PAGES = load_pages(lang)
    out = HERE / i18n.LANGS_UI[lang]["out"]
    out.mkdir(exist_ok=True)
    for old in out.glob("*.html"):
        old.unlink()  # drop pages that no longer exist
    pages = all_pages()
    for i, p in enumerate(pages):
        (out / f"{p['slug']}.html").write_text(render(p, pages, i), encoding="utf-8")
    translated = sum(1 for p in pages if not p.get("fallback") and not p.get("generated"))
    print(f"[{lang}] wrote {len(pages)} pages to {out} ({translated} hand-written)")


def main():
    for lang in i18n.LANGS_UI:
        build(lang)


main()
