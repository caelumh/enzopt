"""Markdown page → one self-contained HTML page (figures embedded), table of contents, figure zoom.

    uv run python scripts/build_guide_html.py              # docs/GUIDE.md → docs/GUIDE.html, docs/guide/* embedded
    uv run python scripts/build_guide_html.py --link-images  # keep relative image paths (smaller file)
    uv run python scripts/build_guide_html.py --page identification  # docs/identification/README.md → .html

Handles the Markdown that the pages use: headings, paragraphs, bold/italic/code/links, nested lists, tables,
block quotes, fenced code, rules and stand-alone images (rendered as figures; a long alt text becomes the caption).
No third-party packages.
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import html
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES = {
    "guide": {
        "src": ROOT / "docs" / "GUIDE.md",
        "out": ROOT / "docs" / "GUIDE.html",
        "label": "안내서",
        "eyebrow": "enzopt · 사용 안내",
        "description": "enzopt의 다섯 단계 계산과 그림 읽는 법, 피펫 표, 실측 데이터 시험 결과를 담은 안내서",
    },
    "identification": {
        "src": ROOT / "docs" / "identification" / "README.md",
        "out": ROOT / "docs" / "IDENTIFICATION.html",
        "label": "식별 레퍼런스",
        "eyebrow": "enzopt · 레퍼런스",
        "description": "반응식과 A340 진행 곡선에서 속도 상수 표본을 찾고 설계로 잇는 과정을 그림 9장으로 보여 주는 "
        "레퍼런스",
    },
}

LIST_RE = re.compile(r"^(\s*)([-*]|\d+\.)\s+(.*)$")
HEAD_RE = re.compile(r"^(#{1,6})\s+(.*)$")
IMG_RE = re.compile(r"^!\[(.*)\]\(([^)\s]+)\)$")
SEP_RE = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
MIME = {".png": "image/png", ".svg": "image/svg+xml", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


# ---------------------------------------------------------------- inline
def inline(s: str) -> str:
    stash: list[str] = []

    def keep(h: str) -> str:
        stash.append(h)
        return f"\x01{len(stash) - 1}\x02"

    s = re.sub(
        r"\[([^\]]+)\]\(([^)\s]+)\)",
        lambda m: keep(f'<a href="{html.escape(m.group(2))}">{inline(m.group(1))}</a>'),
        s,
    )
    s = re.sub(r"`([^`]+)`", lambda m: keep(f"<code>{html.escape(m.group(1))}</code>"), s)
    s = s.replace("\\|", "|").replace("<br>", keep("<br>"))
    s = html.escape(s, quote=False)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<![\w*])\*([A-Za-zℓ]{1,3})\*(?![\w*])", r"<em>\1</em>", s)
    while "\x01" in s:
        s = re.sub(r"\x01(\d+)\x02", lambda m: stash[int(m.group(1))], s)
    return s


def plain(s: str) -> str:
    """Heading text without markup (for the table of contents and anchors)."""
    s = re.sub(r"\s*\(`[^`]*`(?:,\s*`[^`]*`)*\)\s*$", "", s)
    return re.sub(r"[`*]", "", s).strip()


def split_row(line: str) -> list[str]:
    line = line.strip()
    line = line[1:] if line.startswith("|") else line
    line = line[:-1] if line.endswith("|") and not line.endswith("\\|") else line
    cells, cur, code = [], "", False
    for i, ch in enumerate(line):
        if ch == "`":
            code = not code
        if ch == "|" and not code and (i == 0 or line[i - 1] != "\\"):
            cells.append(cur.strip())
            cur = ""
        else:
            cur += ch
    cells.append(cur.strip())
    return cells


# ---------------------------------------------------------------- blocks
class Builder:
    def __init__(self, src: Path, embed: bool, eyebrow: str = "enzopt · 사용 안내", out_dir: Path | None = None):
        self.src = src
        self.embed = embed
        self.out_dir = out_dir or src.parent
        self.eyebrow = eyebrow
        self.toc: list[tuple[int, str, str]] = []
        self.ids: set[str] = set()
        self.title = "enzopt 안내서"
        self.n_fig = 0

    def anchor(self, text: str) -> str:
        base = re.sub(r"[^\w가-힣]+", "-", plain(text)).strip("-").lower() or "s"
        a, k = base, 2
        while a in self.ids:
            a, k = f"{base}-{k}", k + 1
        self.ids.add(a)
        return a

    def image(self, alt: str, src: str) -> str:
        self.n_fig += 1
        path = (self.src.parent / src).resolve()
        if self.embed and path.exists():
            data = base64.b64encode(path.read_bytes()).decode("ascii")
            url = f"data:{MIME.get(path.suffix.lower(), 'application/octet-stream')};base64,{data}"
        else:
            url = Path(os.path.relpath(path, self.out_dir)).as_posix()  # relative to the page, not the Markdown
        short = re.fullmatch(r"그림 \w+", alt.strip()) is not None
        cap = ""
        if not short:
            m = re.match(r"^(그림 \w+)\s+—\s+(.*)$", alt)
            cap = f"<b>{html.escape(m.group(1))}</b> {inline(m.group(2))}" if m else inline(alt)
            cap = f"<figcaption>{cap}</figcaption>"
        kind = " diagram" if path.suffix.lower() == ".svg" else ""
        return (
            f'<figure class="fig{kind}"><button class="zoom" type="button" aria-label="크게 보기">'
            f'<img src="{url}" alt="{html.escape(alt)}" loading="lazy" decoding="async"></button>{cap}</figure>'
        )

    def table(self, rows: list[str]) -> str:
        head = split_row(rows[0])
        body = [split_row(r) for r in rows[2:]]
        wide = " wide" if len(head) > 6 else ""
        h = "".join(f"<th>{inline(c)}</th>" for c in head)
        b = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body)
        return f'<div class="table-wrap{wide}"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'

    def lst(self, lines: list[str], i: int, indent: int) -> tuple[str, int]:
        m = LIST_RE.match(lines[i])
        ordered = m.group(2)[0].isdigit()
        start = int(m.group(2)[:-1]) if ordered else 1
        items: list[str] = []
        while i < len(lines):
            line = lines[i]
            m = LIST_RE.match(line)
            if not m:
                if line.strip() and items and len(line) - len(line.lstrip()) > indent:
                    items[-1] += " " + inline(line.strip())
                    i += 1
                    continue
                break
            ind = len(m.group(1))
            if ind < indent:
                break
            if ind > indent:
                sub, i = self.lst(lines, i, ind)
                items[-1] += sub
                continue
            items.append(inline(m.group(3)))
            i += 1
        tag = "ol" if ordered else "ul"
        st = f' start="{start}"' if ordered and start != 1 else ""
        return f"<{tag}{st}>" + "".join(f"<li>{x}</li>" for x in items) + f"</{tag}>", i

    def build(self, md: str) -> str:
        lines = md.splitlines()
        out: list[str] = []
        para: list[str] = []
        i = 0

        def flush():
            if para:
                out.append(f"<p>{inline(' '.join(para))}</p>")
                para.clear()

        def starts_block(line: str) -> bool:
            return bool(
                HEAD_RE.match(line)
                or LIST_RE.match(line)
                or IMG_RE.match(line.strip())
                or line.startswith(("```", ">", "|"))
                or line.strip() == "---"
            )

        while i < len(lines):
            line = lines[i]
            st = line.strip()
            if not st:
                flush()
                i += 1
                continue
            if st.startswith("```"):
                flush()
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith("```"):
                    j += 1
                code = "\n".join(lines[i + 1 : j])
                out.append(f"<pre><code>{html.escape(code)}</code></pre>")
                i = j + 1
                continue
            m = HEAD_RE.match(line)
            if m:
                flush()
                level, text = len(m.group(1)), m.group(2).strip()
                if level == 1:
                    self.title = plain(text)
                    lead = ""
                    j = i + 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    if j < len(lines) and lines[j].startswith(">"):
                        lead = f'<p class="lead">{inline(lines[j].lstrip("> ").strip())}</p>'
                        i = j
                    out.append(
                        f'<header class="hero"><p class="eyebrow">{html.escape(self.eyebrow)}</p>'
                        f"<h1>{inline(text)}</h1>{lead}</header>"
                    )
                    i += 1
                    continue
                a = self.anchor(text)
                if level in (2, 3):
                    self.toc.append((level, a, plain(text)))
                body = inline(text)
                fm = re.match(r"^그림 (\w+) — (.*)$", text)
                if fm and level == 3:
                    body = f'<span class="fig-no">그림 {html.escape(fm.group(1))}</span> {inline(fm.group(2))}'
                cls = ' class="fig-head"' if fm and level == 3 else ""
                out.append(
                    f'<h{level} id="{a}"{cls}><a class="anchor" href="#{a}" aria-hidden="true">#</a>{body}</h{level}>'
                )
                i += 1
                continue
            if st == "---":
                flush()
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j >= len(lines) or not lines[j].startswith("## "):  # h2 already carries a rule
                    out.append("<hr>")
                i += 1
                continue
            m = IMG_RE.match(st)
            if m:
                flush()
                out.append(self.image(m.group(1), m.group(2)))
                i += 1
                continue
            if line.startswith(">"):
                flush()
                q = []
                while i < len(lines) and lines[i].startswith(">"):
                    q.append(lines[i].lstrip(">").strip())
                    i += 1
                out.append(f"<blockquote><p>{inline(' '.join(q))}</p></blockquote>")
                continue
            if st.startswith("|") and i + 1 < len(lines) and SEP_RE.match(lines[i + 1].strip()):
                flush()
                rows = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    rows.append(lines[i])
                    i += 1
                out.append(self.table(rows))
                continue
            if LIST_RE.match(line):
                flush()
                h, i = self.lst(lines, i, len(LIST_RE.match(line).group(1)))
                out.append(h)
                continue
            if para and starts_block(line):
                flush()
                continue
            para.append(st)
            i += 1
        flush()
        return "\n".join(out)

    def toc_html(self) -> str:
        items, open_sub = [], False
        for level, a, text in self.toc:
            if level == 2:
                if open_sub:
                    items.append("</ul></li>")
                    open_sub = False
                elif items:
                    items.append("</li>")
                items.append(f'<li><a href="#{a}">{html.escape(text)}</a>')
            else:
                if not open_sub:
                    items.append("<ul>")
                    open_sub = True
                items.append(f'<li><a href="#{a}">{html.escape(text)}</a></li>')
        items.append("</ul></li>" if open_sub else "</li>")
        return "<ul>" + "".join(items) + "</ul>"


CSS = """
:root{--bg:#faf9f6;--surface:#ffffff;--text:#1f2328;--muted:#5d646d;--border:#e4e1da;--accent:#0b6aa8;
--accent-soft:#e7f0f8;--code-bg:#f2f0ea;--quote-bg:#eef5fb;--th-bg:#f2f0ea;--row-alt:#fbfaf7;--fig-bg:#ffffff;
--shadow:0 1px 2px rgba(0,0,0,.04),0 4px 16px rgba(0,0,0,.05);color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#121417;--surface:#1a1d21;--text:#e6e7e9;
--muted:#9ba2ab;--border:#2b2f35;--accent:#6fb8ef;--accent-soft:#1b2834;--code-bg:#22262c;--quote-bg:#172431;
--th-bg:#22262c;--row-alt:#171a1e;--fig-bg:#ffffff;--shadow:none;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#121417;--surface:#1a1d21;--text:#e6e7e9;--muted:#9ba2ab;--border:#2b2f35;
--accent:#6fb8ef;--accent-soft:#1b2834;--code-bg:#22262c;--quote-bg:#172431;--th-bg:#22262c;--row-alt:#171a1e;
--fig-bg:#ffffff;--shadow:none;color-scheme:dark}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:72px}
body{margin:0;background:var(--bg);color:var(--text);font:16.5px/1.78 Pretendard,"Pretendard Variable","Noto Sans KR",
"Apple SD Gothic Neo","Malgun Gothic",system-ui,sans-serif;word-break:keep-all;overflow-wrap:break-word;
-webkit-text-size-adjust:100%}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
code{font-family:"JetBrains Mono",Consolas,"D2Coding",ui-monospace,monospace;font-size:.86em;background:var(--code-bg);
padding:.08em .38em;border-radius:5px;word-break:break-all}
pre{background:var(--code-bg);border:1px solid var(--border);border-radius:10px;padding:14px 16px;overflow-x:auto;
line-height:1.55;font-size:13.5px}
pre code{background:none;padding:0;font-size:inherit;word-break:normal;white-space:pre}
.topbar{position:sticky;top:0;z-index:20;display:flex;align-items:center;gap:12px;height:56px;padding:0 16px;
background:color-mix(in srgb,var(--bg) 88%,transparent);backdrop-filter:blur(8px);border-bottom:1px solid var(--border)}
.topbar .brand{font-weight:700;letter-spacing:-.01em;color:var(--text)}
.topbar .brand span{color:var(--muted);font-weight:500;margin-left:6px}
.topbar .spacer{flex:1}
.btn{font:inherit;font-size:14px;color:var(--text);background:var(--surface);border:1px solid var(--border);
border-radius:8px;padding:5px 12px;cursor:pointer}
.btn:hover{border-color:var(--accent)}
.layout{display:grid;grid-template-columns:minmax(0,820px);justify-content:center;max-width:1240px;margin:0 auto;
padding:0 16px}
aside.toc{display:none}
main{min-width:0;padding-bottom:80px}
.hero{padding:44px 0 18px;border-bottom:1px solid var(--border);margin-bottom:8px}
.hero .eyebrow{margin:0 0 6px;color:var(--accent);font-weight:600;font-size:14px;letter-spacing:.02em}
.hero h1{margin:0;font-size:clamp(30px,5vw,42px);line-height:1.2;letter-spacing:-.02em}
.hero .lead{margin:14px 0 0;font-size:clamp(17px,2.4vw,20px);color:var(--muted)}
h2,h3,h4{position:relative;line-height:1.35;letter-spacing:-.01em}
h2{font-size:26px;margin:64px 0 14px;padding-top:18px;border-top:2px solid var(--text)}
h3{font-size:20px;margin:40px 0 10px}
h4{font-size:17px;margin:26px 0 6px;color:var(--muted)}
h2 code,h3 code{font-size:.62em;color:var(--muted);vertical-align:middle}
.anchor{position:absolute;left:-1.1em;opacity:0;color:var(--muted);font-weight:400;text-decoration:none}
h2:hover .anchor,h3:hover .anchor{opacity:1}
.fig-head{padding-top:6px}
.fig-no{display:inline-block;font-size:.72em;font-weight:700;color:#fff;background:var(--accent);border-radius:999px;
padding:.12em .7em;margin-right:.35em;vertical-align:.15em}
:root[data-theme="dark"] .fig-no{color:#0d1116}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]) .fig-no{color:#0d1116}}
p{margin:12px 0}
p>strong:first-child{color:var(--accent)}
ul,ol{padding-left:1.4em;margin:10px 0}li{margin:4px 0}li>ul,li>ol{margin:4px 0}
blockquote{margin:18px 0;padding:12px 18px;background:var(--quote-bg);border-left:4px solid var(--accent);
border-radius:0 10px 10px 0}
blockquote p{margin:0}
hr{border:0;border-top:1px dashed var(--border);margin:44px 0}
.table-wrap{overflow-x:auto;margin:16px 0;border:1px solid var(--border);border-radius:10px;background:var(--surface)}
table{border-collapse:collapse;width:100%;font-size:14.5px;line-height:1.55}
th,td{padding:8px 12px;text-align:left;vertical-align:top;border-bottom:1px solid var(--border)}
th{background:var(--th-bg);font-weight:600;white-space:nowrap}
tbody tr:nth-child(even){background:var(--row-alt)}
tbody tr:last-child td{border-bottom:0}
.table-wrap.wide table{font-size:13px}.table-wrap.wide td{white-space:nowrap}
td:first-child code{white-space:nowrap;word-break:normal}
figure.fig{margin:26px 0}
figure.fig .zoom{display:block;width:100%;padding:12px;margin:0;background:var(--fig-bg);border:1px solid var(--border);
border-radius:12px;box-shadow:var(--shadow);cursor:zoom-in}
figure.fig img{display:block;width:100%;height:auto}
figcaption{margin-top:10px;font-size:14.5px;line-height:1.65;color:var(--muted)}
figcaption b{color:var(--text);margin-right:.3em}
.lightbox{position:fixed;inset:0;z-index:50;display:flex;align-items:center;justify-content:center;padding:16px;
background:rgba(10,12,15,.86);cursor:zoom-out}
.lightbox[hidden]{display:none}
.lightbox img{max-width:min(1600px,100%);max-height:100%;background:#fff;border-radius:8px;padding:8px}
details.toc-mobile{margin:18px 0 0;border:1px solid var(--border);border-radius:10px;background:var(--surface);
padding:8px 14px}
details.toc-mobile summary{cursor:pointer;font-weight:600}
.toc ul,.toc-mobile ul{list-style:none;padding-left:0;margin:6px 0}
.toc ul ul,.toc-mobile ul ul{padding-left:12px;margin:2px 0 6px;border-left:1px solid var(--border)}
.toc a,.toc-mobile a{display:block;padding:3px 8px;border-radius:6px;color:var(--muted);font-size:14px;line-height:1.45}
.toc a:hover,.toc-mobile a:hover{color:var(--text);background:var(--accent-soft);text-decoration:none}
.toc a.active{color:var(--accent);background:var(--accent-soft);font-weight:600}
footer{margin-top:56px;padding-top:16px;border-top:1px solid var(--border);color:var(--muted);font-size:13.5px}
@media (min-width:1100px){
 .layout{grid-template-columns:260px minmax(0,860px);gap:48px}
 aside.toc{display:block;position:sticky;top:56px;align-self:start;max-height:calc(100vh - 56px);overflow-y:auto;
 padding:28px 4px 40px 0}
 aside.toc .toc-title{font-size:12px;font-weight:700;letter-spacing:.08em;color:var(--muted);margin:0 8px 8px}
 details.toc-mobile{display:none}
}
@media (max-width:640px){body{font-size:16px}h2{font-size:22px;margin-top:48px}h3{font-size:18px}
 figure.fig .zoom{padding:6px;border-radius:10px}.anchor{display:none}}
@media print{.topbar,aside.toc,details.toc-mobile,.lightbox{display:none!important}
 body{background:#fff;color:#000}figure.fig{break-inside:avoid}h2,h3{break-after:avoid}}
"""

JS = """
(function(){
 var root=document.documentElement,btn=document.getElementById('theme'),modes=['auto','light','dark'],
 names={auto:'테마: 자동',light:'테마: 밝게',dark:'테마: 어둡게'};
 function get(){try{return localStorage.getItem('enzopt-guide-theme')||'auto'}catch(e){return 'auto'}}
 function apply(m){if(m==='auto'){root.removeAttribute('data-theme')}else{root.setAttribute('data-theme',m)}
  btn.textContent=names[m];try{localStorage.setItem('enzopt-guide-theme',m)}catch(e){}}
 var mode=get();apply(mode);
 btn.addEventListener('click',function(){mode=modes[(modes.indexOf(mode)+1)%3];apply(mode)});
 var lb=document.getElementById('lightbox'),lbi=lb.querySelector('img');
 document.querySelectorAll('figure.fig .zoom').forEach(function(b){b.addEventListener('click',function(){
  var im=b.querySelector('img');lbi.src=im.src;lbi.alt=im.alt;lb.hidden=false})});
 lb.addEventListener('click',function(){lb.hidden=true});
 document.addEventListener('keydown',function(e){if(e.key==='Escape')lb.hidden=true});
 var links={};document.querySelectorAll('aside.toc a').forEach(function(a){links[a.getAttribute('href').slice(1)]=a});
 var cur=null;if('IntersectionObserver' in window){var io=new IntersectionObserver(function(es){es.forEach(function(e){
  if(e.isIntersecting&&links[e.target.id]){if(cur)cur.classList.remove('active');cur=links[e.target.id];
  cur.classList.add('active')}})},{rootMargin:'-60px 0px -70% 0px'});
  document.querySelectorAll('main h2[id],main h3[id]').forEach(function(h){io.observe(h)})}
 document.querySelectorAll('details.toc-mobile a').forEach(function(a){a.addEventListener('click',function(){
  a.closest('details').open=false})});
})();
"""


def render(page: str = "guide", embed: bool = True) -> Path:
    cfg = PAGES[page]
    src, out = cfg["src"], cfg["out"]
    b = Builder(src, embed, cfg["eyebrow"], out.parent)
    body = b.build(src.read_text(encoding="utf-8"))
    toc = b.toc_html()
    today = dt.date.today().isoformat()
    label, desc, src_rel = html.escape(cfg["label"]), html.escape(cfg["description"]), src.relative_to(ROOT).as_posix()
    page = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(b.title)}</title>
<meta name="description" content="{desc}">
<style>{CSS}</style>
</head>
<body>
<div class="topbar"><a class="brand" href="#">enzopt<span>{label}</span></a><div class="spacer"></div>
<button class="btn" id="theme" type="button">테마: 자동</button></div>
<div class="layout">
<aside class="toc" aria-label="목차"><p class="toc-title">목차</p>{toc}</aside>
<main>
{body.replace("</header>", f'</header><details class="toc-mobile"><summary>목차</summary>{toc}</details>', 1)}
<footer>이 페이지는 <code>{src_rel}</code>에서 <code>scripts/build_guide_html.py</code>로 만들었습니다
 · 그림 {b.n_fig}개 · {today}</footer>
</main>
</div>
<div class="lightbox" id="lightbox" hidden><img alt=""></div>
<script>{JS}</script>
</body>
</html>
"""
    out.write_text(page, encoding="utf-8", newline="\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--page", choices=sorted(PAGES), default="guide", help="which Markdown page to build")
    ap.add_argument("--link-images", action="store_true", help="reference the image files instead of embedding them")
    a = ap.parse_args()
    out = render(a.page, embed=not a.link_images)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
