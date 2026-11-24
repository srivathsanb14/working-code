"""ONE storyline -> both decks.  Writes deck/project/slides/*.html + deck.json (online) and pptx/story.json (render.js).

Flow: 1 what is the problem -> 2 why it is hard -> 3 what people usually do, and why not here ->
      4 what we do instead, and why -> 5 how it works (physics, then the search) -> 6 does it work -> 7 next.
"""
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
P = HERE / "project"; SL = P / "slides"
import sys
sys.path.insert(0, str(HERE))
from content13 import S, SECTIONS  # noqa: E402

# =========================================================================== render: online HTML
E = html.escape


def head(eb, t):
    return f'<div class="head">\n<p class="eyebrow">{E(eb)}</p>\n<h2>{E(t)}</h2>\n</div>\n'


def ul(points, size=30):
    return f'<ul class="body" style="font-size:{size}px; line-height:1.35; margin:0; padding-left:1.1em;">' + "".join(f"<li style='margin-bottom:14px'>{E(p)}</li>" for p in points) + "</ul>"


def card(t, b, tag="", tone=""):
    cls = {"accent2": "cool", "accent3": "plain"}.get(tone, "")
    tg = f'<p class="tag">{E(tag)}</p>' if tag else ""
    return f'<div class="card {cls}">{tg}<h3>{E(t)}</h3><p class="txt">{E(b).replace(chr(10), "<br>")}</p></div>'


def callout(t):
    return f'<div class="callout"><p class="ct">{E(t)}</p></div>\n' if t else ""


def cap(t):
    return f'<p class="small" style="font-style:italic;">{E(t)}</p>\n' if t else ""


def im(f, style):
    return f'<img src="project/img/{f}" alt="{E(f)}" style="display:block; object-fit:contain; {style}">'


def table(rows, colw, header=True, mono_first=False, size=26):
    out = f'<table style="font-size:{size}px;">\n'
    if header:
        out += "<tr>" + "".join(f'<th style="width:{int(w * 100)}%">{E(c)}</th>' for c, w in zip(rows[0], colw)) + "</tr>\n"
    for r in rows[1:]:
        out += "<tr>" + "".join(f'<td{" class=mono" if (mono_first and i == 0) else ""}{" style=font-weight:600" if (not header and i == 0) else ""}>{E(c)}</td>' for i, c in enumerate(r)) + "</tr>\n"
    return out + "</table>\n"


alt = False
for s in S:
    k = s["kind"]; cls = ""
    if k in ("statement", "section"):
        cls = "dark"
        if k == "statement":
            body = (f'<div class="head" style="margin-top:150px;">\n<p class="eyebrow">{E(s["eyebrow"])}</p>\n<h1 style="font-size:80px; max-width:1500px;">{E(s["title"])}</h1>\n</div>\n'
                    f'<p class="lead" style="max-width:1400px;">{E(s.get("sub", ""))}</p>\n' + (f'<p class="small">{E(s["foot"])}</p>\n' if s.get("foot") else ""))
        else:
            body = (f'<div class="head" style="margin-top:170px;">\n<p class="eyebrow">Part {E(s["num"])}</p>\n<h1 style="font-size:96px;">{E(s["title"])}</h1>\n</div>\n'
                    f'<p class="lead" style="max-width:1400px; font-size:44px;">{E(s["question"])}</p>\n')
    else:
        alt = not alt; cls = "alt" if alt else ""
        body = head(s["eyebrow"], s["title"])
        if k == "points":
            body += ul(s["points"], 36) + callout(s.get("callout"))
        elif k == "stats":
            body += '<div class="row" style="gap:60px;">' + "".join(f'<div class="stat grow"><p class="num{" blue" if i == 2 else ""}">{E(n)}</p><p class="lbl">{E(l)}</p></div>' for i, (n, l) in enumerate(s["stats"])) + "</div>\n" + callout(s.get("callout")) + cap(s.get("caption"))
        elif k == "image":
            if s["side"] == "right":
                body += ('<div class="row" style="gap:56px; align-items:flex-start;">' + f'<div class="col" style="width:1060px; gap:10px;">{im(s["img"], "width:100%; max-height:560px;")}{cap(s.get("caption"))}</div>'
                         + f'<div class="col grow" style="gap:20px;">{ul(s["points"], 28)}{callout(s.get("callout"))}</div></div>\n')
            else:
                body += im(s["img"], "width:100%; max-height:370px;") + "\n" + cap(s.get("caption")) + ul(s["points"], 26).replace('padding-left:1.1em;"', 'padding-left:1.1em; columns:2; column-gap:60px;"') + callout(s.get("callout"))
        elif k == "cards":
            body += '<div class="cards">' + "".join(card(*c) for c in s["cards"]) + "</div>\n" + callout(s.get("callout"))
        elif k == "table":
            body += table(s["rows"], s["colw"], s.get("header", True), s.get("mono_first", False), s.get("size") or (26 if len(s["rows"]) <= 7 else 21)) + callout(s.get("callout"))
        elif k == "compare":
            body += '<div class="cards">' + card(s["left"][0], "\n".join("• " + p for p in s["left"][1]), "", "accent1") + card(s["right"][0], "\n".join("• " + p for p in s["right"][1]), "", "accent2") + "</div>\n" + callout(s.get("callout"))
        elif k == "code":
            body += ('<div class="row" style="gap:48px; align-items:flex-start;">'
                     f'<pre class="mono" style="width:1180px; margin:0; padding:26px 30px; font-size:19px; line-height:1.4; white-space:pre-wrap; background:#FCFBF8; border:2px solid #C9D3E3; border-radius:14px; color:#13213A;">{E(s["code"])}</pre>'
                     f'<div class="col grow" style="gap:20px;">{ul(s["points"], 23)}</div></div>\n')
        elif k == "eq":
            body += f'<div class="eq big" style="font-size:{s.get("eqsize") or (36 if chr(10) in s["eq"] else 44)}px; text-align:{"left" if chr(10) in s["eq"] else "center"}; white-space:{"pre" if chr(10) in s["eq"] else "normal"};">{E(s["eq"]).replace(chr(10), "<br>")}</div>\n' + ul(s["points"], 28)
        elif k == "flow":
            body += '<div class="row" style="gap:12px;">' + '<x-shape kind="arrow-right" class="arrow" style="align-self:center;"></x-shape>'.join(
                f'<div class="box {"cool" if i == len(s["steps"]) - 1 else ""} grow"><p class="bt">{E(t)}</p><p class="bs">{E(b)}</p></div>' for i, (t, b) in enumerate(s["steps"])) + "</div>\n" + ul(s["points"], 30)
    if s.get("src"):
        col = "#C9D3E3" if cls == "dark" else "#5B6676"
        body += f'<p style="position:absolute; left:120px; right:200px; bottom:30px; font-size:17px; line-height:1.25; color:{col}; margin:0;">Sources: {E(s["src"])}</p>\n'
    c = f' class="{cls}"' if cls else ""
    (SL / f'{s["id"]}.html').write_text(f'<section id="{s["id"]}"{c}>\n{body}<aside>{E(s.get("notes", ""))}</aside>\n</section>\n')

d = json.loads((P / "deck.json").read_text())
old = set(d["order"])
d["order"] = [s["id"] for s in S]
d["sections"] = {f"s{i + 1}": {"description": desc, "start": sid} for i, (sid, desc) in enumerate([("cover", "Opening")] + SECTIONS)}
(P / "deck.json").write_text(json.dumps(d))
(HERE.parent / "pptx" / "story.json").write_text(json.dumps({"slides": S, "sections": [("cover", "Opening")] + SECTIONS}, indent=1))
removed = sorted(old - set(d["order"]))
(HERE / "removed.json").write_text(json.dumps(removed))
print(len(S), "slides; removed from order:", removed)
