// Builds One_Controller_for_Every_Building.pptx: a structured rebuild of the Slides deck.
const pptxgen = require("pptxgenjs");
const path = require("path");
const SKILL = process.env.PPTX_SKILL;
const { applyTheme } = require(path.join(SKILL, "scripts/apply_theme.js"));
const OUT = process.argv[2] || "deck.pptx";

const THEME = {
  name: "Smart Buildings",
  headFontFace: "Calibri",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "13213A", lt1: "FCFBF8", dk2: "2C3748", lt2: "F6F4EF",
    accent1: "D9692A", accent2: "2F72B5", accent3: "5B6676", accent4: "C9D3E3",
    accent5: "EAE5DA", accent6: "8A93A3", hlink: "2F72B5", folHlink: "5B6676",
  },
};
const HEX = THEME.colors;
const MONO = "Courier New";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.title = "One Controller for Every Building";
pres.subject = "A state-space model of room temperature for the NeurIPS 2026 Smart Buildings Challenge";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

const X0 = 0.89, W = 11.55, Y0 = 2.2, YMAX = 6.75;

// ---------------------------------------------------------------- layouts
function layout(name, bg, titleColor, numColor) {
  pres.defineSlideMaster({
    title: name,
    background: { color: bg },
    margin: [0.6, 0.89, 0.75, 0.89],
    objects: [
      { placeholder: { options: { name: "eyebrow", type: "body", x: X0, y: 0.55, w: W, h: 0.36, fontSize: 12, bold: true, color: C.accent1, charSpacing: 2, margin: 0, valign: "top" }, text: "" } },
      { placeholder: { options: { name: "title", type: "title", x: X0, y: 0.92, w: W, h: 1.1, fontSize: 30, bold: true, color: titleColor, margin: 0, valign: "top" }, text: "" } },
    ],
    slideNumber: { x: 12.1, y: 6.95, w: 0.4, h: 0.3, fontSize: 10, color: numColor, align: "right" },
  });
}
layout("CONTENT", C.background2, C.text1, C.accent6);
layout("CONTENT_ALT", C.accent5, C.text1, C.accent6);
pres.defineSlideMaster({
  title: "STATEMENT",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "eyebrow", type: "body", x: X0, y: 1.6, w: W, h: 0.4, fontSize: 13, bold: true, color: C.accent1, charSpacing: 2, margin: 0 }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: X0, y: 2.1, w: 10.6, h: 2.0, fontSize: 40, bold: true, color: C.background2, margin: 0, valign: "top" }, text: "" } },
  ],
  slideNumber: { x: 12.1, y: 6.95, w: 0.4, h: 0.3, fontSize: 10, color: C.accent6, align: "right" },
});

// ---------------------------------------------------------------- helpers
let SECTION = "";
function slide(master, eyebrow, title, notes) {
  const s = pres.addSlide({ masterName: master, sectionTitle: SECTION });
  if (eyebrow) s.addText(eyebrow.toUpperCase(), { placeholder: "eyebrow" });
  if (title) s.addText(title, { placeholder: "title" });
  if (notes) s.addNotes(notes);
  return s;
}
function text(s, t, o) { s.addText(t, Object.assign({ isTextBox: true, margin: 0, fontSize: 14, color: C.text2, valign: "top", paraSpaceAfter: 4 }, o)); }
function card(s, x, y, w, h, { title, body, tag, tone = "accent1", fill = "background1" }) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.12, fill: { color: C[fill] }, line: { color: C.accent5, width: 0.75 }, objectName: `card ${title || tag}` });
  let yy = y + 0.22;
  if (tag) { text(s, tag, { x: x + 0.25, y: yy, w: w - 0.5, h: 0.28, fontSize: 11, fontFace: MONO, color: C[tone] }); yy += 0.32; }
  if (title) {
    s.addShape(pres.ShapeType.ellipse, { x: x + 0.25, y: yy + 0.08, w: 0.16, h: 0.16, fill: { color: C[tone] }, line: { color: C[tone] } });
    text(s, title, { x: x + 0.52, y: yy, w: w - 0.77, h: 0.34, fontSize: 16, bold: true, color: C.text1 }); yy += 0.44;
  }
  if (body) text(s, body, { x: x + 0.25, y: yy, w: w - 0.5, h: y + h - yy - 0.15, fontSize: 13 });
}
function box(s, x, y, w, h, title, sub, tone) {
  const dark = tone === "fill";
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.1, fill: { color: dark ? C.text1 : C.background1 }, line: { color: dark ? C.text1 : (tone ? C[tone] : C.text1), width: 1.5 } });
  text(s, title, { x: x + 0.15, y: y + 0.12, w: w - 0.3, h: 0.32, fontSize: 15, bold: true, color: dark ? C.background2 : C.text1 });
  if (sub) text(s, sub, { x: x + 0.15, y: y + 0.46, w: w - 0.3, h: h - 0.55, fontSize: 12, color: dark ? C.accent4 : C.accent3 });
}
function arrow(s, x, y, w = 0.36) { s.addShape(pres.ShapeType.rightArrow, { x, y, w, h: 0.22, fill: { color: C.accent3 }, line: { color: C.accent3 } }); }
function stat(s, x, y, w, num, label, tone = "accent1", size = 54) {
  text(s, num, { x, y, w, h: 0.95, fontSize: size, bold: true, color: C[tone], fontFace: THEME.headFontFace });
  text(s, label, { x, y: y + 1.0, w, h: 0.9, fontSize: 14 });
}
function eq(s, t, x, y, w, h, size = 15) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.06, fill: { color: C.background1 }, line: { color: C.accent4, width: 0.75 } });
  text(s, t, { x: x + 0.25, y: y + 0.08, w: w - 0.5, h: h - 0.16, fontSize: size, fontFace: MONO, color: C.text1, valign: "middle" });
}
function callout(s, t, x, y, w, h) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h, rectRadius: 0.12, fill: { color: C.text1 }, line: { color: C.text1 } });
  text(s, t, { x: x + 0.3, y: y + 0.18, w: w - 0.6, h: h - 0.36, fontSize: 14, color: C.background2, valign: "middle" });
}
function table(s, rows, x, y, w, colW, fontSize = 13) {
  const head = rows[0].map(t => ({ text: t, options: { bold: true, color: C.text1, fill: { color: C.accent5 } } }));
  const body = rows.slice(1).map(r => r.map(t => (typeof t === "string" ? { text: t } : t)));
  s.addTable([head, ...body], { x, y, w, colW, fontSize, color: C.text2, border: { type: "solid", pt: 0.5, color: "D5CFC2" }, fill: { color: C.background1 }, margin: [0.05, 0.1, 0.05, 0.1], valign: "middle" });
}
function pill(s, t, x, y, w) {
  s.addShape(pres.ShapeType.roundRect, { x, y, w, h: 0.4, rectRadius: 0.2, fill: { color: C.accent5 }, line: { color: C.accent5 } });
  text(s, t, { x, y: y + 0.08, w, h: 0.26, fontSize: 12, bold: true, color: C.text1, align: "center" });
}
// ---- slide bodies (assembled by make_build.sh between head.js and the reused blocks)
const fs = require("fs");
const IMG = process.env.DECK_IMG;
function img(s, file, x, y, maxW, maxH, alt) {
  const b = fs.readFileSync(path.join(IMG, file));
  const iw = b.readUInt32BE(16), ih = b.readUInt32BE(20);
  let w = maxW, h = (maxW * ih) / iw;
  if (h > maxH) { h = maxH; w = (maxH * iw) / ih; }
  s.addImage({ path: path.join(IMG, file), x, y, w, h, altText: alt || file, objectName: alt || file });
  return { w, h };
}
function caption(s, t, x, y, w) { text(s, t, { x, y, w, h: 0.32, fontSize: 11, color: C.accent3, italic: true }); }


// ================================================================ render story.json (one storyline for both decks)
const ST = JSON.parse(fs.readFileSync(process.env.STORY_JSON || path.join(__dirname, "story.json"), "utf8"));
const secStart = Object.fromEntries(ST.sections.map(([id, d]) => [id, d]));
function bullets(s, pts, x, y, w, h, size = 15) {
  s.addText(pts.map((p, i) => ({ text: p, options: { bullet: true, breakLine: i < pts.length - 1 } })),
    { isTextBox: true, x, y, w, h, fontSize: size, color: C.text2, valign: "top", paraSpaceAfter: 8, margin: 0 });
}
let alt = false;
for (const d of ST.slides) {
  if (secStart[d.id]) { SECTION = secStart[d.id]; pres.addSection({ title: SECTION }); }
  const k = d.kind;
  const srcLine = (s, dark) => { if (d.src) text(s, "Sources: " + d.src, { x: X0, y: 6.93, w: 10.9, h: 0.4, fontSize: 9, color: dark ? C.accent4 : C.accent3 }); };
  if (k === "statement") {
    const s = slide("STATEMENT", d.eyebrow, d.title, d.notes);
    if (d.sub) text(s, d.sub, { x: X0, y: 4.35, w: 10.6, h: 1.2, fontSize: 18, color: C.accent4 });
    if (d.foot) text(s, d.foot, { x: X0, y: 5.8, w: 8, h: 0.35, fontSize: 12, color: C.accent4 });
    srcLine(s, true); continue;
  }
  if (k === "section") {
    const s = slide("STATEMENT", "Part " + d.num, d.title, d.notes);
    text(s, d.question, { x: X0, y: 3.6, w: 10.6, h: 1.4, fontSize: 24, color: C.accent4 });
    srcLine(s, true); continue;
  }
  alt = !alt;
  const s = slide(alt ? "CONTENT_ALT" : "CONTENT", d.eyebrow, d.title, d.notes);
  srcLine(s, false);
  const cbox = (t, y, h = 0.85) => { if (t) callout(s, t, X0, y, W, h); };
  if (k === "points") { bullets(s, d.points, X0, Y0, W, 3.6, 20); cbox(d.callout, 5.85); }
  else if (k === "stats") {
    d.stats.forEach(([n, l], i) => stat(s, X0 + i * 3.95, Y0 + 0.1, 3.6, n, l, i === 2 ? "accent2" : "accent1"));
    cbox(d.callout, 4.55, 1.0); if (d.caption) caption(s, d.caption, X0, 5.85, W);
  } else if (k === "image") {
    if (d.side === "right") {
      const r = img(s, d.img, X0, Y0 - 0.1, 6.6, 4.2, d.caption || d.img);
      if (d.caption) caption(s, d.caption, X0, Y0 - 0.1 + r.h + 0.08, 6.6);
      bullets(s, d.points, X0 + 6.95, Y0, W - 6.95, 4.4, 15);
    } else {
      const r = img(s, d.img, X0, Y0 - 0.15, W, 3.25, d.caption || d.img);
      const x = X0 + (W - r.w) / 2; // centre: re-add would duplicate; keep left aligned
      if (d.caption) caption(s, d.caption, X0, Y0 - 0.15 + r.h + 0.05, W);
      bullets(s, d.points, X0, Y0 + r.h + 0.35, W, YMAX - (Y0 + r.h + 0.35), 14);
    }
  } else if (k === "cards") {
    const n = d.cards.length, gap = 0.3, cw = (W - gap * (n - 1)) / n, ch = d.callout ? 3.25 : 4.2;
    d.cards.forEach(([t, b, tag, tone], i) => card(s, X0 + i * (cw + gap), Y0, cw, ch, { title: t, body: b, tag: tag || undefined, tone: tone || "accent1" }));
    cbox(d.callout, Y0 + ch + 0.3);
  } else if (k === "table") {
    const rows = d.header === false ? d.rows.slice(1).map(r => r.map((c, i) => (i === 0 ? { text: c, options: { bold: true } } : c))) : d.rows;
    const rr = d.mono_first ? rows.map((r, j) => (j === 0 ? r : r.map((c, i) => (i === 0 ? { text: c, options: { fontFace: MONO, color: C.text1 } } : c)))) : rows;
    const colW = d.colw.map(f => f * W);
    if (d.header === false) {
      s.addTable(rr, { x: X0, y: Y0 - 0.1, w: W, colW, fontSize: d.rows.length > 7 ? 10.5 : 13, color: C.text2, border: { type: "solid", pt: 0.5, color: "D5CFC2" }, fill: { color: C.background1 }, margin: [0.06, 0.1, 0.06, 0.1], valign: "middle" });
    } else table(s, rr, X0, Y0 - 0.1, W, colW, d.font || (d.rows.length > 7 ? 10.5 : 13));
    cbox(d.callout, d.rows.length > 7 ? 6.2 : 5.75, d.rows.length > 7 ? 0.62 : 0.8);
  } else if (k === "compare") {
    const cw = (W - 0.4) / 2, ch = d.callout ? 3.2 : 4.2;
    card(s, X0, Y0, cw, ch, { title: d.left[0], body: d.left[1].map(p => "• " + p).join("\n"), tone: "accent1" });
    card(s, X0 + cw + 0.4, Y0, cw, ch, { title: d.right[0], body: d.right[1].map(p => "• " + p).join("\n"), tone: "accent2" });
    cbox(d.callout, Y0 + ch + 0.3);
  } else if (k === "code") {
    s.addShape(pres.ShapeType.roundRect, { x: X0, y: Y0 - 0.1, w: 7.5, h: 4.55, rectRadius: 0.08, fill: { color: C.background1 }, line: { color: C.accent4, width: 0.75 } });
    text(s, d.code, { x: X0 + 0.2, y: Y0, w: 7.1, h: 4.35, fontSize: 8.5, fontFace: MONO, color: C.text1 });
    bullets(s, d.points, X0 + 7.8, Y0, W - 7.8, 4.4, 14);
  } else if (k === "eq") {
    const nl = d.eq.split("\n").length, two = nl > 1, eh = two ? 0.45 + 0.3 * nl : 0.9;
    eq(s, d.eq, X0, Y0, W, eh, d.eqpt || (two ? 17 : 22));
    bullets(s, d.points, X0, Y0 + eh + 0.3, W, YMAX - (Y0 + eh + 0.3), two ? 14 : 16);
  } else if (k === "flow") {
    const n = d.steps.length, aw = 0.36, gap = 0.12, bw = (W - (n - 1) * (aw + 2 * gap)) / n;
    d.steps.forEach(([t, b], i) => {
      const x = X0 + i * (bw + aw + 2 * gap);
      box(s, x, Y0, bw, 1.35, t, b, i === n - 1 ? "accent2" : undefined);
      if (i < n - 1) arrow(s, x + bw + gap, Y0 + 0.56, aw);
    });
    if (d.points && d.points.length) bullets(s, d.points, X0, Y0 + 1.85, W, 2.5, 16);
    cbox(d.callout, Y0 + (d.points && d.points.length ? 4.0 : 1.85));
  } else if (k === "figure") {
    const r = img(s, d.img, X0, Y0 - 0.15, W, d.callout ? 3.55 : 4.4, d.title);
    cbox(d.callout, Y0 - 0.15 + r.h + 0.2);
  }
}

(async () => {
  await pres.writeFile({ fileName: OUT });
  await applyTheme(OUT, THEME);
  console.log("wrote " + OUT);
})();
