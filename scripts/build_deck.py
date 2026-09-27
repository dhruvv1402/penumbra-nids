"""Round 1 submission deck, in the landing page's visual language, on the official template's layout.

    uv run --with python-pptx python scripts/build_deck.py [--team team.json]

The template's rules (its first slide): at most 6 slides, the instructions slide removed, every
placeholder replaced, short bullets, and the slide order, titles and layout kept. So every slide here
keeps the template's title and box geometry; what changes is the styling - colour blocks, heavy
condensed type, thick black outlines, offset shadows, and the site's mascots.

Team details come from a JSON file (never committed; see TEAM below for the keys). Anything missing
is drawn as a red [FILL: ...] marker so it cannot slip through unnoticed. Output goes to
artifacts/submission/ (gitignored: it carries names and enrollment numbers).
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "Microsoft_Innovate_2026_Round1_Template.pptx"
OUT_DIR = ROOT / "artifacts" / "submission"

YEL, GRN, RED = RGBColor(0xFF, 0xE1, 0x1E), RGBColor(0x1F, 0xCB, 0x5C), RGBColor(0xF0, 0x2D, 0x2D)
PUR, BLU, BLK = RGBColor(0x7B, 0x7B, 0xF2), RGBColor(0x1E, 0x6F, 0xD9), RGBColor(0x0B, 0x0B, 0x0B)
CRM, WHT, FILLRED = RGBColor(0xFF, 0xF6, 0xDE), RGBColor(0xFF, 0xFF, 0xFF), RGBColor(0xD0, 0x00, 0x00)
DISPLAY, BODY, BODY_BOLD, BODY_BLACK = "Impact", "Segoe UI Semibold", "Segoe UI Bold", "Segoe UI Black"

LIVE = "penumbra-console.kindmoss-9a9e65e5.centralindia.azurecontainerapps.io"
REPO = "github.com/dhruvv1402/penumbra-nids"

TEAM: dict = {
    "team_name": None,
    "team_id": None,
    "problem_title": "Catch the Attack the Signatures Miss",
    "theme": None,
    "deadline": None,
    "leader": None,  # "Name, Enrollment No., Email"
    "members": [],  # [{name, enrollment, programme, email, role}]
    "mentor": None,
}


# --- drawing helpers ------------------------------------------------------------------------------


def _fill(shape, color):
    if color is None:
        shape.fill.background()
    else:
        shape.fill.solid()
        shape.fill.fore_color.rgb = color


def box(
    slide, x, y, w, h, fill=WHT, line=BLK, lw=2.5, radius=0.08, shadow=0.07, shape=MSO_SHAPE.ROUNDED_RECTANGLE
):
    """A flat block with a thick black outline and a hard offset shadow - the site's card."""
    if shadow:
        s = slide.shapes.add_shape(shape, Inches(x + shadow), Inches(y + shadow), Inches(w), Inches(h))
        _fill(s, BLK)
        s.line.fill.background()
        s.shadow.inherit = False
        if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
            s.adjustments[0] = radius
    b = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    _fill(b, fill)
    if line is None:
        b.line.fill.background()
    else:
        b.line.color.rgb = line
        b.line.width = Pt(lw)
    b.shadow.inherit = False
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        b.adjustments[0] = radius
    return b


def text(slide, x, y, w, h, paras, *, anchor=MSO_ANCHOR.TOP, align=PP_ALIGN.LEFT, spacing=1.0, space_after=0):
    """paras: list of paragraphs; each a list of runs (text, font, size, color[, bold])."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, Inches(0.02))
    for i, runs in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        p.space_after = Pt(space_after)
        for run in runs:
            t, font, size, color = run[:4]
            r = p.add_run()
            r.text = t
            r.font.name = font
            r.font.size = Pt(size)
            r.font.color.rgb = color
            r.font.bold = run[4] if len(run) > 4 else False
    return tb


def value(v, what):
    """The value, or a red marker naming what is missing."""
    return (v, BODY_BOLD, 13, BLK) if v else (f"[FILL: {what}]", BODY_BLACK, 13, FILLRED)


def bullets(slide, x, y, w, h, items, *, size=13, dot=BLK, space=5):
    paras = []
    for item in items:
        runs = [("●  ", BODY_BLACK, size - 3, dot)]
        if isinstance(item, tuple):  # (bold lead, rest)
            runs += [(item[0], BODY_BLACK, size, BLK), (item[1], BODY, size, BLK)]
        else:
            runs.append((item, BODY, size, BLK))
        paras.append(runs)
    return text(slide, x, y, w, h, paras, space_after=space, spacing=1.05)


def badge(slide, x, y, label, color, d=0.55, font=DISPLAY):
    c = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y), Inches(d), Inches(d))
    _fill(c, color)
    c.line.color.rgb = BLK
    c.line.width = Pt(2.5)
    c.shadow.inherit = False
    tf = c.text_frame
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, 0)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.name = font
    r.font.size = Pt(20)
    r.font.color.rgb = WHT if color in (RED, BLU, PUR, BLK) else BLK
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE


def card(slide, x, y, w, h, n, title, color, fill=WHT, badge_font=DISPLAY):
    box(slide, x, y, w, h, fill=fill)
    badge(slide, x + 0.25, y + 0.25, n, color, font=badge_font)
    text(slide, x + 0.95, y + 0.25, w - 1.2, 0.55, [[(title, BODY_BLACK, 17, BLK)]], anchor=MSO_ANCHOR.MIDDLE)


def pill(slide, x, y, w, h, label, fill=YEL, size=11, color=BLK, font=BODY_BLACK, shadow=0.05):
    b = box(slide, x, y, w, h, fill=fill, radius=0.5, shadow=shadow, lw=2.25)
    tf = b.text_frame
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, Inches(0.04))
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.word_wrap = True
    lines = label if isinstance(label, list) else [(label, font, size, color)]
    for i, run in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text, r.font.name, r.font.size, r.font.color.rgb = run[0], run[1], Pt(run[2]), run[3]
    return b


def picture(slide, path, x, y, w=None, h=None):
    kw = {}
    if w is not None:
        kw["width"] = Inches(w)
    if h is not None:
        kw["height"] = Inches(h)
    return slide.shapes.add_picture(str(path), Inches(x), Inches(y), **kw)


def framed_shot(slide, path, x, y, w, h):
    box(slide, x, y, w, h, fill=BLK, radius=0.04, shadow=0.07)
    pic = slide.shapes.add_picture(
        str(path), Inches(x + 0.04), Inches(y + 0.04), Inches(w - 0.08), Inches(h - 0.08)
    )
    return pic


def background(slide, color):
    bg = slide.background.fill
    bg.solid()
    bg.fore_color.rgb = color


def chrome(slide, title, n, team_name):
    """Title, round pill and footer - the template's furniture, restyled."""
    background(slide, CRM)
    text(slide, 0.6, 0.3, 9.5, 0.85, [[(title.upper(), DISPLAY, 40, BLK)]], anchor=MSO_ANCHOR.MIDDLE)
    pill(slide, 10.3, 0.47, 2.45, 0.5, "ROUND 1 — IDEA SUBMISSION", fill=YEL, size=9.5)
    foot = [
        (team_name or "[FILL: team name]", BODY_BLACK, 10, BLK if team_name else FILLRED),
        ("   |   Microsoft Innovate 2026 · Bennett University   |   ", BODY, 10, BLK),
        (f"Slide {n} of 6", BODY_BLACK, 10, BLK),
    ]
    text(slide, 0.6, 7.02, 12.1, 0.35, [foot], anchor=MSO_ANCHOR.MIDDLE)


# --- slides ---------------------------------------------------------------------------------------


def title_slide(prs, team, art, logos):
    s = prs.slides.add_slide(prs.slide_layouts[0])
    background(s, YEL)
    # Header: the organisers' logos on white, exactly as the template has them.
    head = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.0))
    _fill(head, WHT)
    head.line.fill.background()
    head.shadow.inherit = False
    picture(s, logos, 0.6, 0.16, 5.6, 0.68)
    text(s, 6.6, 0.16, 6.1, 0.68, [[("School of Computer Science Engineering & Technology", BODY_BLACK, 14, BLK)]],
         anchor=MSO_ANCHOR.MIDDLE, align=PP_ALIGN.RIGHT)  # fmt: skip
    rule = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(1.0), prs.slide_width, Inches(0.045))
    _fill(rule, BLK)
    rule.line.fill.background()

    # Left: event and the fields.
    pill(s, 0.7, 1.33, 3.9, 0.4, "MICROSOFT POWERED HACKATHON", fill=WHT, size=10)
    text(s, 0.7, 1.78, 8.3, 0.9, [[("MICROSOFT INNOVATE 2026", DISPLAY, 50, BLK)]], anchor=MSO_ANCHOR.MIDDLE)
    text(s, 0.7, 2.66, 8.0, 0.4, [[("Innovate  ·  Build  ·  Empower", BODY_BLACK, 16, BLK)]])
    fields = [
        ("TEAM NAME", team["team_name"], "team name"),
        ("TEAM ID", team["team_id"], "team ID from registration"),
        ("PROBLEM STATEMENT", team["problem_title"], "problem title"),
        ("THEME / BUCKET", team["theme"], "theme / bucket"),
        ("TEAM LEADER", team["leader"], "name, enrollment no., email"),
    ]
    for i, (label, v, what) in enumerate(fields):
        y = 3.35 + i * 0.75
        text(s, 0.7, y, 3.1, 0.5, [[(label, BODY_BLACK, 13, BLK)]], anchor=MSO_ANCHOR.MIDDLE)
        box(s, 3.9, y, 4.9, 0.5, fill=WHT, radius=0.3, shadow=0.05, lw=2.25)
        run = value(v, what)
        size = 11 if v and len(v) > 44 else 13
        text(s, 4.08, y, 4.6, 0.5, [[(run[0], run[1], size, run[3])]], anchor=MSO_ANCHOR.MIDDLE)

    # Right: the project, in the site's colours.
    panel = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(9.1), Inches(1.045), Inches(4.233), Inches(6.455))
    _fill(panel, GRN)
    panel.line.fill.background()
    edge = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(9.1), Inches(1.045), Inches(0.045), Inches(6.455))
    _fill(edge, BLK)
    edge.line.fill.background()
    picture(s, art / "eclipse.png", 9.95, 1.25, w=2.55)
    text(
        s,
        9.1,
        3.72,
        4.23,
        0.6,
        [[("ROUND 1", DISPLAY, 40, BLK)]],
        align=PP_ALIGN.CENTER,
        anchor=MSO_ANCHOR.MIDDLE,
    )
    pill(s, 10.13, 4.36, 2.2, 0.42, "IDEA SUBMISSION", fill=WHT, size=11)
    dl = team["deadline"]
    text(s, 9.1, 6.4, 4.23, 0.4,
         [[(f"Submission deadline: {dl}" if dl else "[FILL: submission deadline]", BODY_BOLD, 11, BLK if dl else FILLRED)]],
         align=PP_ALIGN.CENTER)  # fmt: skip


def team_slide(prs, team, art):
    s = prs.slides.add_slide(prs.slide_layouts[0])
    chrome(s, "Team Details", 2, team["team_name"])
    cols = [
        ("S. NO.", 0.75),
        ("FULL NAME", 2.55),
        ("ENROLLMENT NO.", 1.95),
        ("PROGRAMME & YEAR", 2.3),
        ("EMAIL", 2.85),
        ("ROLE", 1.73),
    ]
    members = team["members"] or [{}]
    x0, y0, row_h, head_h = 0.6, 1.45, 0.56, 0.52
    total_h = head_h + row_h * len(members)
    box(s, x0, y0, 12.13, total_h, fill=WHT, radius=0.04, shadow=0.08)
    head = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x0), Inches(y0), Inches(12.13), Inches(head_h))
    _fill(head, BLK)
    head.line.fill.background()
    x = x0
    for name, w in cols:
        text(s, x + 0.12, y0, w - 0.2, head_h, [[(name, BODY_BLACK, 11, YEL)]], anchor=MSO_ANCHOR.MIDDLE)
        x += w
    keys = [
        None,
        ("name", "name"),
        ("enrollment", "enrollment no."),
        ("programme", "programme, year"),
        ("email", "email"),
        ("role", "role"),
    ]
    for r, m in enumerate(members):
        y = y0 + head_h + r * row_h
        if r % 2:
            band = s.shapes.add_shape(
                MSO_SHAPE.RECTANGLE, Inches(x0 + 0.03), Inches(y), Inches(12.07), Inches(row_h)
            )
            _fill(band, CRM)
            band.line.fill.background()
        if r:
            line = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x0), Inches(y), Inches(12.13), Inches(0.03))
            _fill(line, BLK)
            line.line.fill.background()
        x = x0
        for (_col, w), key in zip(cols, keys, strict=True):
            if key is None:
                run = (str(r + 1), DISPLAY, 18, BLK)
            else:
                run = value(m.get(key[0]), key[1])
                run = (run[0], run[1], 11.5 if key[0] == "email" else 12.5, run[3])
            text(s, x + 0.12, y, w - 0.2, row_h, [[run]], anchor=MSO_ANCHOR.MIDDLE)
            x += w
    box(s, 0.6, 5.85, 12.13, 0.75, fill=YEL, radius=0.2, shadow=0.06)
    mentor = team["mentor"]
    text(s, 0.85, 5.85, 11.6, 0.75,
         [[("Faculty Mentor:  ", BODY_BLACK, 14, BLK),
           (mentor or "[FILL: name, department, email - or remove this line]", BODY_BOLD, 14, BLK if mentor else FILLRED)]],
         anchor=MSO_ANCHOR.MIDDLE)  # fmt: skip
    if len(members) <= 3:  # the mascot only where the table leaves room for it
        picture(s, art / "packet.png", 10.9, 4.35, w=1.55)


def problem_slide(prs, team, art):
    s = prs.slides.add_slide(prs.slide_layouts[0])
    chrome(s, "Problem Statement", 3, team["team_name"])
    card(s, 0.6, 1.45, 5.95, 2.6, "1", "What is the problem?", RED)
    bullets(s, 0.95, 2.4, 5.35, 1.55, [
        ("Signatures miss new attacks: ", "an IDS rule exists only after someone has seen the attack."),
        ("ML repeats the failure: ", "on the NSL-KDD benchmark a standard classifier catches only ~5% of never-seen attack types at 1% false alarms."),
        ("Alert floods hide the rest: ", "SOCs receive far more alerts than analysts can triage."),
    ], dot=RED, size=11.5, space=3)  # fmt: skip
    card(s, 6.78, 1.45, 5.95, 2.6, "2", "Who faces it?", BLU)
    bullets(s, 7.13, 2.4, 5.35, 1.55, [
        ("SOC analysts ", "at companies, universities and hospitals."),
        ("Every day: ", "zero-days, new malware, and network traffic that drifts away from what a model learnt."),
    ], dot=BLU, size=12.5)  # fmt: skip
    card(s, 0.6, 4.25, 12.13, 2.5, "3", "Why do current options fall short?", PUR)
    bullets(s, 0.95, 5.2, 11.4, 1.5, [
        ("Signature IDS and supervised ML ", "cannot flag what they have never seen."),
        ("Anomaly detectors flood the SOC with false alarms, ", "and auto-blocking takes the business offline."),
        ("Headline accuracy hides the failure: ", "balanced test sets do not look like real networks."),
    ], dot=PUR, size=13)  # fmt: skip


def solution_slide(prs, team, art, shot):
    s = prs.slides.add_slide(prs.slide_layouts[0])
    chrome(s, "Proposed Solution", 4, team["team_name"])
    box(s, 0.6, 1.45, 12.13, 1.55, fill=YEL, radius=0.1)
    badge(s, 0.85, 1.7, "1", GRN)
    text(s, 1.55, 1.7, 9.0, 0.55, [[("Our idea in one line", BODY_BLACK, 17, BLK)]], anchor=MSO_ANCHOR.MIDDLE)
    text(s, 0.95, 2.3, 11.5, 0.62, [[
        ("We are building ", BODY, 15, BLK), ("Penumbra", BODY_BLACK, 15, BLK),
        (", an ML network detector that helps SOC analysts catch attacks no signature knows, by flagging "
         "what it has never seen, explaining every alert, and never blocking traffic.", BODY, 15, BLK),
    ]])  # fmt: skip

    card(s, 0.6, 3.2, 5.95, 3.55, "2", "How it solves the problem", GRN)
    bullets(s, 0.95, 4.1, 5.35, 2.6, [
        ("Two heads: ", "one model names known attack families; a second learns only normal traffic and flags anything unfamiliar."),
        ("Two lanes: ", "known threats go to an incident queue; novel behaviour to a hunting queue with a fixed daily budget."),
        ("Explained alerts: ", "each alert says why it fired, in plain English, with its MITRE ATT&CK technique."),
    ], dot=GRN, size=12, space=5)  # fmt: skip

    card(s, 6.78, 3.2, 5.95, 3.55, "3", "What makes it different", RED)
    bullets(s, 7.13, 4.1, 2.95, 2.6, [
        ("Sees the unseen: ", "novelty detection, not just known signatures."),
        ("Microsoft stack: ", "built for Azure and Microsoft Sentinel (ASIM alerts)."),
        ("Human in the loop: ", "alerts only, never blocks traffic."),
    ], dot=RED, size=11.5, space=4)  # fmt: skip
    framed_shot(s, shot, 10.18, 4.2, 2.35, 1.47)
    text(s, 10.1, 5.75, 2.5, 0.4, [[("Analyst console (prototype)", BODY_BLACK, 9.5, BLK)]], align=PP_ALIGN.CENTER)


def technical_slide(prs, team, art, shot):
    s = prs.slides.add_slide(prs.slide_layouts[0])
    chrome(s, "Technical Approach", 5, team["team_name"])
    text(s, 0.6, 1.36, 6.0, 0.4, [[("Planned tech stack", BODY_BLACK, 16, BLK)]])
    stack = [
        ("FRONTEND", "Next.js · React", YEL),
        ("BACKEND", "Python · FastAPI", GRN),
        ("DATABASE", "SQLite / PostgreSQL", WHT),
        ("AI / APIS", "scikit-learn · XGBoost", PUR),
        ("CLOUD", "Azure · Sentinel", BLU),
        ("OTHER TOOLS", "Docker · GitHub", RED),
    ]
    for i, (cat, tech, col) in enumerate(stack):
        fg = WHT if col in (BLU, RED, PUR) else BLK
        pill(s, 0.6 + i * 2.04, 1.84, 1.95, 0.62,
             [(cat, BODY_BLACK, 8, fg), (tech, BODY_BOLD, 10, fg)], fill=col)  # fmt: skip

    text(s, 0.6, 2.7, 8.0, 0.4, [[("How the system will work", BODY_BLACK, 16, BLK)]])
    flow = [
        ("NETWORK TRAFFIC", "pcap / NetFlow records", WHT, BLK),
        ("SENSOR + API", "turns traffic into flow features", YEL, BLK),
        ("TWO-HEAD ML", "known + novelty + \u201cI don\u2019t know\u201d lane", BLK, YEL),
        ("SOC + SENTINEL", "alerts, triage, human verdict", GRN, BLK),
    ]
    xs = [0.75, 3.93, 7.11, 10.29]
    for x, (t, d, fill, fg) in zip(xs, flow, strict=True):
        box(s, x, 3.25, 2.55, 1.05, fill=fill, radius=0.12)
        text(s, x + 0.1, 3.3, 2.35, 0.95, [[(t, DISPLAY, 18, fg)], [(d, BODY_BOLD, 10, fg)]],
             align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)  # fmt: skip
    for x in (3.38, 6.56, 9.74):
        a = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(3.6), Inches(0.47), Inches(0.36))
        _fill(a, BLK)
        a.line.fill.background()
        a.shadow.inherit = False

    card(s, 0.6, 4.75, 12.13, 2.0, "\u2713", "What we have started working on", PUR, badge_font="Segoe UI Symbol")
    bullets(s, 0.95, 5.6, 11.4, 1.1, [
        ("Datasets collected: ", "UNSW-NB15, NSL-KDD and CICIDS2017 public intrusion-detection datasets."),
        ("Prototype begun: ", "both detection heads trained and tested on these datasets."),
        ("Alert format drafted ", "in Microsoft Sentinel\u2019s ASIM schema."),
    ], dot=PUR, size=12, space=2)  # fmt: skip


def impact_slide(prs, team, art):
    s = prs.slides.add_slide(prs.slide_layouts[0])
    chrome(s, "Impact and Next Steps", 6, team["team_name"])
    card(s, 0.6, 1.45, 5.95, 2.45, "1", "Expected impact", GRN)
    bullets(s, 0.95, 2.35, 5.3, 1.5, [
        ("SOC analysts ", "see attacks no signature exists for, without an alert flood."),
        ("Target: ", "catch far more never-seen attacks than a classifier alone, at the same false-alarm rate."),
    ], dot=GRN, size=12.5, space=4)  # fmt: skip
    card(s, 6.78, 1.45, 5.95, 2.45, "2", "Feasibility", YEL)
    bullets(s, 7.13, 2.35, 5.3, 1.5, [
        ("Data in hand: ", "three public, labelled intrusion-detection datasets."),
        ("Resources: ", "Azure for Students credit; open-source ML stack."),
        ("Early prototype: ", "both detection heads already trained and tested."),
    ], dot=BLK, size=12.5, space=4)  # fmt: skip
    card(s, 0.6, 4.1, 5.95, 2.65, "3", "Plan for the next round", RED)
    bullets(s, 0.95, 5.0, 5.3, 1.7, [
        ("Step 1: ", "build the two-head detector and the alert API."),
        ("Step 2: ", "add the analyst console and Microsoft Sentinel integration."),
        ("Step 3: ", "demo: attack traffic \u2192 novel alert \u2192 Sentinel incident \u2192 analyst verdict."),
    ], dot=RED, size=12.5, space=4)  # fmt: skip
    card(s, 6.78, 4.1, 5.95, 2.65, "4", "References and data sources", BLU)
    bullets(s, 7.13, 5.0, 5.3, 1.7, [
        ("UNSW-NB15 ", "(Moustafa & Slay, 2015)"),
        ("NSL-KDD ", "(Tavallaee et al., 2009)"),
        ("CICIDS2017 ", "(Sharafaldin et al., 2018)"),
        ("MITRE ATT&CK; ", "Microsoft Sentinel ASIM schema"),
    ], dot=BLU, size=12.5, space=3)  # fmt: skip


# --- assembly -------------------------------------------------------------------------------------


def build(team: dict, art: Path, shots: dict[str, Path], out: Path) -> Path:
    prs = Presentation(str(TEMPLATE))
    logos = art / "tpl_Image_0.png"
    # Remove every template slide (the instructions slide included) and rebuild on its layout.
    sld_ids = prs.slides._sldIdLst  # noqa: SLF001 - python-pptx has no public delete
    for sld in list(sld_ids):
        prs.part.drop_rel(sld.rId)
        sld_ids.remove(sld)
    title_slide(prs, team, art, logos)
    team_slide(prs, team, art)
    problem_slide(prs, team, art)
    solution_slide(prs, team, art, shots["landing"])
    technical_slide(prs, team, art, shots["console"])
    impact_slide(prs, team, art)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", type=Path, help="JSON with the keys in TEAM")
    parser.add_argument("--art", type=Path, required=True, help="directory with the exported artwork PNGs")
    parser.add_argument("--landing-shot", type=Path, required=True)
    parser.add_argument("--console-shot", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "Penumbra_Round1.pptx")
    a = parser.parse_args()
    team = copy.deepcopy(TEAM)
    if a.team:
        team.update(json.loads(a.team.read_text(encoding="utf-8")))
    print(build(team, a.art, {"landing": a.landing_shot, "console": a.console_shot}, a.out))
