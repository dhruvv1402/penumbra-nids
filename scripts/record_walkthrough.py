# ruff: noqa: E501  (captions are prose)
"""Record the demo walkthrough as a video: the fallback if the live demo fails on stage.

Run `penumbra demo` first, then:

    uv run --with playwright python scripts/record_walkthrough.py [--out artifacts/demo]

Uses the locally installed Microsoft Edge (no browser download). Captions are drawn on the page so
the silent video explains itself; narrate over it if the submission wants a voice-over. Output is a
.webm in the output directory (artifacts/ is gitignored).
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

BASE = "http://localhost:3000"

CAPTION_JS = """
(text) => {
  let el = document.getElementById('__walkthrough_caption');
  if (!el) {
    el = document.createElement('div');
    el.id = '__walkthrough_caption';
    Object.assign(el.style, {
      position: 'fixed', left: '50%', bottom: '28px', transform: 'translateX(-50%)',
      maxWidth: '78%', padding: '12px 18px', background: 'rgba(8,10,14,0.92)',
      border: '1px solid #3b82f6', borderRadius: '8px', color: '#e5e7eb',
      font: '15px/1.45 ui-sans-serif, system-ui, sans-serif', zIndex: 99999,
      boxShadow: '0 8px 30px rgba(0,0,0,0.5)', textAlign: 'center',
    });
    document.body.appendChild(el);
  }
  el.textContent = text;
}
"""


def caption(page: Page, text: str, hold: float = 4.0) -> None:
    page.evaluate(CAPTION_JS, text)
    page.wait_for_timeout(int(hold * 1000))


def show(page: Page, text: str) -> None:
    """Scroll the panel whose title contains `text` into view, if it is on the page."""
    target = page.get_by_text(text, exact=False).first
    if target.count():
        target.scroll_into_view_if_needed()
        page.mouse.wheel(0, -60)
        page.wait_for_timeout(1000)


def goto(page: Page, path: str) -> None:
    page.goto(f"{BASE}{path}")
    page.wait_for_timeout(2500)


def main(out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_raw"
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        ctx = browser.new_context(viewport={"width": 1440, "height": 900}, record_video_dir=str(tmp),
                                  record_video_size={"width": 1440, "height": 900})
        page = ctx.new_page()

        goto(page, "/")
        page.evaluate("localStorage.clear()")
        goto(page, "/")
        caption(page, "PENUMBRA - an ML network detector that alerts a SOC and never blocks traffic.", 4)
        page.locator("input").nth(0).fill("senior")
        page.locator("input").nth(1).fill("senior")
        page.get_by_role("button", name="Sign in").click()
        page.wait_for_timeout(3000)

        caption(page, "Two lanes. Known threats are recognised attack families: an SLA'd queue.", 4)
        page.locator("li.slide-in").first.click()
        page.wait_for_timeout(1500)
        caption(page, "Every alert shows why it fired, in plain English, with a cited ATT&CK triage note - built offline.", 6)

        page.locator("button:has-text('hunting')").first.click()
        page.wait_for_timeout(800)
        caption(page, "Hunting: traffic unlike anything in the benign baseline. A fixed daily budget, so it cannot cause alert fatigue.", 5)
        page.locator("li.slide-in").first.click()
        page.wait_for_timeout(1500)
        caption(page, "A novelty alert's note names no technique - the system's own verdict is that it does not recognise this.", 5)

        page.locator("button:has-text('review')").first.click()
        page.wait_for_timeout(800)
        caption(page, "Review: the conformal layer declined to commit. 'I don't know' is a reportable answer, routed to a human.", 5)

        page.locator("button:has-text('incidents')").first.click()
        page.wait_for_timeout(800)
        caption(page, "Real CICIDS2017 source IPs: 94,115 alerts become 203 incidents. 99.93% of attack flows reach an analyst.", 5)
        page.locator("ul li").first.click()
        page.wait_for_timeout(2000)
        caption(page, "One PortScan: 41,866 flows, 1,001 ports - one row in the queue instead of 41,866 tickets.", 6)

        page.locator("button:has-text('known')").first.click()
        page.wait_for_timeout(800)
        page.locator("li.slide-in").nth(2).click()
        page.wait_for_timeout(1500)
        page.get_by_role("button", name="Benign by policy").click()
        page.wait_for_timeout(1500)
        caption(page, "Benign-by-policy becomes a suppression rule: scoped narrower than a family, always expiring, never a training label.", 6)
        form = page.get_by_placeholder("why this is policy")
        if form.count():
            form.fill("nightly internal vulnerability scan")
            page.wait_for_timeout(800)
            page.get_by_role("button", name="Create rule").click()
            page.wait_for_timeout(2000)
            caption(page, "Created, audited, expiring in 30 days. Matching alerts are reclassified at ingest - stored, not dropped.", 5)

        goto(page, "/feedback")
        caption(page, "Analyst verdicts retrain the model - so a stolen account could poison it. Promotion needs a second senior.", 6)
        show(page, "poisoning drill")
        caption(page, "We ran the attack (E7, pre-registered): recall 0.84 -> 0.19, but only at full dose - and the per-family peer check flagged every flip.", 7)

        goto(page, "/evaluation")
        caption(page, "Every number on this page was written by a command, not typed. Mined rules, sequence model, evasion...", 5)
        show(page, "incident correlation")
        caption(page, "Correlation with ground truth: the incident the model calls DoS Hulk is really Friday's DDoS.", 5)
        show(page, "in-distribution vs under shift")
        caption(page, "Calibration holds in-distribution and gets worse under shift - so p_attack says where it was calibrated.", 6)
        show(page, "threshold drift")
        caption(page, "Under drift a 1% target realised 10.2%. Re-fitting thresholds restores 1.06% - and shows the unseen-attack recall had been bought with false positives (E8).", 7)
        caption(page, "A baseline that is 5% attack traffic costs a quarter of unseen-attack recall while the FPR falls. Poisoning the baseline looks like tuning.", 6)
        show(page, "re-baselined on a real network")
        caption(page, "Our own laptops: out of the box every flow alerts. Re-baselined with no labels: ordinary flows 70% -> 4.9%, scan flows detected 92.9%.", 7)
        show(page, "speed")
        caption(page, "One flow scored in 12.6 ms (was 139). Flat-array forests, checked at startup to change no decision.", 5)

        goto(page, "/drift")
        caption(page, "Conformal coverage as a label-free drift alarm: abstention rises as the data moves away from training.", 6)

        goto(page, "/governance")
        caption(page, "No blocking code path, enforced in CI. RBAC from the server's table. A hash-chained audit log, re-verified live.", 6)
        show(page, "audit log")
        caption(page, "Every verdict, suppression and promotion is in the chain - including the rule created a minute ago.", 5)
        caption(page, "Penumbra surfaces; a human decides.", 4)

        video = page.video
        ctx.close()
        browser.close()
        raw = Path(video.path()) if video else None

    final = out / "penumbra-walkthrough.webm"
    if raw and raw.exists():
        shutil.move(str(raw), final)
    shutil.rmtree(tmp, ignore_errors=True)
    return final


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/demo"))
    print(main(parser.parse_args().out))
