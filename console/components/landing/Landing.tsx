"use client";

/**
 * The public landing page: a horizontal-scrolling, colour-block poster.
 *
 * Vertical scroll drives a horizontal track. The page is given the track's width as height, the
 * track is pinned with `position: sticky`, and every frame it eases toward the scroll position
 * rather than jumping to it - that easing is what makes a mouse wheel feel smooth. Reduced motion
 * turns the easing and the idle animations off; below 900px the track becomes an ordinary
 * vertical page, because horizontal scroll on a phone fights the browser.
 *
 * Every number on the page is a measured result from docs/EVALUATION.md or the live deployment;
 * none is illustrative.
 */

import Link from "next/link";
import { useCallback, useEffect, useRef, useState } from "react";

import { Badge, Binoculars, Eclipse, GitHubMark, LogoMark, Packet, QuestionEye, Sparkle, Target } from "./Art";

const REPO = "https://github.com/dhruvv1402/penumbra-nids";
const PANELS = ["hero", "heads", "proof", "how", "misses", "never", "come"] as const;
type PanelId = (typeof PANELS)[number];

export default function Landing() {
  const outer = useRef<HTMLDivElement>(null);
  const track = useRef<HTMLDivElement>(null);
  const [active, setActive] = useState(0);
  const horizontal = useRef(true);

  useEffect(() => {
    const o = outer.current;
    const t = track.current;
    if (!o || !t) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    let current = 0;
    let frame = 0;
    let lastActive = -1;

    const measure = () => {
      horizontal.current = window.innerWidth >= 900;
      if (!horizontal.current) {
        o.style.height = "";
        t.style.transform = "";
        return;
      }
      const max = t.scrollWidth - window.innerWidth;
      o.style.height = `${max + window.innerHeight}px`;
    };

    const panels = Array.from(t.querySelectorAll<HTMLElement>("[data-panel]"));
    const tick = () => {
      if (horizontal.current) {
        const max = Math.max(t.scrollWidth - window.innerWidth, 1);
        const target = Math.min(Math.max(window.scrollY - o.offsetTop, 0), max);
        current = reduce ? target : current + (target - current) * 0.11;
        if (Math.abs(target - current) < 0.2) current = target;
        t.style.transform = `translate3d(${-current}px,0,0)`;
        document.documentElement.style.setProperty("--lx-progress", (current / max).toFixed(4));
        let idx = 0;
        for (let i = 0; i < panels.length; i++) {
          const left = panels[i].offsetLeft - current;
          if (left < window.innerWidth * 0.82) panels[i].classList.add("lx-in");
          if (left <= window.innerWidth * 0.5) idx = i;
        }
        if (idx !== lastActive) {
          lastActive = idx;
          setActive(idx);
        }
      } else {
        for (const p of panels) {
          if (p.getBoundingClientRect().top < window.innerHeight * 0.85) p.classList.add("lx-in");
        }
      }
      frame = requestAnimationFrame(tick);
    };

    measure();
    frame = requestAnimationFrame(tick);
    window.addEventListener("resize", measure);
    // Fonts arriving change widths; measure again once they have.
    void document.fonts?.ready.then(measure);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", measure);
    };
  }, []);

  const go = useCallback((id: PanelId | number) => {
    const t = track.current;
    const o = outer.current;
    if (!t || !o) return;
    const index = typeof id === "number" ? id : PANELS.indexOf(id);
    const panel = t.querySelectorAll<HTMLElement>("[data-panel]")[Math.min(Math.max(index, 0), PANELS.length - 1)];
    if (!panel) return;
    if (horizontal.current) {
      const max = t.scrollWidth - window.innerWidth;
      window.scrollTo({ top: o.offsetTop + Math.min(panel.offsetLeft, max), behavior: "smooth" });
    } else {
      panel.scrollIntoView({ behavior: "smooth" });
    }
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowRight") go(active + 1);
      if (e.key === "ArrowLeft") go(active - 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [active, go]);

  return (
    <div className="lx">
      <Nav go={go} />
      <div ref={outer} className="lx-outer">
        <div className="lx-sticky">
          <div ref={track} className="lx-track">
            <Hero next={() => go(1)} />
            <Heads />
            <Proof />
            <How />
            <Misses />
            <Never />
            <Come />
          </div>
        </div>
      </div>
      <div className="lx-progress" aria-hidden>
        <span />
      </div>
      <div className="lx-dots" aria-label="Sections">
        {PANELS.map((p, i) => (
          <button key={p} onClick={() => go(i)} className={i === active ? "on" : ""} aria-label={`Go to ${p}`} />
        ))}
      </div>
    </div>
  );
}

function Nav({ go }: { go: (id: PanelId) => void }) {
  return (
    <header className="lx-nav">
      <button className="lx-logo" onClick={() => go("hero")} aria-label="Penumbra, back to start">
        <LogoMark className="lx-logo-mark" />
        <span>PENUMBRA</span>
      </button>
      <nav className="lx-pills">
        <button className="lx-pill" onClick={() => go("heads")}>
          HOW
        </button>
        <button className="lx-pill" onClick={() => go("proof")}>
          PROOF
        </button>
        <Link className="lx-pill" href="/score">
          TRY IT
        </Link>
        <Link className="lx-pill lx-pill-hot" href="/console">
          CONSOLE
        </Link>
        <span className="lx-circle" title="Release">
          v1.2
        </span>
        <a className="lx-circle" href={REPO} target="_blank" rel="noreferrer" aria-label="Source on GitHub">
          <GitHubMark className="lx-gh" />
        </a>
      </nav>
    </header>
  );
}

// --- panels -----------------------------------------------------------------------------------------

function Hero({ next }: { next: () => void }) {
  return (
    <section data-panel="hero" className="lx-panel lx-hero lx-in">
      <div className="lx-cell lx-purple lx-center">
        <Pack
          tone="known"
          title="KNOWN"
          kicker="Head 01"
          lines={["labelled attacks + benign", "names the family it knows"]}
          art={<Target className="lx-pack-art" />}
          tilt={-6}
        />
      </div>

      <div className="lx-cell lx-green lx-headline-top">
        <h1 className="lx-mega">
          <span className="lx-rise" style={{ animationDelay: "40ms" }}>
            SEE
          </span>
          <span className="lx-rise" style={{ animationDelay: "140ms" }}>
            THE
          </span>
        </h1>
        <div className="lx-mascot">
          <Eclipse className="lx-eclipse" />
          <Sparkle className="lx-spark s1" />
          <Sparkle className="lx-spark s2" size={16} color="#FFE11E" />
          <Sparkle className="lx-spark s3" size={14} />
        </div>
      </div>

      <div className="lx-cell lx-red lx-center">
        <Pack
          tone="novel"
          title="NOVEL"
          kicker="Head 02"
          lines={["benign traffic only", "flags what it has never seen"]}
          art={<QuestionEye className="lx-pack-art" />}
          tilt={5}
        />
      </div>

      <div className="lx-cell lx-blue lx-center lx-badge-cell">
        <Badge text="#ALERTNEVERBLOCK • #HUMANDECIDES • " className="lx-badge" />
        <Packet className="lx-packet" />
      </div>

      <div className="lx-cell lx-yellow lx-headline-bottom">
        <h1 className="lx-mega lx-mega-wide">
          <span className="lx-rise" style={{ animationDelay: "240ms" }}>
            UNSEEN!
          </span>
        </h1>
        <Binoculars className="lx-binos" />
      </div>

      <button className="lx-cell lx-black lx-center lx-next" onClick={next} aria-label="Next: how it works">
        <svg viewBox="0 0 120 60" className="lx-arrow" aria-hidden>
          <path d="M8 30 H100 M78 8 L104 30 L78 52" fill="none" stroke="#FFE11E" strokeWidth="10" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <span className="lx-scroll-hint">scroll</span>
      </button>
    </section>
  );
}

function Pack({
  tone,
  title,
  kicker,
  lines,
  art,
  tilt,
}: {
  tone: "known" | "novel";
  title: string;
  kicker: string;
  lines: string[];
  art: React.ReactNode;
  tilt: number;
}) {
  return (
    <div className={`lx-pack lx-pack-${tone}`} style={{ "--tilt": `${tilt}deg` } as React.CSSProperties}>
      <div className="lx-pack-top">
        <LogoMark className="lx-pack-logo" />
        <span>PENUMBRA</span>
      </div>
      <div className="lx-pack-window">{art}</div>
      <div className="lx-pack-kicker">{kicker}</div>
      <div className="lx-pack-title">{title}</div>
      {lines.map((l) => (
        <div key={l} className="lx-pack-line">
          {l}
        </div>
      ))}
      <div className="lx-pack-foot">
        <span>ML-NIDS</span>
        <span>NEVER BLOCKS</span>
      </div>
    </div>
  );
}

function Heads() {
  return (
    <section data-panel="heads" className="lx-panel lx-split">
      <div className="lx-cell lx-yellow lx-pad lx-col">
        <p className="lx-kicker">The thesis</p>
        <h2 className="lx-big">
          A CLASSIFIER
          <br />
          CAN&apos;T SEE
          <br />
          WHAT IT
          <br />
          NEVER MET.
        </h2>
        <p className="lx-body">
          Trained on known attack families, a supervised model files anything new under{" "}
          <b>&ldquo;normal&rdquo;</b> - the only bucket it has for &ldquo;nothing I recognise&rdquo;. So Penumbra runs
          two heads, and measures whether the second one earns its keep.
        </p>
      </div>
      <div className="lx-cell lx-purple lx-pad lx-col lx-reveal">
        <div className="lx-duel">
          <div className="lx-duel-card lx-red">
            <Target className="lx-duel-art" />
            <h3>KNOWN</h3>
            <p>labelled attacks + benign</p>
            <p className="lx-dim">&ldquo;which family is this?&rdquo;</p>
          </div>
          <div className="lx-duel-vs">+</div>
          <div className="lx-duel-card lx-green">
            <QuestionEye className="lx-duel-art" />
            <h3>NOVEL</h3>
            <p>benign traffic only</p>
            <p className="lx-dim">&ldquo;is this unlike anything seen?&rdquo;</p>
          </div>
        </div>
        <div className="lx-stat-strip">
          <div>
            <span className="lx-num">5%</span>
            <span>unseen attacks caught by the classifier alone</span>
          </div>
          <div className="lx-strip-arrow">→</div>
          <div>
            <span className="lx-num">35%</span>
            <span>with the novelty head, same 1% false-alarm budget</span>
          </div>
        </div>
      </div>
    </section>
  );
}

const STATS: { value: string; label: string; tone: string; note: string }[] = [
  { value: "94,115→203", label: "alerts became incidents", tone: "lx-yellow", note: "CICIDS2017, real source IPs" },
  { value: "12.6ms", label: "to score one flow", tone: "lx-red", note: "was 139 ms; 0 decisions changed" },
  { value: "92.9%", label: "of scans caught on our own network", tone: "lx-blue", note: "after re-learning normal, no labels" },
  { value: "18.8→2.9%", label: "false alarms, after fixing our own bug", tone: "lx-purple", note: "E9a: threshold fitted on memorised rows" },
  { value: "13", label: "real Microsoft Sentinel incidents", tone: "lx-green", note: "live workspace, every record DvcAction=Allow" },
  { value: "0", label: "endpoints that can block traffic", tone: "lx-black", note: "enforced in CI" },
];

function Proof() {
  return (
    <section data-panel="proof" className="lx-panel lx-proof">
      <div className="lx-proof-head lx-pad">
        <p className="lx-kicker">Measured, not claimed</p>
        <h2 className="lx-big">
          THE
          <br />
          NUMBERS.
        </h2>
        <Sparkle className="lx-spark-static" size={38} color="#FFE11E" />
      </div>
      <div className="lx-tiles">
        {STATS.map((s, i) => (
          <div key={s.label} className={`lx-tile ${s.tone} lx-reveal`} style={{ transitionDelay: `${i * 70}ms` }}>
            <span className="lx-tile-num">{s.value}</span>
            <span className="lx-tile-label">{s.label}</span>
            <span className="lx-tile-note">{s.note}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

const STEPS = [
  { n: "01", t: "SENSE", d: "pcap or flow records in, features out", tone: "lx-yellow" },
  { n: "02", t: "SCORE", d: "known head + novelty head, 12.6 ms a flow", tone: "lx-green" },
  { n: "03", t: "SORT", d: "known-threat lane · hunting lane · review lane", tone: "lx-purple" },
  { n: "04", t: "SHIP", d: "incidents to Microsoft Sentinel as ASIM", tone: "lx-blue" },
  { n: "05", t: "DECIDE", d: "a human. Always.", tone: "lx-red" },
];

function How() {
  return (
    <section data-panel="how" className="lx-panel lx-how lx-pad">
      <p className="lx-kicker">How it works</p>
      <h2 className="lx-big lx-oneline">FLOW → VERDICT → HUMAN</h2>
      <div className="lx-steps">
        {STEPS.map((s, i) => (
          <div key={s.t} className={`lx-step ${s.tone} lx-reveal`} style={{ transitionDelay: `${i * 90}ms` }}>
            <span className="lx-step-n">{s.n}</span>
            <span className="lx-step-t">{s.t}</span>
            <span className="lx-step-d">{s.d}</span>
          </div>
        ))}
      </div>
      <Packet className="lx-how-packet" />
    </section>
  );
}

const MISSES = [
  "Re-learning normal did NOT beat moving thresholds (E8, refuted)",
  "A deep sequence model lost to nine cheap features",
  "Our shipped detector fired on 18.8% of benign traffic - our bug, measured and fixed (E9a)",
  "Weakest families? ~90% of Analysis and Backdoor rows duplicate another family exactly",
  "3 trees + 3 SVMs lost to the forest on all three datasets (E9)",
  "Our own Sentinel claim was wrong - measured, corrected",
];

function Misses() {
  return (
    <section data-panel="misses" className="lx-panel lx-misses lx-red lx-pad">
      <p className="lx-kicker lx-kicker-light">Honesty is a feature</p>
      <h2 className="lx-big lx-ink-cream">
        WE PUBLISH
        <br />
        OUR MISSES.
      </h2>
      <div className="lx-stickers">
        {MISSES.map((m, i) => (
          <span
            key={m}
            className="lx-sticker lx-reveal"
            style={{ "--r": `${(i % 2 ? 1 : -1) * (1.5 + (i % 3))}deg`, transitionDelay: `${i * 60}ms` } as React.CSSProperties}
          >
            {m}
          </span>
        ))}
      </div>
      <p className="lx-body lx-ink-cream lx-narrow">
        Every experiment is written down and committed <b>before</b> it runs. Refuted predictions stay in the
        record.
      </p>
    </section>
  );
}

function Never() {
  return (
    <section data-panel="never" className="lx-panel lx-never lx-black lx-pad">
      <h2 className="lx-giant">
        <span>ALERT.</span>
        <span className="lx-outline">NEVER</span>
        <span>BLOCK.</span>
      </h2>
      <div className="lx-code lx-reveal">
        <div>
          <span className="k">DvcAction</span> = <span className="v">&quot;Allow&quot;</span>
          <span className="c">  // we did not block</span>
        </div>
        <div>
          <span className="k">ThreatConfidence</span> = <span className="v">93</span>
          <span className="c">  // and we were sure</span>
        </div>
        <div>
          <span className="k">EventSeverity</span> = <span className="v">&quot;High&quot;</span>
        </div>
      </div>
      <p className="lx-body lx-ink-cream lx-narrow">
        At a 1% false-positive rate, auto-blocking takes a business offline faster than an attacker could.
        Penumbra surfaces. A human decides.
      </p>
    </section>
  );
}

function Come() {
  return (
    <section data-panel="come" className="lx-panel lx-come lx-yellow lx-pad">
      <div className="lx-come-inner">
        <h2 className="lx-mega lx-come-title">
          COME
          <br />
          SEE.
        </h2>
        <div className="lx-come-side">
          <Eclipse className="lx-come-eclipse" />
          <div className="lx-cta-row">
            <Link className="lx-cta" href="/score">
              RUN THE MODELS ON REAL TRAFFIC →
            </Link>
            <Link className="lx-cta lx-cta-hot" href="/console">
              OPEN THE CONSOLE →
            </Link>
            <a className="lx-cta" href={`${REPO}/blob/main/docs/EVALUATION.md`} target="_blank" rel="noreferrer">
              READ THE PROOF
            </a>
            <a className="lx-cta" href={REPO} target="_blank" rel="noreferrer">
              SOURCE
            </a>
          </div>
          <p className="lx-small">
            &ldquo;View as guest&rdquo; opens a read-only console on real CICIDS2017 and NSL-KDD alerts.
            <br />
            Bennett University Hackathon 2026 · Microsoft track · Problem 26
          </p>
        </div>
      </div>
    </section>
  );
}
