#!/usr/bin/env python3
"""Figures and an HTML report for a sweep directory.

    report.py DIR [--baseline FILE] [--out report.html] [--pdf]

With --baseline, the first figure is the before/after comparison — the one that
answers "did this change help?".
"""
import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from traj_eval import SEQS, SHORT, evaluate                    # noqa: E402
from bench import score, NOISE_FLOOR                           # noqa: E402

# Categorical slots 1-3 of the validated palette: these three clear all-pairs
# colour-vision separation in both themes. Do not substitute by eye.
THEMES = {
    "light": dict(surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e",
                  muted="#8a8984", grid="#e4e3df", gt="#1baf7a",
                  series={"kornia": "#2a78d6", "orb3": "#eb6834", "base": "#8a8984"}),
    "dark":  dict(surface="#1a1a19", primary="#ffffff", secondary="#c3c2b7",
                  muted="#8a8984", grid="#333331", gt="#199e70",
                  series={"kornia": "#3987e5", "orb3": "#d95926", "base": "#8a8984"}),
}
MAIN = "kornia-si_loop"


def style(th):
    plt.rcParams.update({
        "figure.facecolor": th["surface"], "axes.facecolor": th["surface"],
        "savefig.facecolor": th["surface"], "text.color": th["primary"],
        "axes.labelcolor": th["secondary"], "xtick.color": th["secondary"],
        "ytick.color": th["secondary"], "axes.edgecolor": th["grid"],
        "grid.color": th["grid"], "axes.titlecolor": th["primary"],
        "font.size": 9, "axes.titlesize": 10, "axes.linewidth": 0.8,
        "legend.frameon": False, "figure.dpi": 160,
    })


def recede(ax):
    ax.grid(True, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def fig_compare(med, base, th, path):
    """Current against baseline — the iteration figure."""
    seqs = [s for s in SEQS if f"{MAIN}|{s}" in med and f"{MAIN}|{s}" in base]
    if not seqs:
        return False
    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    w, x = 0.38, np.arange(len(seqs))
    b = [base[f"{MAIN}|{s}"]["ate_rmse"] for s in seqs]
    c = [med[f"{MAIN}|{s}"]["ate_rmse"] for s in seqs]
    ax.bar(x-w/2-0.004, b, w*0.94, label="baseline", color=th["series"]["base"], zorder=3)
    ax.bar(x+w/2+0.004, c, w*0.94, label="current", color=th["series"]["kornia"], zorder=3)
    for xi, bv, cv in zip(x, b, c):
        d = (cv-bv)/bv
        if abs(d) > NOISE_FLOOR:
            ax.text(xi, max(bv, cv)*1.12, f"{d*100:+.0f}%", ha="center", fontsize=6.5,
                    color=th["primary"], zorder=4)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[s] for s in seqs], fontsize=8)
    ax.set_ylabel("ATE RMSE (m, log scale)")
    ax.set_title(f"Current against baseline — labelled where the change clears "
                 f"the ±{NOISE_FLOOR*100:.0f}% noise floor", loc="left", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=2, fontsize=8)
    recede(ax); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return True


def fig_ate(med, th, path):
    """ATE per sequence, against the ORB-SLAM3 baseline when it was run."""
    seqs = [s for s in SEQS if f"{MAIN}|{s}" in med]
    if not seqs:
        return False
    has_orb = any(f"orb3|{s}" in med for s in seqs)
    fig, ax = plt.subplots(figsize=(9.6, 3.8))
    w, x = (0.38, np.arange(len(seqs))) if has_orb else (0.6, np.arange(len(seqs)))
    series = [(MAIN, "kornia-slam", "kornia")] + ([("orb3", "ORB-SLAM3", "orb3")] if has_orb else [])
    for i, (cfg, label, tone) in enumerate(series):
        v = [med[f"{cfg}|{s}"]["ate_rmse"] if f"{cfg}|{s}" in med else np.nan for s in seqs]
        sp = [med[f"{cfg}|{s}"]["ate_spread"] if f"{cfg}|{s}" in med else 0 for s in seqs]
        off = ((i-0.5)*w) if has_orb else 0
        ax.bar(x+off, v, w*0.94, label=label, color=th["series"][tone], zorder=3)
        ax.errorbar(x+off, v, yerr=np.array(sp)/2, fmt="none", ecolor=th["primary"],
                    elinewidth=0.9, capsize=2.5, alpha=.6, zorder=4)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[s] for s in seqs], fontsize=8)
    ax.set_ylabel("ATE RMSE (m, log scale)")
    ax.set_title("Absolute trajectory error — median of runs, whiskers show spread",
                 loc="left", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=2, fontsize=8)
    recede(ax); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return True


def fig_traj(sweep, med, th, path):
    """Top-down paths over ground truth, median-ATE run of each cell."""
    runs = json.loads((Path(sweep)/"runs.json").read_text())
    panels = []
    for seq in SEQS:
        entry = {}
        for cfg, key, frame in ((MAIN, "kornia", "camera"), ("orb3", "orb3", "body")):
            rs = sorted([r for r in runs if r["config"] == cfg and r["seq"] == seq],
                        key=lambda r: r["ate_rmse"])
            if not rs:
                continue
            p = Path(sweep)/cfg/seq/rs[len(rs)//2]["run"]/"traj.txt"
            if p.exists():
                try:
                    entry[key] = evaluate(p, seq, frame=frame)
                except Exception:                              # noqa: BLE001
                    pass
        if entry:
            panels.append((seq, entry))
    if not panels:
        return False
    ncol = 4
    nrow = int(np.ceil(len(panels)/ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 2.9*nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (seq, entry) in zip(axes, panels):
        ref = next(iter(entry.values()))
        ax.plot(ref["gt_all"][:, 0], ref["gt_all"][:, 1], color=th["gt"], lw=2.6,
                alpha=0.85, label="ground truth", zorder=1)
        for key, label, z in (("orb3", "ORB-SLAM3", 2), ("kornia", "kornia-slam", 3)):
            if key in entry:
                e = entry[key]["est"]
                ax.plot(e[:, 0], e[:, 1], color=th["series"][key], lw=1.1,
                        alpha=0.9, label=label, zorder=z)
        ate = entry.get("kornia", {}).get("ate_rmse")
        ax.set_title(f"{SHORT[seq]}" + (f"   {ate:.2f} m" if ate else ""),
                     loc="left", fontsize=9)
        ax.set_aspect("equal"); ax.tick_params(labelsize=7); recede(ax)
    for ax in axes[len(panels):]:
        ax.axis("off")
    h, l = axes[0].get_legend_handles_labels()
    order = ["ground truth", "kornia-slam", "ORB-SLAM3"]
    pairs = sorted(zip(h, l), key=lambda hl: order.index(hl[1]) if hl[1] in order else 9)
    fig.legend([a for a, _ in pairs], [b for _, b in pairs], loc="lower center",
               ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Estimated paths over ground truth (x–y, metres, rigidly aligned)",
                 x=0.02, ha="left", fontsize=11, color=th["primary"])
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return True


def fig_ablation(med, th, path):
    """One subsystem changed at a time — only drawn if ablations were run."""
    order = [(f"kornia-{c}", l) for c, l in
             (("s_only", "stereo only"), ("si_noloop", "SI, no loop"),
              ("si_loop", "SI + loop"), ("mi_loop", "mono-inertial"))]
    seqs = [s for s in SEQS if sum(f"{c}|{s}" in med for c, _ in order) >= 2]
    if len(set(c for c, _ in order if any(f"{c}|{s}" in med for s in SEQS))) < 2:
        return False
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    n = len(order); w = 0.8/n; x = np.arange(len(seqs))
    for i, ((cfg, label), sh) in enumerate(zip(order, (0.35, 0.6, 1.0, 0.8))):
        v = [med[f"{cfg}|{s}"]["ate_rmse"] if f"{cfg}|{s}" in med else np.nan for s in seqs]
        ax.bar(x+(i-(n-1)/2)*w, v, w*0.9, label=label, zorder=3,
               color=th["series"]["kornia"], alpha=sh)
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[s] for s in seqs], fontsize=8)
    ax.set_ylabel("ATE RMSE (m, log scale)")
    ax.set_title("Ablations — same sequences, one subsystem changed", loc="left", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.005), ncol=4, fontsize=7.5)
    recede(ax); ax.grid(axis="x", visible=False)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    return True


def build(sweep, baseline, out, want_pdf):
    sweep = Path(sweep)
    runs, med = score(sweep)
    (sweep/"runs.json").write_text(json.dumps(runs, indent=1))
    (sweep/"medians.json").write_text(json.dumps(med, indent=1))
    base = json.loads(Path(baseline).read_text()) if baseline else None

    figs = sweep/"figs"
    figs.mkdir(exist_ok=True)
    built = {}
    for tname, th in THEMES.items():
        style(th)
        for name, fn in (("compare", lambda p: base and fig_compare(med, base, th, p)),
                         ("ate", lambda p: fig_ate(med, th, p)),
                         ("ablation", lambda p: fig_ablation(med, th, p)),
                         ("traj", lambda p: fig_traj(sweep, med, th, p))):
            ok = fn(figs/f"{name}-{tname}.png")
            built[name] = built.get(name, False) or bool(ok)

    def embed(name, caption):
        if not built.get(name):
            return ""
        def uri(v):
            return "data:image/png;base64," + base64.b64encode(
                (figs/f"{name}-{v}.png").read_bytes()).decode()
        return (f'<figure class="fig"><img class="fig-light" src="{uri("light")}" alt="{caption}">'
                f'<img class="fig-dark" src="{uri("dark")}" alt="{caption}">'
                f'<figcaption>{caption}</figcaption></figure>')

    has_orb = any(k.startswith("orb3|") for k in med)
    rows = ""
    for s_ in SEQS:
        k = f"{MAIN}|{s_}"
        if k not in med:
            continue
        e = med[k]
        cells = [f'<th scope="row">{SHORT[s_]}</th>',
                 f'<td class="num">{e["ate_rmse"]:.3f}'
                 f'<span class="pm">±{e["ate_spread"]:.3f}</span></td>']
        if base:
            if k in base:
                d = (e["ate_rmse"] - base[k]["ate_rmse"]) / base[k]["ate_rmse"]
                cls = "good" if d < -NOISE_FLOOR else ("warn" if d > NOISE_FLOOR else "")
                cells.append(f'<td class="num {cls}">{d * 100:+.1f}%</td>')
            else:
                cells.append('<td class="num none">—</td>')
        if has_orb:
            o = med.get(f"orb3|{s_}")
            cells.append(f'<td class="num">{o["ate_rmse"]:.3f}</td>' if o
                         else '<td class="num none">—</td>')
        cells += [f'<td class="num">{e["ate_rot_rmse"]:.2f}</td>',
                  f'<td class="num">{e["rpe_rot_rmse"]:.2f}</td>',
                  f'<td class="num">{(e["path_ratio"] - 1) * 100:+.1f}%</td>',
                  f'<td class="num">{e["n_runs"]}</td>']
        rows += "<tr>" + "".join(cells) + "</tr>\n"

    css = (Path(__file__).resolve().parent/"report.css").read_text()
    html = f"""<title>SLAM Benchmark</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Serif:ital,wght@0,400;0,600;1,400&display=swap">
<style>{css}</style>
<div class="wrap">
<header>
  <p class="eyebrow">EuRoC MAV · stereo-inertial</p>
  <h1>kornia-slam benchmark</h1>
  <p class="sub">{len(runs)} runs scored from <code>{sweep.name}</code>
     {'against a stored baseline' if base else ''}.</p>
</header>
<hr class="rule">
{embed("compare", f"Current against baseline. Labels appear only where the change clears the ±{NOISE_FLOOR*100:.0f}% run-to-run noise floor.")}
{embed("ate", "ATE per sequence; whiskers span the repeat runs.")}
<section>
  <div class="tbl-wrap"><table class="tbl"><thead><tr>
    <th scope="col">Sequence</th><th scope="col" class="num">ATE (m)</th>
    {'<th scope="col" class="num">vs base</th>' if base else ''}
    {'<th scope="col" class="num">ORB3 (m)</th>' if has_orb else ''}
    <th scope="col" class="num">attitude °</th><th scope="col" class="num">rot drift °/s</th>
    <th scope="col" class="num">path error</th><th scope="col" class="num">runs</th>
  </tr></thead><tbody>{rows}</tbody></table></div>
  <p class="fine">ATE is translation RMSE after a rigid Umeyama fit — not Sim(3), which would
  absorb path-length error into a scale factor and hide it. <em>Rot drift</em> is relative
  rotation error over a one-second interval: local error that no global alignment can flatter.
  Run-to-run spread is around {NOISE_FLOOR*100:.0f}% of ATE, so smaller differences are not
  evidence of anything.</p>
</section>
{embed("ablation", "kornia-slam against itself, one subsystem changed at a time.")}
{embed("traj", "Top-down paths over ground truth, median-ATE run of each sequence.")}
</div>
"""
    Path(out).write_text(html)
    print(f"report → {out}")
    if want_pdf:
        make_pdf(Path(out))
    return 0


def make_pdf(html_path):
    """Render via headless Chrome, pinning the light theme for paper."""
    import shutil, subprocess, tempfile
    chrome = next((c for c in ("google-chrome", "chromium", "chromium-browser")
                   if shutil.which(c)), None)
    if not chrome:
        print("no chrome/chromium on PATH — skipping PDF", file=sys.stderr)
        return
    frag = html_path.read_text()
    head, style_open = frag.split("<style>", 1)
    style_body, body = style_open.split("</style>", 1)
    doc = (f'<!doctype html><html lang="en" data-theme="light"><head>'
           f'<meta charset="utf-8">{head}<style>{style_body}\n'
           '@page { size:A4 portrait; margin:14mm 13mm 16mm; }\n'
           'body { padding:0 !important; font-size:11.2pt; } .wrap { max-width:none; }\n'
           '@media print { * { -webkit-print-color-adjust:exact !important;'
           ' print-color-adjust:exact !important; }\n'
           '  .fig,.note,.stats,figure,.tbl-wrap { break-inside:avoid; }\n'
           '  table.tbl { min-width:0; font-size:8.6pt; }\n'
           '  h1,h2,h3,.eyebrow { break-after:avoid; } figcaption { break-before:avoid; } }'
           f'</style></head><body>{body}</body></html>')
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
        f.write(doc)
        tmp = f.name
    pdf = html_path.with_suffix(".pdf")
    subprocess.call([chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
                     "--virtual-time-budget=25000",
                     "--run-all-compositor-stages-before-draw",
                     f"--print-to-pdf={pdf}", "--no-pdf-header-footer",
                     f"file://{tmp}"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"pdf → {pdf}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--baseline")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pdf", action="store_true")
    a = ap.parse_args()
    sys.exit(build(a.dir, a.baseline, a.out or str(Path(a.dir)/"report.html"), a.pdf))
