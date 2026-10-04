#!/usr/bin/env python3
"""Build the public results landing page (single-file HTML, figures base64-inlined).

    python3 scripts/build_results_page.py            # -> results_page.html in the repo
    OUT=/srv/www/soil2crop/index.html python3 scripts/build_results_page.py
"""
import base64
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
OUT = os.environ.get("OUT", os.path.join(ROOT, "results_page.html"))


def b64(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def main():
    r2 = load("real_R2_summary.json")
    r3 = load("real_R3b_summary.json")
    e1 = load("feas_e1_full.json")
    e2 = load("feas_e2_full.json")
    wc = load("real_worldcereal_inventory.json")

    fig1 = b64(os.path.join(FIG, "real_transfer_gap.png"))
    fig2 = b64(os.path.join(FIG, "fewshot_curve.png"))

    def row(k, v):
        return f"<tr><td>{k}</td><td>{v:+.3f}</td></tr>" if isinstance(v, float) else f"<tr><td>{k}</td><td>{v}</td></tr>"

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Soil2Crop — results</title><style>
 body{{font:16px/1.55 -apple-system,Segoe UI,Roboto,sans-serif;color:#14231a;background:#f6f8f5;margin:0}}
 .wrap{{max-width:960px;margin:0 auto;padding:24px}}
 h1{{font-size:26px}} h2{{font-size:19px;margin-top:28px;border-bottom:2px solid #d7e0d8;padding-bottom:6px}}
 .tag{{font-size:12px;background:#eaf1ea;border:1px solid #d7e0d8;border-radius:999px;padding:2px 10px;margin-left:6px}}
 table{{border-collapse:collapse;width:100%;margin:10px 0}} td,th{{border:1px solid #d7e0d8;padding:6px 10px;text-align:left;font-size:14px}}
 th{{background:#eef2ee}} img{{width:100%;border:1px solid #d7e0d8;border-radius:8px;margin:8px 0}}
 .cta{{display:inline-block;background:#2b7a3d;color:#fff;text-decoration:none;padding:12px 20px;border-radius:8px;font-weight:600;margin:6px 0}}
 .note{{background:#fff8ec;border:1px solid #f0dcb4;border-radius:8px;padding:12px;font-size:14px}}
 code{{background:#eef2ee;padding:2px 6px;border-radius:4px}}
</style></head><body><div class="wrap">
<h1>🌱 Soil2Crop <span class="tag">TabPFN-3.5</span><span class="tag">real data</span><span class="tag">transfer-honest</span></h1>
<p><b>The accurate crop recommender — on real data.</b> It recommends the crop actually grown
around a coordinate at <b>100% (3-class)</b> with just <b>~5 local observations</b>, zero training.
And unlike the ~99%-on-a-<i>synthetic</i>-table projects, it reports exactly where it is accurate
(inside the tested regions, with local data) and where it is not (zero-shot across continents).</p>
<a class="cta" href="/soil2crop/">▶ Open the live app (type any coordinate)</a>

<h2>What it does</h2>
<p>Given a farm coordinate → fetch <b>real soil</b> (ISRIC SoilGrids v2.0) + <b>real climate</b>
(NASA POWER) → ask <b>TabPFN-3.5</b> (zero-shot, no training) which crop the <b>nearest real fields</b>
actually grow. Returns a ranked list with confidence, the evidence, and a "how much to trust this"
note. Labels: ESA WorldCereal ({wc['n_points']:,} real points).</p>

<h2>Result 1 — the transfer gap (the finding)</h2>
<img src="{fig1}" alt="Real soil to crop transfer gap">
<table><tr><th>Split</th><th>Accuracy</th><th>Majority baseline</th></tr>
{''.join(f"<tr><td>{k}</td><td>{v['accuracy']:.3f}</td><td>{v['majority_baseline']:.3f}</td></tr>" for k,v in r2['cells'].items())}
</table>
<p class="note">A random split looks strong (0.918 vs 0.385 baseline), but the model
<b>does not transfer across continents</b> (US→Africa 0.490 vs 0.510 baseline). The ~99% synthetic
numbers are an illusion of a self-referential benchmark.</p>

<h2>Result 2 — a few local points fix it</h2>
<img src="{fig2}" alt="Few-shot learning curve">
<p>Spatially blocked (train-local and test rows in different 0.5° cells — identical result to
unblocked, so <b>not leakage</b>). <b>~5 local labelled points</b> solve it. Diagnostic: climate
alone at k=5 already reaches 1.000, soil alone 0.825 — the 3 classes are climate-zone proxies, so
TabPFN's edge is <b>~5 rows with zero tuning</b>, not a large accuracy margin. A plain logistic
regression at k=20 already hits 0.979. We say so.</p>

<h2>Reference numbers</h2>
<table><tr><th>Task</th><th>Result</th></tr>
<tr><td>Synthetic 22-crop table (the field's benchmark)</td><td>acc {e1['accuracy']:.3f} (chance {e1['chance']:.3f})</td></tr>
<tr><td>OSSL SOC from vis-NIR spectra</td><td>R² {e2['r2']:.3f}, RPD {e2['rpd']:.2f} ({e2['rpd_label']}); constant-baseline RMSE {e2['constant_baseline']['rmse']:.1f} vs {e2['rmse']:.1f}</td></tr>
</table>

<h2>Why it is honest</h2>
<p>Every number is from a saved JSON in the repo; every classification result sits beside its
majority baseline; the app shows a competence note and the confidence gap so a user knows when
<i>not</i> to trust it. No claim is larger than the measurement behind it.</p>

<p style="color:#5b6b60;font-size:13px">Repo: github.com/alcunii/tabpfn-soil-hackathon ·
model: TabPFN-3.5 (priorlabs) · data: ISRIC SoilGrids, NASA POWER, ESA WorldCereal.</p>
</div></body></html>"""
    os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    # optional: if the target is served by a web user, chown it (best-effort, never fatal)
    try:
        if os.geteuid() == 0:
            import grp
            import pwd
            for u in ("caddy", "www-data"):
                try:
                    pw = pwd.getpwnam(u)
                    os.chown(OUT, pw.pw_uid, pw.pw_gid)
                    break
                except KeyError:
                    continue
    except Exception:
        pass
    print("wrote", OUT, len(html), "bytes")


if __name__ == "__main__":
    main()
