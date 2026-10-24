"""Build the submission report (HTML -> PDF via headless Chrome).

    python report/build_report.py --account <omnicampusアカウント名>

Inputs: outputs/eval.json, outputs/results.json, report/figures/*.png (see --figures).
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"
FIG = REPORT / "figures"


def chart(perc: dict, path: Path):
    for f in font_manager.findSystemFonts():
        if "NotoSansCJK-Regular" in f:
            font_manager.fontManager.addfont(f)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
    fig, ax = plt.subplots(figsize=(5.2, 2.6), dpi=200)
    labels = ["OWLv2のみ\n(全270件)", "OWLv2のみ\n(物体180件)", "OWLv2+3D検証\n(全270件)"]
    vals = [perc["label_acc_owl_only"] * 100, perc["label_acc_owl_only_objects"] * 100, perc["label_acc_hybrid"] * 100]
    bars = ax.bar(labels, vals, color=["#9aa5b8", "#c3ccdb", "#2f7de1"], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.1f}%", ha="center", fontsize=10)
    ax.set_ylim(0, 112)
    ax.set_ylabel("ラベル正解率 (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def build(account: str, video_url: str):
    ev = json.loads((ROOT / "outputs/eval.json").read_text())
    demo = json.loads((ROOT / "outputs/results.json").read_text())
    perc, task = ev["perception"], ev["task"]
    chart(perc, FIG / "accuracy.png")
    demo_rows = "".join(
        f"<tr><td>{html.escape(r['title'])}</td><td>{r['success']}/{r['total']}</td><td>{r['stats']['picks']}</td>"
        f"<td>{r['stats']['grasp_failures']}</td><td>{', '.join(r['stats']['skipped']) or '–'}</td>"
        f"<td>{r['stats']['owl_corrected']}/{r['stats']['owl_boxes']}</td></tr>"
        for r in demo
    )
    infer = [s for r in demo for s in r["stats"]["infer_seconds"]]
    avg_infer = sum(infer) / len(infer)
    n_ep_ok = round(task["episode_success_rate"] * task["trials"])
    tpl = (REPORT / "template.html").read_text()
    out = tpl.format(
        account=html.escape(account),
        video_url=html.escape(video_url),
        acc_owl=f"{perc['label_acc_owl_only'] * 100:.1f}",
        acc_owl_obj=f"{perc['label_acc_owl_only_objects'] * 100:.1f}",
        acc_hyb=f"{perc['label_acc_hybrid'] * 100:.1f}",
        pos_mean=perc["pos_err_mm_mean"], pos_max=perc["pos_err_mm_max"], yaw=perc["cube_yaw_err_deg_mean"],
        scenes=perc["scenes"], items=perc["items"],
        trials=task["trials"], obj_rate=f"{task['object_success_rate'] * 100:.1f}", ep_ok=n_ep_ok,
        ep_rate=f"{task['episode_success_rate'] * 100:.1f}",
        demo_rows=demo_rows, infer=f"{avg_infer:.1f}",
    )
    html_path = REPORT / "report.html"
    html_path.write_text(out)
    pdf = REPORT / f"説明資料_PAI最終課題_{account}.pdf"
    chrome = shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")
    pdf.unlink(missing_ok=True)
    cmd = [chrome, "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
           f"--user-data-dir={REPORT / '.chrome'}", f"--print-to-pdf={pdf}", html_path.as_uri()]
    try:
        subprocess.run(cmd, capture_output=True, timeout=90)
    except subprocess.TimeoutExpired:
        pass  # headless Chrome sometimes lingers after writing the PDF
    shutil.rmtree(REPORT / ".chrome", ignore_errors=True)
    if not pdf.exists():
        raise SystemExit("PDF generation failed")
    print(pdf)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--account", default="omnicampusアカウント名")
    ap.add_argument("--video-url", default="（YouTube の URL をここに記載）")
    a = ap.parse_args()
    build(a.account, a.video_url)
