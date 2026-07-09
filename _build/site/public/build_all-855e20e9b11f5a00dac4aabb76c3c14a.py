"""Build all step scripts: python -> .tex -> pdflatex -> .pdf -> .jpg"""
import os
import subprocess
import sys
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
PDFLATEX = Path(os.environ["APPDATA"]) / "TinyTeX" / "bin" / "windows" / "pdflatex.exe"
OUT = Path(r"D:\gis\visual\figures")

STEPS = ["step1_minimal", "step2_encoder", "step3_unet"]
JPG_NAMES = {
    "step1_minimal": "nn-sub-minimal.jpg",
    "step2_encoder": "nn-sub-encoder.jpg",
    "step3_unet": "nn-unet-main.jpg",
}

for step in STEPS:
    print(f"=== {step}")
    # 1) generate the .tex
    r = subprocess.run([sys.executable, f"{step}.py"], cwd=HERE, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr)
        raise SystemExit(f"{step}.py failed")
    # 2) compile with pdflatex
    r = subprocess.run(
        [str(PDFLATEX), "-interaction=nonstopmode", "-halt-on-error", f"{step}.tex"],
        cwd=HERE, capture_output=True, text=True,
    )
    if not (HERE / f"{step}.pdf").exists():
        print(r.stdout[-3000:])
        raise SystemExit(f"pdflatex failed for {step}")
    # 3) render PDF page -> JPG
    doc = pymupdf.open(HERE / f"{step}.pdf")
    page = doc[0]
    zoom = 2.2 if step != "step3_unet" else 1.8
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    out_path = OUT / JPG_NAMES[step]
    pix.save(out_path, jpg_quality=90)
    print(f"  -> {out_path.name}: {pix.width}x{pix.height}, {out_path.stat().st_size // 1024} KB")

print("done")
