"""Build all node-link figures: matplotlib steps + TikZ compile + PDF -> JPG."""
import os
import subprocess
import sys
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "figures"
PDFLATEX = Path(os.environ["APPDATA"]) / "TinyTeX" / "bin" / "windows" / "pdflatex.exe"

# 1) the three matplotlib renders
r = subprocess.run([sys.executable, "mlp_matplotlib.py"], cwd=HERE, capture_output=True, text=True)
print(r.stdout.strip())
if r.returncode != 0:
    print(r.stderr)
    raise SystemExit("matplotlib script failed")

# 2) the TikZ version
r = subprocess.run([str(PDFLATEX), "-interaction=nonstopmode", "-halt-on-error", "mlp_tikz.tex"],
                   cwd=HERE, capture_output=True, text=True)
if not (HERE / "mlp_tikz.pdf").exists():
    print(r.stdout[-3000:])
    raise SystemExit("pdflatex failed")

doc = pymupdf.open(HERE / "mlp_tikz.pdf")
pix = doc[0].get_pixmap(matrix=pymupdf.Matrix(2.4, 2.4), alpha=False)
pix.save(OUT / "nl-tikz.jpg", jpg_quality=90)
print(f"saved nl-tikz.jpg: {pix.width}x{pix.height}")
print("done")
