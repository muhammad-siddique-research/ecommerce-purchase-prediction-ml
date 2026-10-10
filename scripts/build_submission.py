"""Build a self-contained submission ZIP for the university predictive modeling mini-project."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
FOLDERS = ("data", "notebooks", "src", "results")
EXTRA = ("README.md", "requirements.txt", "CITATION.cff", "reports/research_report_final.pdf", "reports/research_report.md")
ZIP_OUT = ROOT / "dist" / "Predictive_Modeling_Mini_Project_Submission.zip"
PREFIX = "Predictive_Modeling_Mini_Project"

def main() -> None:
    raw = ROOT / "data" / "online_shoppers_intention.csv"
    with raw.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 12330, "Real UCI dataset must contain 12,330 sessions"
    assert sum(str(r["Revenue"]).strip().lower() == "true" for r in rows) == 1908, "Revenue prevalence mismatch"
    assert len(rows[0]) == 18, "Dataset must contain 17 features and the Revenue target"

    nb_file = ROOT / "notebooks" / "Algorithm_Evaluation_Extravaganza_UCI.ipynb"
    nb = json.loads(nb_file.read_text(encoding="utf-8"))
    assert sum(bool(c.get("outputs")) for c in nb["cells"] if c["cell_type"] == "code") == 13, "Executed notebook outputs missing"
    markdown = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert "[INSERT" not in markdown and "[MODEL]" not in markdown and "[VALUE]" not in markdown, "Notebook placeholders remain"

    files = [ROOT / n for n in EXTRA]
    for dirname in FOLDERS:
        files.extend(p for p in (ROOT / dirname).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    # Preserve all measured figures/CSVs, while excluding compiled caches or artifacts.
    files = sorted(set(files))
    assert len(list((ROOT / "results" / "figures").glob("*.png"))) == 7, "Seven visualizations required"
    assert (ROOT / "results" / "outlier_screening.csv") in files, "Screening report missing"
    for f in files:
        assert f.exists() and f.stat().st_size > 0, f"Missing/empty artifact: {f}"
    ZIP_OUT.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=8) as z:
        for f in files:
            z.write(f, arcname=f"{PREFIX}/{f.relative_to(ROOT).as_posix()}")
        z.writestr(f"{PREFIX}/SUBMISSION_README.txt", (
            "Predictive Modeling Mini Project - university submission\n"
            "Researcher: Muhammad Siddique\n"
            "Original data: UCI Online Shoppers Purchasing Intention Dataset (CC BY 4.0)\n"
            "Dataset citation: https://doi.org/10.24432/C5F88Q\n"
            "Start with reports/research_report_final.pdf and notebooks/Algorithm_Evaluation_Extravaganza_UCI.ipynb.\n"
            "Run locally: pip install -r requirements.txt\n"
            "Then: python -m src.run_experiment --csv data/online_shoppers_intention.csv\n"
            "Outliers: python -m src.outlier_audit --csv data/online_shoppers_intention.csv\n"
            "The saved results come from the executed notebook, not from synthetic tests.\n"
            "Submission URL: https://github.com/muhammad-siddique-research/ecommerce-purchase-prediction-ml\n"
        ))
    with zipfile.ZipFile(ZIP_OUT) as z:
        assert z.testzip() is None
        print(f"Created {ZIP_OUT} ({ZIP_OUT.stat().st_size:,} bytes; {len(z.namelist())} files)")

if __name__ == "__main__":
    main()
