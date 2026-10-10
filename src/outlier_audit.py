"""Independent, non-destructive Tukey-IQR screen for selected UCI features.

Run from the repository root:
    python -m src.outlier_audit --csv data/online_shoppers_intention.csv

The audit reports candidate outliers. It does NOT remove records or alter trained models.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

FEATURES = ("ProductRelated_Duration", "PageValues")

def audit(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    rows = []
    for feature in FEATURES:
        values = pd.to_numeric(df[feature], errors="raise").dropna()
        q1, median, q3 = (float(values.quantile(q)) for q in (0.25, 0.5, 0.75))
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        rows.append({
            "Feature": feature, "Q1": q1, "Median": median, "Q3": q3,
            "IQR": iqr, "Lower_fence": lower, "Upper_fence": upper,
            "Flagged_below": int((values < lower).sum()),
            "Flagged_above": int((values > upper).sum()),
            "Total_rows": len(values), "Maximum": float(values.max()),
            "Decision": (
                "Retained plausible extreme engagement" if feature == "ProductRelated_Duration"
                else "Retained zero-inflated feature; IQR flags all positive values"
            ),
        })
    return pd.DataFrame(rows)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/online_shoppers_intention.csv")
    parser.add_argument("--output", default=None, help="Optional CSV output path")
    args = parser.parse_args()
    result = audit(args.csv)
    print(result.to_string(index=False))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output, index=False)

if __name__ == "__main__":
    main()
