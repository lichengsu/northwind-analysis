"""Regenerate all figures, tables, and headline numbers."""

import json
from pathlib import Path

from analysis import chart_style, load_data, q1_discounts, q2_shipping, q3_sales_reps

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "docs" / "figures"
OUT_DIR = ROOT / "outputs"


def main():
    """Run all analyses and save their figures and tables.

    Args:
        None.

    Returns:
        None. Writes ``outputs/key_numbers.json`` and prints it.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart_style.apply()

    nw = load_data.load()
    results = {
        "data_checks": load_data.sanity_checks(nw),
        "q1_discounts": q1_discounts.run(nw, FIG_DIR, OUT_DIR),
        "q2_shipping": q2_shipping.run(nw, FIG_DIR, OUT_DIR),
        "q3_sales_reps": q3_sales_reps.run(nw, FIG_DIR, OUT_DIR),
    }

    with open(OUT_DIR / "key_numbers.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
