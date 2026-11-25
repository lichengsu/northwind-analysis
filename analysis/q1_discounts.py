"""Question 1: do deeper discounts buy bigger orders?"""

import numpy as np
import pandas as pd

from analysis import chart_style as cs

STANDARD_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
CAP = 0.10
N_BOOT = 2000
SEED = 561


def label_level(discount):
    """Return the display label for a line's discount.

    Args:
        discount (float): Discount as a fraction rounded to two
            decimals, e.g. 0.15.

    Returns:
        str: ``"0%"`` to ``"25%"`` for the standard levels, otherwise
        ``"Other (1-6%)"``.
    """
    if discount in STANDARD_LEVELS:
        return f"{discount:.0%}"
    return "Other (1-6%)"


def bootstrap_median_ci(values, rng):
    """Estimate a 95% bootstrap interval for the median.

    Args:
        values (array-like of float): Sample to resample.
        rng (numpy.random.Generator): Random source; advanced by
            ``N_BOOT`` resamples of the sample size.

    Returns:
        numpy.ndarray: The 2.5th and 97.5th percentiles of the
        bootstrap medians.
    """
    values = np.asarray(values)
    draws = rng.choice(values, size=(N_BOOT, len(values)), replace=True)
    medians = np.median(draws, axis=1)
    return np.percentile(medians, [2.5, 97.5])


def build_table(lines):
    """Summarize order quantity and discount cost by discount level.

    Each line's quantity is divided by the same product's median
    quantity on full-price lines, so products of different sizes can
    be compared.

    Args:
        lines (pandas.DataFrame): Order lines from ``Northwind.lines``.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame]: The summary table,
        one row per level from 0% to 25% with "Other (1-6%)" last;
        and a copy of ``lines`` with ``Level``, ``BaselineQty``, and
        ``RelativeQty`` added (missing when the product was never sold
        at full price).
    """
    li = lines.copy()
    li["Level"] = li["Discount"].map(label_level)

    # Each product's typical order size when sold at full price.
    baseline = (
        li[li["Discount"] == 0].groupby("ProductID")["Quantity"].median().rename("BaselineQty")
    )
    li = li.merge(baseline, left_on="ProductID", right_index=True, how="left")
    li["RelativeQty"] = li["Quantity"] / li["BaselineQty"]

    rng = np.random.default_rng(SEED)
    rows = []
    order = [f"{d:.0%}" for d in STANDARD_LEVELS] + ["Other (1-6%)"]
    for level in order:
        grp = li[li["Level"] == level]
        rel = grp["RelativeQty"].dropna()
        low, high = bootstrap_median_ci(rel, rng)
        rows.append({
            "DiscountLevel": level,
            "Lines": len(grp),
            "LinesWithBaseline": len(rel),
            "MedianQuantity": grp["Quantity"].median(),
            "MedianRelativeQty": rel.median(),
            "RelativeQtyCI_Low": low,
            "RelativeQtyCI_High": high,
            "Revenue": grp["Revenue"].sum(),
            "DiscountGiven": grp["DiscountValue"].sum(),
        })
    table = pd.DataFrame(rows)
    table["ShareOfDiscountGiven"] = table["DiscountGiven"] / table["DiscountGiven"].sum()
    return table, li


def plot(table, path):
    """Draw median relative quantity by discount level with intervals.

    Args:
        table (pandas.DataFrame): Summary from ``build_table``; the
            "Other (1-6%)" row is left out of the chart.
        path (pathlib.Path): Output PNG path.

    Returns:
        None. Writes the figure to ``path``.
    """
    t = table[table["DiscountLevel"] != "Other (1-6%)"].reset_index(drop=True)
    x = np.arange(len(t))
    colors = [cs.NEUTRAL] + [cs.BLUE] * (len(t) - 1)

    fig, ax = cs.plt.subplots(figsize=(7.2, 4.2))
    ax.axhline(1.0, color=cs.AXIS, linewidth=1, zorder=1)
    ax.vlines(x, t["RelativeQtyCI_Low"], t["RelativeQtyCI_High"], color=colors, linewidth=2, zorder=2)
    ax.scatter(x, t["MedianRelativeQty"], s=70, color=colors, edgecolor=cs.SURFACE, linewidth=2, zorder=3)
    for xi, row in t.iterrows():
        ax.annotate(f"{row['MedianRelativeQty']:.2f}x", (xi, row["MedianRelativeQty"]),
                    xytext=(10, 0), textcoords="offset points", va="center",
                    fontsize=9, color=cs.INK)
        ax.annotate(f"n={row['Lines']}", (xi, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, 6), textcoords="offset points", ha="center",
                    fontsize=8, color=cs.MUTED)
    ax.set_xticks(x, t["DiscountLevel"])
    ax.set_xlabel("Discount on the order line")
    ax.set_ylabel("Quantity vs. product's full-price median (x)")
    ax.set_ylim(0.6, max(2.0, t["RelativeQtyCI_High"].max() + 0.1))
    ax.set_title("A 5% discount lifts order size about as much as 10-20% does", pad=24)
    ax.text(0, 1.025, "Dots: median line quantity relative to the same product sold at 0%. "
            "Bars: 95% bootstrap interval.", transform=ax.transAxes, fontsize=8.5,
            color=cs.INK_SECONDARY)
    cs.save(fig, path)


def cap_scenario(lines):
    """Estimate the discount saved if no line exceeded ``CAP``.

    Assumes quantities would not change under the cap, so the result
    is an upper bound.

    Args:
        lines (pandas.DataFrame): Order lines from ``Northwind.lines``.

    Returns:
        tuple[pandas.DataFrame, float]: Lines discounted above
        ``CAP``, and the dollars saved by cutting them to ``CAP``.
    """
    over = lines[lines["Discount"] > CAP]
    savings = (over["GrossValue"] * (over["Discount"] - CAP)).sum()
    return over, savings


def run(nw, fig_dir, out_dir):
    """Run the Question 1 analysis and save its outputs.

    Args:
        nw (Northwind): Loaded data.
        fig_dir (pathlib.Path): Folder for ``q1_discount_lift.png``.
        out_dir (pathlib.Path): Folder for ``q1_discount_levels.csv``.

    Returns:
        dict[str, int | float]: Headline numbers for the webpage.
    """
    table, li = build_table(nw.lines)
    table.round(4).to_csv(out_dir / "q1_discount_levels.csv", index=False)
    plot(table, fig_dir / "q1_discount_lift.png")

    over, savings = cap_scenario(nw.lines)
    discounted = li[li["Discount"] > 0]
    orders_with_discount = (nw.orders["DiscountValue"] > 0).mean()
    return {
        "discount_lines": int(len(discounted)),
        "discount_line_share": round(len(discounted) / len(li), 4),
        "orders_with_any_discount_share": round(float(orders_with_discount), 4),
        "discount_given_total": round(float(li["DiscountValue"].sum()), 2),
        "discount_given_share_of_gross": round(float(li["DiscountValue"].sum() / li["GrossValue"].sum()), 4),
        "median_rel_qty_discounted": round(float(discounted["RelativeQty"].median()), 3),
        "median_rel_qty_full_price": round(float(li.loc[li["Discount"] == 0, "RelativeQty"].median()), 3),
        "lines_above_cap": int(len(over)),
        "discount_given_above_cap_lines": round(float(over["DiscountValue"].sum()), 2),
        "savings_if_capped_at_10pct": round(float(savings), 2),
    }
