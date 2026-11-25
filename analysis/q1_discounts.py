"""Question 1: how much volume discounts buy, and whether they pay."""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from analysis import chart_style as cs

STANDARD_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
CAP = 0.10
N_BOOT = 2000
N_SIM = 4000
SEED = 561
# Northwind has no cost data, so gross margin at list price is assumed.
BASE_MARGIN = 0.30
MARGINS = [0.20, 0.30, 0.40]
MODELS = {
    "across": "LogRelativeQty ~ AnyDiscount + DiscountPts",
    "within": "LogRelativeQty ~ AnyDiscount + DiscountPts + C(CustomerID)",
}


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


def model_data(li, orders):
    """Prepare standard-discount lines for the lift regressions.

    Args:
        li (pandas.DataFrame): Lines with ``RelativeQty`` from
            ``build_table``.
        orders (pandas.DataFrame): Orders, used for ``CustomerID``.

    Returns:
        pandas.DataFrame: Lines at a standard discount level whose
        product has a full-price baseline, with ``CustomerID``,
        ``LogRelativeQty``, ``AnyDiscount`` (0/1), and
        ``DiscountPts`` (discount in percentage points) added.
    """
    df = li.merge(orders[["OrderID", "CustomerID"]], on="OrderID")
    df = df[df["Discount"].isin(STANDARD_LEVELS)].dropna(subset=["RelativeQty"]).copy()
    df["LogRelativeQty"] = np.log(df["RelativeQty"])
    df["AnyDiscount"] = (df["Discount"] > 0).astype(int)
    df["DiscountPts"] = df["Discount"] * 100
    return df


def fit_lift_models(df):
    """Fit the across-customer and within-customer lift regressions.

    Both regress log relative quantity on a discount indicator and
    the discount depth, with standard errors clustered by order. The
    within-customer model adds customer fixed effects, so it compares
    each customer's discounted lines with that customer's own
    full-price lines.

    Args:
        df (pandas.DataFrame): Output of ``model_data``.

    Returns:
        dict[str, dict]: For ``"across"`` and ``"within"``: ``params``
        and ``cov`` (numpy arrays for AnyDiscount and DiscountPts),
        ``pvalues`` (dict), and ``nobs`` (int).
    """
    fits = {}
    keys = ["AnyDiscount", "DiscountPts"]
    for name, formula in MODELS.items():
        m = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["OrderID"]})
        fits[name] = {
            "params": m.params[keys].to_numpy(),
            "cov": m.cov_params().loc[keys, keys].to_numpy(),
            "pvalues": m.pvalues[keys].to_dict(),
            "nobs": int(m.nobs),
        }
    return fits


def lift(discount_pts, params):
    """Return the modeled quantity multiplier at a discount depth.

    Args:
        discount_pts (numpy.ndarray): Discounts in percentage points.
        params (numpy.ndarray): AnyDiscount and DiscountPts
            coefficients, shape ``(2,)`` or ``(n_draws, 2)``.

    Returns:
        numpy.ndarray: Quantity relative to full price; 1 at 0%.
    """
    params = np.atleast_2d(params)
    d = np.asarray(discount_pts, dtype=float)
    log_lift = params[:, [0]] * (d > 0) + params[:, [1]] * d
    return np.exp(log_lift).squeeze()


def profit_change(discount_pts, params, margin):
    """Return gross profit per line relative to selling at full price.

    With list price ``p``, unit cost ``p * (1 - margin)``, and quantity
    multiplied by the modeled lift, profit changes by
    ``lift * (margin - d) / margin - 1``.

    Args:
        discount_pts (numpy.ndarray): Discounts in percentage points.
        params (numpy.ndarray): Lift coefficients, as in ``lift``.
        margin (float): Gross margin at list price, e.g. 0.30.

    Returns:
        numpy.ndarray: Fractional profit change; -0.2 means 20% less
        gross profit than the same line at full price.
    """
    d = np.asarray(discount_pts, dtype=float) / 100
    return lift(discount_pts, params) * (margin - d) / margin - 1


def break_even_discount(params, margin):
    """Find the largest discount that does not reduce gross profit.

    Only the observed range, 5% to 25%, is searched; the models say
    nothing about discounts below 5%.

    Args:
        params (numpy.ndarray): Lift coefficients, shape ``(2,)``.
        margin (float): Gross margin at list price.

    Returns:
        float: Break-even discount in percentage points, to 0.1 pt;
        0.0 if every discount from 5% to 25% loses profit.
    """
    grid = np.arange(5, 25.01, 0.1)
    ok = grid[profit_change(grid, params, margin) >= 0]
    return round(float(ok.max()), 1) if len(ok) else 0.0


def profit_curves(fits, margin, rng):
    """Compute profit-change curves with 95% bands for each model.

    Bands come from ``N_SIM`` draws of the two coefficients from their
    estimated (clustered) sampling distribution.

    Args:
        fits (dict): Output of ``fit_lift_models``.
        margin (float): Gross margin at list price.
        rng (numpy.random.Generator): Random source for the draws.

    Returns:
        pandas.DataFrame: One row per 0.5-point step from 5% to 25%
        with the estimate, lower, and upper profit change per model.
    """
    grid = np.arange(5, 25.01, 0.5)
    out = pd.DataFrame({"DiscountPts": grid})
    for name, fit in fits.items():
        draws = rng.multivariate_normal(fit["params"], fit["cov"], size=N_SIM)
        sims = profit_change(grid, draws, margin)
        out[f"{name}_est"] = profit_change(grid, fit["params"], margin)
        out[f"{name}_low"] = np.percentile(sims, 2.5, axis=0)
        out[f"{name}_high"] = np.percentile(sims, 97.5, axis=0)
    return out


def profit_impact(df, params, margin):
    """Estimate gross profit gained or lost on actual discounted lines.

    Compares each discounted line's profit with the profit it would
    have earned at full price on the quantity implied by removing the
    modeled lift.

    Args:
        df (pandas.DataFrame): Output of ``model_data``.
        params (numpy.ndarray): Lift coefficients, shape ``(2,)``.
        margin (float): Gross margin at list price.

    Returns:
        float: Total dollar change in gross profit; negative means
        discounts cost more profit than the extra volume earned.
    """
    disc = df[df["AnyDiscount"] == 1]
    actual = disc["GrossValue"] * (margin - disc["Discount"])
    full_price = disc["GrossValue"] / lift(disc["DiscountPts"].to_numpy(), params) * margin
    return float((actual - full_price).sum())


def plot_profit(curves, fits, margin, path):
    """Draw profit change by discount depth for both lift estimates.

    Args:
        curves (pandas.DataFrame): Output of ``profit_curves``.
        fits (dict): Output of ``fit_lift_models``.
        margin (float): Gross margin the curves assume.
        path (pathlib.Path): Output PNG path.

    Returns:
        None. Writes the figure to ``path``.
    """
    labels = {
        "across": ("All customers pooled", cs.BLUE),
        "within": ("Same customer, with vs. without discount", cs.ORANGE),
    }
    fig, ax = cs.plt.subplots(figsize=(7.6, 4.6))
    ax.axhline(0, color=cs.INK_SECONDARY, linewidth=1, zorder=1)
    x = curves["DiscountPts"]
    for name, (label, color) in labels.items():
        lift_pct = (np.exp(fits[name]["params"][0]) - 1) * 100
        ax.fill_between(x, curves[f"{name}_low"] * 100, curves[f"{name}_high"] * 100,
                        color=color, alpha=0.15, linewidth=0, zorder=2)
        ax.plot(x, curves[f"{name}_est"] * 100, color=color, linewidth=2, zorder=3,
                label=f"{label} (lift {lift_pct:+.0f}%)")
    ax.scatter([0], [0], s=60, color=cs.NEUTRAL, edgecolor=cs.SURFACE, linewidth=2, zorder=4)
    ax.annotate("Full price", (0, 0), xytext=(8, 8), textcoords="offset points",
                fontsize=8.5, color=cs.INK_SECONDARY)
    ax.set_xlim(-0.8, 25.8)
    ax.set_ylim(-100, 50)
    ax.set_xticks([0, 5, 10, 15, 20, 25], ["0%", "5%", "10%", "15%", "20%", "25%"])
    ax.set_xlabel("Discount on the order line (Northwind only uses 5% to 25%)")
    ax.set_ylabel("Gross profit vs. full price (%)")
    ax.set_title("Only a 5% discount can pay for itself, and only on the generous estimate",
                 pad=24)
    ax.text(0, 1.025, f"Assumes a {margin:.0%} gross margin at list price. "
            "Lines: estimated effect; bands: 95% interval.",
            transform=ax.transAxes, fontsize=8.5, color=cs.INK_SECONDARY)
    ax.legend(loc="lower left")
    cs.save(fig, path)


def run(nw, fig_dir, out_dir):
    """Run the Question 1 analysis and save its outputs.

    Args:
        nw (Northwind): Loaded data.
        fig_dir (pathlib.Path): Folder for ``q1_discount_lift.png``
            and ``q1_profit_by_discount.png``.
        out_dir (pathlib.Path): Folder for ``q1_discount_levels.csv``,
            ``q1_lift_models.csv``, and ``q1_break_even.csv``.

    Returns:
        dict[str, Any]: Headline numbers for the webpage.
    """
    table, li = build_table(nw.lines)
    table.round(4).to_csv(out_dir / "q1_discount_levels.csv", index=False)
    plot(table, fig_dir / "q1_discount_lift.png")

    df = model_data(li, nw.orders)
    fits = fit_lift_models(df)
    rng = np.random.default_rng(SEED)
    curves = profit_curves(fits, BASE_MARGIN, rng)
    plot_profit(curves, fits, BASE_MARGIN, fig_dir / "q1_profit_by_discount.png")

    model_rows, be_rows = [], []
    for name, fit in fits.items():
        se = np.sqrt(np.diag(fit["cov"]))
        model_rows.append({
            "Model": name,
            "Lines": fit["nobs"],
            "LiftAnyDiscount": np.exp(fit["params"][0]) - 1,
            "LiftAnyDiscount_CI_Low": np.exp(fit["params"][0] - 1.96 * se[0]) - 1,
            "LiftAnyDiscount_CI_High": np.exp(fit["params"][0] + 1.96 * se[0]) - 1,
            "P_AnyDiscount": fit["pvalues"]["AnyDiscount"],
            "LiftPerExtraPoint": np.exp(fit["params"][1]) - 1,
            "P_PerExtraPoint": fit["pvalues"]["DiscountPts"],
        })
        for margin in MARGINS:
            be_rows.append({
                "Model": name,
                "Margin": margin,
                "BreakEvenDiscountPts": break_even_discount(fit["params"], margin),
                "ProfitImpactOnDiscountedLines": profit_impact(df, fit["params"], margin),
            })
    models = pd.DataFrame(model_rows)
    break_even = pd.DataFrame(be_rows)
    models.round(4).to_csv(out_dir / "q1_lift_models.csv", index=False)
    break_even.round(2).to_csv(out_dir / "q1_break_even.csv", index=False)
    curves.round(4).to_csv(out_dir / "q1_profit_curves.csv", index=False)

    at_base = break_even[break_even["Margin"] == BASE_MARGIN].set_index("Model")
    by_level = curves.set_index("DiscountPts")

    over, savings = cap_scenario(nw.lines)
    discounted = li[li["Discount"] > 0]
    orders_with_discount = (nw.orders["DiscountValue"] > 0).mean()
    return {
        "lift_models": models.round(4).to_dict(orient="records"),
        "assumed_margin": BASE_MARGIN,
        "break_even_pts_at_base_margin": at_base["BreakEvenDiscountPts"].to_dict(),
        "profit_impact_at_base_margin": at_base["ProfitImpactOnDiscountedLines"].round(2).to_dict(),
        "profit_change_at_levels_across": {
            f"{lv}%": round(float(by_level.loc[lv, "across_est"]), 4) for lv in [5, 10, 15, 20, 25]
        },
        "profit_change_at_levels_within": {
            f"{lv}%": round(float(by_level.loc[lv, "within_est"]), 4) for lv in [5, 10, 15, 20, 25]
        },
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
