"""Question 2: what drives late shipments, and is any shipper better?"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.proportion import proportion_confint

from analysis import chart_style as cs

PROMISE_LABELS = {14: "14 days\n(rush)", 28: "28 days\n(standard)", 42: "42 days\n(relaxed)"}
SLOW_DISPATCH_DAYS = 14
FEDERAL_ID = 3


def shipped_orders(nw):
    """Return shipped orders with freight share and shipper name.

    Args:
        nw (Northwind): Loaded data.

    Returns:
        pandas.DataFrame: Copy of shipped orders with ``Late`` as bool,
        ``Slow`` (more than 14 days to dispatch), ``Federal`` (0/1),
        ``FreightShare`` (freight / net revenue), and ``Shipper``.
    """
    o = nw.orders[nw.orders["Shipped"]].copy()
    o["Late"] = o["Late"].astype(bool)
    o["Slow"] = o["DaysToShip"] > SLOW_DISPATCH_DAYS
    o["Federal"] = (o["ShipVia"] == FEDERAL_ID).astype(int)
    o["FreightShare"] = o["Freight"] / o["Revenue"]
    names = nw.shippers.set_index("ShipperID")["CompanyName"]
    o["Shipper"] = o["ShipVia"].map(names)
    return o


def summarize(o, by):
    """Compute late rate, dispatch time, and freight by group.

    Args:
        o (pandas.DataFrame): Orders from ``shipped_orders``.
        by (str | list[str]): Column or columns to group by.

    Returns:
        pandas.DataFrame: One row per group with order, late, and
        slow-dispatch counts and rates, 95% Wilson intervals for both
        rates, median and 90th-percentile days to ship, median
        freight, and median freight share of order value.
    """
    t = o.groupby(by).agg(
        Orders=("OrderID", "size"),
        LateOrders=("Late", "sum"),
        LateRate=("Late", "mean"),
        SlowOrders=("Slow", "sum"),
        SlowRate=("Slow", "mean"),
        MedianDaysToShip=("DaysToShip", "median"),
        P90DaysToShip=("DaysToShip", lambda s: s.quantile(0.9)),
        MedianFreight=("Freight", "median"),
        MedianFreightShareOfValue=("FreightShare", "median"),
    ).reset_index()
    for col in ["Late", "Slow"]:
        low, high = proportion_confint(t[f"{col}Orders"], t["Orders"], method="wilson")
        t[f"{col}Rate_CI_Low"] = low
        t[f"{col}Rate_CI_High"] = high
    return t


def shipper_tests(o):
    """Test whether Federal Shipping differs from the other shippers.

    Each outcome gets a chi-square test across all three shippers, a
    Fisher exact test of Federal against the other two combined, and
    a logistic regression that controls for the promised window (and,
    for slow dispatch, order value and line count). Orders promised in
    42 days are left out of the late-rate regression because none of
    them were late.

    Args:
        o (pandas.DataFrame): Orders from ``shipped_orders``.

    Returns:
        pandas.DataFrame: One row per test with the outcome, test
        name, Federal's rate, the others' rate, the effect (odds ratio
        where one applies), and the p-value.
    """
    rows = []
    fed, other = o[o["Federal"] == 1], o[o["Federal"] == 0]
    controls = {
        "Late": "C(PromisedDays, Treatment(28))",
        "Slow": "C(PromisedDays, Treatment(28)) + np.log(Revenue) + LineCount",
    }
    for outcome, label in [("Late", "Shipped late"), ("Slow", "Over 14 days to dispatch")]:
        base = {"Outcome": label, "FederalRate": fed[outcome].mean(),
                "OthersRate": other[outcome].mean()}
        _, p3, _, _ = stats.chi2_contingency(pd.crosstab(o["ShipVia"], o[outcome]))
        rows.append({**base, "Test": "Chi-square, all three shippers",
                     "Effect": np.nan, "PValue": p3})
        odds, p_fisher = stats.fisher_exact(pd.crosstab(o["Federal"], o[outcome]))
        rows.append({**base, "Test": "Fisher exact, Federal vs. others",
                     "Effect": odds, "PValue": p_fisher})
        data = o if outcome == "Slow" else o[o["PromisedDays"] != 42]
        data = data.assign(**{outcome: data[outcome].astype(int)})
        model = smf.logit(f"{outcome} ~ Federal + {controls[outcome]}", data=data).fit(disp=0)
        rows.append({**base, "Test": "Logistic regression with controls",
                     "Effect": float(np.exp(model.params["Federal"])),
                     "PValue": float(model.pvalues["Federal"])})
    mw = stats.mannwhitneyu(fed["DaysToShip"], other["DaysToShip"])
    rows.append({"Outcome": "Days to dispatch", "Test": "Mann-Whitney, Federal vs. others",
                 "FederalRate": fed["DaysToShip"].median(),
                 "OthersRate": other["DaysToShip"].median(),
                 "Effect": np.nan, "PValue": mw.pvalue})
    kw = stats.kruskal(*[g["FreightShare"] for _, g in o.groupby("ShipVia")])
    rows.append({"Outcome": "Freight as share of order value",
                 "Test": "Kruskal-Wallis, all three shippers",
                 "FederalRate": fed["FreightShare"].median(),
                 "OthersRate": other["FreightShare"].median(),
                 "Effect": np.nan, "PValue": kw.pvalue})
    return pd.DataFrame(rows)


def plot_late_by_promise(by_promise, path):
    """Draw late rate by promised delivery window.

    Args:
        by_promise (pandas.DataFrame): ``summarize`` output grouped by
            ``PromisedDays``; the 14-day bar is highlighted.
        path (pathlib.Path): Output PNG path.

    Returns:
        None. Writes the figure to ``path``.
    """
    fig, ax = cs.plt.subplots(figsize=(6.2, 4.2))
    x = np.arange(len(by_promise))
    colors = [cs.ORANGE if d == 14 else cs.BLUE for d in by_promise["PromisedDays"]]
    ax.bar(x, by_promise["LateRate"] * 100, width=0.55, color=colors,
           edgecolor=cs.SURFACE, linewidth=2)
    for xi, row in zip(x, by_promise.itertuples()):
        ax.annotate(f"{row.LateRate:.1%}\n{row.LateOrders} of {row.Orders}",
                    (xi, row.LateRate * 100), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=cs.INK)
    ax.set_xticks(x, [PROMISE_LABELS[d] for d in by_promise["PromisedDays"]])
    ax.set_ylabel("Orders shipped after the required date (%)")
    ax.set_ylim(0, 22)
    ax.grid(axis="x", visible=False)
    ax.set_title("Rush orders are late 5x as often as standard orders")
    cs.save(fig, path)


def plot_shipper_comparison(by_shipper, tests, path):
    """Draw late and slow-dispatch rates by shipper with 95% intervals.

    Args:
        by_shipper (pandas.DataFrame): ``summarize`` output grouped by
            ``ShipVia`` and ``Shipper``.
        tests (pandas.DataFrame): Output of ``shipper_tests``, used
            for the Fisher p-values shown under each panel title.
        path (pathlib.Path): Output PNG path.

    Returns:
        None. Writes the figure to ``path``.
    """
    fisher = tests[tests["Test"].str.startswith("Fisher")].set_index("Outcome")["PValue"]
    panels = [
        ("Late", "Shipped after the required date", fisher["Shipped late"]),
        ("Slow", f"Took over {SLOW_DISPATCH_DAYS} days to dispatch",
         fisher["Over 14 days to dispatch"]),
    ]
    fig, axes = cs.plt.subplots(1, 2, figsize=(9, 4.4), sharey=True,
                                gridspec_kw={"wspace": 0.12})
    x = np.arange(len(by_shipper))
    colors = [cs.BLUE if s == FEDERAL_ID else cs.NEUTRAL for s in by_shipper["ShipVia"]]
    for ax, (col, title, p) in zip(axes, panels):
        rate = by_shipper[f"{col}Rate"] * 100
        err = [rate - by_shipper[f"{col}Rate_CI_Low"] * 100,
               by_shipper[f"{col}Rate_CI_High"] * 100 - rate]
        ax.bar(x, rate, width=0.55, color=colors, edgecolor=cs.SURFACE, linewidth=2)
        ax.errorbar(x, rate, yerr=err, fmt="none", ecolor=cs.INK_SECONDARY,
                    elinewidth=1.2, capsize=4)
        for xi, row in zip(x, by_shipper.itertuples()):
            n = getattr(row, f"{col}Orders")
            top = getattr(row, f"{col}Rate_CI_High") * 100
            ax.annotate(f"{getattr(row, f'{col}Rate'):.1%}\n{n} of {row.Orders}", (xi, top),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, color=cs.INK)
        ax.set_xticks(x, by_shipper["Shipper"])
        verdict = "significant" if p < 0.05 else "not significant"
        ax.set_title(f"{title}\nFederal vs. others: p = {p:.2f} ({verdict})", fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Share of shipped orders (%), with 95% interval")
    axes[0].set_ylim(0, 25)
    fig.suptitle("Federal Shipping stalls fewer orders; its late-rate edge could be chance",
                 x=0.075, ha="left", fontsize=12, fontweight="bold", y=1.04)
    cs.save(fig, path)


def plot_dispatch_times(o, path):
    """Draw a histogram of days to dispatch, split by on-time and late.

    Args:
        o (pandas.DataFrame): Orders from ``shipped_orders``.
        path (pathlib.Path): Output PNG path.

    Returns:
        None. Writes the figure to ``path``.
    """
    fig, ax = cs.plt.subplots(figsize=(9, 4))
    bins = np.arange(0, o["DaysToShip"].max() + 2) - 0.5
    on_time = o.loc[~o["Late"], "DaysToShip"]
    late = o.loc[o["Late"], "DaysToShip"]
    ax.hist([on_time, late], bins=bins, stacked=True, color=[cs.BLUE, cs.ORANGE],
            edgecolor=cs.SURFACE, linewidth=0.8, label=["Shipped on time", "Shipped late"])
    ax.axvline(SLOW_DISPATCH_DAYS + 0.5, color=cs.INK_SECONDARY, linewidth=1)
    slow_share = o["Slow"].mean()
    ax.annotate(f"{slow_share:.0%} of orders take more than {SLOW_DISPATCH_DAYS} days\n"
                f"to leave the warehouse; every late order is in this tail",
                (SLOW_DISPATCH_DAYS + 1.2, ax.get_ylim()[1] * 0.85), fontsize=9,
                color=cs.INK_SECONDARY, va="top")
    ax.set_xlabel("Days from order to dispatch")
    ax.set_ylabel("Orders")
    ax.set_title("Most orders leave within a week; a long tail waits 2-5 weeks")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    cs.save(fig, path)


def run(nw, fig_dir, out_dir):
    """Run the Question 2 analysis and save its outputs.

    Args:
        nw (Northwind): Loaded data.
        fig_dir (pathlib.Path): Folder for the three shipping charts.
        out_dir (pathlib.Path): Folder for the summary and test CSVs.

    Returns:
        dict[str, Any]: Headline numbers for the webpage, including
        per-shipper rates and the Federal Shipping test results.
    """
    o = shipped_orders(nw)
    by_promise = summarize(o, "PromisedDays")
    by_shipper = summarize(o, ["ShipVia", "Shipper"])
    tests = shipper_tests(o)
    by_promise.round(4).to_csv(out_dir / "q2_by_promised_window.csv", index=False)
    by_shipper.round(4).to_csv(out_dir / "q2_by_shipper.csv", index=False)
    tests.round(4).to_csv(out_dir / "q2_shipper_tests.csv", index=False)

    plot_late_by_promise(by_promise, fig_dir / "q2_late_by_promise.png")
    plot_shipper_comparison(by_shipper, tests, fig_dir / "q2_shipper_comparison.png")
    plot_dispatch_times(o, fig_dir / "q2_dispatch_times.png")

    late = o[o["Late"]]
    rush = by_promise.set_index("PromisedDays")
    shippers = by_shipper.set_index("Shipper")
    return {
        "shipped_orders": int(len(o)),
        "late_orders": int(o["Late"].sum()),
        "late_rate": round(float(o["Late"].mean()), 4),
        "late_rate_rush_14d": round(float(rush.loc[14, "LateRate"]), 4),
        "late_rate_standard_28d": round(float(rush.loc[28, "LateRate"]), 4),
        "late_rate_relaxed_42d": round(float(rush.loc[42, "LateRate"]), 4),
        "median_days_to_ship_rush": float(rush.loc[14, "MedianDaysToShip"]),
        "median_days_to_ship_standard": float(rush.loc[28, "MedianDaysToShip"]),
        "late_rate_by_shipper": shippers["LateRate"].round(4).to_dict(),
        "slow_rate_by_shipper": shippers["SlowRate"].round(4).to_dict(),
        "shipper_tests": tests.round(4).to_dict(orient="records"),
        "median_days_to_ship_all": float(o["DaysToShip"].median()),
        "orders_slower_than_14d": int(o["Slow"].sum()),
        "share_slower_than_14d": round(float(o["Slow"].mean()), 4),
        "min_days_to_ship_among_late": int(late["DaysToShip"].min()),
        "median_days_late_among_late": float(late["DaysLate"].median()),
        "freight_share_of_value_median": round(float(o["FreightShare"].median()), 4),
    }
