"""Question 2: what drives late shipments?"""

import numpy as np
import pandas as pd

from analysis import chart_style as cs

PROMISE_LABELS = {14: "14 days\n(rush)", 28: "28 days\n(standard)", 42: "42 days\n(relaxed)"}
SLOW_DISPATCH_DAYS = 14


def shipped_orders(nw):
    """Return shipped orders with freight share and shipper name.

    Args:
        nw (Northwind): Loaded data.

    Returns:
        pandas.DataFrame: Copy of shipped orders with ``Late`` as bool,
        ``FreightShare`` (freight / net revenue), and ``Shipper``.
    """
    o =nw.orders[nw.orders["Shipped"]].copy()
    o["Late"] = o["Late"].astype(bool)
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
        pandas.DataFrame: One row per group with order and late
        counts, late rate, median and 90th-percentile days to ship,
        median freight, and median freight share of order value.
    """
    return o.groupby(by).agg(
        Orders=("OrderID", "size"),
        LateOrders=("Late", "sum"),
        LateRate=("Late", "mean"),
        MedianDaysToShip=("DaysToShip", "median"),
        P90DaysToShip=("DaysToShip", lambda s: s.quantile(0.9)),
        MedianFreight=("Freight", "median"),
        MedianFreightShareOfValue=("FreightShare", "median"),
    ).reset_index()


def plot_late_rates(by_promise, by_shipper, path):
    """Draw late rates by promised window and by shipper side by side.

    Args:
        by_promise (pandas.DataFrame): ``summarize`` output grouped by
            ``PromisedDays``; the 14-day bar is highlighted.
        by_shipper (pandas.DataFrame): ``summarize`` output grouped by
            ``ShipVia`` and ``Shipper``.
        path (pathlib.Path): Output PNG path.

    Returns:
        None. Writes the figure to ``path``.
    """
    fig, axes = cs.plt.subplots(1, 2, figsize=(9, 4.2), sharey=True,
                                gridspec_kw={"wspace": 0.12})
    panels = [
        (axes[0], by_promise, [PROMISE_LABELS[d] for d in by_promise["PromisedDays"]],
         "By promised delivery window", by_promise["PromisedDays"].eq(14)),
        (axes[1], by_shipper, list(by_shipper["Shipper"]), "By shipper",
         pd.Series(False, index=by_shipper.index)),
    ]
    for ax, df, labels, title, highlight in panels:
        x = np.arange(len(df))
        colors = [cs.ORANGE if h else cs.BLUE for h in highlight]
        ax.bar(x, df["LateRate"] * 100, width=0.55, color=colors, edgecolor=cs.SURFACE, linewidth=2)
        for xi, row in zip(x, df.itertuples()):
            ax.annotate(f"{row.LateRate:.1%}\n{row.LateOrders} of {row.Orders}",
                        (xi, row.LateRate * 100), xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8.5, color=cs.INK)
        ax.set_xticks(x, labels)
        ax.set_title(title, fontsize=10.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Orders shipped after the required date (%)")
    axes[0].set_ylim(0, 22)
    fig.suptitle("Rush orders are late 5x as often; the shipper makes little difference",
                 x=0.075, ha="left", fontsize=12, fontweight="bold", y=1.02)
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
    slow_share = (o["DaysToShip"] > SLOW_DISPATCH_DAYS).mean()
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
        fig_dir (pathlib.Path): Folder for the two shipping charts.
        out_dir (pathlib.Path): Folder for the two summary CSVs.

    Returns:
        dict[str, Any]: Headline numbers for the webpage. Values are
        numbers except ``late_rate_by_shipper``, which maps shipper
        name to late rate.
    """
    o = shipped_orders(nw)
    by_promise = summarize(o, "PromisedDays")
    by_shipper = summarize(o, ["ShipVia", "Shipper"])
    by_promise.round(4).to_csv(out_dir / "q2_by_promised_window.csv", index=False)
    by_shipper.round(4).to_csv(out_dir / "q2_by_shipper.csv", index=False)

    plot_late_rates(by_promise, by_shipper, fig_dir / "q2_late_rates.png")
    plot_dispatch_times(o, fig_dir / "q2_dispatch_times.png")

    late = o[o["Late"]]
    slow = o["DaysToShip"] > SLOW_DISPATCH_DAYS
    rush = by_promise.set_index("PromisedDays")
    return {
        "shipped_orders": int(len(o)),
        "late_orders": int(o["Late"].sum()),
        "late_rate": round(float(o["Late"].mean()), 4),
        "late_rate_rush_14d": round(float(rush.loc[14, "LateRate"]), 4),
        "late_rate_standard_28d": round(float(rush.loc[28, "LateRate"]), 4),
        "late_rate_relaxed_42d": round(float(rush.loc[42, "LateRate"]), 4),
        "median_days_to_ship_rush": float(rush.loc[14, "MedianDaysToShip"]),
        "median_days_to_ship_standard": float(rush.loc[28, "MedianDaysToShip"]),
        "late_rate_by_shipper": {r.Shipper: round(r.LateRate, 4) for r in by_shipper.itertuples()},
        "median_days_to_ship_all": float(o["DaysToShip"].median()),
        "orders_slower_than_14d": int(slow.sum()),
        "share_slower_than_14d": round(float(slow.mean()), 4),
        "min_days_to_ship_among_late": int(late["DaysToShip"].min()),
        "median_days_late_among_late": float(late["DaysLate"].median()),
        "freight_share_of_value_median": round(float(o["FreightShare"].median()), 4),
    }
