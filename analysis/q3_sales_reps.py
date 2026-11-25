"""Question 3: London's sales gap and the case for discounts."""

import numpy as np
import pandas as pd
from scipy import stats

from analysis import chart_style as cs

OFFICE_COLORS = {"Seattle (US)": cs.BLUE, "London (UK)": cs.ORANGE}
SIZE_TIERS = ["Smallest 25%", "Second 25%", "Third 25%", "Largest 25%"]


def rep_table(nw):
    """Summarize sales, discounting, and lateness per employee.

    Args:
        nw (Northwind): Loaded data.

    Returns:
        pandas.DataFrame: One row per employee, highest revenue first.
        ``DiscountRate`` is discount dollars over list value, and
        ``TenureYearsAtStart`` is years from hire date to the first
        order in the data.
    """
    o = nw.orders
    data_start = o["OrderDate"].min()
    shipped = o[o["Shipped"]]

    reps = o.groupby("EmployeeID").agg(
        Orders=("OrderID", "size"),
        Customers=("CustomerID", "nunique"),
        Revenue=("Revenue", "sum"),
        GrossValue=("GrossValue", "sum"),
        DiscountGiven=("DiscountValue", "sum"),
        AvgOrderValue=("Revenue", "mean"),
        ShareOfOrdersDiscounted=("DiscountValue", lambda s: (s > 0).mean()),
    )
    reps["DiscountRate"] = reps["DiscountGiven"] / reps["GrossValue"]
    reps["LateRate"] = shipped.groupby("EmployeeID")["Late"].mean().astype(float)
    reps["RevenueShare"] = reps["Revenue"] / reps["Revenue"].sum()

    emp = nw.employees.set_index("EmployeeID")[["Name", "LastName", "Title", "Office", "HireDate"]]
    reps = emp.join(reps)
    reps["TenureYearsAtStart"] = (data_start - reps["HireDate"]).dt.days / 365.25
    return reps.sort_values("Revenue", ascending=False).reset_index()


def office_table(reps):
    """Aggregate employee results to the Seattle and London offices.

    Revenue per person is split into orders per person times average
    order value, to show which of the two drives any gap.

    Args:
        reps (pandas.DataFrame): Output of ``rep_table``.

    Returns:
        pandas.DataFrame: One row per office with headcount, orders,
        revenue, discount rate, revenue and orders per person, average
        customers per person, and average order value.
    """
    t = reps.groupby("Office").agg(
        People=("EmployeeID", "size"),
        Orders=("Orders", "sum"),
        Revenue=("Revenue", "sum"),
        GrossValue=("GrossValue", "sum"),
        DiscountGiven=("DiscountGiven", "sum"),
        CustomersPerPerson=("Customers", "mean"),
    )
    t["DiscountRate"] = t["DiscountGiven"] / t["GrossValue"]
    t["RevenuePerPerson"] = t["Revenue"] / t["People"]
    t["OrdersPerPerson"] = t["Orders"] / t["People"]
    t["AvgOrderValue"] = t["Revenue"] / t["Orders"]
    return t.reset_index()


def orders_with_office(nw):
    """Return orders with the office of the employee who took them.

    Args:
        nw (Northwind): Loaded data.

    Returns:
        pandas.DataFrame: Copy of ``nw.orders`` with ``Office`` added.
    """
    return nw.orders.merge(nw.employees[["EmployeeID", "Office"]], on="EmployeeID")


def customer_discount_table(nw):
    """Summarize discount depth by customer size quartile.

    Customers are ranked by list value (before discounts), so the
    discounts themselves cannot push a customer into a lower tier.

    Args:
        nw (Northwind): Loaded data.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame]: One row per size
        quartile with customer count, list value, discount given, and
        discount rate; and one row per customer with ``GrossValue``,
        ``DiscountGiven``, ``DiscountRate``, and ``SizeTier``.
    """
    o = nw.orders
    cust = o.groupby("CustomerID").agg(GrossValue=("GrossValue", "sum"),
                                       DiscountGiven=("DiscountValue", "sum"))
    cust["DiscountRate"] = cust["DiscountGiven"] / cust["GrossValue"]
    cust["SizeTier"] = pd.qcut(cust["GrossValue"].rank(method="first"), 4, labels=SIZE_TIERS)
    tiers = cust.groupby("SizeTier", observed=True).agg(
        Customers=("GrossValue", "size"),
        GrossValue=("GrossValue", "sum"),
        DiscountGiven=("DiscountGiven", "sum"),
        ShareGettingAnyDiscount=("DiscountGiven", lambda s: (s > 0).mean()),
    )
    tiers["DiscountRate"] = tiers["DiscountGiven"] / tiers["GrossValue"]
    tiers["ShareOfAllDiscount"] = tiers["DiscountGiven"] / tiers["DiscountGiven"].sum()
    return tiers.reset_index(), cust.reset_index()


def shared_customer_rates(nw):
    """Compare discount rates the two offices give the same customers.

    Args:
        nw (Northwind): Loaded data.

    Returns:
        pandas.DataFrame: One row per customer who bought from both
        offices, with one discount-rate column per office.
    """
    o = orders_with_office(nw)
    co = o.groupby(["CustomerID", "Office"]).agg(GrossValue=("GrossValue", "sum"),
                                                 DiscountGiven=("DiscountValue", "sum"))
    co["DiscountRate"] = co["DiscountGiven"] / co["GrossValue"]
    return co["DiscountRate"].unstack().dropna()


def plot_office_gap(offices, path):
    """Draw orders per person, average order value, and discount rate.

    Args:
        offices (pandas.DataFrame): Output of ``office_table``.
        path (pathlib.Path): Output PNG path.

    Returns:
        None. Writes the figure to ``path``.
    """
    t = offices.set_index("Office").loc[list(OFFICE_COLORS)]
    panels = [
        ("OrdersPerPerson", "Orders per person", "{:.0f}"),
        ("AvgOrderValue", "Average order value ($)", "${:,.0f}"),
        ("DiscountRate", "Discount given (% of list value)", "{:.1%}"),
    ]
    fig, axes = cs.plt.subplots(1, 3, figsize=(9.6, 3.8), gridspec_kw={"wspace": 0.35})
    x = np.arange(len(t))
    colors = list(OFFICE_COLORS.values())
    for ax, (col, title, fmt) in zip(axes, panels):
        values = t[col] * (100 if col == "DiscountRate" else 1)
        ax.bar(x, values, width=0.6, color=colors, edgecolor=cs.SURFACE, linewidth=2)
        for xi, v in zip(x, t[col]):
            ax.annotate(fmt.format(v), (xi, v * (100 if col == "DiscountRate" else 1)),
                        xytext=(0, 4), textcoords="offset points", ha="center",
                        va="bottom", fontsize=9, color=cs.INK)
        ax.set_xticks(x, [o.split(" ")[0] for o in t.index])
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, values.max() * 1.22)
        ax.grid(axis="x", visible=False)
    fig.suptitle("London's gap is fewer orders, not smaller ones, despite deeper discounts",
                 x=0.075, ha="left", fontsize=12, fontweight="bold", y=1.05)
    cs.save(fig, path)


def plot_discount_by_size(tiers, rho, p, path):
    """Draw the discount rate customers receive by size quartile.

    Args:
        tiers (pandas.DataFrame): First output of
            ``customer_discount_table``.
        rho (float): Spearman correlation of customer size and
            discount rate, shown in the subtitle.
        p (float): P-value of that correlation.
        path (pathlib.Path): Output PNG path.

    Returns:
        None. Writes the figure to ``path``.
    """
    fig, ax = cs.plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(len(tiers))
    ax.bar(x, tiers["DiscountRate"] * 100, width=0.55, color=cs.BLUE,
           edgecolor=cs.SURFACE, linewidth=2)
    for xi, row in zip(x, tiers.itertuples()):
        ax.annotate(f"{row.DiscountRate:.1%}\n{row.ShareOfAllDiscount:.0%} of all discount $",
                    (xi, row.DiscountRate * 100), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=cs.INK)
    ax.set_xticks(x, [f"{t}\nof customers" for t in tiers["SizeTier"]])
    ax.set_ylabel("Discount given (% of list value)")
    ax.set_ylim(0, (tiers["DiscountRate"].max() * 100) * 1.35)
    ax.grid(axis="x", visible=False)
    ax.set_title("Discounts go mostly to the largest accounts", pad=24)
    p_text = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.text(0, 1.025, f"Customers ranked by list value of purchases. "
            f"Spearman rho = {rho:.2f}, {p_text}.",
            transform=ax.transAxes, fontsize=8.5, color=cs.INK_SECONDARY)
    cs.save(fig, path)


def run(nw, fig_dir, out_dir):
    """Run the Question 3 analysis and save its outputs.

    Args:
        nw (Northwind): Loaded data.
        fig_dir (pathlib.Path): Folder for ``q3_office_gap.png`` and
            ``q3_discount_by_customer_size.png``.
        out_dir (pathlib.Path): Folder for the rep, office, and
            customer-tier CSVs.

    Returns:
        dict[str, Any]: Headline numbers for the webpage: office
        decomposition, discount correlations, and test results.
    """
    reps = rep_table(nw)
    offices = office_table(reps)
    tiers, cust = customer_discount_table(nw)
    shared = shared_customer_rates(nw)

    export = reps.drop(columns=["LastName"]).assign(HireDate=reps["HireDate"].dt.date)
    export.round(4).to_csv(out_dir / "q3_sales_reps.csv", index=False)
    offices.round(4).to_csv(out_dir / "q3_offices.csv", index=False)
    tiers.round(4).to_csv(out_dir / "q3_discount_by_customer_size.csv", index=False)

    rho_cust, p_cust = stats.spearmanr(cust["GrossValue"], cust["DiscountRate"])
    rho_rep, p_rep = stats.spearmanr(reps["DiscountRate"], reps["Revenue"])
    wilcoxon = stats.wilcoxon(shared["London (UK)"], shared["Seattle (US)"])
    o = orders_with_office(nw)
    aov_test = stats.mannwhitneyu(*[g["Revenue"] for _, g in o.groupby("Office")])

    plot_office_gap(offices, fig_dir / "q3_office_gap.png")
    plot_discount_by_size(tiers, rho_cust, p_cust, fig_dir / "q3_discount_by_customer_size.png")

    off = offices.set_index("Office")
    us, uk = off.loc["Seattle (US)"], off.loc["London (UK)"]
    top_tier = tiers.set_index("SizeTier").loc["Largest 25%"]
    bottom_half = tiers.set_index("SizeTier").loc[["Smallest 25%", "Second 25%"]]
    return {
        "top_rep": reps.iloc[0]["Name"],
        "top_rep_revenue": round(float(reps.iloc[0]["Revenue"]), 2),
        "highest_discount_rep": reps.loc[reps["DiscountRate"].idxmax(), "Name"],
        "highest_discount_rate": round(float(reps["DiscountRate"].max()), 4),
        "us_revenue_per_person": round(float(us["RevenuePerPerson"]), 2),
        "uk_revenue_per_person": round(float(uk["RevenuePerPerson"]), 2),
        "us_orders_per_person": round(float(us["OrdersPerPerson"]), 1),
        "uk_orders_per_person": round(float(uk["OrdersPerPerson"]), 1),
        "us_customers_per_person": round(float(us["CustomersPerPerson"]), 1),
        "uk_customers_per_person": round(float(uk["CustomersPerPerson"]), 1),
        "us_avg_order_value": round(float(us["AvgOrderValue"]), 2),
        "uk_avg_order_value": round(float(uk["AvgOrderValue"]), 2),
        "avg_order_value_mannwhitney_p": round(float(aov_test.pvalue), 4),
        "us_discount_rate": round(float(us["DiscountRate"]), 4),
        "uk_discount_rate": round(float(uk["DiscountRate"]), 4),
        "customers_served_by_both_offices": int(len(shared)),
        "customers_total": int(o["CustomerID"].nunique()),
        "shared_customers_mean_rate_us": round(float(shared["Seattle (US)"].mean()), 4),
        "shared_customers_mean_rate_uk": round(float(shared["London (UK)"].mean()), 4),
        "shared_customers_wilcoxon_p": round(float(wilcoxon.pvalue), 4),
        "spearman_customer_size_vs_discount_rate": round(float(rho_cust), 2),
        "spearman_customer_p": float(p_cust),
        "largest_quartile_discount_rate": round(float(top_tier["DiscountRate"]), 4),
        "largest_quartile_share_of_discount": round(float(top_tier["ShareOfAllDiscount"]), 4),
        "bottom_half_discount_rate": round(float(bottom_half["DiscountGiven"].sum()
                                                 / bottom_half["GrossValue"].sum()), 4),
        "spearman_rep_discount_vs_revenue": round(float(rho_rep), 2),
        "spearman_rep_p": round(float(p_rep), 4),
        "tenure_range_years": [round(float(reps["TenureYearsAtStart"].min()), 1),
                               round(float(reps["TenureYearsAtStart"].max()), 1)],
    }
