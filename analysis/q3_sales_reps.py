"""Question 3: how do sales staff and offices compare?"""

from analysis import chart_style as cs

OFFICE_COLORS = {"Seattle (US)": cs.BLUE, "London (UK)": cs.ORANGE}
LABEL_OFFSETS = {
    "Peacock": (8, 4), "Leverling": (8, 2), "Davolio": (8, -10), "Fuller": (8, 2),
    "Callahan": (8, -4), "King": (-8, 6), "Dodsworth": (8, 4), "Suyama": (8, -8),
    "Buchanan": (8, -2),
}


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

    Args:
        reps (pandas.DataFrame): Output of ``rep_table``.

    Returns:
        pandas.DataFrame: One row per office with headcount, orders,
        revenue, discount rate, and revenue per person.
    """
    t =reps.groupby("Office").agg(
        People=("EmployeeID", "size"),
        Orders=("Orders", "sum"),
        Revenue=("Revenue", "sum"),
        GrossValue=("GrossValue", "sum"),
        DiscountGiven=("DiscountGiven", "sum"),
    )
    t["DiscountRate"] = t["DiscountGiven"] / t["GrossValue"]
    t["RevenuePerPerson"] = t["Revenue"] / t["People"]
    return t.reset_index()


def plot(reps, path):
    """Draw each employee's discount rate against net revenue.

    Args:
        reps (pandas.DataFrame): Output of ``rep_table``.
        path (pathlib.Path): Output PNG path.

    Returns:
        None. Writes the figure to ``path``.
    """
    fig, ax = cs.plt.subplots(figsize=(7.6, 4.8))
    for office, color in OFFICE_COLORS.items():
        sub = reps[reps["Office"] == office]
        ax.scatter(sub["DiscountRate"] * 100, sub["Revenue"] / 1000, s=sub["Orders"] * 1.4,
                   color=color, alpha=0.9, edgecolor=cs.SURFACE, linewidth=2, label=office, zorder=3)
    for row in reps.itertuples():
        dx, dy = LABEL_OFFSETS.get(row.LastName, (8, 0))
        ax.annotate(row.LastName, (row.DiscountRate * 100, row.Revenue / 1000), xytext=(dx, dy),
                    textcoords="offset points", fontsize=8.5, color=cs.INK,
                    ha="right" if dx < 0 else "left", va="center")
    ax.set_xlabel("Discount given, % of list value of the rep's sales")
    ax.set_ylabel("Net revenue, Jul 1996 - May 1998 ($K)")
    ax.set_xlim(3, 13)
    ax.set_ylim(0, 260)
    ax.set_title("Seattle sells about twice as much per person as London, with smaller discounts",
                 pad=24)
    ax.text(0, 1.025, "Each dot is one employee who took orders; dot size = number of orders.",
            transform=ax.transAxes, fontsize=8.5, color=cs.INK_SECONDARY)
    legend = ax.legend(loc="lower right")
    for handle in legend.legend_handles:
        handle.set_sizes([60])
    cs.save(fig, path)


def run(nw, fig_dir, out_dir):
    """Run the Question 3 analysis and save its outputs.

    Args:
        nw (Northwind): Loaded data.
        fig_dir (pathlib.Path): Folder for
            ``q3_discount_vs_revenue.png``.
        out_dir (pathlib.Path): Folder for ``q3_sales_reps.csv`` and
            ``q3_offices.csv``.

    Returns:
        dict[str, Any]: Headline numbers for the webpage: top and
        bottom sellers, rank correlations, and office totals.
    """
    reps = rep_table(nw)
    offices = office_table(reps)
    export = reps.drop(columns=["LastName"]).assign(HireDate=reps["HireDate"].dt.date)
    export.round(4).to_csv(out_dir / "q3_sales_reps.csv", index=False)
    offices.round(4).to_csv(out_dir / "q3_offices.csv", index=False)
    plot(reps, fig_dir / "q3_discount_vs_revenue.png")

    o = nw.orders.merge(nw.employees[["EmployeeID", "Office"]], on="EmployeeID")
    shared_customers = int((o.groupby("CustomerID")["Office"].nunique() > 1).sum())

    rank_corr = reps[["DiscountRate", "Revenue", "TenureYearsAtStart", "AvgOrderValue"]].corr(method="spearman")
    top, bottom = reps.iloc[0], reps.iloc[-1]
    most_disc = reps.loc[reps["DiscountRate"].idxmax()]
    least_disc = reps.loc[reps["DiscountRate"].idxmin()]
    off = offices.set_index("Office")
    return {
        "top_rep": top["Name"],
        "top_rep_revenue": round(float(top["Revenue"]), 2),
        "bottom_rep": bottom["Name"],
        "bottom_rep_revenue": round(float(bottom["Revenue"]), 2),
        "revenue_ratio_top_to_bottom": round(float(top["Revenue"] / bottom["Revenue"]), 2),
        "highest_discount_rep": most_disc["Name"],
        "highest_discount_rate": round(float(most_disc["DiscountRate"]), 4),
        "lowest_discount_rep": least_disc["Name"],
        "lowest_discount_rate": round(float(least_disc["DiscountRate"]), 4),
        "spearman_discount_vs_revenue": round(float(rank_corr.loc["DiscountRate", "Revenue"]), 2),
        "spearman_tenure_vs_revenue": round(float(rank_corr.loc["TenureYearsAtStart", "Revenue"]), 2),
        "spearman_discount_vs_avg_order_value": round(float(rank_corr.loc["DiscountRate", "AvgOrderValue"]), 2),
        "us_revenue_per_person": round(float(off.loc["Seattle (US)", "RevenuePerPerson"]), 2),
        "uk_revenue_per_person": round(float(off.loc["London (UK)", "RevenuePerPerson"]), 2),
        "us_discount_rate": round(float(off.loc["Seattle (US)", "DiscountRate"]), 4),
        "uk_discount_rate": round(float(off.loc["London (UK)", "DiscountRate"]), 4),
        "customers_served_by_both_offices": shared_customers,
        "customers_total": int(o["CustomerID"].nunique()),
        "tenure_range_years": [round(float(reps["TenureYearsAtStart"].min()), 1),
                               round(float(reps["TenureYearsAtStart"].max()), 1)],
    }
