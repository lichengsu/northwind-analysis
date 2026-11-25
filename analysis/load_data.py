"""Load the Northwind workbook and add revenue and shipping fields."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "Northwind Database.xlsx"

# The Product sheet carries leftover columns from earlier class exercises
# (Cheap, NewPrice, IsCheap, ...). Only these are used.
PRODUCT_COLUMNS = [
    "ProductID", "ProductName", "SupplierID", "CategoryID", "QuantityPerUnit",
    "Price", "UnitsInStock", "UnitsOnOrder", "ReorderLevel", "Discontinued",
]


@dataclass
class Northwind:
    """Cleaned Northwind tables used by every analysis.

    Attributes:
        orders (pandas.DataFrame): One row per order with order totals
            and shipping fields such as ``Late`` and ``DaysToShip``.
        lines (pandas.DataFrame): One row per order line with
            ``GrossValue``, ``DiscountValue``, and ``Revenue``.
        employees (pandas.DataFrame): Employees with ``Name`` and
            ``Office`` added.
        shippers (pandas.DataFrame): Shipping companies.
        products (pandas.DataFrame): Products, limited to
            ``PRODUCT_COLUMNS``.
        customers (pandas.DataFrame): Customers, unchanged.
    """

    orders: pd.DataFrame
    lines: pd.DataFrame
    employees: pd.DataFrame
    shippers: pd.DataFrame
    products: pd.DataFrame
    customers: pd.DataFrame


def load(path=DATA_PATH):
    """Read the workbook and add revenue and shipping fields.

    Discounts are rounded to two decimals first because the sheet
    stores them as single-precision floats.

    Args:
        path (str | pathlib.Path): Northwind Excel workbook; defaults
            to ``data/Northwind Database.xlsx`` in the repository.

    Returns:
        Northwind: Cleaned tables. For orders not yet shipped,
        ``Late``, ``DaysToShip``, and ``DaysLate`` are missing.
    """
    sheets = pd.read_excel(path, sheet_name=None)

    lines = sheets["Orders Details"].copy()
    # Discounts are stored as single-precision floats (0.05 reads as
    # 0.0500000007); round to the intended whole-percent values.
    lines["Discount"] = lines["Discount"].round(2)
    lines["GrossValue"] = lines["UnitPrice"] * lines["Quantity"]
    lines["DiscountValue"] = lines["GrossValue"] * lines["Discount"]
    lines["Revenue"] = lines["GrossValue"] - lines["DiscountValue"]

    order_totals = lines.groupby("OrderID").agg(
        Revenue=("Revenue", "sum"),
        GrossValue=("GrossValue", "sum"),
        DiscountValue=("DiscountValue", "sum"),
        LineCount=("ProductID", "size"),
    )

    orders = sheets["Orders"].merge(order_totals, left_on="OrderID", right_index=True)
    orders["Shipped"] = orders["ShippedDate"].notna()
    orders["PromisedDays"] = (orders["RequiredDate"] - orders["OrderDate"]).dt.days
    orders["DaysToShip"] = (orders["ShippedDate"] - orders["OrderDate"]).dt.days
    orders["DaysLate"] = (orders["ShippedDate"] - orders["RequiredDate"]).dt.days
    # Unshipped orders are neither late nor on time yet, so leave them missing.
    orders["Late"] = (orders["DaysLate"] > 0).where(orders["Shipped"])

    employees = sheets["Employees"].copy()
    employees["Name"] = employees["FirstName"] + " " + employees["LastName"]
    employees["Office"] = employees["Country"].map({"USA": "Seattle (US)", "UK": "London (UK)"})

    return Northwind(
        orders=orders,
        lines=lines,
        employees=employees,
        shippers=sheets["Shippers"].copy(),
        products=sheets["Product"][PRODUCT_COLUMNS].copy(),
        customers=sheets["Customers"].copy(),
    )


def sanity_checks(nw):
    """Summarize the loaded data and assert basic integrity.

    Args:
        nw (Northwind): Tables returned by ``load``.

    Returns:
        dict[str, int | float | str]: Row counts, date range, total
        revenue and discount, and integrity-check counts.

    Raises:
        AssertionError: If an order line references an unknown
            product, an order has no lines, or order totals differ
            from line totals by a cent or more.
    """
    o, li = nw.orders, nw.lines
    checks = {
        "orders": len(o),
        "order_lines": len(li),
        "first_order": o["OrderDate"].min().date().isoformat(),
        "last_order": o["OrderDate"].max().date().isoformat(),
        "unshipped_orders": int((~o["Shipped"]).sum()),
        "total_revenue": round(float(li["Revenue"].sum()), 2),
        "total_discount_value": round(float(li["DiscountValue"].sum()), 2),
        "lines_without_product": int((~li["ProductID"].isin(nw.products["ProductID"])).sum()),
        "orders_without_lines": int((~nw.orders["OrderID"].isin(li["OrderID"])).sum()),
    }
    assert checks["lines_without_product"] == 0, "order lines reference unknown products"
    assert checks["orders_without_lines"] == 0, "orders with no line items"
    assert abs(o["Revenue"].sum() - li["Revenue"].sum()) < 0.01, "order totals do not add up"
    return checks
