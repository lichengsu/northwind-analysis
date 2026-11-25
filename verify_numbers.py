"""Check headline numbers against an independent openpyxl recount."""

import json
from collections import defaultdict
from pathlib import Path

from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Northwind Database.xlsx"
KEY_NUMBERS = ROOT / "outputs" / "key_numbers.json"


def rows(ws):
    """Iterate over a worksheet's data rows as dictionaries.

    Args:
        ws (openpyxl.worksheet._read_only.ReadOnlyWorksheet): Sheet
            whose first row is the header.

    Yields:
        dict[str, Any]: Next row keyed by column name; rows with an
        empty first cell are skipped.
    """
    it =ws.iter_rows(values_only=True)
    header = [h for h in next(it) if h is not None]
    for r in it:
        if r[0] is None:
            continue
        yield dict(zip(header, r))


def recompute():
    """Recompute headline numbers directly from the workbook.

    Uses openpyxl and plain Python rather than pandas so it shares no
    code with the main pipeline.

    Args:
        None.

    Returns:
        dict[str, int | float]: Total revenue and discount, share of
        discount at 15% or more, late order count, rush late rate,
        and revenue per person and discount rate for each office.
    """
    wb =load_workbook(DATA, read_only=True, data_only=True)
    lines = list(rows(wb["Orders Details"]))
    orders = list(rows(wb["Orders"]))
    employees = {e["EmployeeID"]: e for e in rows(wb["Employees"])}

    revenue_by_order = defaultdict(float)
    gross_by_order = defaultdict(float)
    discount_by_order = defaultdict(float)
    deep_discount_value = 0.0
    total_discount_value = 0.0
    for li in lines:
        disc = round(li["Discount"], 2)
        gross = li["UnitPrice"] * li["Quantity"]
        revenue_by_order[li["OrderID"]] += gross * (1 - disc)
        gross_by_order[li["OrderID"]] += gross
        discount_by_order[li["OrderID"]] += gross * disc
        total_discount_value += gross * disc
        if disc >= 0.15:
            deep_discount_value += gross * disc

    shipped = [o for o in orders if o["ShippedDate"] is not None]
    late = [o for o in shipped if o["ShippedDate"] > o["RequiredDate"]]
    rush = [o for o in shipped if (o["RequiredDate"] - o["OrderDate"]).days == 14]
    rush_late = [o for o in rush if o["ShippedDate"] > o["RequiredDate"]]

    office_rev, office_gross, office_disc, office_people = (defaultdict(float) for _ in range(4))
    for o in orders:
        country = employees[o["EmployeeID"]]["Country"]
        office_rev[country] += revenue_by_order[o["OrderID"]]
        office_gross[country] += gross_by_order[o["OrderID"]]
        office_disc[country] += discount_by_order[o["OrderID"]]
    for e in employees.values():
        office_people[e["Country"]] += 1

    return {
        "total_revenue": round(sum(revenue_by_order.values()), 2),
        "discount_given_total": round(total_discount_value, 2),
        "share_of_discount_at_15pct_or_more": round(deep_discount_value / total_discount_value, 4),
        "late_orders": len(late),
        "late_rate_rush_14d": round(len(rush_late) / len(rush), 4),
        "us_revenue_per_person": round(office_rev["USA"] / office_people["USA"], 2),
        "uk_revenue_per_person": round(office_rev["UK"] / office_people["UK"], 2),
        "us_discount_rate": round(office_disc["USA"] / office_gross["USA"], 4),
        "uk_discount_rate": round(office_disc["UK"] / office_gross["UK"], 4),
    }


def main():
    """Compare recomputed numbers with ``outputs/key_numbers.json``.

    Args:
        None.

    Returns:
        None. Prints one line per check, then exits with status 1 if
        any value differs by 0.01 or more, otherwise status 0.
    """
    pipeline =json.loads(KEY_NUMBERS.read_text(encoding="utf-8"))
    expected = {
        "total_revenue": pipeline["data_checks"]["total_revenue"],
        "discount_given_total": pipeline["q1_discounts"]["discount_given_total"],
        "late_orders": pipeline["q2_shipping"]["late_orders"],
        "late_rate_rush_14d": pipeline["q2_shipping"]["late_rate_rush_14d"],
        "us_revenue_per_person": pipeline["q3_sales_reps"]["us_revenue_per_person"],
        "uk_revenue_per_person": pipeline["q3_sales_reps"]["uk_revenue_per_person"],
        "us_discount_rate": pipeline["q3_sales_reps"]["us_discount_rate"],
        "uk_discount_rate": pipeline["q3_sales_reps"]["uk_discount_rate"],
    }
    independent = recompute()
    failures = 0
    for key, value in expected.items():
        ok = abs(independent[key] - value) < 0.01
        failures += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {key}: pipeline={value} independent={independent[key]}")
    print(f"(info) share_of_discount_at_15pct_or_more: {independent['share_of_discount_at_15pct_or_more']}")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
