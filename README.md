# Northwind Sales Review

Analysis code for 15.561 Homework 3. It answers three business questions about Northwind Traders (July 1996 – May 1998) from the course's Northwind dataset:

1. **Discounts:** do deeper discounts buy bigger orders?
2. **Shipping:** what drives late shipments, the shipper or the promised deadline?
3. **Sales staff:** how do employees and offices compare, and does heavier discounting go with higher sales?

**Webpage with the findings:** https://lichengsu.github.io/northwind-analysis/

## Reproduce the results

You need Python 3.10 or newer.

```bash
git clone https://github.com/lichengsu/northwind-analysis.git
cd northwind-analysis
python -m venv .venv
```

Activate the environment (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on macOS/Linux), then:

```bash
pip install -r requirements.txt
python run_all.py
python verify_numbers.py
```

`run_all.py` reads `data/Northwind Database.xlsx` and regenerates every figure and table on the webpage. It also prints the headline numbers and saves them to `outputs/key_numbers.json`. `verify_numbers.py` recomputes the main figures directly from the spreadsheet with a different library (openpyxl, not pandas) and exits with an error if any figure disagrees.

Results are deterministic. The bootstrap intervals in Question 1 use a fixed random seed (561).

## Where each result on the webpage comes from

| Webpage item | Produced by | Output file |
|---|---|---|
| Q1 chart | `analysis/q1_discounts.py` | `docs/figures/q1_discount_lift.png` |
| Q1 table | `analysis/q1_discounts.py` | `outputs/q1_discount_levels.csv` |
| Q1 headline numbers ($88,666 given, $36,083 cap scenario) | `analysis/q1_discounts.py` | `outputs/key_numbers.json` → `q1_discounts` |
| Q2 late-rate chart | `analysis/q2_shipping.py` | `docs/figures/q2_late_rates.png` |
| Q2 dispatch-time chart | `analysis/q2_shipping.py` | `docs/figures/q2_dispatch_times.png` |
| Q2 table | `analysis/q2_shipping.py` | `outputs/q2_by_promised_window.csv`, `outputs/q2_by_shipper.csv` |
| Q3 chart | `analysis/q3_sales_reps.py` | `docs/figures/q3_discount_vs_revenue.png` |
| Q3 tables | `analysis/q3_sales_reps.py` | `outputs/q3_sales_reps.csv`, `outputs/q3_offices.csv` |
| Data notes (830 orders, $1,265,793 revenue, ...) | `analysis/load_data.py` | `outputs/key_numbers.json` → `data_checks` |

## Repository layout

```
data/                 Northwind Database.xlsx (course dataset, unmodified)
analysis/
  load_data.py        Loads the sheets, computes revenue and shipping fields, runs sanity checks
  chart_style.py      Shared matplotlib style and colors
  q1_discounts.py     Question 1
  q2_shipping.py      Question 2
  q3_sales_reps.py    Question 3
run_all.py            Runs everything and writes figures, tables, and key_numbers.json
verify_numbers.py     Independent recomputation of the headline numbers
docs/                 The webpage (served by GitHub Pages) and its figures
outputs/              CSV tables and key_numbers.json
```

## Definitions and data handling

- **Net revenue** = `UnitPrice × Quantity × (1 − Discount)` from the Orders Details sheet. It uses the price paid on the order, not the current list price in the Product sheet.
- **Discount** values are stored as single-precision floats (0.05 reads as 0.0500000007), so they are rounded to two decimals on load.
- **Late** = shipped after `RequiredDate`. The 21 orders with no `ShippedDate` (placed 8 Apr – 6 May 1998) are excluded from shipping metrics.
- **Promised window** = `RequiredDate − OrderDate`. It is always 14, 28, or 42 days.
- The Product sheet has leftover columns from earlier exercises (`Cheap`, `NewPrice`, `IsCheap`, ...). They are ignored.
- 1998 is a partial year (January to 6 May), so no analysis compares calendar-year totals.
