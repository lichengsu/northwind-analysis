# Northwind Sales Review

Analysis code for 15.561 Homework 3. It answers three business questions about Northwind Traders (July 1996 – May 1998) from the course's Northwind dataset:

1. **Discounts:** how much extra volume does a discount buy, and is it worth the margin it costs?
2. **Shipping:** what drives late shipments, and is Federal Shipping significantly better than the other shippers?
3. **Sales offices:** why does London sell less than Seattle, and are the discounts reps give justified?

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

Results are deterministic. The bootstrap intervals and simulation bands in Question 1 use a fixed random seed (561).

## Where each result on the webpage comes from

| Webpage item | Produced by | Output file |
|---|---|---|
| Q1 lift-by-level chart | `analysis/q1_discounts.py` | `docs/figures/q1_discount_lift.png`, `outputs/q1_discount_levels.csv` |
| Q1 lift regression table | `analysis/q1_discounts.py` | `outputs/q1_lift_models.csv` |
| Q1 profit/loss chart and table | `analysis/q1_discounts.py` | `docs/figures/q1_profit_by_discount.png`, `outputs/q1_profit_curves.csv` |
| Q1 break-even and total profit effect | `analysis/q1_discounts.py` | `outputs/q1_break_even.csv` |
| Q2 late rate by promised window | `analysis/q2_shipping.py` | `docs/figures/q2_late_by_promise.png`, `outputs/q2_by_promised_window.csv` |
| Q2 dispatch-time histogram | `analysis/q2_shipping.py` | `docs/figures/q2_dispatch_times.png` |
| Q2 shipper comparison chart | `analysis/q2_shipping.py` | `docs/figures/q2_shipper_comparison.png`, `outputs/q2_by_shipper.csv` |
| Q2 significance tests | `analysis/q2_shipping.py` | `outputs/q2_shipper_tests.csv` |
| Q3 office gap chart and table | `analysis/q3_sales_reps.py` | `docs/figures/q3_office_gap.png`, `outputs/q3_offices.csv` |
| Q3 discount by customer size | `analysis/q3_sales_reps.py` | `docs/figures/q3_discount_by_customer_size.png`, `outputs/q3_discount_by_customer_size.csv` |
| Q3 employee table | `analysis/q3_sales_reps.py` | `outputs/q3_sales_reps.csv` |
| Test results quoted in the text (p-values, correlations) | all three modules | `outputs/key_numbers.json` |
| Data notes (830 orders, $1,265,793 revenue, ...) | `analysis/load_data.py` | `outputs/key_numbers.json` → `data_checks` |

## Repository layout

```
data/                 Northwind Database.xlsx (course dataset, unmodified)
analysis/
  load_data.py        Loads the sheets, computes revenue and shipping fields, runs sanity checks
  chart_style.py      Shared matplotlib style and colors
  q1_discounts.py     Question 1: lift by discount level, lift regressions, profit/loss
  q2_shipping.py      Question 2: late rates, dispatch times, shipper significance tests
  q3_sales_reps.py    Question 3: office gap, discounts by customer size, employee table
run_all.py            Runs everything and writes figures, tables, and key_numbers.json
verify_numbers.py     Independent recomputation of the headline numbers
docs/                 The webpage (served by GitHub Pages) and its figures
outputs/              CSV tables and key_numbers.json
```

## Definitions and data handling

- **Net revenue** = `UnitPrice × Quantity × (1 − Discount)` from the Orders Details sheet. It uses the price paid on the order, not the current list price in the Product sheet.
- **Discount** values are stored as single-precision floats (0.05 reads as 0.0500000007), so they are rounded to two decimals on load.
- **Relative quantity** = a line's quantity ÷ the same product's median quantity on full-price lines. The Q1 regressions use its logarithm, with standard errors clustered by order. The "same customer" model adds customer fixed effects.
- **Gross margin is assumed** to be 30% at list price, with 20% and 40% as alternatives, because the dataset has no cost data. Profit change for a discounted line = `lift × (margin − discount) ÷ margin − 1`. Change `BASE_MARGIN` and `MARGINS` in `analysis/q1_discounts.py` to try other values.
- **Late** = shipped after `RequiredDate`. **Stalled** = more than 14 days from order to dispatch. The 21 orders with no `ShippedDate` (placed 8 Apr – 6 May 1998) are excluded from shipping metrics.
- **Customer size quartiles** (Q3) rank customers by list value before discounts.
- **Promised window** = `RequiredDate − OrderDate`. It is always 14, 28, or 42 days.
- The Product sheet has leftover columns from earlier exercises (`Cheap`, `NewPrice`, `IsCheap`, ...). They are ignored.
- 1998 is a partial year (January to 6 May), so no analysis compares calendar-year totals.
