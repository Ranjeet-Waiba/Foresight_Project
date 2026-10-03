# Project FORESIGHT - Demand & Inventory Intelligence

## Client and objective
NorthBay Living is treated as a D2C home and lifestyle client. This repository implements a reproducible weekly SKU forecast, stockout/overstock risk scoring, a Streamlit planning dashboard, and a FastAPI scoring service.

## Dataset used
**NorthBay Living Synthetic FORESIGHT Dataset v1.0** is used in this package.

Raw tables:
- `sales_daily.csv`: SKU-day demand, revenue, price, promotion flag.
- `sku_master.csv`: category, subcategory, launch date, cost, list price.
- `calendar.csv`: week, month, season, holiday and promotional events.
- `inventory_snapshots.csv`: periodic on-hand, on-order, lead time and reorder point.


## Headline results
- Mean rolling-origin model WAPE: **0.145**
- Mean seasonal-naive WAPE: **0.232**
- Relative WAPE improvement: **37.5%**
- Total modelled sales at risk: **Rs 6,319,015**
- Total modelled locked capital: **Rs 11,546,815**

## Reproduce end to end
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/run_all.py
```


## Run the dashboard
```bash
streamlit run app/streamlit_app.py
```

## Run the scoring service
```bash
uvicorn service.main:app --reload
```
Then open `/docs` for the interactive API documentation. Example endpoint: `/score/SKU-0001`. Bad or unknown SKU input returns a controlled HTTP error.



## Methodology
The forecast horizon is eight weeks. Forecast accuracy is assessed using WAPE, with signed bias as a secondary check. A seasonal-naive baseline is calculated before the more complex model. Validation uses three rolling-origin backtest windows rather than a random split. Features use only past information: lagged weekly demand, rolling demand statistics, seasonal terms, month and promotion status.

Risk logic is intentionally transparent. Stockout risk compares forecast demand over lead time with on-hand plus on-order stock and a safety level. Overstock risk compares current on-hand stock with forecast weeks of supply. Each SKU is mapped to one of four actions: Reorder now, Markdown / clear, Watch / volatile, or Healthy.



