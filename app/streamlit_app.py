from pathlib import Path
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
P = ROOT / "data" / "processed"

st.set_page_config(page_title="FORESIGHT Planning Dashboard", layout="wide")
st.title("Project FORESIGHT - Demand & Inventory Intelligence")

risk_path = OUT / "risk_scores.csv"
fc_path = OUT / "forecast_8week.csv"
weekly_path = P / "weekly_sales.csv"
if not (risk_path.exists() and fc_path.exists() and weekly_path.exists()):
    st.warning("Processed outputs are missing. Run `python src/run_all.py` first.")
    st.stop()

risk = pd.read_csv(risk_path)
fc = pd.read_csv(fc_path, parse_dates=["week_start"])
hist = pd.read_csv(weekly_path, parse_dates=["week_start"])

cats = ["All"] + sorted(risk.category.dropna().unique().tolist())
cat = st.sidebar.selectbox("Category", cats)
filtered = risk if cat == "All" else risk[risk.category == cat]
sku = st.sidebar.selectbox("SKU", filtered.sku_id.tolist())

r = risk[risk.sku_id == sku].iloc[0]
cols = st.columns(4)
cols[0].metric("Action", r.action)
cols[1].metric("8-week forecast", f"{r.forecast_8w:,.0f} units")
cols[2].metric("Sales at risk", f"Rs {r.sales_at_risk_rs:,.0f}")
cols[3].metric("Locked capital", f"Rs {r.locked_capital_rs:,.0f}")

st.subheader("Actual history and forecast")
h = hist[hist.sku_id == sku][["week_start","units_sold"]].tail(40).rename(columns={"units_sold":"Actual"}).set_index("week_start")
f = fc[fc.sku_id == sku][["week_start","forecast_units"]].rename(columns={"forecast_units":"Forecast"}).set_index("week_start")
st.line_chart(h.join(f, how="outer"))

st.subheader("Prioritised actions")
show = filtered.sort_values("value_at_stake_rs", ascending=False)[["sku_id","category","action","stockout_score","overstock_score","sales_at_risk_rs","locked_capital_rs"]]
st.dataframe(show, use_container_width=True, hide_index=True)
