from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
O = ROOT / "outputs"

def main():
    fc = pd.read_csv(O / "forecast_8week.csv", parse_dates=["week_start"])
    inv = pd.read_csv(P / "inventory_clean.csv", parse_dates=["date"])
    latest_inv = inv.sort_values("date").groupby("sku_id", as_index=False).tail(1)
    fsum = fc.groupby("sku_id", as_index=False).agg(
        forecast_8w=("forecast_units","sum"),
        forecast_weekly=("forecast_units","mean"),
        upper_8w=("upper_80","sum"),
        unit_cost=("unit_cost","first"), list_price=("list_price","first"),
        category=("category","first"), subcategory=("subcategory","first")
    )
    d = fsum.merge(latest_inv, on="sku_id", how="left")
    d["lead_time_weeks"] = d.lead_time_days/7
    d["lead_time_demand"] = d.forecast_weekly*d.lead_time_weeks
    d["available_units"] = d.on_hand_units.fillna(0)+d.on_order_units.fillna(0)
    d["safety_level"] = np.maximum(d.reorder_point.fillna(0), d.lead_time_demand*1.15)
    d["stockout_score"] = ((d.safety_level-d.available_units)/d.safety_level.replace(0,np.nan)).fillna(0).clip(0,1)
    d["weeks_of_supply"] = d.on_hand_units.fillna(0)/d.forecast_weekly.replace(0,np.nan)
    d["overstock_score"] = ((d.weeks_of_supply-8)/12).replace([np.inf,-np.inf],1).fillna(1).clip(0,1)
    d["stockout_high"] = d.stockout_score >= 0.5
    d["overstock_high"] = d.overstock_score >= 0.5
    d["action"] = np.select(
        [d.stockout_high & ~d.overstock_high, ~d.stockout_high & d.overstock_high, d.stockout_high & d.overstock_high],
        ["Reorder now", "Markdown / clear", "Watch / volatile"], default="Healthy")
    potential_lost_units = np.maximum(0, d.lead_time_demand-d.available_units)
    d["sales_at_risk_rs"] = potential_lost_units*d.list_price
    excess_units = np.maximum(0, d.on_hand_units.fillna(0)-d.forecast_8w*1.25)
    d["locked_capital_rs"] = excess_units*d.unit_cost
    d["value_at_stake_rs"] = d.sales_at_risk_rs+d.locked_capital_rs
    cols = ["sku_id","category","subcategory","forecast_8w","on_hand_units","on_order_units","lead_time_days","reorder_point","stockout_score","overstock_score","action","sales_at_risk_rs","locked_capital_rs","value_at_stake_rs"]
    d[cols].sort_values("value_at_stake_rs", ascending=False).to_csv(O / "risk_scores.csv", index=False)
    print(d.action.value_counts())
    print("Total sales at risk Rs:", round(d.sales_at_risk_rs.sum(),2))
    print("Locked capital Rs:", round(d.locked_capital_rs.sum(),2))

if __name__ == "__main__":
    main()
