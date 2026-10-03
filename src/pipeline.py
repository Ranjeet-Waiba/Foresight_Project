from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

CATEGORY_MAP = {
    "home decor": "Decor",
    "decor": "Decor",
    "furnishings": "Furnishings",
    "small appliances": "Small Appliances",
}

def load_clean():
    sales = pd.read_csv(RAW / "sales_daily.csv", parse_dates=["date"])
    sku = pd.read_csv(RAW / "sku_master.csv", parse_dates=["launch_date"])
    cal = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
    inv = pd.read_csv(RAW / "inventory_snapshots.csv", parse_dates=["date"])

    report = {}
    report["sales_rows_raw"] = len(sales)
    report["sales_duplicates_removed"] = int(sales.duplicated(["date","sku_id"]).sum())
    sales = sales.drop_duplicates(["date","sku_id"], keep="first").copy()
    report["sku_duplicates_removed"] = int(sku.duplicated("sku_id").sum())
    sku = sku.drop_duplicates("sku_id", keep="first").copy()
    report["inventory_duplicates_removed"] = int(inv.duplicated(["date","sku_id"]).sum())
    inv = inv.drop_duplicates(["date","sku_id"], keep="last").copy()

    sku["category"] = sku["category"].astype(str).str.strip().str.lower().map(CATEGORY_MAP).fillna(sku["category"].astype(str).str.strip().str.title())
    sku["subcategory"] = sku["subcategory"].astype(str).str.strip().str.title()

    # Price missingness: fill from list price. Revenue missingness: units x price.
    sales = sales.merge(sku[["sku_id","list_price"]], on="sku_id", how="left")
    report["unit_price_missing_filled"] = int(sales["unit_price"].isna().sum())
    sales["unit_price"] = sales["unit_price"].fillna(sales["list_price"])
    report["revenue_missing_filled"] = int(sales["revenue"].isna().sum())
    sales["revenue"] = sales["revenue"].fillna(sales["units_sold"] * sales["unit_price"])
    sales = sales.drop(columns="list_price")

    # Inventory reorder-point missingness: SKU median then overall median.
    report["reorder_point_missing_filled"] = int(inv["reorder_point"].isna().sum())
    inv["reorder_point"] = inv.groupby("sku_id")["reorder_point"].transform(lambda s: s.fillna(s.median()))
    inv["reorder_point"] = inv["reorder_point"].fillna(inv["reorder_point"].median()).round().astype(int)

    merged = sales.merge(sku, on="sku_id", how="left", validate="many_to_one")
    merged = merged.merge(cal, on="date", how="left", validate="many_to_one")
    merged = merged.sort_values(["sku_id","date"]).reset_index(drop=True)
    merged["week_start"] = merged["date"] - pd.to_timedelta(merged["date"].dt.dayofweek, unit="D")

    weekly = (merged.groupby(["sku_id","week_start"], as_index=False)
              .agg(units_sold=("units_sold","sum"), revenue=("revenue","sum"),
                   avg_unit_price=("unit_price","mean"), promo_days=("promo_flag","sum"),
                   category=("category","first"), subcategory=("subcategory","first"),
                   unit_cost=("unit_cost","first"), list_price=("list_price","first")))
    weekly["promo_flag"] = (weekly["promo_days"] > 0).astype(int)
    weekly["weekofyear"] = weekly["week_start"].dt.isocalendar().week.astype(int)
    weekly["month"] = weekly["week_start"].dt.month

    merged.to_csv(PROCESSED / "analysis_ready_daily.csv", index=False)
    weekly.to_csv(PROCESSED / "weekly_sales.csv", index=False)
    inv.to_csv(PROCESSED / "inventory_clean.csv", index=False)
    pd.Series(report).to_json(PROCESSED / "data_quality.json", indent=2)
    return merged, weekly, inv, report

if __name__ == "__main__":
    _, weekly, _, report = load_clean()
    print("Pipeline complete")
    print(report)
    print(f"Weekly rows: {len(weekly):,}")
