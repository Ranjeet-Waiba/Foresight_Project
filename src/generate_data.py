from pathlib import Path
import numpy as np
import pandas as pd

SEED = 42
N_SKUS = 180
START = "2024-01-01"
END = "2026-08-31"

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(SEED)
dates = pd.date_range(START, END, freq="D")

cats = {
    "Furnishings": ["Cushions", "Throws", "Bedding"],
    "Decor": ["Lighting", "Wall Decor", "Planters"],
    "Small Appliances": ["Kitchen", "Cleaning", "Climate"],
}

sku_rows = []
for i in range(1, N_SKUS + 1):
    category = rng.choice(list(cats))
    subcategory = rng.choice(cats[category])
    unit_cost = float(np.round(rng.uniform(180, 4500), 2))
    markup = rng.uniform(1.45, 2.4)
    list_price = float(np.round(unit_cost * markup, 2))
    launch_offset = int(rng.integers(0, 420))
    launch_date = pd.Timestamp(START) + pd.Timedelta(days=launch_offset)
    sku_rows.append([f"SKU-{i:04d}", category, subcategory, launch_date.date(), unit_cost, list_price])
sku_master = pd.DataFrame(sku_rows, columns=["sku_id", "category", "subcategory", "launch_date", "unit_cost", "list_price"])

# Deliberately imperfect labels and one duplicate to exercise cleaning logic.
idx = rng.choice(sku_master.index, size=8, replace=False)
sku_master.loc[idx[:3], "category"] = "Home Decor"
sku_master.loc[idx[3:6], "category"] = "small appliances"
sku_master.loc[idx[6:], "subcategory"] = sku_master.loc[idx[6:], "subcategory"].str.lower()
sku_master = pd.concat([sku_master, sku_master.iloc[[5]]], ignore_index=True)
sku_master.to_csv(RAW / "sku_master.csv", index=False)

calendar = pd.DataFrame({"date": dates})
calendar["week"] = calendar["date"].dt.isocalendar().week.astype(int)
calendar["month"] = calendar["date"].dt.month
calendar["season"] = np.select(
    [calendar.month.isin([12,1,2]), calendar.month.isin([3,4,5]), calendar.month.isin([6,7,8])],
    ["Winter", "Spring", "Summer"], default="Autumn"
)
calendar["is_holiday"] = 0
calendar["promo_event"] = "None"
for y in sorted(calendar.date.dt.year.unique()):
    masks = {
        "New Year": (calendar.date.between(f"{y}-01-01", f"{y}-01-07")),
        "Mid-Year Sale": (calendar.date.between(f"{y}-06-15", f"{y}-06-30")),
        "Festive Sale": (calendar.date.between(f"{y}-10-15", f"{y}-11-15")),
        "Year End": (calendar.date.between(f"{y}-12-15", f"{y}-12-31")),
    }
    for event, m in masks.items():
        calendar.loc[m, "promo_event"] = event
        calendar.loc[m, "is_holiday"] = 1 if event in {"New Year", "Year End"} else calendar.loc[m, "is_holiday"]
calendar.to_csv(RAW / "calendar.csv", index=False)

# Generate daily sales with weekly and annual seasonality, promotion uplift, trend and zero-demand SKUs.
base_master = sku_master.drop_duplicates("sku_id", keep="first").copy()
rows = []
for _, s in base_master.iterrows():
    launch = pd.Timestamp(s.launch_date)
    base = rng.lognormal(mean=1.55, sigma=0.55)
    sku_phase = rng.uniform(0, 2*np.pi)
    trend = rng.uniform(-0.00035, 0.00065)
    dead = rng.random() < 0.06
    volatile = rng.random() < 0.12
    for d in dates:
        if d < launch:
            continue
        dow_factor = 1.18 if d.dayofweek in [4,5,6] else 0.92
        annual = 1 + 0.18*np.sin(2*np.pi*d.dayofyear/365.25 + sku_phase)
        day_index = (d - launch).days
        trend_factor = max(0.55, 1 + trend*day_index)
        promo = int((d.month == 6 and d.day >= 15) or (d.month in [10,11] and (d.month==10 or d.day<=15)) or (d.month==12 and d.day>=15))
        promo_factor = 1.55 if promo else 1.0
        lam = base*dow_factor*annual*trend_factor*promo_factor
        if dead and d > pd.Timestamp("2025-11-01"):
            lam *= 0.05
        if volatile:
            lam *= rng.lognormal(0, 0.28)
        units = int(rng.poisson(max(lam, 0.03)))
        discount = rng.uniform(0.05, 0.18) if promo else rng.uniform(0, 0.03)
        unit_price = float(np.round(float(s.list_price)*(1-discount), 2))
        revenue = float(np.round(units*unit_price, 2))
        rows.append([d.date(), s.sku_id, units, revenue, unit_price, promo])
sales = pd.DataFrame(rows, columns=["date", "sku_id", "units_sold", "revenue", "unit_price", "promo_flag"])
# Inject small realistic quality issues.
miss = rng.choice(sales.index, size=max(1, int(len(sales)*0.004)), replace=False)
sales.loc[miss, "unit_price"] = np.nan
miss_rev = rng.choice(sales.index, size=max(1, int(len(sales)*0.002)), replace=False)
sales.loc[miss_rev, "revenue"] = np.nan
dup = sales.sample(n=max(1, int(len(sales)*0.0015)), random_state=SEED)
sales = pd.concat([sales, dup], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
sales.to_csv(RAW / "sales_daily.csv", index=False)

# Weekly inventory snapshots, linked loosely to trailing demand.
sales_cleanish = sales.drop_duplicates(["date","sku_id"]).copy()
sales_cleanish["date"] = pd.to_datetime(sales_cleanish["date"])
sales_cleanish["week_end"] = sales_cleanish["date"] + pd.to_timedelta(6-sales_cleanish["date"].dt.dayofweek, unit="D")
weekly_demand = sales_cleanish.groupby(["sku_id","week_end"], as_index=False)["units_sold"].sum()
weekly_lookup = {(r.sku_id, pd.Timestamp(r.week_end)): float(r.units_sold) for r in weekly_demand.itertuples()}
weekly_dates = pd.date_range(START, END, freq="W-SUN")
inv_rows = []
for _, s in base_master.iterrows():
    lead = int(rng.integers(5, 29))
    reorder = int(rng.integers(18, 75))
    on_hand = int(rng.integers(35, 180))
    recent_weeks = []
    for d in weekly_dates:
        if d < pd.Timestamp(s.launch_date):
            continue
        wd = weekly_lookup.get((s.sku_id, pd.Timestamp(d)), 0.0)
        recent_weeks.append(wd)
        recent = np.mean(recent_weeks[-4:]) if recent_weeks else rng.uniform(2,10)
        on_hand = max(0, int(on_hand - wd + rng.integers(0, 9)))
        on_order = 0
        if on_hand < reorder:
            on_order = int(max(reorder*2, recent*4) + rng.integers(0, 30))
            if rng.random() < 0.55:
                on_hand += int(on_order*rng.uniform(0.5, 1.0))
                on_order = 0
        inv_rows.append([d.date(), s.sku_id, on_hand, on_order, lead, reorder])
inv = pd.DataFrame(inv_rows, columns=["date","sku_id","on_hand_units","on_order_units","lead_time_days","reorder_point"])
if len(inv):
    bad = rng.choice(inv.index, size=max(1, int(len(inv)*0.003)), replace=False)
    inv.loc[bad, "reorder_point"] = np.nan
    inv = pd.concat([inv, inv.iloc[[10]]], ignore_index=True)
inv.to_csv(RAW / "inventory_snapshots.csv", index=False)

print(f"Generated dataset in {RAW}")
print(f"sales_daily: {len(sales):,} rows")
print(f"sku_master: {len(sku_master):,} rows")
print(f"calendar: {len(calendar):,} rows")
print(f"inventory_snapshots: {len(inv):,} rows")
