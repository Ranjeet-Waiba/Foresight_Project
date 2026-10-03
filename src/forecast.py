from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs"
MODELS = ROOT / "models"
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)

FEATURES = ["lag_1","lag_2","lag_4","lag_13","lag_52","roll_4","roll_13","week_sin","week_cos","month","promo_flag"]

def wape(y, p):
    denom = np.abs(y).sum()
    return float(np.abs(y-p).sum() / denom) if denom else np.nan

def add_features(df):
    df = df.sort_values(["sku_id","week_start"]).copy()
    g = df.groupby("sku_id", group_keys=False)["units_sold"]
    for lag in [1,2,4,13,52]:
        df[f"lag_{lag}"] = g.shift(lag)
    df["roll_4"] = g.transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    df["roll_13"] = g.transform(lambda s: s.shift(1).rolling(13, min_periods=4).mean())
    week = df["week_start"].dt.isocalendar().week.astype(int)
    df["week_sin"] = np.sin(2*np.pi*week/52.18)
    df["week_cos"] = np.cos(2*np.pi*week/52.18)
    return df

def baseline_predict(df):
    return df["lag_52"].fillna(df["lag_4"]).fillna(df["lag_1"]).fillna(0).clip(lower=0)

def main():
    weekly = pd.read_csv(PROCESSED / "weekly_sales.csv", parse_dates=["week_start"])
    feat = add_features(weekly)
    min_date, max_date = feat.week_start.min(), feat.week_start.max()
    test_starts = [max_date - pd.Timedelta(weeks=24), max_date - pd.Timedelta(weeks=16), max_date - pd.Timedelta(weeks=8)]
    folds = []
    for fold, start in enumerate(test_starts, 1):
        test_end = start + pd.Timedelta(weeks=7)
        train = feat[feat.week_start < start].dropna(subset=["lag_4","roll_4"])
        test = feat[(feat.week_start >= start) & (feat.week_start <= test_end)].dropna(subset=["lag_4","roll_4"])
        if train.empty or test.empty:
            continue
        model = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.07, max_depth=7, max_iter=180, l2_regularization=0.2, random_state=42)
        Xtr = train[FEATURES].fillna(0)
        ytr = train.units_sold.clip(lower=0)
        model.fit(Xtr, ytr)
        pred = np.clip(model.predict(test[FEATURES].fillna(0)), 0, None)
        base = baseline_predict(test).to_numpy()
        folds.append({"fold": fold, "start": str(start.date()), "end": str(test_end.date()), "model_wape": wape(test.units_sold.to_numpy(), pred), "baseline_wape": wape(test.units_sold.to_numpy(), base), "bias": float((pred-test.units_sold.to_numpy()).sum()/max(test.units_sold.sum(),1))})
    metrics = pd.DataFrame(folds)
    metrics.to_csv(OUT / "backtest_metrics.csv", index=False)

    train_all = feat.dropna(subset=["lag_4","roll_4"]).copy()
    model = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.07, max_depth=7, max_iter=180, l2_regularization=0.2, random_state=42)
    model.fit(train_all[FEATURES].fillna(0), train_all.units_sold.clip(lower=0))
    joblib.dump(model, MODELS / "demand_model.joblib")

    # Recursive 8-week SKU forecasts.
    hist = weekly[["sku_id","week_start","units_sold","promo_flag","category","subcategory","unit_cost","list_price"]].copy()
    future_rows = []
    last_week = hist.week_start.max()
    for step in range(1,9):
        dt = last_week + pd.Timedelta(weeks=step)
        for sku, grp in hist.groupby("sku_id"):
            s = grp.sort_values("week_start")
            vals = s.units_sold.to_numpy()
            lag = lambda k: vals[-k] if len(vals)>=k else (vals[-1] if len(vals) else 0)
            roll4 = vals[-4:].mean() if len(vals)>=2 else lag(1)
            roll13 = vals[-13:].mean() if len(vals)>=4 else roll4
            wk = int(dt.isocalendar().week)
            promo = int((dt.month==6 and dt.day>=15) or (dt.month in [10,11]) or (dt.month==12 and dt.day>=15))
            row = {"lag_1":lag(1),"lag_2":lag(2),"lag_4":lag(4),"lag_13":lag(13),"lag_52":lag(52),"roll_4":roll4,"roll_13":roll13,"week_sin":np.sin(2*np.pi*wk/52.18),"week_cos":np.cos(2*np.pi*wk/52.18),"month":dt.month,"promo_flag":promo}
            p = float(max(0, model.predict(pd.DataFrame([row], columns=FEATURES))[0]))
            std = float(np.std(vals[-13:])) if len(vals)>3 else max(1.0, p*0.2)
            lo = max(0.0, p - 1.28*std)
            hi = p + 1.28*std
            meta = s.iloc[-1]
            future_rows.append([sku, dt, step, p, lo, hi, meta.category, meta.subcategory, meta.unit_cost, meta.list_price])
            hist = pd.concat([hist, pd.DataFrame([[sku,dt,p,promo,meta.category,meta.subcategory,meta.unit_cost,meta.list_price]], columns=hist.columns)], ignore_index=True)
    fc = pd.DataFrame(future_rows, columns=["sku_id","week_start","horizon_week","forecast_units","lower_80","upper_80","category","subcategory","unit_cost","list_price"])
    fc.to_csv(OUT / "forecast_8week.csv", index=False)

    summary = {
        "model": "HistGradientBoostingRegressor (Poisson)",
        "features": FEATURES,
        "horizon_weeks": 8,
        "backtest_folds": len(metrics),
        "mean_model_wape": float(metrics.model_wape.mean()),
        "mean_baseline_wape": float(metrics.baseline_wape.mean()),
        "model_beats_baseline": bool(metrics.model_wape.mean() < metrics.baseline_wape.mean()),
        "mean_bias": float(metrics.bias.mean())
    }
    (OUT / "model_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
