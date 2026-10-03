from pathlib import Path
from fastapi import FastAPI, HTTPException
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
app = FastAPI(title="FORESIGHT Scoring Service", version="1.0")

@app.get("/health")
def health():
    return {"status":"ok"}

@app.get("/score/{sku_id}")
def score(sku_id: str):
    risk_path, fc_path = OUT/"risk_scores.csv", OUT/"forecast_8week.csv"
    if not (risk_path.exists() and fc_path.exists()):
        raise HTTPException(status_code=503, detail="Model outputs missing. Run python src/run_all.py")
    risk = pd.read_csv(risk_path)
    fc = pd.read_csv(fc_path)
    r = risk[risk.sku_id == sku_id]
    if r.empty:
        raise HTTPException(status_code=404, detail=f"Unknown SKU: {sku_id}")
    return {"sku_id": sku_id, "risk": r.iloc[0].to_dict(), "forecast": fc[fc.sku_id==sku_id].to_dict(orient="records")}
