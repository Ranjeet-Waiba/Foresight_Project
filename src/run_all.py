import subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for script in ["generate_data.py", "pipeline.py", "forecast.py", "risk.py"]:
    print(f"\n=== Running {script} ===")
    subprocess.run([sys.executable, str(ROOT/"src"/script)], check=True)
print("\nFORESIGHT pipeline completed successfully.")
