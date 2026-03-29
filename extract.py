"""
Extrae los CSVs de CICIDS2017 desde el zip a data/raw/.
Ejecutar una sola vez antes de train.py y evaluate.py.

Uso:
    python extract.py
"""

import zipfile
import os
from pathlib import Path

ZIP_PATH = "CICIDS2017/CSVs/GeneratedLabelledFlows.zip"
OUT_DIR = Path("data/raw")


def extract():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH) as zf:
        members = zf.namelist()
        csv_members = [m for m in members if m.endswith(".csv")]
        for member in csv_members:
            filename = Path(member).name
            dest = OUT_DIR / filename
            if dest.exists():
                print(f"Ya existe, omitiendo: {filename}")
                continue
            print(f"Extrayendo: {filename}")
            with zf.open(member) as src, open(dest, "wb") as dst:
                dst.write(src.read())
    print(f"\nCSVs disponibles en {OUT_DIR}:")
    for f in sorted(OUT_DIR.glob("*.csv")):
        size_mb = f.stat().st_size / 1_048_576
        print(f"  {f.name}  ({size_mb:.0f} MB)")


if __name__ == "__main__":
    extract()
