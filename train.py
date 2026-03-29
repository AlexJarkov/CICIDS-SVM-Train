"""
Entrenamiento del modelo One-Class SVM sobre tráfico BENIGN del lunes de CICIDS2017.

El modelo aprende la frontera del tráfico normal. En inferencia, cualquier flujo
que caiga fuera de esa frontera se clasifica como anomalía (ataque).

Uso:
    python train.py

Salida:
    models/ocsvm_model.joblib  — dict con scaler, model y lista de features
"""

import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

MONDAY_CSV = "data/raw/Monday-WorkingHours.pcap_ISCX.csv"
MODEL_OUTPUT = "models/ocsvm_model.joblib"

METADATA_COLS = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Protocol",
    "Timestamp",
]
LABEL_COL = "Label"


def load_benign(path: str) -> pd.DataFrame:
    print(f"  Leyendo {path} ...")
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()

    if LABEL_COL in df.columns:
        total = len(df)
        df = df[df[LABEL_COL].str.strip() == "BENIGN"].copy()
        print(f"  Filas BENIGN: {len(df):,} / {total:,}")
        df = df.drop(columns=[LABEL_COL])
    else:
        print(f"  Sin columna '{LABEL_COL}' — se usan todos los flujos")

    drop_cols = [c for c in METADATA_COLS if c in df.columns]
    df = df.drop(columns=drop_cols)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    CICFlowMeter genera inf y NaN en divisiones por cero (flujos de duración 0).
    Se eliminan esas filas para no contaminar el escalador ni el modelo.
    """
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    removed = before - len(df)
    if removed:
        print(f"  Filas eliminadas (inf/NaN): {removed:,}")
    return df.astype(np.float64)


def train(df: pd.DataFrame) -> tuple[StandardScaler, OneClassSVM]:
    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    model = OneClassSVM(
        kernel="rbf",
        nu=0.05,        # fracción esperada de outliers en datos de entrenamiento
        gamma="scale",  # 1 / (n_features * X.var())
    )
    print(f"  Ajustando OneClassSVM sobre {len(X):,} muestras con {X.shape[1]} features...")
    model.fit(X)
    return scaler, model


if __name__ == "__main__":
    print("=" * 60)
    print("CICIDS2017 — Entrenamiento One-Class SVM")
    print("=" * 60)

    print("\n[1/3] Cargando datos de entrenamiento (lunes, solo BENIGN)")
    df = load_benign(MONDAY_CSV)
    df = clean(df)
    print(f"  Flujos listos: {len(df):,}  |  Features: {df.shape[1]}")

    print("\n[2/3] Entrenando modelo...")
    scaler, model = train(df)
    print("  Entrenamiento completado.")

    print("\n[3/3] Guardando artefacto...")
    artifact = {
        "scaler": scaler,
        "model": model,
        "features": list(df.columns),
    }
    joblib.dump(artifact, MODEL_OUTPUT)
    print(f"  Guardado en: {MODEL_OUTPUT}")

    print("\nListo. Ejecuta evaluate.py para medir detección sobre días con ataques.")
    sys.exit(0)
