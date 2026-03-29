"""
Evaluación del modelo One-Class SVM sobre los días de CICIDS2017 que contienen ataques.

El modelo fue entrenado solo con tráfico BENIGN (lunes). Aquí se mide su capacidad
de detectar tráfico de ataque como anomalía en los días restantes.

Interpretación de predicciones del OneClassSVM:
  +1  →  normal  (dentro de la frontera aprendida)
  -1  →  anomalía detectada (fuera de la frontera) → reportado como ATTACK

Uso:
    python evaluate.py

Requiere:
    models/ocsvm_model.joblib  (generado por train.py)
    data/raw/*.csv             (generado por extract.py)
"""

import sys
from pathlib import Path

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

MODEL_PATH = "models/ocsvm_model.joblib"
LABEL_COL = "Label"
METADATA_COLS = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Protocol",
    "Timestamp",
]

# Cada entrada es (etiqueta_dia, [lista de CSVs]).
# Thursday y Friday están partidos en varios archivos.
ATTACK_DAYS = [
    (
        "Tuesday",
        ["data/raw/Tuesday-WorkingHours.pcap_ISCX.csv"],
    ),
    (
        "Wednesday",
        ["data/raw/Wednesday-workingHours.pcap_ISCX.csv"],
    ),
    (
        "Thursday",
        [
            "data/raw/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
            "data/raw/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
        ],
    ),
    (
        "Friday",
        [
            "data/raw/Friday-WorkingHours-Morning.pcap_ISCX.csv",
            "data/raw/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
            "data/raw/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        ],
    ),
]


def load_artifact():
    artifact = joblib.load(MODEL_PATH)
    return artifact["scaler"], artifact["model"], artifact["features"]


def load_day(paths: list[str], feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """
    Carga y concatena uno o varios CSVs de un mismo día.
    Devuelve (X, y_true) donde y_true: 1=ataque, 0=normal.
    """
    frames = []
    label_frames = []

    for path in paths:
        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip()

        if LABEL_COL in df.columns:
            label_frames.append(df[LABEL_COL].str.strip())
        else:
            label_frames.append(pd.Series(["BENIGN"] * len(df), index=df.index))

        drop_cols = [c for c in METADATA_COLS + [LABEL_COL] if c in df.columns]
        df = df.drop(columns=drop_cols)
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)
    labels_all = pd.concat(label_frames, ignore_index=True)

    # Limpiar inf/NaN y alinear índices entre features y etiquetas
    df_all = df_all.replace([np.inf, -np.inf], np.nan)
    valid_mask = df_all.notna().all(axis=1)
    df_all = df_all[valid_mask].reset_index(drop=True)
    labels_all = labels_all[valid_mask].reset_index(drop=True)

    # Alinear columnas con las del modelo (mismo orden)
    missing = set(feature_cols) - set(df_all.columns)
    if missing:
        raise ValueError(f"Columnas del modelo ausentes en el CSV: {missing}")
    df_all = df_all[feature_cols].astype(np.float64)

    y = (labels_all != "BENIGN").astype(int)  # 1=ataque, 0=normal
    return df_all, y


def evaluate_day(
    day_name: str,
    paths: list[str],
    scaler,
    model,
    feature_cols: list[str],
):
    print(f"\n{'─' * 60}")
    print(f"  {day_name}")
    files_str = ", ".join(Path(p).name for p in paths)
    print(f"  Archivos: {files_str}")

    X, y_true = load_day(paths, feature_cols)
    print(f"  Flujos: {len(X):,}  |  Ataques reales: {y_true.sum():,} ({y_true.mean()*100:.1f}%)")

    X_scaled = scaler.transform(X)

    # +1 = normal, -1 = anomalía → convertir a etiquetas binarias (1=ataque detectado)
    raw_pred = model.predict(X_scaled)
    y_pred = (raw_pred == -1).astype(int)

    print(classification_report(y_true, y_pred, target_names=["BENIGN", "ATTACK"], digits=4))

    if y_true.nunique() == 2:
        scores = model.decision_function(X_scaled)
        # decision_function: valores más negativos = más anómalo
        auc = roc_auc_score(y_true, -scores)
        print(f"  ROC-AUC: {auc:.4f}")

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["BENIGN", "ATTACK"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(f"{day_name} — Confusion Matrix")
    plt.tight_layout()
    out_path = f"models/{day_name.lower()}_confusion_matrix.png"
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"  Matriz de confusión guardada en: {out_path}")


def evaluate(scaler, model, feature_cols: list[str]):
    missing_files = []
    for _, paths in ATTACK_DAYS:
        for p in paths:
            if not Path(p).exists():
                missing_files.append(p)

    if missing_files:
        print("\nERROR: Faltan los siguientes CSVs. Ejecuta extract.py primero.")
        for f in missing_files:
            print(f"  {f}")
        sys.exit(1)

    for day_name, paths in ATTACK_DAYS:
        evaluate_day(day_name, paths, scaler, model, feature_cols)


if __name__ == "__main__":
    print("=" * 60)
    print("CICIDS2017 — Evaluación One-Class SVM")
    print("=" * 60)

    print(f"\nCargando modelo desde {MODEL_PATH} ...")
    scaler, model, feature_cols = load_artifact()
    print(f"Features del modelo: {len(feature_cols)}")

    evaluate(scaler, model, feature_cols)

    print(f"\n{'=' * 60}")
    print("Evaluación completada.")
    sys.exit(0)
