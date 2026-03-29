# Entrenamiento: One-Class SVM con CICIDS2017

Proyecto separado para entrenar el modelo de detección de anomalías que consumirá `nids-detector`.

---

## Contexto

El sistema IDS (`ids-balena`) captura tráfico de red y lo almacena en PostgreSQL con 78 features
extraídas por `FlowManager`, cuyos nombres de columna son idénticos a los del CSV de CICIDS2017.
El modelo entrenado aquí se monta en el contenedor `nids-detector` para inferencia en tiempo real.

Se usa **One-Class SVM** porque:
- La base de datos no almacena etiqueta de ataque (`label` fue eliminada del schema)
- El sistema está diseñado para detección de anomalías: se aprende la frontera del tráfico normal
- En producción solo se observa tráfico sin etiquetar

---

## Dataset: CICIDS2017

Descarga desde: `https://www.unb.ca/cic/datasets/ids-2017.html`

| Archivo | Flujos | Etiquetas presentes | Uso |
|---|---|---|---|
| `Monday-WorkingHours.pcap_ISCX.csv` | ~529 000 | Solo `BENIGN` | **Entrenamiento** |
| `Tuesday-WorkingHours.pcap_ISCX.csv` | ~445 000 | BENIGN + FTP/SSH Patator | Test |
| `Wednesday-WorkingHours.pcap_ISCX.csv` | ~692 000 | BENIGN + DoS (Hulk, GoldenEye, Slowloris, Slowhttptest) | Test |
| `Thursday-WorkingHours.pcap_ISCX.csv` | ~170 000 | BENIGN + Web Attacks + Infiltration | Test |
| `Friday-WorkingHours.pcap_ISCX.csv` | ~286 000 | BENIGN + Botnet + DDoS + PortScan | Test |

Solo el archivo del **lunes** (100% BENIGN) se usa para entrenar. Los demás son exclusivamente
para medir la capacidad del modelo de separar tráfico normal de ataques.

---

## Estructura del proyecto

```
cicids2017-ocsvm/
├── data/
│   └── raw/                          # CSVs originales de CICIDS2017 (no commitear)
├── models/
│   └── ocsvm_model.joblib            # Artefacto final para nids-detector
├── train.py                          # Script de entrenamiento
├── evaluate.py                       # Script de evaluación contra días con ataques
├── requirements.txt
└── README.md
```

---

## Dependencias

```
# requirements.txt
scikit-learn==1.5.2
pandas==2.2.3
numpy==1.26.4
joblib==1.4.2
matplotlib==3.9.2       # solo para curvas ROC/confusion matrix en evaluate.py
```

---

## Features

De las 78 columnas de `cicids_flows`, se descartan las 7 columnas de metadatos no numéricas
y se trabaja con las **71 features numéricas**:

```python
# train.py
METADATA_COLS = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Protocol",
    "Timestamp",
]

# Si el CSV de CICIDS2017 trae columna Label, se usa para filtrar y luego se descarta
LABEL_COL = "Label"
```

Las 71 features restantes cubren:
- Duración y tasas (bytes/s, paquetes/s, forward/backward)
- Estadísticas de longitud de paquete (min, max, mean, std)
- Inter-Arrival Time (flow, fwd, bwd: mean, std, min, max, total)
- Conteo de flags TCP (FIN, SYN, RST, PSH, ACK, URG, ECE, CWR)
- Métricas de subflow, bulk transfer, ventana TCP inicial
- Tiempos activo/inactivo (mean, std, min, max)

---

## Script de entrenamiento: `train.py`

```python
import pandas as pd
import numpy as np
import joblib
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

MONDAY_CSV = "data/raw/Monday-WorkingHours.pcap_ISCX.csv"
MODEL_OUTPUT = "models/ocsvm_model.joblib"

METADATA_COLS = [
    "Flow ID", "Source IP", "Source Port",
    "Destination IP", "Destination Port", "Protocol", "Timestamp",
]
LABEL_COL = "Label"

def load_benign(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    # Normalizar nombres: el CSV tiene espacios extra en algunos headers
    df.columns = df.columns.str.strip()
    # El lunes es 100% BENIGN, pero filtrar por si acaso
    if LABEL_COL in df.columns:
        df = df[df[LABEL_COL].str.strip() == "BENIGN"].drop(columns=[LABEL_COL])
    drop_cols = [c for c in METADATA_COLS if c in df.columns]
    df = df.drop(columns=drop_cols)
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    # CICFlowMeter genera inf y NaN en divisiones por cero (flujos de 0 duración)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df.astype(np.float64)

def train(df: pd.DataFrame) -> tuple:
    scaler = StandardScaler()
    X = scaler.fit_transform(df)

    model = OneClassSVM(
        kernel="rbf",
        nu=0.05,       # fracción esperada de outliers en datos de entrenamiento
        gamma="scale", # equivale a 1 / (n_features * X.var())
    )
    model.fit(X)
    return scaler, model

if __name__ == "__main__":
    print("Cargando datos...")
    df = load_benign(MONDAY_CSV)
    df = clean(df)
    print(f"Flujos para entrenamiento: {len(df):,} | Features: {df.shape[1]}")

    print("Entrenando One-Class SVM...")
    scaler, model = train(df)

    joblib.dump({"scaler": scaler, "model": model, "features": list(df.columns)}, MODEL_OUTPUT)
    print(f"Modelo guardado en {MODEL_OUTPUT}")
```

---

## Script de evaluación: `evaluate.py`

```python
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, roc_auc_score

ATTACK_CSVS = [
    "data/raw/Tuesday-WorkingHours.pcap_ISCX.csv",
    "data/raw/Wednesday-WorkingHours.pcap_ISCX.csv",
    "data/raw/Thursday-WorkingHours.pcap_ISCX.csv",
    "data/raw/Friday-WorkingHours.pcap_ISCX.csv",
]
MODEL_PATH = "models/ocsvm_model.joblib"
LABEL_COL = "Label"
METADATA_COLS = [
    "Flow ID", "Source IP", "Source Port",
    "Destination IP", "Destination Port", "Protocol", "Timestamp",
]

def load_artifact():
    artifact = joblib.load(MODEL_PATH)
    return artifact["scaler"], artifact["model"], artifact["features"]

def load_day(path: str, feature_cols: list) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip()
    labels = df[LABEL_COL].str.strip() if LABEL_COL in df.columns else None
    drop_cols = [c for c in METADATA_COLS + [LABEL_COL] if c in df.columns]
    df = df.drop(columns=drop_cols)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    # Alinear columnas con las del modelo
    df = df[feature_cols]
    y = (labels.loc[df.index] != "BENIGN").astype(int)  # 1=ataque, 0=normal
    return df.astype(np.float64), y

def evaluate(scaler, model, feature_cols):
    for csv in ATTACK_CSVS:
        X, y_true = load_day(csv, feature_cols)
        X_scaled = scaler.transform(X)
        # OneClassSVM: +1 = normal, -1 = anomalía
        raw_pred = model.predict(X_scaled)
        y_pred = (raw_pred == -1).astype(int)  # 1=anomalía detectada

        print(f"\n--- {csv.split('/')[-1]} ---")
        print(classification_report(y_true, y_pred, target_names=["BENIGN", "ATTACK"]))

        if y_true.nunique() == 2:
            scores = model.decision_function(X_scaled)
            auc = roc_auc_score(y_true, -scores)  # score negativo = más anómalo
            print(f"ROC-AUC: {auc:.4f}")

if __name__ == "__main__":
    scaler, model, feature_cols = load_artifact()
    evaluate(scaler, model, feature_cols)
```

---

## Ajuste de hiperparámetros

El parámetro más sensible es `nu`:

| `nu` | Comportamiento |
|---|---|
| `0.01` | Muy conservador — pocos falsos positivos, puede perder ataques sutiles |
| `0.05` | Balance recomendado para empezar |
| `0.10` | Más sensible — detecta más ataques pero genera más falsas alarmas |

Para ajustar, ejecutar `evaluate.py` con distintos valores y comparar F1-score sobre la clase `ATTACK`.
No usar los días de ataque para seleccionar `nu` directamente — solo para validación final.

---

## Artefacto de salida

`models/ocsvm_model.joblib` contiene un dict con tres claves:

```python
{
    "scaler": StandardScaler,   # fit sobre Monday BENIGN
    "model":  OneClassSVM,      # entrenado sobre Monday BENIGN normalizado
    "features": list[str],      # lista de 71 nombres de columna en orden
}
```

Este archivo se monta en el contenedor `nids-detector` via volumen Docker.
Ver `docs/nids-detector.md` para la integración.
