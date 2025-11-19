from fastapi import FastAPI, HTTPException, Query, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import load_model

# Activar ejecución eager global y en el modelo
tf.config.run_functions_eagerly(True)

# =========================================
# CONFIGURACIÓN GLOBAL
# =========================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

PANEL_CSV = DATA_DIR / "panel_diario_por_producto.csv"
FEATURES_JSON = DATA_DIR / "features_used.json"
MODEL_PATH = MODELS_DIR / "lstm_model.h5"

LOOKBACK = 14

print("[INIT] Cargando panel diario...")
panel = pd.read_csv(PANEL_CSV, parse_dates=["date"])

print("[INIT] Cargando features usadas...")
with open(FEATURES_JSON, "r", encoding="utf-8") as f:
    feature_cols = json.load(f)["feature_cols"]

print("[INIT] Cargando modelo LSTM...")
# Cargamos sin compilar y recompilamos con run_eagerly=True
model = load_model(MODEL_PATH, compile=False)
model.compile(
    optimizer="adam",
    loss="mse",
    run_eagerly=True
)
print("[INIT] Todo cargado correctamente.")


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def build_sequences_from_panel(panel_path: Path, lookback: int = 14, horizon: int = 1):
    """
    Construye X, y usando EXACTAMENTE las mismas columnas con las que se entrenó
    el modelo original (feature_cols de features_used.json).
    """
    df = pd.read_csv(panel_path, parse_dates=["date"])
    df = df.sort_values(["product_id", "date"]).reset_index(drop=True)

    # Usar las mismas features que el modelo
    used_features = [c for c in feature_cols if c in df.columns]
    if not used_features:
        raise ValueError(
            f"Ninguna de las columnas {feature_cols} está en el CSV panel_diario_por_producto."
        )

    X_list, y_list, meta = [], [], []

    for pid, g in df.groupby("product_id"):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < lookback + horizon:
            continue

        # (n_dias, n_features)
        feat = g[used_features].to_numpy(np.float32)
        tgt = g["sale_amount"].to_numpy(np.float32)

        for t in range(lookback, len(g) - horizon + 1):
            X_list.append(feat[t - lookback:t])
            y_list.append(tgt[t + horizon - 1])
            meta.append((pid, g["date"].iloc[t + horizon - 1]))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    print(
        f"[RETRAIN] Secuencias construidas: X.shape={X.shape}, y.shape={y.shape}, "
        f"n_features={len(used_features)}"
    )

    return X, y


def build_window(
    panel_df: pd.DataFrame,
    product_id: str,
    ref_date,
    feature_cols,
    lookback: int = LOOKBACK,
):
    """
    Construye la ventana histórica para un producto dado y fecha de referencia.
    Asegura que 'date' sea datetime para evitar errores de comparación.
    """
    ref_date = pd.to_datetime(ref_date)

    # Filtrar producto y asegurar tipo datetime
    g = panel_df[panel_df["product_id"] == product_id].copy()
    if "date" in g.columns:
        g["date"] = pd.to_datetime(g["date"])

    g = g.sort_values("date")
    g_hist = g[g["date"] < ref_date].copy()

    if g_hist.empty:
        raise ValueError(f"No hay historial para {product_id} antes de {ref_date.date()}.")

    lookback_eff = min(lookback, len(g_hist))
    window = g_hist[feature_cols].tail(lookback_eff).to_numpy(np.float32)
    X_input = window.reshape(1, lookback_eff, len(feature_cols))
    last_row = g_hist.iloc[-1]
    return X_input, last_row


def recommend_order(
    product_id: str,
    ref_date,
    lead_time_days: int,
    service_z: float,
    panel_df: pd.DataFrame,
    model,
    feature_cols,
    lookback: int = LOOKBACK,
):
    X_input, last_row = build_window(panel_df, product_id, ref_date, feature_cols, lookback)

    y_pred_1day = float(model.predict(X_input, verbose=0)[0, 0])

    demand_during_lead = y_pred_1day * lead_time_days
    safety_stock = service_z * y_pred_1day

    current_stock = float(last_row.get("stock", last_row.get("quantity_available", 0.0)))

    base_target = demand_during_lead + safety_stock
    reorder_point = float(last_row.get("reorder_point", 0.0))
    opt_stock = float(last_row.get("optimal_stock_level", base_target))

    target_stock = max(base_target, reorder_point, opt_stock)
    qty_to_order = target_stock - current_stock

    if qty_to_order <= 0 and y_pred_1day > 0:
        qty_to_order = max(1.0, round(y_pred_1day))
    else:
        qty_to_order = max(0.0, round(qty_to_order, 2))

    info = {
        "product_id": product_id,
        "product_name": last_row.get("product_name", ""),
        "ref_date": pd.to_datetime(ref_date).date().isoformat(),
        "predicted_demand_1day": round(y_pred_1day, 2),
        "lead_time_days": lead_time_days,
        "demand_during_lead": round(demand_during_lead, 2),
        "safety_stock": round(safety_stock, 2),
        "target_stock": round(target_stock, 2),
        "current_stock": round(current_stock, 2),
        "recommended_order": round(qty_to_order, 2),
    }

    return float(qty_to_order), info


def recommend_orders_all(
    ref_date,
    lead_time_days: int,
    service_z: float,
    panel_df: pd.DataFrame,
    model,
    feature_cols,
    lookback: int = LOOKBACK,
):
    ref_date = pd.to_datetime(ref_date)

    products = sorted(panel_df["product_id"].unique())
    rows = []

    for pid in products:
        try:
            qty, info = recommend_order(
                product_id=pid,
                ref_date=ref_date,
                lead_time_days=lead_time_days,
                service_z=service_z,
                panel_df=panel_df,
                model=model,
                feature_cols=feature_cols,
                lookback=lookback,
            )
            if info["recommended_order"] > 0:
                rows.append(info)
        except ValueError:
            continue

    df_orders = pd.DataFrame(rows)
    if not df_orders.empty:
        df_orders = df_orders.sort_values("recommended_order", ascending=False)
    return df_orders


# ============================================================
# MODELOS Pydantic
# ============================================================

class Recommendation(BaseModel):
    product_id: str
    product_name: str
    ref_date: str
    predicted_demand_1day: float
    lead_time_days: int
    demand_during_lead: float
    safety_stock: float
    target_stock: float
    current_stock: float
    recommended_order: float


class RecommendationsResponse(BaseModel):
    ref_date: str
    lead_time_days: int
    service_z: float
    items: List[Recommendation]


# ============================================================
# APP FASTAPI
# ============================================================

app = FastAPI(
    title="Stock Forecast API",
    description="API para recomendar pedidos de reposición por producto y fecha.",
    version="1.0.0",
)

# ========= CORS =========
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todos los orígenes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok", "n_products": int(panel["product_id"].nunique())}


@app.get("/recomendacion/{product_id}", response_model=Recommendation)
def get_recommendation(
    product_id: str,
    fecha: str = Query(..., description="Fecha YYYY-MM-DD"),
    lead_time_days: int = Query(5, ge=1),
    service_z: float = Query(1.28),
):
    try:
        _, info = recommend_order(
            product_id=product_id,
            ref_date=fecha,
            lead_time_days=lead_time_days,
            service_z=service_z,
            panel_df=panel,
            model=model,
            feature_cols=feature_cols,
            lookback=LOOKBACK,
        )
        return Recommendation(**info)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/recomendaciones", response_model=RecommendationsResponse)
def get_recommendations_all(
    fecha: str = Query(..., description="Fecha YYYY-MM-DD"),
    lead_time_days: int = Query(5, ge=1),
    service_z: float = Query(1.28),
):
    df_orders = recommend_orders_all(
        ref_date=fecha,
        lead_time_days=lead_time_days,
        service_z=service_z,
        panel_df=panel,
        model=model,
        feature_cols=feature_cols,
        lookback=LOOKBACK,
    )

    items: List[Recommendation] = []
    if not df_orders.empty:
        for _, row in df_orders.iterrows():
            items.append(Recommendation(**row.to_dict()))

    return RecommendationsResponse(
        ref_date=fecha,
        lead_time_days=lead_time_days,
        service_z=service_z,
        items=items,
    )


# ============================================================
# ENDPOINT DE REENTRENAMIENTO CON CSV
# ============================================================

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    try:
        print("Archivo recibido:", file.filename)

        # Ruta temporal
        temp_file = DATA_DIR / "temp.csv"
        with open(temp_file, "wb") as f:
            f.write(await file.read())

        print("Archivo guardado en:", temp_file)

        # Leer CSV en UTF-8
        df_new = pd.read_csv(temp_file, encoding="utf-8")
        print("CSV cargado correctamente. Filas nuevas:", len(df_new))

        # ========= NORMALIZAR COLUMNA DATE EN EL CSV NUEVO =========
        if "date" in df_new.columns:
            # Tus nuevos CSV vienen como "YYYY-MM-DD"
            df_new["date"] = pd.to_datetime(
                df_new["date"],
                format="%Y-%m-%d",   # fuerza exactamente este formato
                errors="coerce"
            )

            # Validar que no haya fechas inválidas
            if df_new["date"].isna().any():
                raise HTTPException(
                    status_code=400,
                    detail="El CSV contiene fechas en formato inválido. Debe ser YYYY-MM-DD."
                )

        # --- Validar columnas mínimas ---
        required_min = ["product_id", "product_name", "date", "sale_amount"]
        required_features = [c for c in feature_cols]
        required_all = list(dict.fromkeys(required_min + required_features))

        missing = [c for c in required_all if c not in df_new.columns]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"El CSV no contiene todas las columnas necesarias. Faltan: {missing}",
            )

        # ========= CARGAR PANEL EXISTENTE Y NORMALIZAR DATE =========
        df_existing = pd.read_csv(PANEL_CSV)  # sin parse_dates aquí

        if "date" in df_existing.columns:
            df_existing["date"] = pd.to_datetime(
                df_existing["date"],
                errors="coerce"
            )

        # Asegurarnos de que tampoco tenga NaT raros
        if df_existing["date"].isna().any():
            # Si quieres, aquí podrías limpiar o lanzar error;
            # por ahora simplemente rellenamos con 0 para evitar crasheos
            df_existing = df_existing.dropna(subset=["date"])

        # ========= UNIFICAR COLUMNAS Y COMBINAR =========
        all_cols = sorted(set(df_existing.columns) | set(df_new.columns))
        df_existing = df_existing.reindex(columns=all_cols)
        df_new = df_new.reindex(columns=all_cols)

        df_combined = (
            pd.concat([df_existing, df_new], ignore_index=True)
            .sort_values(["product_id", "date"])
            .drop_duplicates(subset=["product_id", "date"], keep="last")
        )

        # Asegurar que la date combinada sigue siendo datetime
        if "date" in df_combined.columns:
            df_combined["date"] = pd.to_datetime(df_combined["date"], errors="coerce")

        # Guardar panel actualizado
        df_combined.to_csv(PANEL_CSV, index=False, encoding="utf-8-sig")
        print("Panel actualizado guardado en:", PANEL_CSV)

        # Actualizar el panel global en memoria
        global panel
        panel = df_combined

        # ========= RECONSTRUIR SECUENCIAS Y REENTRENAR =========
        X_train, y_train = build_sequences_from_panel(
            PANEL_CSV, lookback=LOOKBACK, horizon=1
        )

        n_features_data = X_train.shape[2]
        n_features_model = model.input_shape[2]
        if n_features_data != n_features_model:
            raise RuntimeError(
                f"Mismatch de features: datos={n_features_data}, modelo={n_features_model}. "
                "Revisa que feature_cols y el CSV tengan las mismas columnas."
            )

        print(
            f"Iniciando reentrenamiento incremental: X_train={X_train.shape}, y_train={y_train.shape}"
        )

        model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=1)

        # Guardar el modelo actualizado
        model.save(MODEL_PATH)
        print("Modelo reentrenado y guardado en:", MODEL_PATH)

        return {
            "filename": file.filename,
            "status": "success",
            "message": "CSV cargado y reentrenamiento iniciado correctamente.",
        }

    except HTTPException:
        # Re-lanzar errores de validación para que lleguen tal cual al frontend
        raise
    except Exception as e:
        print(f"Error al procesar el archivo: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al cargar o reentrenar el modelo: {e}",
        )
