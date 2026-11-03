import os
import sys
import pandas as pd
from typing import List, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from catboost import CatBoostClassifier, Pool

# Добавляем в sys.path путь к data_science/src для импорта feature_engineering
DATA_SCIENCE_SRC = os.getenv("DATA_SCIENCE_SRC", "./data_science/src")
if DATA_SCIENCE_SRC not in sys.path:
    sys.path.append(DATA_SCIENCE_SRC)

from feature_engineering import prepare_features, get_categorical_features

app = FastAPI(title="Churn Copilot Prediction Service")

model: CatBoostClassifier = None
df_features: pd.DataFrame = None
cat_features: list = get_categorical_features()

class FeatureImpact(BaseModel):
    feature: str
    feature_value: Union[str, int, float, None]
    risk_contribution_pct: float

class ChurnResponse(BaseModel):
    client_id: int
    churn_probability: float
    top_reasons: List[FeatureImpact]

class BatchRequest(BaseModel):
    client_ids: List[int]

class BatchItem(BaseModel):
    client_id: int
    churn_probability: float

@app.on_event("startup")
def load_model_and_data():
    global model, df_features
    
    model_path = os.getenv("MODEL_PATH", "/opt/data_science/artifacts/best_model.cbm")
    data_path = os.getenv("DATA_PATH", "/opt/data_science/data/raw/test.csv")
    
    if not os.path.exists(model_path):
        raise RuntimeError(f"Модель не найдена по пути {model_path}.")
    if not os.path.exists(data_path):
        raise RuntimeError(f"Тестовые данные не найдены по пути {data_path}")
        
    model = CatBoostClassifier()
    model.load_model(model_path)
    
    X, _ = prepare_features(data_path, target_col='Exited')
    df_features = X.reset_index(drop=True)


@app.get("/predict", response_model=ChurnResponse)
def predict_churn(client_id: int):
    if model is None or df_features is None:
        raise HTTPException(status_code=503, detail="Модель или данные не загружены")
        
    if client_id < 0 or client_id >= len(df_features):
        raise HTTPException(
            status_code=404, 
            detail=f"Client ID {client_id} не найден. Доступный диапазон: 0 - {len(df_features) - 1}"
        )
        
    X_row = df_features.iloc[[client_id]]
    proba = model.predict_proba(X_row)[0][1]
    
    pool = Pool(X_row, cat_features=cat_features)
    shap_values = model.get_feature_importance(pool, type='ShapValues')
    
    sv = shap_values[0, :-1]
    
    # Собираем только те факторы, которые увеличивают риск (impact > 0)
    risk_factors = []
    total_risk_shap = 0.0
    
    for i, col_name in enumerate(X_row.columns):
        if sv[i] > 0:
            val = X_row.iloc[0][col_name]
            if hasattr(val, 'item'):
                val = val.item()
            elif pd.isna(val):
                val = None
                
            risk_factors.append({
                "feature": col_name,
                "feature_value": val,
                "raw_shap": float(sv[i])
            })
            total_risk_shap += float(sv[i])  # Считаем общую сумму положительных SHAP
            
    # Переводим в проценты (долю в общем риске)
    for item in risk_factors:
        if total_risk_shap > 0:
            # Считаем процент, округляем до 1 знака после запятой
            item["risk_contribution_pct"] = round((item["raw_shap"] / total_risk_shap) * 100, 1)
        else:
            item["risk_contribution_pct"] = 0.0
            
    # Сортируем по убыванию доли (от самого сильного фактора к слабому)
    risk_factors.sort(key=lambda x: x["risk_contribution_pct"], reverse=True)
    
    # Берем топ-5
    top_risks = risk_factors[:5]
    
    # Формируем ответ
    response_reasons = [
        FeatureImpact(
            feature=item["feature"], 
            feature_value=item["feature_value"],
            risk_contribution_pct=item["risk_contribution_pct"]
        ) for item in top_risks
    ]
    
    return ChurnResponse(
        client_id=client_id,
        churn_probability=float(round(proba, 4)),
        top_reasons=response_reasons
    )


@app.post("/predict_batch", response_model=List[BatchItem])
def predict_batch(body: BatchRequest):
    """Вероятности пачкой: один прогон CatBoost по всем строкам (для серверных
    фильтров рисков в agent_service — без SHAP и построчных вызовов)."""
    if model is None or df_features is None:
        raise HTTPException(status_code=503, detail="Модель или данные не загружены")

    n = len(df_features)
    invalid = sorted({cid for cid in body.client_ids if cid < 0 or cid >= n})
    if invalid:
        preview = ", ".join(map(str, invalid[:10]))
        raise HTTPException(
            status_code=404,
            detail=f"Client IDs вне диапазона 0 - {n - 1}: {preview}"
        )

    proba = model.predict_proba(df_features.iloc[body.client_ids])[:, 1]
    return [
        BatchItem(client_id=cid, churn_probability=float(round(p, 4)))
        for cid, p in zip(body.client_ids, proba)
    ]
