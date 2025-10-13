"""
FastAPI - Deteksi Kanker Payudara (COMPLETE VERSION)
2 MODELS IN 1 API:
1. Full Model WITH KET (F1-F17 + P1-P11 + Ket) → 29 features → 3 classes
2. Risk Screening Model (F1-F17) → 17 features → 2 classes

Support: 0/1 atau Ya/Tidak

Author: AI Assistant  
Date: 2025-10-13
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Union
import pickle
import numpy as np
from datetime import datetime
import uvicorn

# ===========================================================================================
# INITIALIZE FASTAPI
# ===========================================================================================

app = FastAPI(
    title="API Deteksi Kanker Payudara - Complete",
    description="API dengan 2 model: Full Diagnosis (29 features with Ket) & Risk Screening (17 features)",
    version="4.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===========================================================================================
# LOAD MODELS (2 MODELS)
# ===========================================================================================

print("=" * 80)
print("🔄 Loading Models...")
print("=" * 80)

# MODEL 1: Full Model WITH KET (F1-F17 + P1-P11 + Ket) - 29 features
try:
    with open('model_production_with_ket.pkl', 'rb') as f:
        model_full_data = pickle.load(f)
    
    model_full = model_full_data['model']
    le_full = model_full_data['label_encoder']
    features_full = model_full_data['feature_columns']
    has_ket = model_full_data.get('has_ket_feature', False)
    
    print("✅ Model Full (with Ket) berhasil dimuat!")
    print(f"   Features: {len(features_full)} (F1-F17 + P1-P11 + Ket)")
    print(f"   Has Ket: {'✅ Yes' if has_ket else '❌ No'}")
    print(f"   Classes: {list(le_full.classes_)}")
    MODEL_FULL_LOADED = True
except Exception as e:
    print(f"❌ Model Full ERROR: {e}")
    MODEL_FULL_LOADED = False

# MODEL 2: Risk Screening Model (F1-F17 only) - 17 features
try:
    with open('model_risk_screening_improved.pkl', 'rb') as f:
        model_risk_data = pickle.load(f)
    
    model_risk = model_risk_data['model']
    le_risk = model_risk_data['label_encoder']
    features_risk = model_risk_data['feature_columns']
    
    print("✅ Model Risk Screening berhasil dimuat!")
    print(f"   Features: {len(features_risk)} (F1-F17 only)")
    print(f"   Classes: {list(le_risk.classes_)}")
    MODEL_RISK_LOADED = True
except Exception as e:
    print(f"❌ Model Risk ERROR: {e}")
    MODEL_RISK_LOADED = False

print("=" * 80)
print(f"📊 Status:")
print(f"   Model Full (with Ket): {'✅ Ready' if MODEL_FULL_LOADED else '❌ Not loaded'}")
print(f"   Model Risk: {'✅ Ready' if MODEL_RISK_LOADED else '❌ Not loaded'}")
print("=" * 80)

# ===========================================================================================
# PYDANTIC MODELS
# ===========================================================================================

class PatientDataFull(BaseModel):
    """Model untuk Full Diagnosis (29 features WITH KET)"""
    nama: Optional[str] = None
    umur: Optional[int] = None
    
    # F1-F17
    F1: Union[str, int]; F2: Union[str, int]; F3: Union[str, int]
    F4: Union[str, int]; F5: Union[str, int]; F6: Union[str, int]
    F7: Union[str, int]; F8: Union[str, int]; F9: Union[str, int]
    F10: Union[str, int]; F11: Union[str, int]; F12: Union[str, int]
    F13: Union[str, int]; F14: Union[str, int]; F15: Union[str, int]
    F16: Union[str, int]; F17: Union[str, int]
    
    # P1-P11
    P1: Union[str, int]; P2: Union[str, int]; P3: Union[str, int]
    P4: Union[str, int]; P5: Union[str, int]; P6: Union[str, int]
    P7: Union[str, int]; P8: Union[str, int]; P9: Union[str, int]
    P10: Union[str, int]; P11: Union[str, int]
    
    # KET - NEW!
    Ket: Union[str, int] = Field(0, description="Keterangan: 0=tidak ada isi, 1=ada isi")
    
    class Config:
        schema_extra = {
            "example": {
                "nama": "Ibu Siti", "umur": 45,
                "F1": 1, "F2": 0, "F3": 1, "F4": 0, "F5": 0, "F6": 0,
                "F7": 1, "F8": 0, "F9": 0, "F10": 0, "F11": 0, "F12": 1,
                "F13": 0, "F14": 0, "F15": 0, "F16": 0, "F17": 0,
                "P1": 1, "P2": 1, "P3": 1, "P4": 1, "P5": 1, "P6": 0,
                "P7": 1, "P8": 1, "P9": 0, "P10": 0, "P11": 1,
                "Ket": 1
            }
        }

class RiskFactorsInput(BaseModel):
    """Model untuk Risk Screening (17 features)"""
    nama: Optional[str] = None
    umur: Optional[int] = None
    
    # F1-F17 only
    F1: Union[str, int]; F2: Union[str, int]; F3: Union[str, int]
    F4: Union[str, int]; F5: Union[str, int]; F6: Union[str, int]
    F7: Union[str, int]; F8: Union[str, int]; F9: Union[str, int]
    F10: Union[str, int]; F11: Union[str, int]; F12: Union[str, int]
    F13: Union[str, int]; F14: Union[str, int]; F15: Union[str, int]
    F16: Union[str, int]; F17: Union[str, int]
    
    class Config:
        schema_extra = {
            "example": {
                "nama": "Ibu Ani", "umur": 40,
                "F1": 1, "F2": 0, "F3": 1, "F4": 0, "F5": 1, "F6": 0,
                "F7": 1, "F8": 0, "F9": 0, "F10": 0, "F11": 0, "F12": 1,
                "F13": 0, "F14": 0, "F15": 0, "F16": 0, "F17": 0
            }
        }

# ===========================================================================================
# HELPER FUNCTIONS
# ===========================================================================================

def convert_to_binary(value: Union[str, int]) -> int:
    """Convert input (Ya/Tidak/0/1) ke binary"""
    if isinstance(value, int):
        return 1 if value > 0 else 0
    if isinstance(value, str):
        v = value.lower().strip()
        if v in ['ya', 'yes', 'y', 'true']:
            return 1
        elif v in ['tidak', 'no', 'n', 'false']:
            return 0
        try:
            return 1 if int(value) > 0 else 0
        except:
            return 0
    return 0

# ===========================================================================================
# API ENDPOINTS - GENERAL
# ===========================================================================================

@app.get("/")
async def root():
    """Homepage API"""
    return {
        "message": "API Deteksi Kanker Payudara - Complete",
        "version": "4.0.0",
        "models": {
            "model_full": {
                "status": "Ready" if MODEL_FULL_LOADED else "Not loaded",
                "features": "29 (F1-F17 + P1-P11 + Ket)",
                "has_ket": has_ket if MODEL_FULL_LOADED else False,
                "output": "3 classes (Normal, Jinak, Ganas)",
                "endpoint": "/predict-full"
            },
            "model_risk": {
                "status": "Ready" if MODEL_RISK_LOADED else "Not loaded",
                "features": "17 (F1-F17)",
                "output": "2 classes (Tidak Suspect, Suspect)",
                "endpoint": "/predict-risk"
            }
        },
        "documentation": "/docs"
    }

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "models": {
            "full": MODEL_FULL_LOADED,
            "risk": MODEL_RISK_LOADED
        },
        "timestamp": datetime.now().isoformat()
    }

# ===========================================================================================
# ENDPOINT 1: FULL DIAGNOSIS (29 FEATURES WITH KET) → 3 CLASSES
# ===========================================================================================

@app.post("/predict-full")
async def predict_full(data: PatientDataFull):
    """
    Full Diagnosis dengan 29 features (F1-F17 + P1-P11 + Ket)
    Output: 3 classes (Normal, Suspect Kelainan Jinak, Suspect Kelainan Ganas)
    """
    
    if not MODEL_FULL_LOADED:
        raise HTTPException(status_code=503, detail="Model Full not loaded")
    
    try:
        data_dict = data.dict()
        
        # Encode (handle Ket specially)
        data_encoded = []
        for feat in features_full:
            if feat == 'Ket':
                # Ket is already 0/1
                value = data_dict.get('Ket', 0)
                data_encoded.append(convert_to_binary(value))
            else:
                # F1-F17, P1-P11
                value = data_dict.get(feat, 0)
                data_encoded.append(convert_to_binary(value))
        
        X_new = np.array(data_encoded).reshape(1, -1)
        
        # Predict
        pred = model_full.predict(X_new)[0]
        pred_proba = model_full.predict_proba(X_new)[0]
        hasil = le_full.inverse_transform([pred])[0]
        
        # Probabilities
        probabilities = dict(zip(le_full.classes_, [float(p) for p in pred_proba]))
        
        # Risk assessment
        ganas_idx = list(le_full.classes_).index('Suspect Kelainan Ganas')
        prob_ganas = pred_proba[ganas_idx]
        
        if hasil == "Normal":
            risk_level = "Low"
            color = "green"
            if prob_ganas >= 0.3:
                warning = "⚠️ Probabilitas Ganas ≥30%"
                recommendation = "Segera konsultasi dokter (USG/Mammografi)"
            elif prob_ganas >= 0.2:
                warning = "ℹ️ Probabilitas Ganas ≥20%"
                recommendation = "Pemeriksaan rutin berkala"
            else:
                warning = None
                recommendation = "Tetap jaga pola hidup sehat dan SADARI rutin"
        elif hasil == "Suspect Kelainan Jinak":
            risk_level = "Medium"
            color = "yellow"
            warning = "⚠️ Terdeteksi suspek kelainan jinak"
            recommendation = "Segera konsultasi dokter (USG/Mammografi)"
        else:
            risk_level = "High"
            color = "red"
            warning = "🚨 PERHATIAN: Terdeteksi suspek kelainan GANAS"
            recommendation = "🏥 SEGERA konsultasi dokter onkologi (Biopsi/MRI)"
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "model_type": "Full Diagnosis (29 features with Ket)",
            "patient_info": {"nama": data.nama, "umur": data.umur} if data.nama or data.umur else None,
            "prediction": hasil,
            "risk_level": risk_level,
            "color_code": color,
            "probabilities": probabilities,
            "probability_ganas": f"{prob_ganas*100:.2f}%",
            "warning": warning,
            "recommendation": recommendation,
            "ket_provided": data_dict.get('Ket', 0) == 1
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ===========================================================================================
# ENDPOINT 2: RISK SCREENING (17 FEATURES) → 2 CLASSES
# ===========================================================================================

@app.post("/predict-risk")
async def predict_risk(data: RiskFactorsInput):
    """
    Risk Screening dengan 17 features (F1-F17 saja)
    Output: 2 classes (Tidak Suspect, Suspect)
    """
    
    if not MODEL_RISK_LOADED:
        raise HTTPException(status_code=503, detail="Model Risk not loaded")
    
    try:
        data_dict = data.dict()
        
        # Encode
        data_encoded = [convert_to_binary(data_dict.get(f, 0)) for f in features_risk]
        X_new = np.array(data_encoded).reshape(1, -1)
        
        # Predict
        pred = model_risk.predict(X_new)[0]
        pred_proba = model_risk.predict_proba(X_new)[0]
        hasil = le_risk.inverse_transform([pred])[0]
        
        # Probabilities
        probabilities = dict(zip(le_risk.classes_, [float(p) for p in pred_proba]))
        
        # Risk factors count
        risk_count = sum(data_encoded)
        
        # Risk assessment
        if hasil == "Tidak Suspect":
            risk_level = "Low"
            color = "green"
            recommendation = "✅ Tidak terdeteksi faktor risiko signifikan. Tetap jaga pola hidup sehat."
            next_step = "Pemeriksaan rutin berkala"
        else:
            prob_suspect = probabilities["Suspect"]
            if prob_suspect >= 0.8:
                risk_level = "High"
                color = "red"
                recommendation = "🚨 Faktor risiko tinggi. SEGERA lakukan pemeriksaan fisik lengkap (P1-P11)."
                next_step = "SEGERA: Konsultasi dokter + USG/Mammografi"
            elif prob_suspect >= 0.6:
                risk_level = "Medium"
                color = "orange"
                recommendation = "⚠️ Faktor risiko sedang. Disarankan pemeriksaan fisik (P1-P11)."
                next_step = "Konsultasi dokter dalam 1-2 minggu"
            else:
                risk_level = "Medium-Low"
                color = "yellow"
                recommendation = "ℹ️ Beberapa faktor risiko terdeteksi. Monitoring berkala diperlukan."
                next_step = "Monitoring + pemeriksaan jika ada gejala"
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "model_type": "Risk Screening (17 features)",
            "patient_info": {"nama": data.nama, "umur": data.umur} if data.nama or data.umur else None,
            "result": hasil,
            "risk_level": risk_level,
            "color_code": color,
            "probabilities": probabilities,
            "risk_factors_count": f"{risk_count}/17",
            "recommendation": recommendation,
            "next_step": next_step,
            "note": "Ini risk screening awal. Untuk diagnosis lebih lanjut, lengkapi dengan pemeriksaan fisik (P1-P11)."
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# ===========================================================================================
# ENDPOINT 3: SMART PREDICT (AUTO-DETECT)
# ===========================================================================================

@app.post("/predict")
async def predict_smart(data: dict):
    """
    Smart Predict: Auto-detect apakah data lengkap atau hanya F1-F17
    
    - Jika ada P1-P11 (dan optional Ket) → gunakan Model Full
    - Jika hanya F1-F17 → gunakan Model Risk Screening
    """
    
    # Check if P1-P11 present
    has_p_features = any(f"P{i}" in data for i in range(1, 12))
    
    if has_p_features:
        # Route ke Full Model (with Ket)
        if not MODEL_FULL_LOADED:
            raise HTTPException(status_code=503, detail="Model Full not loaded")
        
        # Add default Ket if not provided
        if 'Ket' not in data:
            data['Ket'] = 0
        
        try:
            patient_data = PatientDataFull(**data)
            return await predict_full(patient_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid data format: {str(e)}")
    else:
        # Route ke Risk Screening Model
        if not MODEL_RISK_LOADED:
            raise HTTPException(status_code=503, detail="Model Risk not loaded")
        
        try:
            risk_data = RiskFactorsInput(**data)
            return await predict_risk(risk_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid data format: {str(e)}")

# ===========================================================================================
# RUN SERVER
# ===========================================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 Starting Complete API Server...")
    print("=" * 80)
    print("📍 Server: http://127.0.0.1:8000")
    print("📍 Docs: http://127.0.0.1:8000/docs")
    print("=" * 80)
    print("\n📋 Available Endpoints:")
    print("   POST /predict-full  → Full diagnosis (29 features WITH Ket)")
    print("   POST /predict-risk  → Risk screening (17 features)")
    print("   POST /predict       → Smart auto-detect")
    print("=" * 80)
    print("\n✨ Features:")
    print("   ✅ 2 Models dalam 1 API")
    print("   ✅ Model Full: 29 features (F1-F17 + P1-P11 + Ket)")
    print("   ✅ Model Risk: 17 features (F1-F17)")
    print("   ✅ Support format 0/1 atau Ya/Tidak")
    print("   ✅ Auto-detect endpoint /predict")
    print("\n💡 Tekan CTRL+C untuk stop\n")
    
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
