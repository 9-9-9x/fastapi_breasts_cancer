"""
FastAPI - Deteksi Kanker Payudara (FIXED FEATURE COUNT VERSION)
Model Full: 28 features (F1-F17 + P1-P11) tanpa Ket
Model Risk: 17 features (F1-F17)
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import Union, Dict, Any, Optional
import pickle
import numpy as np
from datetime import datetime
import uvicorn
import traceback
import joblib

# ===========================================================================================
# INITIALIZE FASTAPI
# ===========================================================================================

app = FastAPI(
    title="API Deteksi Kanker Payudara - Complete",
    description="API dengan 2 model: Full Diagnosis (28 features) & Risk Screening (17 features)",
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
# CUSTOM LABEL ENCODER CLASS
# ===========================================================================================

class SimpleLabelEncoder:
    """Simple label encoder untuk model FR dengan 2 kelas: Tidak Suspect, Suspect"""
    def __init__(self):
        self.classes_ = np.array(['Tidak Suspect', 'Suspect'])
    
    def inverse_transform(self, labels):
        if hasattr(labels, '__iter__'):
            return [self.classes_[label] for label in labels]
        else:
            return self.classes_[labels]
    
    def transform(self, labels):
        label_to_idx = {'Tidak Suspect': 0, 'Suspect': 1}
        if hasattr(labels, '__iter__'):
            return [label_to_idx[label] for label in labels]
        else:
            return label_to_idx[labels]

# ===========================================================================================
# LOAD MODELS (2 MODELS) - DENGAN DEBUG INFO
# ===========================================================================================

print("=" * 80)
print("🔄 Loading Models...")
print("=" * 80)

# VARIABEL GLOBAL
model_full = None
le_full = None
features_full = []
expected_features_full = 28  # TANPA Ket

model_risk = None
le_risk = None
features_risk = []

MODEL_FULL_LOADED = False
MODEL_RISK_LOADED = False

# DEBUG: Cek isi model file
def debug_model_file(filename):
    print(f"\n🔍 Debugging {filename}:")
    try:
        data = joblib.load(filename)
        print(f"   Type: {type(data)}")
        if isinstance(data, dict):
            print(f"   Keys: {list(data.keys())}")
            if 'feature_columns' in data:
                print(f"   Features: {len(data['feature_columns'])}")
                print(f"   Feature list: {data['feature_columns']}")
                print(f"   Has 'Ket' in features: {'Ket' in data['feature_columns']}")
        elif hasattr(data, 'feature_names_in_'):
            print(f"   Model features: {data.feature_names_in_}")
            print(f"   Number of features: {len(data.feature_names_in_)}")
        return data
    except Exception as e:
        print(f"   Error: {e}")
        return None

# MODEL 1: Full Model - 28 features TANPA Ket
print("\n📦 Loading Full Model...")
try:
    model_full_data = debug_model_file('naive_bayes_FULL_20251229_183259.pkl')
    
    if isinstance(model_full_data, dict):
        model_full = model_full_data.get('model')
        le_full = model_full_data.get('label_encoder')
        features_full = model_full_data.get('feature_columns', [])
    else:
        model_full = model_full_data
        # Coba ambil features dari model
        if hasattr(model_full, 'feature_names_in_'):
            features_full = model_full.feature_names_in_.tolist()
    
    # DEBUG: Print informasi features
    print(f"\n📊 Full Model Features Analysis:")
    print(f"   Total features loaded: {len(features_full)}")
    print(f"   Features: {features_full}")
    print(f"   Has 'Ket': {'Ket' in features_full}")
    
    # HAPUS 'Ket' dari features jika ada (karena model hanya 28 features)
    original_count = len(features_full)
    features_full = [f for f in features_full if f != 'Ket']
    print(f"   After removing 'Ket': {len(features_full)} features")
    
    if le_full is None and hasattr(model_full, 'classes_'):
        # Buat label encoder sederhana
        le_full = type('TempLabelEncoder', (), {})()
        le_full.classes_ = model_full.classes_
        le_full.inverse_transform = lambda x: [model_full.classes_[i] for i in x]
    
    print("✅ Model Full berhasil dimuat!")
    print(f"   Features: {len(features_full)} (F1-F17 + P1-P11)")
    print(f"   Expected: 28 features")
    print(f"   Classes: {model_full.classes_}")
    MODEL_FULL_LOADED = True
    
except Exception as e:
    print(f"❌ Model Full ERROR: {e}")
    print(f"   Traceback: {traceback.format_exc()}")
    MODEL_FULL_LOADED = False

# MODEL 2: Risk Screening Model
print("\n📦 Loading Risk Model...")
try:
    model_risk_data = debug_model_file('fr_model_20251229_181516.pkl')
    
    if isinstance(model_risk_data, dict):
        model_risk = model_risk_data.get('model')
        features_risk = model_risk_data.get('feature_names', [])
    else:
        model_risk = model_risk_data
        if hasattr(model_risk, 'feature_names_in_'):
            features_risk = model_risk.feature_names_in_.tolist()
    
    # Buat label encoder untuk risk model
    le_risk = SimpleLabelEncoder()
    
    # Default jika kosong
    if not features_risk:
        features_risk = [f"F{i}" for i in range(1, 18)]
    
    print("✅ Model Risk Screening berhasil dimuat!")
    print(f"   Features: {len(features_risk)} (F1-F17)")
    print(f"   Classes: {le_risk.classes_}")
    MODEL_RISK_LOADED = True
    
except Exception as e:
    print(f"❌ Model Risk ERROR: {e}")
    print(f"   Traceback: {traceback.format_exc()}")
    MODEL_RISK_LOADED = False

print("=" * 80)
print(f"📊 Status:")
print(f"   Model Full: {'✅ Ready' if MODEL_FULL_LOADED else '❌ Not loaded'} ({len(features_full) if features_full else 0} features)")
print(f"   Model Risk: {'✅ Ready' if MODEL_RISK_LOADED else '❌ Not loaded'} ({len(features_risk) if features_risk else 0} features)")
print("=" * 80)

# ===========================================================================================
# PYDANTIC MODELS - PERBAIKAN: Ket jadi optional
# ===========================================================================================

class PatientDataFull(BaseModel):
    """Model untuk Full Diagnosis (28 features TANPA Ket)"""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "F1": 1, "F2": 0, "F3": 1, "F4": 0, "F5": 0, "F6": 0,
                "F7": 1, "F8": 0, "F9": 0, "F10": 0, "F11": 0, "F12": 1,
                "F13": 0, "F14": 0, "F15": 0, "F16": 0, "F17": 0,
                "P1": 1, "P2": 1, "P3": 1, "P4": 1, "P5": 1, "P6": 0,
                "P7": 1, "P8": 1, "P9": 0, "P10": 0, "P11": 1
            }
        }
    )
    
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
    
    # KET menjadi optional, karena model tidak membutuhkannya
    Ket: Optional[Union[str, int]] = Field(default=None, description="Keterangan: tidak digunakan dalam model")

class RiskFactorsInput(BaseModel):
    """Model untuk Risk Screening (17 features)"""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "F1": 1, "F2": 0, "F3": 1, "F4": 0, "F5": 1, "F6": 0,
                "F7": 1, "F8": 0, "F9": 0, "F10": 0, "F11": 0, "F12": 1,
                "F13": 0, "F14": 0, "F15": 0, "F16": 0, "F17": 0
            }
        }
    )
    
    # F1-F17 only
    F1: Union[str, int]; F2: Union[str, int]; F3: Union[str, int]
    F4: Union[str, int]; F5: Union[str, int]; F6: Union[str, int]
    F7: Union[str, int]; F8: Union[str, int]; F9: Union[str, int]
    F10: Union[str, int]; F11: Union[str, int]; F12: Union[str, int]
    F13: Union[str, int]; F14: Union[str, int]; F15: Union[str, int]
    F16: Union[str, int]; F17: Union[str, int]

# ===========================================================================================
# HELPER FUNCTIONS - DIPERBAIKI
# ===========================================================================================

def convert_to_binary(value: Union[str, int]) -> int:
    """Convert input (Ya/Tidak/0/1) ke binary"""
    if value is None:
        return 0
    if isinstance(value, int):
        return 1 if value > 0 else 0
    if isinstance(value, str):
        v = value.lower().strip()
        if v in ['ya', 'yes', 'y', 'true', '1', 'benar', 't']:
            return 1
        elif v in ['tidak', 'no', 'n', 'false', '0', 'salah']:
            return 0
        try:
            return 1 if int(value) > 0 else 0
        except:
            return 0
    return 0

def prepare_full_features(data_dict: Dict) -> np.ndarray:
    """Siapkan features untuk Full Model (28 features TANPA Ket)"""
    print(f"\n🔧 Preparing Full Model features...")
    
    # Urutkan features sesuai dengan model
    # Model mengharapkan: F1-F17, lalu P1-P11
    input_data = []
    
    # F1-F17
    for i in range(1, 18):
        key = f"F{i}"
        value = data_dict.get(key, 0)
        binary_value = convert_to_binary(value)
        input_data.append(binary_value)
        print(f"   {key}: {value} -> {binary_value}")
    
    # P1-P11
    for i in range(1, 12):
        key = f"P{i}"
        value = data_dict.get(key, 0)
        binary_value = convert_to_binary(value)
        input_data.append(binary_value)
        print(f"   {key}: {value} -> {binary_value}")
    
    # IGNORE Ket - tidak digunakan
    if 'Ket' in data_dict:
        print(f"   Ket: {data_dict['Ket']} -> IGNORED (not used in model)")
    
    print(f"   Total features: {len(input_data)}")
    print(f"   Features array: {input_data}")
    
    return np.array(input_data).reshape(1, -1)

def prepare_risk_features(data_dict: Dict) -> np.ndarray:
    """Siapkan features untuk Risk Model (17 features)"""
    print(f"\n🔧 Preparing Risk Model features...")
    
    input_data = []
    
    # Hanya F1-F17
    for i in range(1, 18):
        key = f"F{i}"
        value = data_dict.get(key, 0)
        binary_value = convert_to_binary(value)
        input_data.append(binary_value)
        print(f"   {key}: {value} -> {binary_value}")
    
    print(f"   Total features: {len(input_data)}")
    
    return np.array(input_data).reshape(1, -1)

# ===========================================================================================
# API ENDPOINTS - DIPERBAIKI
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
                "features": f"{len(features_full)} (F1-F17 + P1-P11)",
                "note": "Tidak termasuk Ket (28 features total)",
                "output": "3 classes (Normal, Suspect Kelainan Jinak, Suspect Kelainan Ganas)",
                "endpoint": "/predict-full"
            },
            "model_risk": {
                "status": "Ready" if MODEL_RISK_LOADED else "Not loaded",
                "features": f"{len(features_risk)} (F1-F17)",
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
        "status": "healthy" if (MODEL_FULL_LOADED or MODEL_RISK_LOADED) else "degraded",
        "models": {
            "full": MODEL_FULL_LOADED,
            "risk": MODEL_RISK_LOADED
        },
        "features": {
            "full": len(features_full) if features_full else 0,
            "risk": len(features_risk) if features_risk else 0
        },
        "timestamp": datetime.now().isoformat()
    }

# ===========================================================================================
# ENDPOINT 1: FULL DIAGNOSIS (28 FEATURES TANPA KET)
# ===========================================================================================

@app.post("/predict-full")
async def predict_full(data: PatientDataFull):
    """Full Diagnosis dengan 28 features (F1-F17 + P1-P11) TANPA Ket"""
    
    if not MODEL_FULL_LOADED:
        raise HTTPException(status_code=503, detail="Model Full not loaded")
    
    try:
        data_dict = data.model_dump()
        print(f"\n🎯 Full Model Prediction Request")
        print(f"   Received keys: {list(data_dict.keys())}")
        
        # Prepare features (TANPA Ket)
        X = prepare_full_features(data_dict)
        print(f"   X shape: {X.shape}")
        print(f"   Expected: (1, 28)")
        
        # Predict
        prediction = model_full.predict(X)[0]
        
        # Convert prediction to label
        if hasattr(le_full, 'inverse_transform'):
            result = le_full.inverse_transform([prediction])[0]
        else:
            result = le_full.classes_[prediction]
        
        print(f"   Raw prediction: {prediction}")
        print(f"   Result: {result}")
        
        # Get probabilities
        if hasattr(model_full, 'predict_proba'):
            proba = model_full.predict_proba(X)[0]
        else:
            proba = [0.33, 0.33, 0.34]
        
        # Create probabilities dictionary
        probabilities = {}
        for i, cls in enumerate(le_full.classes_):
            probabilities[str(cls)] = float(proba[i])
        
        print(f"   Probabilities: {probabilities}")
        
        # Risk assessment
        ganas_prob = probabilities.get("Suspect Kelainan Ganas", 0.0)
        
        if result == "Normal":
            risk_level = "Low"
            color = "green"
            if ganas_prob >= 0.3:
                warning = "⚠️ Probabilitas Ganas ≥30%"
                recommendation = "Segera konsultasi dokter (USG/Mammografi)"
            elif ganas_prob >= 0.2:
                warning = "ℹ️ Probabilitas Ganas ≥20%"
                recommendation = "Pemeriksaan rutin berkala"
            else:
                warning = None
                recommendation = "Tetap jaga pola hidup sehat dan SADARI rutin"
        elif result == "Suspect Kelainan Jinak":
            risk_level = "Medium"
            color = "yellow"
            warning = "⚠️ Terdeteksi suspek kelainan jinak"
            recommendation = "Segera konsultasi dokter (USG/Mammografi)"
        else:
            risk_level = "High"
            color = "red"
            warning = "🚨 PERHATIAN: Terdeteksi suspek kelainan GANAS"
            recommendation = "🏥 SEGERA konsultasi dokter onkologi (Biopsi/MRI)"
        
        response = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "model_type": "Full Diagnosis (28 features)",
            "prediction": result,
            "risk_level": risk_level,
            "color_code": color,
            "probabilities": probabilities,
            "probability_ganas": f"{ganas_prob*100:.2f}%",
            "warning": warning,
            "recommendation": recommendation,
            "note": "Model menggunakan 28 features (F1-F17 + P1-P11) tanpa Ket"
        }
        
        print(f"✅ Prediction successful: {result}")
        return response
        
    except Exception as e:
        error_detail = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(f"❌ Prediction error: {error_detail}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ===========================================================================================
# ENDPOINT 2: RISK SCREENING (17 FEATURES)
# ===========================================================================================

@app.post("/predict-risk")
async def predict_risk(data: RiskFactorsInput):
    """Risk Screening dengan 17 features (F1-F17 saja)"""
    if not MODEL_RISK_LOADED:
        raise HTTPException(status_code=503, detail="Model Risk not loaded")
    
    try:
        data_dict = data.model_dump()
        print(f"\n🎯 Risk Model Prediction Request")
        
        # Prepare features
        X = prepare_risk_features(data_dict)
        print(f"   X shape: {X.shape}")
        print(f"   Expected: (1, 17)")
        
        # Predict
        prediction = model_risk.predict(X)[0]
        
        # Untuk FR model, mapping: 0 = Tidak Suspect, 1 = Suspect
        if prediction == 0:
            result = "Tidak Suspect"
        else:
            result = "Suspect"
        
        print(f"   Raw prediction: {prediction}")
        print(f"   Result: {result}")
        
        # Get probabilities
        if hasattr(model_risk, 'predict_proba'):
            proba = model_risk.predict_proba(X)[0]
        else:
            proba = [1.0, 0.0] if result == "Tidak Suspect" else [0.0, 1.0]
        
        # Create probabilities dictionary
        probabilities = {
            "Tidak Suspect": float(proba[0]),
            "Suspect": float(proba[1])
        }
        
        print(f"   Probabilities: {probabilities}")
        
        # Count risk factors
        risk_count = sum([convert_to_binary(data_dict.get(f, 0)) for f in features_risk])
        
        # Risk assessment
        if result == "Tidak Suspect":
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
        
        response = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "model_type": "Risk Screening (17 features)",
            "result": result,
            "risk_level": risk_level,
            "color_code": color,
            "probabilities": probabilities,
            "risk_factors_count": f"{risk_count}/{len(features_risk)}",
            "recommendation": recommendation,
            "next_step": next_step,
            "note": "Ini risk screening awal. Untuk diagnosis lebih lanjut, lengkapi dengan pemeriksaan fisik (P1-P11)."
        }
        
        print(f"✅ Prediction successful: {result}")
        return response
    
    except Exception as e:
        error_detail = f"Error: {str(e)}\nTraceback: {traceback.format_exc()}"
        print(f"❌ Prediction error: {error_detail}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# ===========================================================================================
# ENDPOINT 3: SMART PREDICT (AUTO-DETECT) - DIPERBAIKI
# ===========================================================================================

@app.post("/predict")
async def predict_smart(data: dict):
    """
    Smart Predict: Auto-detect apakah data lengkap atau hanya F1-F17
    PERBAIKAN: Tangani Ket dengan benar
    """
    
    print(f"\n🤖 Smart Predict Request")
    print(f"   Received keys: {list(data.keys())}")
    
    # Check if P1-P11 present
    has_p_features = any(f"P{i}" in data for i in range(1, 12))
    
    if has_p_features:
        print(f"   Has P features: YES → Route to Full Model")
        # Route ke Full Model
        if not MODEL_FULL_LOADED:
            raise HTTPException(status_code=503, detail="Model Full not loaded")
        
        # Buang Ket jika ada, karena model Full tidak butuh Ket
        if 'Ket' in data:
            print(f"   Removing Ket from data (not used in Full Model)")
            data_without_ket = {k: v for k, v in data.items() if k != 'Ket'}
        else:
            data_without_ket = data
        
        try:
            patient_data = PatientDataFull(**data_without_ket)
            return await predict_full(patient_data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid data format: {str(e)}")
    else:
        print(f"   Has P features: NO → Route to Risk Model")
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
    print("   GET  /                → Homepage")
    print("   GET  /health          → Health check")
    print("   POST /predict-full    → Full diagnosis (28 features: F1-F17 + P1-P11)")
    print("   POST /predict-risk    → Risk screening (17 features: F1-F17)")
    print("   POST /predict         → Smart auto-detect")
    print("=" * 80)
    print("\n⚠️  IMPORTANT:")
    print("   Full Model: 28 features (F1-F17 + P1-P11), NO Ket")
    print("   Risk Model: 17 features (F1-F17)")
    print("\n💡 Tekan CTRL+C untuk stop\n")
    
    try:
        uvicorn.run(app, host="127.0.0.1", port=8001, reload=True)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        print(traceback.format_exc())