import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("MODEL NAIVE BAYES PRODUCTION - DENGAN FEATURE 'KET'")
print("29 FEATURES: F1-F17 + P1-P11 + Ket")
print("=" * 100)

# ===========================================================================================
# STEP 0: LOAD DATA
# ===========================================================================================
print("\n" + "=" * 100)
print("STEP 0: LOAD DATA DARI EXCEL")
print("=" * 100)

file_path = "sample.xlsx"
df = pd.read_excel(file_path, sheet_name='Semua Data', skiprows=1)

df_valid = df[df['No'].notna()].copy()
df_valid = df_valid[pd.to_numeric(df_valid['No'], errors='coerce').notna()].copy()

print(f"\n✅ Data berhasil dimuat dari: {file_path}")
print(f"✅ Total data pasien: {len(df_valid)} pasien")

# ===========================================================================================
# STEP 1: PREPROCESSING (WITH "KET" FEATURE)
# ===========================================================================================
print("\n" + "=" * 100)
print("STEP 1: PREPROCESSING DATA (WITH 'KET')")
print("=" * 100)

# Pilih features F1-F17 dan P1-P11
feature_cols_base = [col for col in df_valid.columns if col.startswith('F') or col.startswith('P')]
feature_cols_base = [col for col in feature_cols_base if len(col) <= 3]

print(f"\n📊 Base features: {len(feature_cols_base)}")
print(f"   Faktor Risiko (F): {len([c for c in feature_cols_base if c.startswith('F')])}")
print(f"   Parameter Pemeriksaan (P): {len([c for c in feature_cols_base if c.startswith('P')])}")

# Encode base features
X_base = df_valid[feature_cols_base].copy().fillna('Tidak')
X_encoded = X_base.copy()
for col in X_encoded.columns:
    X_encoded[col] = X_encoded[col].map({'Ya': 1, 'Tidak': 0}).fillna(0)

# ADD "KET" FEATURE
print(f"\n🔧 Adding 'Ket' feature...")

# Check if "Ket." or "Ket" column exists
ket_col_name = None
for possible_name in ['Ket.', 'Ket', 'Keterangan', 'ket']:
    if possible_name in df_valid.columns:
        ket_col_name = possible_name
        break

if ket_col_name:
    print(f"   Found column: '{ket_col_name}'")
    
    # Encode Ket: 1 if filled, 0 if empty/"-"
    def encode_ket(value):
        if pd.isna(value):
            return 0  # Empty → 0
        if isinstance(value, str):
            value_clean = value.strip()
            if value_clean == '' or value_clean == '-':
                return 0  # Empty or "-" → 0
            else:
                return 1  # Has content → 1
        return 0
    
    X_encoded['Ket'] = df_valid[ket_col_name].apply(encode_ket)
    
    # Count distribution
    ket_counts = X_encoded['Ket'].value_counts()
    print(f"   Ket = 0 (Tidak ada isi): {ket_counts.get(0, 0)} pasien")
    print(f"   Ket = 1 (Ada isi): {ket_counts.get(1, 0)} pasien")
else:
    print(f"   ⚠️  WARNING: Column 'Ket' not found in data!")
    print(f"   Available columns: {list(df_valid.columns)}")
    print(f"   Continuing without 'Ket' feature...")

# Final feature list
feature_cols = list(X_encoded.columns)
print(f"\n✅ Total features: {len(feature_cols)}")
if 'Ket' in feature_cols:
    print(f"   ✅ Feature 'Ket' added successfully!")

# Target
target_col = 'Hasil Pemeriksaan'
y = df_valid[target_col].copy()

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\n📊 Label mapping:")
for label, encoded in zip(le.classes_, le.transform(le.classes_)):
    print(f"   {encoded}: {label}")

print(f"\n📈 Distribusi kelas:")
unique, counts = np.unique(y_encoded, return_counts=True)
for label_idx, count in zip(unique, counts):
    label_name = le.inverse_transform([label_idx])[0]
    print(f"   {label_name}: {count} sampel ({count/len(y_encoded)*100:.1f}%)")

# ===========================================================================================
# STEP 2: SPLIT DATA
# ===========================================================================================
print("\n" + "=" * 100)
print("STEP 2: SPLIT DATA (80% Training, 20% Testing)")
print("=" * 100)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_encoded
)

print(f"\n📊 Training: {X_train.shape[0]} samples")
print(f"📊 Testing: {X_test.shape[0]} samples")

# ===========================================================================================
# STEP 3: APPLY SMOTE
# ===========================================================================================
print("\n" + "=" * 100)
print("STEP 3: APPLY SMOTE")
print("=" * 100)

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"\n✅ SMOTE berhasil diterapkan!")
print(f"   SEBELUM SMOTE: {X_train.shape[0]} sampel")
print(f"   SESUDAH SMOTE: {X_train_smote.shape[0]} sampel")
print(f"   Penambahan: {X_train_smote.shape[0] - X_train.shape[0]} synthetic samples")

# ===========================================================================================
# STEP 4: TRAINING MODEL
# ===========================================================================================
print("\n" + "=" * 100)
print("STEP 4: TRAINING MODEL NAIVE BAYES")
print("=" * 100)

model = BernoulliNB()

print(f"\n🤖 Model: Bernoulli Naive Bayes")
print(f"   Features: {len(feature_cols)} (F1-F17 + P1-P11 + Ket)")

print(f"\n⏳ Training model...")
model.fit(X_train_smote, y_train_smote)
print(f"✅ Training selesai!")

# ===========================================================================================
# STEP 5: EVALUASI MODEL
# ===========================================================================================
print("\n" + "=" * 100)
print("STEP 5: EVALUASI MODEL")
print("=" * 100)

# Accuracies
y_train_pred = model.predict(X_train_smote)
train_accuracy = accuracy_score(y_train_smote, y_train_pred)

y_test_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"\n📈 AKURASI MODEL:")
print(f"   Training Accuracy: {train_accuracy*100:.2f}%")
print(f"   Testing Accuracy: {test_accuracy*100:.2f}%")
print(f"   Gap: {abs(train_accuracy - test_accuracy)*100:.2f}%")

# Classification Report
print(f"\n📊 CLASSIFICATION REPORT (Testing Set):\n")
print(classification_report(y_test, y_test_pred, target_names=le.classes_, digits=4))

# Confusion Matrix
print(f"\n🔍 CONFUSION MATRIX (Testing Set):\n")
cm = confusion_matrix(y_test, y_test_pred)
cm_df = pd.DataFrame(cm, 
                     index=[f"True: {c}" for c in le.classes_], 
                     columns=[f"Pred: {c}" for c in le.classes_])
print(cm_df)

# Ganas class metrics
ganas_idx = list(le.classes_).index('Suspect Kelainan Ganas')
tp_ganas = cm[ganas_idx, ganas_idx]
fn_ganas = cm[ganas_idx, :].sum() - tp_ganas
fp_ganas = cm[:, ganas_idx].sum() - tp_ganas
tn_ganas = cm.sum() - tp_ganas - fn_ganas - fp_ganas

recall_ganas = tp_ganas / (tp_ganas + fn_ganas) if (tp_ganas + fn_ganas) > 0 else 0
precision_ganas = tp_ganas / (tp_ganas + fp_ganas) if (tp_ganas + fp_ganas) > 0 else 0
f1_ganas = 2 * (precision_ganas * recall_ganas) / (precision_ganas + recall_ganas) if (precision_ganas + recall_ganas) > 0 else 0

print(f"\n⚠️  FOKUS ANALISIS: KELAS GANAS")
print(f"   Recall: {recall_ganas*100:.2f}%")
print(f"   Precision: {precision_ganas*100:.2f}%")
print(f"   F1-Score: {f1_ganas:.4f}")

# ===========================================================================================
# STEP 6: CROSS-VALIDATION
# ===========================================================================================
print("\n" + "=" * 100)
print("STEP 6: CROSS-VALIDATION (5-FOLD)")
print("=" * 100)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(BernoulliNB(), X_encoded, y_encoded, cv=skf, scoring='accuracy')

print(f"\n📊 Cross-Validation:")
print(f"   Scores: {[f'{s*100:.2f}%' for s in cv_scores]}")
print(f"   Mean: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ===========================================================================================
# STEP 7: COMPARE WITH/WITHOUT KET
# ===========================================================================================
if 'Ket' in feature_cols:
    print("\n" + "=" * 100)
    print("STEP 7: COMPARE WITH/WITHOUT 'KET' FEATURE")
    print("=" * 100)
    
    # Train without Ket
    feature_cols_no_ket = [f for f in feature_cols if f != 'Ket']
    X_train_no_ket = X_train[feature_cols_no_ket]
    X_test_no_ket = X_test[feature_cols_no_ket]
    
    X_train_no_ket_smote, _ = smote.fit_resample(X_train_no_ket, y_train)
    
    model_no_ket = BernoulliNB()
    model_no_ket.fit(X_train_no_ket_smote, y_train_smote)
    
    y_pred_no_ket = model_no_ket.predict(X_test_no_ket)
    accuracy_no_ket = accuracy_score(y_test, y_pred_no_ket)
    
    print(f"\n📊 COMPARISON:")
    print(f"   WITHOUT Ket (28 features): {accuracy_no_ket*100:.2f}%")
    print(f"   WITH Ket (29 features):    {test_accuracy*100:.2f}%")
    print(f"   Difference: {(test_accuracy - accuracy_no_ket)*100:+.2f}%")
    
    if test_accuracy > accuracy_no_ket:
        print(f"\n   ✅ 'Ket' feature improves accuracy!")
    elif test_accuracy < accuracy_no_ket:
        print(f"\n   ⚠️  'Ket' feature decreases accuracy")
    else:
        print(f"\n   → No significant difference")

# ===========================================================================================
# STEP 8: SAVE MODEL
# ===========================================================================================
print("\n" + "=" * 100)
print("STEP 8: SAVE MODEL WITH 'KET' FEATURE")
print("=" * 100)

model_data = {
    'model': model,
    'label_encoder': le,
    'feature_columns': feature_cols,
    'has_ket_feature': 'Ket' in feature_cols,
    'threshold': 0.5,
    'training_accuracy': train_accuracy,
    'testing_accuracy': test_accuracy,
    'recall_ganas': recall_ganas,
    'precision_ganas': precision_ganas,
    'f1_ganas': f1_ganas,
    'confusion_matrix': cm,
    'cv_mean': cv_scores.mean(),
    'cv_std': cv_scores.std(),
    'used_smote': True,
    'date_created': '2025-10-13',
    'version': '2.0_with_ket'
}

model_filename = 'model_production_with_ket.pkl'
with open(model_filename, 'wb') as f:
    pickle.dump(model_data, f)

print(f"\n✅ Model saved: {model_filename}")
print(f"\n📊 MODEL SUMMARY:")
print(f"   Features: {len(feature_cols)} (F1-F17 + P1-P11 + Ket)")
print(f"   Has 'Ket': {'✅ Yes' if 'Ket' in feature_cols else '❌ No'}")
print(f"   Testing Accuracy: {test_accuracy*100:.2f}%")
print(f"   Recall Ganas: {recall_ganas*100:.2f}%")
print(f"   CV Mean: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ===========================================================================================
# EXAMPLE USAGE
# ===========================================================================================
print("\n" + "=" * 100)
print("CONTOH PENGGUNAAN MODEL")
print("=" * 100)

print(f"""
# Load model
import pickle
with open('{model_filename}', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
le = model_data['label_encoder']
feature_cols = model_data['feature_columns']

# Input data pasien (29 features)
data_pasien = {{
    'F1': 'Ya', 'F2': 'Tidak', ..., 'F17': 'Tidak',  # 17 features
    'P1': 'Ya', 'P2': 'Ya', ..., 'P11': 'Ya',        # 11 features
    'Ket': 1  # 0 jika tidak ada keterangan, 1 jika ada isi
}}

# Encode
data_encoded = []
for feat in feature_cols:
    if feat == 'Ket':
        data_encoded.append(data_pasien.get('Ket', 0))  # Integer langsung
    else:
        value = data_pasien.get(feat, 'Tidak')
        data_encoded.append(1 if value == 'Ya' else 0)

X_new = np.array(data_encoded).reshape(1, -1)

# Predict
hasil = model.predict(X_new)[0]
proba = model.predict_proba(X_new)[0]

print(f"Hasil: {{le.inverse_transform([hasil])[0]}}")
print(f"Probabilitas: {{dict(zip(le.classes_, proba))}}")
""")

print("\n" + "=" * 100)
print("SELESAI!")
print("=" * 100)
