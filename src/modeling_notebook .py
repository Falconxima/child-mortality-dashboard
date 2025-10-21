# %% [markdown]
# # 📈 Modeling - Child Mortality Prediction
# 
# **Author:** Raihan Aprilialdy Risanto  
# **Institution:** Universitas Negeri Jakarta  
# **Date:** 2024
# 
# ## Objectives
# 1. Preprocess data for modeling (imputation, scaling, encoding)
# 2. Train a regression model (Random Forest) to predict Under-5 Mortality Rate
# 3. Evaluate model performance
# 4. Analyze feature importance
# 5. Save the trained model

# Import libraries
import pandas as pd
import numpy as np
import os
import joblib
import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- PERBAIKAN IMPORT: Langsung tanpa try-except ---
# Tambahkan src ke path
try:
    sys.path.append(str(Path.cwd().parent / 'src'))
    print("✅ Added src directory to sys.path")
except Exception as e:
    print(f"⚠️ Warning: Could not add src to path: {e}")

# Impor langsung dari config dan utils
try:
    from config import MERGED_DATA_IMPUTED, MODELS_DIR, REPORTS_DIR, TARGET_VARIABLE 
    from utils import * 
    print("✅ Configuration and utilities imported successfully")
except ImportError as e:
    print(f"❌ FATAL ERROR: Could not import from config.py or utils.py: {e}")
    print("   Pastikan file config.py dan utils.py ada di folder src.")
    exit()
# --- SELESAI PERBAIKAN IMPORT ---

# --- DEBUG PRINT (Boleh dihapus nanti) ---
print(f"DEBUG: MERGED_DATA_IMPUTED path = {MERGED_DATA_IMPUTED}") 

print("✅ Libraries loaded successfully")

# Konfigurasi Target sudah diimpor
# TARGET_VARIABLE = 'under_five_mortality_rate' 

# Definisikan BASE_DIR jika diperlukan (meski path sudah dari config)
BASE_DIR = Path.cwd().parent # Asumsi notebook/script ada di src atau notebooks

print(f"📁 Project Base Directory (estimated): {BASE_DIR}")
print(f"💾 Model output directory: {MODELS_DIR}") 
print(f"📄 Summary output directory: {REPORTS_DIR}")

# Pastikan folder output ada (pindahkan ke sini)
try:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"⚠️ Warning: Could not create output directories: {e}")

# %% [markdown]
# ## 1. Load Data

# %% [markdown]
# ## 1. Load Data

# %%
# Load the imputed dataset
print(f"\nMencoba memuat data dari: {MERGED_DATA_IMPUTED}")
try:
    # --- PERBAIKAN: Pindahkan pemuatan data KE LUAR try ---
    df = pd.read_csv(MERGED_DATA_IMPUTED)
    print(f"Dataset berhasil dimuat!")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())

except FileNotFoundError: # <-- Jaga-jaga jika file dihapus manual
    print(f"❌ FATAL ERROR: File {MERGED_DATA_IMPUTED} tidak ditemukan. Jalankan dataprep.py lagi.")
    exit()
except NameError: # <-- Ini seharusnya tidak terjadi lagi
    print(f"❌ FATAL ERROR: Variabel MERGED_DATA_IMPUTED tidak ditemukan. Cek import!")
    exit()
except Exception as e: # Tangkap error lain saat loading
    print(f"❌ FATAL ERROR saat load data: {e}")
    exit()

# %%
# Basic information (pindahkan ini agar tetap jalan jika load sukses)
try:
    get_data_info(df, "Child Mortality Dataset Overview")
except NameError:
     print("DEBUG: Fungsi get_data_info tidak ditemukan (cek utils.py)")
except Exception as e:
     print(f"Error di get_data_info: {e}")

# %% [markdown]
# ## 2. Define Features and Target

# %%
# --- PERBAIKAN: Definisikan fitur secara eksplisit ---
# Buang kolom ID/metadata dan kolom target
IDENTIFIER_COLS = ['isocode', 'country', 'sdgsubregion', 'unicefregion', 
                   'unicefprogrammeregion', 'uncertaintybounds', 'year']

# Ambil semua kolom numerik yang tersisa sebagai fitur
NUMERIC_FEATURES = df.select_dtypes(include=np.number).columns.tolist()
NUMERIC_FEATURES = [col for col in NUMERIC_FEATURES if col != TARGET_VARIABLE and col != 'year']

# --- PERBAIKAN: Gunakan 'sdgregion', BUKAN 'country' ---
CATEGORICAL_FEATURES = ['sdgregion'] 

# Pastikan fitur ada di dataframe
NUMERIC_FEATURES = [col for col in NUMERIC_FEATURES if col in df.columns]
CATEGORICAL_FEATURES = [col for col in CATEGORICAL_FEATURES if col in df.columns]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Pastikan data target ada
if TARGET_VARIABLE not in df.columns:
    print(f"❌ Error: Target variable '{TARGET_VARIABLE}' not found in the dataset!")
    exit()

# Ambil data yang relevan
df_model = df[FEATURES + [TARGET_VARIABLE]].copy()

# Set tipe kategori
for col in CATEGORICAL_FEATURES:
     df_model[col] = df_model[col].astype('category')

print(f"\n🎯 Target Variable (y): {TARGET_VARIABLE}")
print(f"🔢 Numeric Features (X): {NUMERIC_FEATURES}")
print(f"🔠 Categorical Features (X): {CATEGORICAL_FEATURES}")

# Hapus baris di mana TARGET kosong (seharusnya sudah bersih dari dataprep)
df_model.dropna(subset=[TARGET_VARIABLE], inplace=True)
print(f"\nData shape after dropping rows with missing target: {df_model.shape}")

# %% [markdown]
# ## 3. Preprocessing Pipeline
# Menyiapkan data untuk model:
# - Mengisi nilai kosong (imputation)
# - Menyesuaikan skala angka (scaling)
# - Mengubah kategori menjadi angka (one-hot encoding)

# %%
print("\nBuilding preprocessing pipeline...")

# Pipeline untuk data numerik
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')), # Isi NaN sisa dengan rata-rata
    ('scaler', StandardScaler()) # Skalakan data
])

# Pipeline untuk data kategorikal
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')), # Isi NaN kategori
    ('onehot', OneHotEncoder(handle_unknown='ignore')) # Ubah jadi biner
])

# Gabungkan pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, NUMERIC_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES)
    ],
    remainder='passthrough' # Biarkan kolom lain jika ada
)

print("✅ Preprocessing pipeline created.")

# %% [markdown]
# ## 4. Split Data (Train/Test)
# Membagi data menjadi bagian Latihan (untuk melatih model) dan bagian Tes (untuk menguji performa model). Kita gunakan pembagian acak karena data kita tidak berurutan sempurna.

# %%
print("\nSplitting data into Training and Testing sets (Random Split)...")

X = df_model[FEATURES]
y = df_model[TARGET_VARIABLE]

# --- PERBAIKAN: Gunakan Random Split (train_test_split) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 

print(f"Data latih: {X_train.shape[0]} samples")
print(f"Data tes:   {X_test.shape[0]} samples")

# %% [markdown]
# ## 5. Train Model
# Melatih model `RandomForestRegressor` menggunakan data latihan.

# %%
print("\nTraining RandomForestRegressor model...")

# Definisikan model (parameter bisa di-tuning lebih lanjut)
model = RandomForestRegressor(n_estimators=100, # Jumlah 'pohon'
                              random_state=42, 
                              n_jobs=-1, # Gunakan semua core CPU
                              max_depth=15, # Batasi kedalaman pohon
                              min_samples_split=5, # Minimal sampel untuk membelah node
                              min_samples_leaf=3, # Minimal sampel di daun
                              oob_score=True) # Gunakan Out-of-Bag score untuk validasi cepat

# Buat pipeline lengkap: Preprocessing + Model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', model)
])

# Latih pipeline!
pipeline.fit(X_train, y_train)

print("✅ Model training complete.")
# Tampilkan OOB Score (estimasi R^2 pada data yang tidak terlihat saat training)
if hasattr(pipeline.named_steps['regressor'], 'oob_score_'):
    print(f"   Out-of-Bag (OOB) R^2 Score Estimate: {pipeline.named_steps['regressor'].oob_score_:.4f}")


# %% [markdown]
# ## 6. Evaluate Model
# Mengukur seberapa baik model kita bekerja pada data tes yang belum pernah dilihat sebelumnya.

# %%
print("\nEvaluating model performance on the test set...")
y_pred = pipeline.predict(X_test)

# Hitung metrik evaluasi
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n--- Evaluation Results ---")
print(f"R-squared (R2):     {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print("------------------------")
print("Notes:")
print("  R-squared (R²): Measures how much variance the model explains (0 to 1, higher is better).")
print("  MAE: Average absolute difference between prediction and actual value (lower is better).")
print("  RMSE: Root of the average squared difference (lower is better, sensitive to large errors).")

# Simpan hasil evaluasi
summary_path = REPORTS_DIR / "model_evaluation_metrics.txt"
try:
    with open(summary_path, 'w') as f:
        f.write("--- Evaluation Results: RandomForestRegressor ---\n")
        f.write(f"Target Variable: {TARGET_VARIABLE}\n")
        f.write(f"Data Split: Random 80% Train / 20% Test\n")
        f.write(f"Features Used: {len(FEATURES)}\n")
        f.write("-----------------------------------------------\n")
        f.write(f"R-squared (R2):     {r2:.4f}\n")
        f.write(f"Mean Absolute Error (MAE): {mae:.4f}\n")
        f.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}\n")
    print(f"\n✅ Evaluation metrics saved to: {summary_path}")
except Exception as e:
    print(f"\n❌ Error saving evaluation metrics: {e}")

# %% [markdown]
# ## 7. Feature Importance
# Melihat fitur mana yang dianggap paling penting oleh model dalam membuat prediksi.

# %%
print("\nAnalyzing feature importance...")

try:
    # Ambil nama fitur setelah preprocessing
    ohe_categories = pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].categories_
    ohe_feature_names = pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(CATEGORICAL_FEATURES)
    
    # Gabungkan nama fitur numerik dan kategorikal (setelah OHE)
    all_feature_names = NUMERIC_FEATURES + list(ohe_feature_names)
    
    # Ambil nilai importance dari model
    importances = pipeline.named_steps['regressor'].feature_importances_
    
    # Pastikan panjangnya cocok (kadang bisa error jika ada fitur yang hilang)
    if len(all_feature_names) != len(importances):
         print(f"⚠️ Warning: Length mismatch between feature names ({len(all_feature_names)}) and importances ({len(importances)}). Skipping detailed importance.")
         # Coba tampilkan importance mentah jika memungkinkan
         raw_importances = pd.Series(importances).sort_values(ascending=False)
         print("\nRaw feature importances (top 10):")
         print(raw_importances.head(10))
         
    else:
        # Buat DataFrame
        feat_importance_df = pd.DataFrame({
            'feature': all_feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feat_importance_df.head(10))
        
        # Simpan ke CSV
        importance_path = REPORTS_DIR / "feature_importance.csv"
        feat_importance_df.to_csv(importance_path, index=False)
        print(f"\n✅ Feature importance saved to: {importance_path}")

        # Visualisasi Feature Importance
        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feat_importance_df.head(15), palette='viridis')
        plt.title('Top 15 Most Important Features', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(REPORTS_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

except Exception as e:
    print(f"❌ Error calculating or visualizing feature importance: {e}")
    # Print raw importances if available
    if 'importances' in locals():
         print("\nRaw feature importances:")
         print(pd.Series(importances).sort_values(ascending=False))


# %% [markdown]
# ## 8. Save Model
# Menyimpan pipeline model yang sudah dilatih agar bisa digunakan lagi nanti tanpa perlu training ulang.

# %%
print("\nSaving the trained model pipeline...")
model_path = MODELS_DIR / "random_forest_u5mr_pipeline.joblib"
try:
    joblib.dump(pipeline, model_path)
    print(f"✅ Model pipeline saved successfully to: {model_path}")
except Exception as e:
    print(f"❌ Error saving model: {e}")

# %% [markdown]
# ---
# ## ✅ Modeling Complete!
# 
# **Next Steps:**
# 1. Analyze the evaluation metrics (R², MAE, RMSE).
# 2. Interpret the feature importances.
# 3. Use the saved model (`.joblib`) for predictions or simulations (like in `intervention_simulation.py`).
# 4. Consider further model tuning (hyperparameter optimization) or trying different algorithms if performance needs improvement.

# %%
print("\n✅ Modeling script finished.")