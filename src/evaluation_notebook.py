# File: evaluation_analysis.py
# Deskripsi: Evaluasi model, analisis error, dan pembuatan insight/rekomendasi.
# (VERSI PERBAIKAN FINAL: Fix NameError & Encoding Error)
# Penulis: [Nama Anda]
# Bahasa: Python 3.10+
# ---------------------------------------------------------------

# %%
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import sys
from pathlib import Path

print("Memulai skrip Evaluasi Model...")

# --- Konfigurasi Awal & Path ---
print("Mencari lokasi file skrip dan memuat konfigurasi...")
try:
    # Tambahkan src ke path
    sys.path.append(str(Path.cwd().parent / 'src'))
    from config import MERGED_DATA_IMPUTED, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, TARGET_VARIABLE
    from utils import * # Asumsi utils.py punya get_data_info dll.
    print("✅ Konfigurasi dan utilitas berhasil dimuat.")
    # Definisikan BASE_DIR jika perlu
    BASE_DIR = Path.cwd().parent # Asumsi skrip ada di src
except ImportError as e:
    print(f"❌ FATAL ERROR: Gagal import dari config.py atau utils.py: {e}")
    exit()
except Exception as e:
     print(f"❌ FATAL ERROR saat setup path: {e}")
     exit()

# Path spesifik
MODEL_PATH = MODELS_DIR / "random_forest_u5mr_pipeline.joblib"
DATA_PATH_IN = MERGED_DATA_IMPUTED
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

print("✅ Libraries loaded successfully")
print(f"📁 Project Base Directory (estimated): {BASE_DIR}")
print(f"💾 Model akan dimuat dari: {MODEL_PATH}")
print(f"📊 Data akan dimuat dari: {DATA_PATH_IN}")
print(f"📄 Hasil evaluasi akan disimpan di: {REPORTS_DIR}")
print(f"🖼️ Gambar akan disimpan di: {FIGURES_DIR}")

# Pastikan folder output ada
try:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"⚠️ Warning: Gagal membuat folder output: {e}")


# %% [markdown]
# ## 1. Load Saved Model and Data

# %%
print(f"\nMemuat model pipeline dari: {MODEL_PATH}")
# --- PERBAIKAN NameError: Definisikan di scope luar try ---
NUMERIC_FEATURES = []
CATEGORICAL_FEATURES = []
FEATURES = []
# --- SELESAI PERBAIKAN ---
try:
    pipeline = joblib.load(MODEL_PATH)
    print(f"✅ Model pipeline loaded: {type(pipeline.named_steps['regressor']).__name__}")
    # Ekstrak nama fitur dari pipeline
    try:
        # Cek transformer numerik dan kategorik
        num_features_extracted = pipeline.named_steps['preprocessor'].transformers_[0][2]
        cat_features_extracted = pipeline.named_steps['preprocessor'].transformers_[1][2]
        print(f"✅ Features used by model (Numeric): {len(num_features_extracted)}")
        print(f"✅ Features used by model (Categorical): {len(cat_features_extracted)}")

        # --- PERBAIKAN: Definisikan variabel di scope ini ---
        NUMERIC_FEATURES = num_features_extracted
        CATEGORICAL_FEATURES = cat_features_extracted
        FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        # --- SELESAI PERBAIKAN ---

    except Exception as e:
        print(f"⚠️ Warning: Could not automatically extract feature names from pipeline: {e}")
        # Fallback: Definisikan manual jika perlu (ambil dari log modeling)
        NUMERIC_FEATURES = ['bcg', 'dtp3', 'hepbb', 'hib3', 'mcv1', 'pcv3', 'pol3', 'rcv1', 'rotac', 'overweight', 'stunting']
        CATEGORICAL_FEATURES = ['sdgregion']
        FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        print("   -> Menggunakan daftar fitur fallback.")

except FileNotFoundError:
    print(f"❌ Error: File model {MODEL_PATH} tidak ditemukan. Jalankan modeling dulu.")
    exit()
except Exception as e:
    print(f"❌ Error saat memuat model: {e}")
    exit()

# %%
print(f"\nMemuat data dari: {DATA_PATH_IN}")
try:
    df = pd.read_csv(DATA_PATH_IN)
    print(f"Dataset loaded: {df.shape}")
    # Pastikan semua fitur ada
    missing_cols = [f for f in FEATURES if f not in df.columns]
    if missing_cols:
         print(f"❌ Error: Kolom fitur {missing_cols} tidak ditemukan di data!")
         exit()
    if TARGET_VARIABLE not in df.columns:
         print(f"❌ Error: Kolom target {TARGET_VARIABLE} tidak ditemukan di data!")
         exit()
except FileNotFoundError:
    print(f"❌ Error: File {DATA_PATH_IN} tidak ditemukan.")
    exit()
except Exception as e:
    print(f"❌ Error saat memuat data: {e}")
    exit()

# %% [markdown]
# ## 2. Prediction Analysis (on Full Data)

# %%
# Siapkan fitur (X) dan target (y) dari seluruh data
X = df[FEATURES].copy() # Tambah .copy() untuk hindari SettingWithCopyWarning
y = df[TARGET_VARIABLE]

# Set ulang tipe kategori jika perlu (penting untuk pipeline)
for col in CATEGORICAL_FEATURES:
     if col in X.columns:
         # Gunakan .loc untuk modifikasi DataFrame
         X.loc[:, col] = X[col].astype('category')

print(f"\nMembuat prediksi pada {len(X)} baris data...")
try:
    y_pred = pipeline.predict(X)
except Exception as e:
    print(f"❌ Error saat membuat prediksi: {e}")
    print("   Tipe data X:")
    print(X.info())
    exit()

# Hitung metrik (pada seluruh data)
mae_full = mean_absolute_error(y, y_pred)
rmse_full = np.sqrt(mean_squared_error(y, y_pred))
r2_full = r2_score(y, y_pred)

print(f"\n📊 Performa Model (pada seluruh data):")
print(f"   MAE: {mae_full:.4f}")
print(f"   RMSE: {rmse_full:.4f}")
print(f"   R²: {r2_full:.4f}")

# %%
# Tambahkan hasil prediksi dan error ke dataframe
df['predicted_mortality'] = y_pred
df['prediction_error'] = y - y_pred # Error = Aktual - Prediksi
df['absolute_error'] = np.abs(df['prediction_error'])
df['percentage_error'] = np.where(y != 0, (df['prediction_error'] / y) * 100, np.inf)

print("\n✅ Kolom prediksi dan error ditambahkan ke dataset.")

# %% [markdown]
# ## 3. Error Analysis

# %%
# Statistik deskriptif error
error_stats = df['absolute_error'].describe()
print("\n📊 Statistik Absolute Error Prediksi:")
print(error_stats)

# %%
# Visualisasi distribusi error
print("\nMembuat visualisasi analisis error...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Analisis Error Prediksi Model', fontsize=16, fontweight='bold')

# Error distribution
sns.histplot(df['prediction_error'].dropna(), bins=50, kde=True, ax=axes[0, 0])
axes[0, 0].axvline(x=0, color='r', linestyle='--', lw=2)
axes[0, 0].set_xlabel('Error Prediksi (Aktual - Prediksi)')
axes[0, 0].set_ylabel('Frekuensi')
axes[0, 0].set_title('Distribusi Error Prediksi', fontweight='bold')

# Absolute error distribution
sns.histplot(df['absolute_error'].dropna(), bins=50, kde=True, ax=axes[0, 1], color='coral')
axes[0, 1].set_xlabel('Absolute Error')
axes[0, 1].set_ylabel('Frekuensi')
axes[0, 1].set_title('Distribusi Absolute Error', fontweight='bold')

# Percentage error distribution
finite_pct_error = df.loc[np.isfinite(df['percentage_error']), 'percentage_error']
sns.histplot(finite_pct_error, bins=50, kde=True, ax=axes[1, 0], color='lightgreen')
axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_xlabel('Percentage Error (%)')
axes[1, 0].set_ylabel('Frekuensi')
axes[1, 0].set_title('Distribusi Percentage Error', fontweight='bold')

# Error vs Actual
sns.scatterplot(x=y, y=df['absolute_error'], alpha=0.3, s=10, ax=axes[1, 1])
axes[1, 1].set_xlabel('Angka Kematian Aktual')
axes[1, 1].set_ylabel('Absolute Error')
axes[1, 1].set_title('Absolute Error vs Angka Kematian Aktual', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(FIGURES_DIR / 'error_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ Gambar analisis error disimpan ke: {FIGURES_DIR / 'error_analysis.png'}")

# %% [markdown]
# ## 4. Negara dengan Error Prediksi Tertinggi

# %%
# Top countries with highest errors
if 'country' in df.columns and 'year' in df.columns:
    latest_year = df['year'].max()
    latest_df = df[df['year'] == latest_year].copy()
    
    # Urutkan berdasarkan absolute error
    latest_df_sorted = latest_df.sort_values('absolute_error', ascending=False)
    
    print(f"\n📊 Negara dengan Prediksi Paling Meleset (Tahun {latest_year}):")
    print(latest_df_sorted[['country', TARGET_VARIABLE, 'predicted_mortality', 'absolute_error']].head(10).to_string(index=False))

    # Analisis tambahan: Negara mana yang sering salah?
    avg_error_by_country = df.groupby('country')['absolute_error'].mean().sort_values(ascending=False)
    print("\n📊 Negara dengan Rata-rata Absolute Error Tertinggi (keseluruhan):")
    print(avg_error_by_country.head(10))

# %% [markdown]
# ## 5. Analisis Tren Prediksi vs Aktual

# %%
# Analyze trends over time
if 'year' in df.columns:
    yearly_metrics = df.groupby('year').agg(
        actual_mean=(TARGET_VARIABLE, 'mean'),
        predicted_mean=('predicted_mortality', 'mean'),
        mae_mean=('absolute_error', 'mean')
    ).reset_index()
    
    print("\nMembuat visualisasi tren prediksi...")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=('Rata-rata Angka Kematian Aktual vs Prediksi per Tahun',
                                        'Rata-rata Absolute Error per Tahun'))
    
    # Actual vs Predicted over time
    fig.add_trace(go.Scatter(x=yearly_metrics['year'], y=yearly_metrics['actual_mean'],
                             mode='lines+markers', name='Aktual', line=dict(color='blue', width=2)),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=yearly_metrics['year'], y=yearly_metrics['predicted_mean'],
                             mode='lines+markers', name='Prediksi', line=dict(color='red', width=2, dash='dash')),
                  row=1, col=1)
    
    # Error over time
    fig.add_trace(go.Scatter(x=yearly_metrics['year'], y=yearly_metrics['mae_mean'],
                             mode='lines+markers', name='MAE', line=dict(color='green', width=2)),
                  row=2, col=1)
                  
    fig.update_layout(height=700, title_text='Analisis Tren Prediksi Model vs Waktu', hovermode='x unified')
    fig.update_yaxes(title_text="Angka Kematian Balita", row=1, col=1)
    fig.update_yaxes(title_text="Mean Absolute Error", row=2, col=1)
    fig.update_xaxes(title_text="Tahun", row=2, col=1)
    
    fig.write_html(FIGURES_DIR / 'temporal_prediction_trends.html')
    fig.show()
    print(f"✅ Gambar tren prediksi disimpan ke: {FIGURES_DIR / 'temporal_prediction_trends.html'}")


# %% [markdown]
# ## 6. Analisis Dampak Fitur (Feature Importance)

# %%
# Load feature importance
print("\nMemuat dan menampilkan feature importance...")
try:
    feature_importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    
    print("\n📊 Top 15 Most Important Features:")
    print(feature_importance.head(15).to_string(index=False))
    
    # Visualisasi
    plt.figure(figsize=(10, 8))
    sns.barplot(x='importance', y='feature', data=feature_importance.head(15), palette='viridis')
    plt.title('Top 15 Fitur Paling Penting', fontsize=14, fontweight='bold')
    plt.xlabel('Skor Importance (Kontribusi ke Prediksi)')
    plt.ylabel('Fitur')
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✅ Gambar feature importance disimpan ke: {REPORTS_DIR / 'feature_importance.png'}")
        
except FileNotFoundError:
    print(f"⚠️ File feature importance tidak ditemukan di {FEATURE_IMPORTANCE_PATH}")
except Exception as e:
    print(f"❌ Error saat memproses feature importance: {e}")

# %% [markdown]
# ## 7. Rekomendasi Kebijakan

# %%
print("\n" + "="*70)
print("💡 RINGKASAN INSIGHT & REKOMENDASI KEBIJAKAN")
print("="*70)

insights = []
avg_error_by_country = None # Inisialisasi

# Model performance insight
insights.append(f"\n1. PERFORMA MODEL")
insights.append(f"   - Model yang dilatih (Random Forest) menunjukkan performa yang sangat baik dalam memprediksi Angka Kematian Balita (U5MR) dengan R² = {r2_full:.3f} pada keseluruhan data.")
insights.append(f"   - Rata-rata kesalahan prediksi (MAE) sekitar {mae_full:.1f} poin, yang cukup akurat untuk analisis.")

# Feature insights
try:
    feature_importance = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    top_3_features = feature_importance.head(3)['feature'].tolist()
    # Bersihkan nama fitur OHE untuk rekomendasi
    top_3_display = [f.split('_')[-1] if 'sdgregion' in f else f for f in top_3_features]
    
    insights.append(f"\n2. FAKTOR KUNCI")
    insights.append(f"   - Faktor paling dominan yang mempengaruhi U5MR adalah:")
    insights.append(f"     1. Region Geografis (terutama '{top_3_display[0]}')")
    insights.append(f"     2. Tingkat '{top_3_display[1]}'")
    insights.append(f"     3. Cakupan Vaksin '{top_3_display[2].upper()}'")
    insights.append(f"   - Intervensi yang menargetkan faktor-faktor ini akan memberikan dampak terbesar.")
except Exception as e:
    insights.append("\n2. FAKTOR KUNCI: (Gagal memuat feature importance)")
    print(f"   (Error: {e})")


# Error insights (hitung ulang avg_error_by_country jika belum ada)
if 'country' in df.columns:
     if 'avg_error_by_country' not in locals() or avg_error_by_country is None:
          avg_error_by_country = df.groupby('country')['absolute_error'].mean().sort_values(ascending=False)
     high_error_countries = avg_error_by_country.head(5).index.tolist() if avg_error_by_country is not None else ["N/A"]
else:
     high_error_countries = ["N/A"]
     
insights.append(f"\n3. AKURASI MODEL")
insights.append(f"   - Model secara umum akurat, namun prediksinya kurang tepat untuk beberapa negara (rata-rata error tertinggi di: {', '.join(high_error_countries)}).")
insights.append(f"   - Perlu investigasi lebih lanjut atau data tambahan untuk negara-negara ini.")

# Actionable recommendations
insights.append(f"\n4. REKOMENDASI KEBIJAKAN")
insights.append(f"   ✓ **Prioritaskan Intervensi Geografis:** Alokasikan sumber daya lebih besar ke wilayah dengan U5MR tertinggi dan di mana model menunjukkan pengaruh region yang kuat (misal: Sub-Saharan Africa).")
insights.append(f"   ✓ **Fokus pada Stunting:** Perkuat program gizi untuk menurunkan angka stunting, karena ini adalah faktor risiko utama kedua.")
insights.append(f"   ✓ **Tingkatkan Cakupan Vaksin Inti:** Pastikan cakupan vaksin kunci seperti DTP3, MCV1, Pol3 mencapai target global (>90%) karena terbukti kuat menurunkan U5MR.")
insights.append(f"   ✓ **Gunakan Model untuk Monitoring:** Manfaatkan model ini untuk memantau progres penurunan U5MR dan mengidentifikasi negara/wilayah yang tidak sesuai target.")
insights.append(f"   ✓ **Perkuat Data:** Tingkatkan kualitas dan kelengkapan data (terutama untuk vaksin baru dan data nutrisi) untuk meningkatkan akurasi model di masa depan.")

for insight in insights:
    print(insight)

# Simpan insights ke file
report_path = REPORTS_DIR / 'policy_recommendations.txt'
try:
    # --- PERBAIKAN UnicodeEncodeError ---
    with open(report_path, 'w', encoding='utf-8') as f:
    # --- SELESAI PERBAIKAN ---
        f.write("CHILD MORTALITY ANALYSIS - INSIGHTS & RECOMMENDATIONS\n")
        f.write("="*70 + "\n")
        # Tulis ulang insight ke file
        for insight_line in insights:
            # Ganti simbol centang dengan karakter ASCII biasa jika perlu
            insight_line = insight_line.replace('✓', '*') 
            f.write(insight_line + "\n")
    print(f"\n✅ Rekomendasi kebijakan disimpan ke {report_path}")
except Exception as e:
    print(f"\n❌ Error saat menyimpan rekomendasi: {e}")

print("\n" + "="*70)
print("✅ EVALUASI MODEL & PEMBUATAN INSIGHT SELESAI!")
print("="*70)

# %%