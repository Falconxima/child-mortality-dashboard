# File: intervention_simulation.py
# Deskripsi: Menjalankan simulasi intervensi (scenario analysis)
#            menggunakan model yang sudah dilatih.
# Penulis: [Nama Anda]
# Bahasa: Python 3.10+
# ---------------------------------------------------------------

import pandas as pd
import numpy as np
import os
import joblib
import sys
from pathlib import Path

print("Memulai skrip Simulasi Intervensi...")

# --- Konfigurasi Awal & Path ---
print("Mencari lokasi file skrip dan memuat konfigurasi...")
try:
    # Tambahkan src ke path
    sys.path.append(str(Path.cwd().parent / 'src'))
    from config import MERGED_DATA_IMPUTED, MODELS_DIR, REPORTS_DIR, TARGET_VARIABLE, START_YEAR, END_YEAR
    print("✅ Konfigurasi berhasil dimuat.")
    # Definisikan BASE_DIR jika perlu
    BASE_DIR = Path.cwd().parent # Asumsi skrip ada di src
except ImportError as e:
    print(f"❌ FATAL ERROR: Gagal import dari config.py: {e}")
    print("   Pastikan config.py ada di folder src dan bisa diakses.")
    exit()
except Exception as e:
     print(f"❌ FATAL ERROR saat setup path: {e}")
     exit()

# --- Path I/O ---
DATA_PATH_IN = MERGED_DATA_IMPUTED
MODEL_PATH = MODELS_DIR / "random_forest_u5mr_pipeline.joblib" # Nama file model dari skrip modeling
SUMMARY_DIR = REPORTS_DIR

print(f"📁 Project Base Directory (estimated): {BASE_DIR}")
print(f"💾 Model akan dimuat dari: {MODEL_PATH}")
print(f"📄 Hasil simulasi akan disimpan di: {SUMMARY_DIR}")

# Membuat folder output jika belum ada (meski config sudah)
try:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"⚠️ Warning: Gagal membuat folder output {SUMMARY_DIR}: {e}")

# ---------------------------------------------------------------
# 1️⃣ Load Model dan Data
# ---------------------------------------------------------------
print(f"\nMemuat model dari: {MODEL_PATH}")
try:
    pipeline = joblib.load(MODEL_PATH)
    print("✅ Model berhasil dimuat.")
except FileNotFoundError:
    print(f"❌ Error: File model {MODEL_PATH} tidak ditemukan. Jalankan modeling_notebook .py dulu.")
    exit()
except Exception as e:
    print(f"❌ Error saat memuat model: {e}")
    exit()

print(f"\nMemuat data dari: {DATA_PATH_IN}")
try:
    df = pd.read_csv(DATA_PATH_IN)
    print("✅ Dataset berhasil dimuat.")
except FileNotFoundError:
    print(f"❌ Error: File {DATA_PATH_IN} tidak ditemukan. Jalankan dataprep.py dulu.")
    exit()
except Exception as e:
    print(f"❌ Error saat memuat data: {e}")
    exit()

# ---------------------------------------------------------------
# 2️⃣ Siapkan Data untuk Simulasi
# ---------------------------------------------------------------
# Kita gunakan data beberapa tahun terakhir (misal, 5 tahun)
# untuk melihat dampak intervensi pada kondisi 'saat ini'
SIMULATION_YEAR_START = max(START_YEAR, df['year'].max() - 4) # Ambil 5 tahun terakhir, tapi minimal START_YEAR
df_sim = df[df['year'] >= SIMULATION_YEAR_START].copy()

if df_sim.empty:
    print(f"❌ Error: Tidak ada data untuk simulasi (tahun >= {SIMULATION_YEAR_START}). Cek data input.")
    exit()

# Ambil fitur yang dibutuhkan oleh model
# (Harus SAMA PERSIS dengan yang dipakai di modeling_notebook .py)
# Dari log modeling Anda:
NUMERIC_FEATURES = ['bcg', 'dtp3', 'hepbb', 'hib3', 'mcv1', 'pcv3', 'pol3', 'rcv1', 'rotac', 'overweight', 'stunting']
CATEGORICAL_FEATURES = ['sdgregion']

# Pastikan fitur ada
NUMERIC_FEATURES = [col for col in NUMERIC_FEATURES if col in df.columns]
CATEGORICAL_FEATURES = [col for col in CATEGORICAL_FEATURES if col in df.columns]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Cek apakah fitur penting ada
required_features = ['dtp3', 'stunting', 'sdgregion'] # Fitur penting dari importance
missing_req = [f for f in required_features if f not in df_sim.columns]
if missing_req:
    print(f"❌ Error: Fitur penting ({missing_req}) tidak ditemukan di data untuk simulasi.")
    exit()

# Siapkan data X (fitur)
X_original = df_sim[FEATURES].copy() # Copy untuk modifikasi nanti
y_original_actual = df_sim[TARGET_VARIABLE] # Target asli (opsional, untuk perbandingan)

# Set ulang tipe kategori jika perlu
for col in CATEGORICAL_FEATURES:
     X_original[col] = X_original[col].astype('category')


print(f"\nData untuk simulasi (baseline) disiapkan: {X_original.shape[0]} baris")
print(f"Tahun data simulasi: {df_sim['year'].min()} - {df_sim['year'].max()}")

# ---------------------------------------------------------------
# 3️⃣ Jalankan Prediksi Baseline
# ---------------------------------------------------------------
print("\n--- 3. Menjalankan Prediksi Baseline (Tanpa Intervensi) ---")
try:
    y_pred_baseline = pipeline.predict(X_original)
    u5mr_baseline_avg = np.mean(y_pred_baseline)
    print(f"✅ Prediksi baseline selesai.")
    print(f"   Rata-rata U5MR aktual (data simulasi): {np.mean(y_original_actual):.2f}")
    print(f"   Rata-rata U5MR prediksi (baseline):   {u5mr_baseline_avg:.2f}")
except Exception as e:
    print(f"❌ Error saat prediksi baseline: {e}")
    exit()

# ---------------------------------------------------------------
# 4️⃣ Buat Skenario Intervensi
# ---------------------------------------------------------------
print("\n--- 4. Membuat Skenario Intervensi: Naikkan DTP3 ke 90% ---")

# Target intervensi
TARGET_IMMUNIZATION_RATE = 90.0
INTERVENTION_FEATURE = 'dtp3' # Fokus pada DTP3 (vaksin penting menurut model)

# Salin data asli untuk dimodifikasi
X_scenario = X_original.copy()

# ---- Intervensi ----
if INTERVENTION_FEATURE in X_scenario.columns:
    # 1. Ubah fitur 'dtp3'
    # Jika nilainya di bawah target, naikkan ke target. Jika NaN, biarkan (diurus imputer)
    mask = X_scenario[INTERVENTION_FEATURE] < TARGET_IMMUNIZATION_RATE
    X_scenario.loc[mask, INTERVENTION_FEATURE] = TARGET_IMMUNIZATION_RATE
    changed_count = mask.sum()
    print(f"   {changed_count} baris data mengalami peningkatan {INTERVENTION_FEATURE}.")

    # 2. (Opsional) Sesuaikan fitur turunan jika ada
    # Jika Anda membuat fitur lag/rolling di 'features.py' dan memakainya di model,
    # Anda mungkin perlu menghitung ulang fitur itu di sini berdasarkan nilai 'dtp3' yang baru.
    # Contoh (jika pakai 'dtp3_lag1'):
    # if 'dtp3_lag1' in X_scenario.columns:
    #     mask_lag = X_scenario['dtp3_lag1'] < TARGET_IMMUNIZATION_RATE
    #     X_scenario.loc[mask_lag, 'dtp3_lag1'] = TARGET_IMMUNIZATION_RATE
    #     print(f"   Fitur 'dtp3_lag1' juga disesuaikan.")

else:
    print(f"⚠️ Peringatan: Fitur intervensi '{INTERVENTION_FEATURE}' tidak ditemukan. Skenario sama dengan baseline.")
    changed_count = 0

print(f"✅ Skenario intervensi dibuat.")

# ---------------------------------------------------------------
# 5️⃣ Jalankan Prediksi Skenario
# ---------------------------------------------------------------
print("\n--- 5. Menjalankan Prediksi Skenario (Dengan Intervensi) ---")
try:
    y_pred_scenario = pipeline.predict(X_scenario)
    u5mr_scenario_avg = np.mean(y_pred_scenario)
    print(f"✅ Prediksi skenario selesai.")
    print(f"   Rata-rata U5MR prediksi (skenario): {u5mr_scenario_avg:.2f}")
except Exception as e:
    print(f"❌ Error saat prediksi skenario: {e}")
    exit()

# ---------------------------------------------------------------
# 6️⃣ Analisis Dampak
# ---------------------------------------------------------------
penurunan_u5mr = u5mr_baseline_avg - u5mr_scenario_avg
penurunan_persen = (penurunan_u5mr / u5mr_baseline_avg) * 100 if u5mr_baseline_avg != 0 else 0

print("\n" + "="*60)
print("📊 ANALISIS DAMPAK INTERVENSI")
print("="*60)
print(f"Skenario: Meningkatkan cakupan {INTERVENTION_FEATURE.upper()} menjadi {TARGET_IMMUNIZATION_RATE}%")
print(f"Data Tahun: {df_sim['year'].min()} - {df_sim['year'].max()}")
print("--------------------------------------------------")
print(f"Rata-rata U5MR Prediksi (Baseline):    {u5mr_baseline_avg:.2f}")
print(f"Rata-rata U5MR Prediksi (Skenario):    {u5mr_scenario_avg:.2f}")
print("--------------------------------------------------")
print(f"📉 Estimasi Penurunan U5MR:      {penurunan_u5mr:.2f} poin")
print(f"📉 Estimasi Penurunan Persentase: {penurunan_persen:.1f} %")
print("--------------------------------------------------")
print(f"({changed_count} dari {X_original.shape[0]} data poin terdampak langsung oleh intervensi)")

# Simpan hasil simulasi ke file .txt
summary_filename = f"simulation_summary_{INTERVENTION_FEATURE}_to_{int(TARGET_IMMUNIZATION_RATE)}.txt"
summary_path = SUMMARY_DIR / summary_filename
try:
    with open(summary_path, 'w') as f:
        f.write("--- Hasil Simulasi Intervensi ---\n")
        f.write(f"Model: {MODEL_PATH.name}\n")
        f.write(f"Data: Tahun >= {SIMULATION_YEAR_START} from {DATA_PATH_IN.name}\n")
        f.write(f"Skenario: Meningkatkan cakupan {INTERVENTION_FEATURE.upper()} menjadi {TARGET_IMMUNIZATION_RATE}%\n")
        f.write("--------------------------------------------------\n")
        f.write(f"Rata-rata U5MR Prediksi (Baseline):    {u5mr_baseline_avg:.2f}\n")
        f.write(f"Rata-rata U5MR Prediksi (Skenario):    {u5mr_scenario_avg:.2f}\n")
        f.write("--------------------------------------------------\n")
        f.write(f"Estimasi Penurunan U5MR:      {penurunan_u5mr:.2f} poin\n")
        f.write(f"Estimasi Penurunan Persentase: {penurunan_persen:.1f} %\n")
        f.write("--------------------------------------------------\n")
        f.write(f"({changed_count} dari {X_original.shape[0]} data poin terdampak langsung)\n")
    print(f"\n✅ Hasil simulasi disimpan di: {summary_path}")
except Exception as e:
    print(f"\n❌ Error saat menyimpan hasil simulasi: {e}")

print("\n✅ Skrip Simulasi Intervensi selesai.")