# File: config.py
"""
Configuration file for Child Mortality Analysis Project
Contains all paths and global settings
"""
import os
from pathlib import Path

# --- Path Dasar ---
# Asumsi config.py ada di dalam folder 'src'
# BASE_DIR akan menunjuk ke folder 'child_mortality_analysis'
BASE_DIR = Path(__file__).resolve().parent.parent

# --- Direktori Data ---
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# --- Direktori Output ---
OUTPUT_DIR = BASE_DIR / 'outputs' # Folder utama untuk semua output
FIGURES_DIR = OUTPUT_DIR / 'figures' # Untuk gambar/plot dari EDA/Evaluasi
REPORTS_DIR = OUTPUT_DIR / 'reports' # Untuk file teks/csv laporan (metrik, rekomendasi)
MODELS_DIR = OUTPUT_DIR / 'models' # Untuk menyimpan file model (.joblib/.pkl)

# --- Membuat Direktori Otomatis ---
# Pastikan semua folder output ada saat skrip dijalankan
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, FIGURES_DIR, REPORTS_DIR, MODELS_DIR]:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create directory {directory}. Error: {e}")

# --- Path File Data Mentah (Contoh, sesuaikan jika nama file beda) ---
MORTALITY_FILE = RAW_DATA_DIR / 'Under-five_Mortality_Rates_2024.xlsx'
IMMUNIZATION_FILE = RAW_DATA_DIR / 'wuenic2024rev_web-update.xlsx'
NUTRITION_FILE = RAW_DATA_DIR / 'jme_database_country_model_2025.xlsx'

# --- Path File Data Olahan (Hasil dari dataprep.py) ---
MERGED_DATA_FULL = PROCESSED_DATA_DIR / 'merged_data_full.csv'
MERGED_DATA_IMPUTED = PROCESSED_DATA_DIR / 'merged_data_full_yearly_imputed.csv'
# Jika ada file hasil feature engineering:
# FEATURED_DATASET = PROCESSED_DATA_DIR / 'featured_dataset.csv'

# --- Path File Model (Hasil dari modeling_notebook .py) ---
# Gunakan nama file yang konsisten
SAVED_MODEL_PIPELINE = MODELS_DIR / 'random_forest_u5mr_pipeline.joblib'
# Jika menyimpan scaler/features terpisah (sebenarnya tidak perlu jika pakai pipeline):
# SCALER_FILE = MODELS_DIR / 'scaler.pkl'
# FEATURE_NAMES_FILE = MODELS_DIR / 'feature_names.pkl'

# --- Pengaturan Global ---
RANDOM_STATE = 42 # Untuk reproducibility
TEST_SIZE = 0.2 # Ukuran data tes (misal: 20%)

# Rentang Tahun Analisis (sesuaikan dengan data Anda)
START_YEAR = 2000
END_YEAR = 2023 # Data kita hanya sampai 2023

# Variabel Target
TARGET_VARIABLE = 'under_five_mortality_rate'

# --- (Opsional) Daftar Fitur Kunci ---
# Ini bisa berguna untuk konsistensi antar skrip
KEY_IMMUNIZATION_FEATURES = ['bcg', 'dtp3', 'mcv1', 'pol3', 'hepbb', 'hib3', 'pcv3', 'rcv1', 'rotac']
KEY_NUTRITION_FEATURES = ['stunting', 'overweight']
KEY_REGION_FEATURE = ['sdgregion']


# --- Konfirmasi Loading ---
# Pesan ini akan muncul setiap kali config.py diimpor
print(f"✅ Configuration loaded successfully")
print(f"📁 Base directory: {BASE_DIR}")
print(f"📊 Raw data directory: {RAW_DATA_DIR}")
print(f"💾 Processed data directory: {PROCESSED_DATA_DIR}")
print(f"💾 Model directory: {MODELS_DIR}") # Tambahkan print ini