# config.py
"""Konfigurasi proyek Child Mortality Analysis."""

from pathlib import Path
import os
from typing import Final

# --- Path Utama ---
BASE_DIR: Final[Path] = Path(__file__).resolve().parent.parent

DATA_DIR: Final[Path] = BASE_DIR / "data"
RAW_DATA_DIR: Final[Path] = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Final[Path] = DATA_DIR / "processed"

OUTPUT_DIR: Final[Path] = BASE_DIR / "outputs"
FIGURES_DIR: Final[Path] = OUTPUT_DIR / "figures"
REPORTS_DIR: Final[Path] = OUTPUT_DIR / "reports"
MODELS_DIR: Final[Path] = OUTPUT_DIR / "models"

# --- Pastikan Semua Folder Dibuat Otomatis ---
_REQUIRED_DIRS = [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    FIGURES_DIR,
    REPORTS_DIR,
    MODELS_DIR,
]

for _d in _REQUIRED_DIRS:
    _d.mkdir(parents=True, exist_ok=True)

# --- File Data Raw (bisa diubah via environment variable) ---
MORTALITY_FILE = Path(os.getenv("MORTALITY_FILE", RAW_DATA_DIR / "Under-five_Mortality_Rates_2024.xlsx"))
IMMUNIZATION_FILE = Path(os.getenv("IMMUNIZATION_FILE", RAW_DATA_DIR / "wuenic2024rev_web-update.xlsx"))
NUTRITION_FILE = Path(os.getenv("NUTRITION_FILE", RAW_DATA_DIR / "jme_database_country_model_2025.xlsx"))

# --- File Data Hasil Olahan ---
MERGED_DATA_FULL = PROCESSED_DATA_DIR / "merged_data_full.csv"
MERGED_DATA_IMPUTED = PROCESSED_DATA_DIR / "merged_data_full_yearly_imputed.csv"

# --- File Model ---
SAVED_MODEL_PIPELINE = MODELS_DIR / "random_forest_u5mr_pipeline.joblib"

# --- Pengaturan Global ---
RANDOM_STATE: Final[int] = int(os.getenv("RANDOM_STATE", 42))
TEST_SIZE: Final[float] = float(os.getenv("TEST_SIZE", 0.2))

START_YEAR: Final[int] = int(os.getenv("START_YEAR", 2000))
END_YEAR: Final[int] = int(os.getenv("END_YEAR", 2023))

TARGET_VARIABLE: Final[str] = os.getenv("TARGET_VARIABLE", "under_five_mortality_rate")

# --- Daftar Fitur Penting ---
KEY_IMMUNIZATION_FEATURES = ["bcg", "dtp3", "mcv1", "pol3", "hepbb", "hib3", "pcv3", "rcv1", "rotac"]
KEY_NUTRITION_FEATURES = ["stunting", "overweight"]
KEY_REGION_FEATURE = ["sdgregion"]

# --- Peringatan Jika File Mentah Tidak Ditemukan ---
def _warn_if_missing_files():
    missing = [p for p in (MORTALITY_FILE, IMMUNIZATION_FILE, NUTRITION_FILE) if not p.exists()]
    if missing:
        print("⚠️  Peringatan: Beberapa file data mentah tidak ditemukan:")
        for p in missing:
            print(f"    - {p}")

_warn_if_missing_files()
