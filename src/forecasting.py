# File: forecasting.py
# Deskripsi: Meramal tren U5MR per negara menggunakan Prophet + Evaluasi.
# (VERSI PERBAIKAN: Menambahkan if __name__ == '__main__' untuk Windows multiprocessing)
# Penulis: [Nama Anda]
# Bahasa: Python 3.10+
# ---------------------------------------------------------------

import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from multiprocessing import freeze_support # Import freeze_support
import warnings
warnings.filterwarnings('ignore')

print("Memulai skrip Forecasting (Prophet) dengan Evaluasi...")

# --- Konfigurasi Awal & Path ---
print("Mencari lokasi file skrip dan memuat konfigurasi...")
try:
    SRC_DIR = Path(__file__).resolve().parent
    # Sesuaikan path agar bisa import dari 'src' jika script ini di 'src'
    sys.path.append(str(SRC_DIR.parent)) # Menambah folder 'child_mortality_analysis'
    from src.config import MERGED_DATA_IMPUTED, FIGURES_DIR, REPORTS_DIR, TARGET_VARIABLE, START_YEAR, END_YEAR
    print("✅ Konfigurasi berhasil dimuat.")
    BASE_DIR = SRC_DIR.parent # Folder child_mortality_analysis
except ImportError as e:
    # Coba path alternatif jika struktur berbeda
    try:
        sys.path.append(str(Path.cwd())) # Menambah folder CWD
        from config import MERGED_DATA_IMPUTED, FIGURES_DIR, REPORTS_DIR, TARGET_VARIABLE, START_YEAR, END_YEAR
        print("✅ Konfigurasi berhasil dimuat (dari CWD).")
        BASE_DIR = Path.cwd() # Asumsi dijalankan dari root proyek
    except ImportError:
        print(f"❌ FATAL ERROR: Gagal import dari config.py: {e}")
        exit()
except Exception as e:
     print(f"❌ FATAL ERROR saat setup path: {e}")
     exit()

try:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
except Exception as e:
    print(f"⚠️ Warning: Gagal membuat folder output: {e}")

# --- Parameter Forecasting & Evaluasi ---
NEGARA_CONTOH = ['Indonesia', 'Nigeria', 'Finland']
TAHUN_RAMALAN = 30
CV_INITIAL = '5475 days' # ~15 tahun
CV_PERIOD = '365 days'  # Geser 1 tahun
CV_HORIZON = '1095 days' # Evaluasi 3 tahun

print(f"📁 Project Base Directory: {BASE_DIR}")
print(f"💾 Data akan dimuat dari: {MERGED_DATA_IMPUTED}")
print(f"🎯 Target Variable: {TARGET_VARIABLE}")
print(f"🔮 Meramal untuk negara: {NEGARA_CONTOH}")
print(f"⏳ Meramal {TAHUN_RAMALAN} tahun ke depan.")
print(f"⏱️ Cross-Validation: initial='{CV_INITIAL}', period='{CV_PERIOD}', horizon='{CV_HORIZON}'")

# ---------------------------------------------------------------
# 1. Load Data (di luar if __name__ == '__main__')
# ---------------------------------------------------------------
print(f"\nMemuat data dari: {MERGED_DATA_IMPUTED}")
try:
    df = pd.read_csv(MERGED_DATA_IMPUTED)
    df.rename(columns={'under_five_mortality_rate': TARGET_VARIABLE}, inplace=True, errors='ignore')
    print(f"✅ Dataset berhasil dimuat (Shape: {df.shape})")
except FileNotFoundError:
    print(f"❌ Error: File data {MERGED_DATA_IMPUTED} tidak ditemukan.")
    exit()
except Exception as e:
    print(f"❌ Error saat memuat data: {e}")
    exit()

if TARGET_VARIABLE not in df.columns:
    print(f"❌ Error: Kolom target '{TARGET_VARIABLE}' tidak ditemukan di data!")
    exit()

# ---------------------------------------------------------------
# 2. Proses Forecasting & Evaluasi per Negara (di dalam if __name__ == '__main__')
# ---------------------------------------------------------------
# --- 👇👇👇 PERBAIKAN DIMULAI DI SINI 👇👇👇 ---
if __name__ == '__main__':
    # --- Tambahkan freeze_support() untuk Windows ---
    freeze_support()
    # --- SELESAI PENAMBAHAN ---

    # --- 👇 SEMUA KODE DI BAWAH INI DIGESER (INDENT) 👇 ---
    all_forecasts = {}
    all_metrics = {}

    for negara in NEGARA_CONTOH:
        print("\n" + "="*40)
        print(f"🔮 Memproses Negara: {negara}")
        print("="*40)

        df_country = df[df['country'] == negara].copy()
        if df_country.empty:
            print(f"⚠️ Data untuk {negara} tidak ditemukan. Dilewati.")
            continue

        df_prophet = df_country[['year', TARGET_VARIABLE]].copy()
        df_prophet['ds'] = pd.to_datetime(df_prophet['year'], format='%Y')
        df_prophet = df_prophet.rename(columns={TARGET_VARIABLE: 'y'})
        df_prophet = df_prophet[['ds', 'y']].drop_duplicates(subset=['ds']).sort_values('ds')

        # Tingkatkan batas minimum data untuk CV yang lebih stabil
        min_data_for_cv = int(pd.Timedelta(CV_INITIAL).days / 365) + int(pd.Timedelta(CV_HORIZON).days / 365) + 2
        if len(df_prophet) < min_data_for_cv:
            print(f"⚠️ Data historis untuk {negara} ({len(df_prophet)} poin) tidak cukup untuk cross-validation (min: {min_data_for_cv}). CV Dilewati.")
            # Tetap lanjutkan forecasting tanpa evaluasi CV
            run_cv = False
        else:
            run_cv = True

        print(f"   Data historis ditemukan: {len(df_prophet)} tahun ({df_prophet['ds'].min().year} - {df_prophet['ds'].max().year})")

        model = Prophet(yearly_seasonality=True, interval_width=0.95)

        try:
            model.fit(df_prophet)
            print("   ✅ Model Prophet berhasil dilatih.")
        except Exception as e:
            print(f"   ❌ Gagal melatih model Prophet untuk {negara}: {e}")
            continue

        # --- Bagian Cross-Validation ---
        if run_cv:
            print("   ⏱️ Menjalankan Cross-Validation...")
            try:
                # Coba dulu dengan parallel="processes"
                df_cv = cross_validation(model, initial=CV_INITIAL, period=CV_PERIOD, horizon=CV_HORIZON,
                                         parallel="processes")

                df_p = performance_metrics(df_cv)
                print("   ✅ Cross-Validation (parallel) selesai.")
                print("   📊 Metrik Performa (contoh horizon 1095 hari / 3 tahun):")
                # Tampilkan metrik untuk horizon target
                horizon_timedelta = pd.Timedelta(CV_HORIZON)
                metrics_at_horizon = df_p[df_p['horizon'] == horizon_timedelta]
                if not metrics_at_horizon.empty:
                     print(metrics_at_horizon.round(3))
                else:
                     print("      -> Tidak ada data tepat di horizon target, tampilkan horizon terdekat:")
                     # Cari horizon terdekat jika tidak pas
                     closest_horizon_idx = (df_p['horizon'] - horizon_timedelta).abs().idxmin()
                     print(df_p.loc[[closest_horizon_idx]].round(3))


                all_metrics[negara] = df_p

                metrics_filename = f"forecast_metrics_{negara.lower().replace(' ', '_')}.csv"
                metrics_path = REPORTS_DIR / metrics_filename
                df_p.to_csv(metrics_path, index=False)
                print(f"   💾 Metrik evaluasi lengkap disimpan di: {metrics_path}")

            except RuntimeError as e: # Tangkap error multiprocessing spesifik
                if "An attempt has been made to start a new process" in str(e):
                    print(f"   ⚠️ Gagal menjalankan CV parallel untuk {negara} (masalah multiprocessing Windows): {e}")
                    print("      -> Mencoba ulang CV tanpa parallel (sekuensial)...")
                    try:
                        df_cv = cross_validation(model, initial=CV_INITIAL, period=CV_PERIOD, horizon=CV_HORIZON,
                                                 parallel=None) # Jalankan sekuensial
                        df_p = performance_metrics(df_cv)
                        print("   ✅ Cross-Validation (tanpa parallel) selesai.")
                        print("   📊 Metrik Performa (contoh horizon 1095 hari / 3 tahun):")
                        horizon_timedelta = pd.Timedelta(CV_HORIZON)
                        metrics_at_horizon = df_p[df_p['horizon'] == horizon_timedelta]
                        if not metrics_at_horizon.empty:
                             print(metrics_at_horizon.round(3))
                        else:
                             print("      -> Tidak ada data tepat di horizon target, tampilkan horizon terdekat:")
                             closest_horizon_idx = (df_p['horizon'] - horizon_timedelta).abs().idxmin()
                             print(df_p.loc[[closest_horizon_idx]].round(3))

                        all_metrics[negara] = df_p
                        metrics_filename = f"forecast_metrics_{negara.lower().replace(' ', '_')}.csv"
                        metrics_path = REPORTS_DIR / metrics_filename
                        df_p.to_csv(metrics_path, index=False)
                        print(f"   💾 Metrik evaluasi lengkap disimpan di: {metrics_path}")
                    except Exception as e2:
                        print(f"   ❌ Gagal menjalankan CV bahkan tanpa parallel: {e2}")
                else:
                     # Error lain saat CV
                     print(f"   ❌ Gagal menjalankan Cross-Validation untuk {negara}: {e}")
            except Exception as e:
                 # Error umum lain saat CV
                 print(f"   ❌ Gagal menjalankan Cross-Validation untuk {negara}: {e}")
        else:
             print("   ⏭️ Cross-Validation dilewati karena data tidak cukup.")


        # --- Bagian Forecasting & Plotting ---
        print(f"\n   ⏳ Membuat peramalan {TAHUN_RAMALAN} tahun ke depan...")
        try:
            future = model.make_future_dataframe(periods=TAHUN_RAMALAN, freq='AS')
            print(f"      Membuat dataframe masa depan sampai tahun: {future['ds'].max().year}")
            forecast = model.predict(future)
            print("      ✅ Prediksi berhasil dibuat.")

            all_forecasts[negara] = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

            fig = model.plot(forecast)
            ax = fig.gca()
            ax.set_title(f"Peramalan Angka Kematian Balita - {negara} ({TAHUN_RAMALAN} Thn)", fontsize=16)
            ax.set_xlabel("Tahun", fontsize=12)
            ax.set_ylabel("AKB (per 1.000 kelahiran)", fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.6)

            plot_filename = f"forecast_{negara.lower().replace(' ', '_')}.png"
            plot_path = FIGURES_DIR / plot_filename
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"   🖼️ Plot peramalan disimpan di: {plot_path}")
            plt.close(fig)

        except Exception as e:
            print(f"   ❌ Gagal membuat prediksi atau plot untuk {negara}: {e}")

    # ---------------------------------------------------------------
    # 3. Selesai (masih di dalam if __name__ == '__main__')
    # ---------------------------------------------------------------
    print("\n" + "="*40)
    print("✅ Skrip Forecasting (Prophet) & Evaluasi Selesai.")
    print("="*40)
    print(f"Hasil plot peramalan disimpan di folder: {FIGURES_DIR}")
    print(f"Hasil metrik evaluasi per negara disimpan di folder: {REPORTS_DIR}")

    try:
        if all_metrics:
            all_metrics_df = pd.concat(all_metrics, names=['country', 'index']).reset_index()
            all_metrics_path = REPORTS_DIR / "all_forecast_metrics_summary.csv"
            all_metrics_df.to_csv(all_metrics_path, index=False)
            print(f"💾 Ringkasan semua metrik evaluasi disimpan di: {all_metrics_path}")
    except Exception as e:
         print(f"⚠️ Gagal menyimpan ringkasan metrik: {e}")

# --- 👆👆👆 AKHIR DARI BLOK 'if __name__ == '__main__': 👆👆👆 ---