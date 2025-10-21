"""
Data Preparation Script for Child Mortality Analysis
This script loads, cleans, and merges data from multiple sources
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Asumsi Anda punya file config.py dan utils.py
from config import *
from utils import *

def load_mortality_data():
    """
    Load and process Under-Five Mortality Rates data
    (VERSI PERBAIKAN: Hardcoded header, filter Median, perbaikan tahun .5)
    
    Returns:
        DataFrame with columns: Country, Year, under_five_mortality_rate
    """
    print("\n" + "="*60)
    print("📊 Loading Mortality Data")
    print("="*60)
    
    try:
        # --- PERBAIKAN 1: Langsung baca skiprows=14 ---
        # Buang logika auto-detect yang gagal
        df = pd.read_excel(MORTALITY_FILE, sheet_name='U5MR Country estimates', skiprows=14)
        print(f"✅ Membaca data dengan header di baris 15 (skiprows=14)")

        # --- PERBAIKAN 2: Filter HANYA 'Median' ---
        # Ini membuang data triplikat (Lower/Upper bound)
        if 'Uncertainty.Bounds*' in df.columns:
            df = df[df['Uncertainty.Bounds*'] == 'Median'].copy()
            print(f"✅ Data difilter, hanya mengambil 'Median'. Ukuran baru: {df.shape}")
        else:
            print("⚠️ Peringatan: Kolom 'Uncertainty.Bounds*' tidak ditemukan. Tidak bisa filter Median.")

    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None
    
    if df is None or len(df) == 0:
        print("❌ Failed to load data")
        return None
    
    print(f"✅ Loaded: Shape {df.shape}")
    print("\nColumns:", df.columns.tolist())
    
    # Clean up: Remove rows that are all NaN
    df = df.dropna(how='all')
    
    # Identify country column
    country_col = None
    for col in df.columns:
        if str(col).lower() in ['country.name', 'country', 'country/area']:
             country_col = col
             break
    
    if country_col is None:
        country_col = 'Country.Name' # Fallback
    
    print(f"\n✅ Using column '{country_col}' as Country")
    
    # Rename country column
    df = df.rename(columns={country_col: 'Country'})
    
    # Remove rows where Country is NaN or contains header text
    df = df[df['Country'].notna()]
    df = df[~df['Country'].astype(str).str.contains('Country|Unnamed|NaN', case=False, na=False)]
    
    # Find year columns
    year_cols = []
    for col in df.columns:
        if col == 'Country':
            continue
        try:
            col_str = str(col)
            
            # --- PERBAIKAN 3: Logika pencarian tahun .5 ---
            if col_str.endswith('.5'):
                year_part = col_str.split('.')[0]
                if year_part.isdigit():
                    year_val = int(year_part)
                    if 1950 <= year_val <= 2030:
                        year_cols.append(col)
            elif col_str.isdigit() and len(col_str) == 4:
                year_val = int(col_str)
                if 1950 <= year_val <= 2030:
                    year_cols.append(col)
            elif isinstance(col, (int, float)):
                if 1950 <= col <= 2030:
                    year_cols.append(col)
        except:
            continue
    
    print(f"\n✅ Found {len(year_cols)} year columns: {year_cols[:5]}...")
    
    if len(year_cols) == 0:
        print("❌ No year columns found!")
        print("Available columns:", df.columns.tolist())
        return None
    
    # Melt to long format
    id_vars = [col for col in df.columns if col not in year_cols] # Ambil semua ID vars
    df_long = df.melt(
        id_vars=id_vars,
        value_vars=year_cols,
        var_name='Year',
        value_name='under_five_mortality_rate'
    )
    
    # --- PERBAIKAN 4: Bersihkan '.5' dari kolom Tahun ---
    df_long['Year'] = df_long['Year'].astype(str).str.replace('.5', '', regex=False)
    df_long['Year'] = pd.to_numeric(df_long['Year'], errors='coerce')
    df_long = df_long.dropna(subset=['Year'])
    df_long['Year'] = df_long['Year'].astype(int)
    
    # Remove rows with missing mortality values
    df_long = df_long.dropna(subset=['under_five_mortality_rate'])
    
    # Convert mortality to numeric
    df_long['under_five_mortality_rate'] = pd.to_numeric(df_long['under_five_mortality_rate'], errors='coerce')
    df_long = df_long.dropna(subset=['under_five_mortality_rate'])
    
    # Normalize country names
    df_long = normalize_country_names(df_long)
    
    # Filter year range
    if 'Year' in df_long.columns:
        df_long = df_long[(df_long['Year'] >= START_YEAR) & (df_long['Year'] <= END_YEAR)]
    
    print(f"\n✅ Mortality data processed")
    print(f"   Shape: {df_long.shape}")
    print(f"   Countries: {df_long['Country'].nunique()}")
    print(f"   Year range: {df_long['Year'].min()} - {df_long['Year'].max()}")
    
    return df_long

def load_immunization_data():
    """
    Load and process WUENIC Immunization data from multiple sheets.
    (VERSI PERBAIKAN 2: Logika pencarian tahun yang lebih baik)
    
    Returns:
        DataFrame with columns: Country, Year, [immunization indicators]
    """
    print("\n" + "="*60)
    print("💉 Loading Immunization Data")
    print("="*60)
    
    VACCINE_SHEETS_TO_LOAD = ['BCG', 'DTP3', 'MCV1', 'POL3', 'HEPBB', 'HIB3', 'PCV3', 'RCV1', 'ROTAC']
    
    all_vaccine_data = []

    for sheet_name in VACCINE_SHEETS_TO_LOAD:
        try:
            df_sheet = pd.read_excel(IMMUNIZATION_FILE, sheet_name=sheet_name)
            
            if 'vaccine' not in df_sheet.columns:
                 df_sheet['vaccine'] = sheet_name

            print(f"✅ Berhasil memuat sheet: {sheet_name}")

            # --- PERBAIKAN LOGIKA PENCARIAN TAHUN ---
            year_cols = []
            id_col_names = ['country', 'iso3', 'vaccine', 'unicef_region']
            
            for col in df_sheet.columns:
                # Lewati kolom ID
                if str(col).lower() in id_col_names:
                    continue
                
                try:
                    col_str = str(col)
                    
                    # Cek jika string adalah angka 4 digit (cth: '1980')
                    if col_str.isdigit() and len(col_str) == 4:
                        year_val = int(col_str)
                        if 1950 <= year_val <= 2030:
                            year_cols.append(col)
                    # Cek jika kolomnya adalah float/int (cth: 1980.0 atau 1980)
                    elif isinstance(col, (int, float)):
                        year_val = int(col) # konversi 1980.0 -> 1980
                        if 1950 <= year_val <= 2030:
                            year_cols.append(col)
                except:
                    continue
            # --- SELESAI PERBAIKAN ---

            if not year_cols:
                print(f"⚠️ Tidak ada kolom tahun ditemukan di sheet {sheet_name}. Dilewati.")
                continue
            
            print(f"   -> Ditemukan {len(year_cols)} kolom tahun.")

            # Ambil kolom ID yang benar-benar ada di sheet
            id_vars = [col for col in df_sheet.columns if str(col).lower() in id_col_names]

            df_long = df_sheet.melt(
                id_vars=id_vars,
                value_vars=year_cols,
                var_name='Year',
                value_name='Coverage'
            )
            all_vaccine_data.append(df_long)
            
        except Exception as e:
            print(f"❌ Gagal memuat atau memproses sheet: {sheet_name}. Error: {e}")
            
    if not all_vaccine_data:
        print("❌ Tidak ada data imunisasi yang berhasil dimuat.")
        return None

    # Gabungkan semua data vaksin yang sudah di-melt
    df_long_all = pd.concat(all_vaccine_data, ignore_index=True)
    
    # Bersihkan kolom
    df_long_all['Year'] = pd.to_numeric(df_long_all['Year'], errors='coerce')
    df_long_all['Coverage'] = pd.to_numeric(df_long_all['Coverage'], errors='coerce')
    
    # Ganti nama kolom 'country' SEBELUM dropna
    if 'country' in df_long_all.columns:
        df_long_all = df_long_all.rename(columns={'country': 'Country'})

    df_long_all = df_long_all.dropna(subset=['Year', 'Coverage', 'Country'])
    df_long_all['Year'] = df_long_all['Year'].astype(int)

    print("✅ Semua sheet vaksin digabungkan. Mempivoting data...")
    
    # Pivot vaccines to columns
    df_wide = df_long_all.pivot_table(
        index=['Country', 'Year'],
        columns='vaccine',
        values='Coverage',
        aggfunc='first'
    ).reset_index()
    
    df_wide.columns.name = None
    df = df_wide
    
    # Normalize country names
    df = normalize_country_names(df)
    
    # Filter year range
    if 'Year' in df.columns:
        df = df[(df['Year'] >= START_YEAR) & (df['Year'] <= END_YEAR)]
    
    print(f"\n✅ Immunization data processed")
    print(f"   Shape: {df.shape}")
    print(f"   Countries: {df['Country'].nunique()}")
    if 'Year' in df.columns:
        print(f"   Year range: {df['Year'].min()} - {df['Year'].max()}")
    print(f"   Vaksin dimuat: {df.columns.tolist()[2:]}")
    
    return df

def load_nutrition_data():
    """
    Load and process JME Nutrition data
    (VERSI PERBAIKAN: Membaca sheet spesifik, tidak auto-detect)
    
    Returns:
        DataFrame with columns: Country, Year, [nutrition indicators]
    """
    print("\n" + "="*60)
    print("🍎 Loading Nutrition Data")
    print("="*60)
    
    # --- PERBAIKAN: Langsung targetkan sheet yang relevan ---
    SHEETS_TO_TRY = ['Stunting Prevalence', 'Overweight Prevalence']
    
    df_all_nutrition = []

    for sheet in SHEETS_TO_TRY:
        try:
            df = pd.read_excel(NUTRITION_FILE, sheet_name=sheet)
            print(f"✅ Berhasil memuat sheet: {sheet}")
            
            # Data ini sudah 'long', cari kolom yang tepat
            df.columns = df.columns.str.lower() # standarisasi
            
            # Cari kolom
            country_col = next(c for c in df.columns if 'country' in c or 'area' in c)
            year_col = next(c for c in df.columns if 'year' in c)
            value_col = next(c for c in df.columns if 'point' in c or 'estimate' in c)
            indicator_col = next(c for c in df.columns if 'indicator' in c)

            # Ganti nama
            df = df.rename(columns={
                country_col: 'Country',
                year_col: 'Year',
                value_col: 'Value',
                indicator_col: 'Indicator'
            })
            
            # Ambil kolom penting
            df = df[['Country', 'Year', 'Indicator', 'Value']]
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
            df = df.dropna(subset=['Value'])
            
            df_all_nutrition.append(df)
            
        except Exception as e:
            print(f"❌ Gagal memuat atau memproses sheet: {sheet}. Error: {e}")
            
    if not df_all_nutrition:
        print("❌ Tidak ada data nutrisi yang berhasil dimuat.")
        return None
    
    # Gabungkan semua data nutrisi (jika ada Stunting dan Overweight)
    df_long = pd.concat(df_all_nutrition, ignore_index=True)

    print("✅ Data nutrisi digabungkan. Mempivoting data...")
    
    # Pivot indicators to columns
    df_wide = df_long.pivot_table(
        index=['Country', 'Year'],
        columns='Indicator',
        values='Value',
        aggfunc='first'
    ).reset_index()
    
    df_wide.columns.name = None
    df = df_wide
    
    # Normalize country names
    df = normalize_country_names(df)
    
    # Convert Year to int if it's a column
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        df = df.dropna(subset=['Year'])
        df['Year'] = df['Year'].astype(int)
        
        # Filter year range
        df = df[(df['Year'] >= START_YEAR) & (df['Year'] <= END_YEAR)]
    
    print(f"\n✅ Nutrition data processed")
    print(f"   Shape: {df.shape}")
    print(f"   Countries: {df['Country'].nunique()}")
    if 'Year' in df.columns:
        print(f"   Year range: {df['Year'].min()} - {df['Year'].max()}")
    print(f"   Indikator dimuat: {df.columns.tolist()[2:]}")
    
    return df

def merge_all_data(mortality_df, immunization_df, nutrition_df):
    """
    Merge all datasets on Country and Year
    (Strategi: 'left' merge, dimulai dari 'mortality' sebagai data utama)
    
    Returns:
        Merged DataFrame
    """
    print("\n" + "="*60)
    print("🔗 Merging All Datasets")
    print("="*60)
    
    # Mulai dengan data mortalitas (data utama)
    if mortality_df is None:
        print("❌ Data mortalitas tidak ada. Tidak bisa melanjutkan merge.")
        return None
        
    merged = mortality_df.copy()
    print(f"Memulai dengan data mortalitas: {merged.shape}")
    
    # Merge immunization data
    if immunization_df is not None and 'Year' in immunization_df.columns:
        merged = merged.merge(
            immunization_df,
            on=['Country', 'Year'],
            how='left' # 'left' merge menjaga semua baris mortalitas
        )
        print(f"Setelah merging imunisasi: {merged.shape}")
    else:
        print("⚠️ Data imunisasi tidak ada atau kosong. Dilewati.")
    
    # Merge nutrition data
    if nutrition_df is not None and 'Year' in nutrition_df.columns:
        merged = merged.merge(
            nutrition_df,
            on=['Country', 'Year'],
            how='left' # 'left' merge menjaga semua baris mortalitas
        )
        print(f"Setelah merging nutrisi: {merged.shape}")
    else:
        print("⚠️ Data nutrisi tidak ada atau kosong. Dilewati.")
    
    print(f"\n✅ Merge complete!")
    print(f"   Final shape: {merged.shape}")
    print(f"   Countries: {merged['Country'].nunique()}")
    print(f"   Years: {merged['Year'].nunique()}")
    print(f"   Year range: {merged['Year'].min()} - {merged['Year'].max()}")
    
    return merged

def clean_column_names(df):
    """
    Clean and standardize column names
    """
    df = df.copy()
    
    # Convert to lowercase and replace spaces
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_')
    
    # Remove special characters
    df.columns = df.columns.str.replace(r'[^\w\s]', '', regex=True)
    
    return df

def main():
    """
    Main function to run the entire data preparation pipeline
    """
    print("\n" + "="*70)
    print("🚀 CHILD MORTALITY ANALYSIS - DATA PREPARATION PIPELINE")
    print("="*70)
    
    # Step 1: Load mortality data
    mortality_df = load_mortality_data()
    
    if mortality_df is None:
        print("\n❌ Failed to load mortality data. Exiting.")
        return
    
    # Step 2: Load immunization data
    immunization_df = load_immunization_data()
    
    # Step 3: Load nutrition data
    nutrition_df = load_nutrition_data()
    
    # Step 4: Merge all datasets
    merged_df = merge_all_data(mortality_df, immunization_df, nutrition_df)
    
    # Step 5: Clean column names
    merged_df = clean_column_names(merged_df)
    
    # Step 6: Save merged data (before imputation)
    print("\n" + "="*60)
    print("💾 Saving Merged Data")
    print("="*60)
    save_dataframe(merged_df, MERGED_DATA_FULL)
    
    # Step 7: Impute missing values
    print("\n" + "="*60)
    print("🔧 Imputing Missing Values")
    print("="*60)
    
    print("\nMissing values before imputation:")
    missing_before = merged_df.isnull().sum()[merged_df.isnull().sum() > 0]
    print(missing_before)
    
    merged_df_imputed = impute_missing_values(
        merged_df,
        method='interpolate',
        group_by='country'
    )
    
    print("\nMissing values after imputation:")
    missing_after = merged_df_imputed.isnull().sum()[merged_df_imputed.isnull().sum() > 0]
    if len(missing_after) > 0:
        print(missing_after)
    else:
        print("✅ No missing values!")
    
    # Step 8: Save imputed data
    save_dataframe(merged_df_imputed, MERGED_DATA_IMPUTED)
    
    # Step 9: Generate summary
    print("\n" + "="*60)
    print("📋 Data Summary")
    print("="*60)
    get_data_info(merged_df_imputed, "Final Merged Dataset")
    
    # Step 10: Create feature summary
    feature_summary = create_feature_summary(merged_df_imputed)
    print("\n📊 Feature Summary:")
    print(feature_summary)
    
    # Save feature summary
    feature_summary.to_csv(REPORTS_DIR / 'feature_summary.csv')
    print(f"\n✅ Feature summary saved to {REPORTS_DIR / 'feature_summary.csv'}")
    
    print("\n" + "="*70)
    print("✅ DATA PREPARATION COMPLETE!")
    print("="*70)
    print(f"\n📁 Output files:")
    print(f"   1. {MERGED_DATA_FULL}")
    print(f"   2. {MERGED_DATA_IMPUTED}")
    print(f"   3. {REPORTS_DIR / 'feature_summary.csv'}")
    print(f"\n🎯 Next steps:")
    print(f"   1. Run EDA notebook: notebooks/02_eda.ipynb")
    print(f"   2. Build models: notebooks/03_modeling.ipynb")
    print(f"   3. Launch dashboard: streamlit run app.py")

if __name__ == "__main__":
    main()