# %% [markdown]
# # 📊 Exploratory Data Analysis - Child Mortality (Versi Publik)
# 
# **Author:** Raihan Aprilialdy Risanto  
# **Institution:** Universitas Negeri Jakarta  
# **Date:** 2024
# 
# ## Objectives
# 1. Understand data distribution and patterns (Simplified Visuals)
# 2. Identify key factors related to child mortality
# 3. Analyze trends over time and across the world
# 4. Generate insights understandable by a general audience

# %%
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

import sys
from pathlib import Path
sys.path.append(str(Path.cwd().parent / 'src'))

from config import * # Asumsi config.py mendefinisikan path seperti MERGED_DATA_IMPUTED, FIGURES_DIR
from utils import * # Asumsi utils.py punya fungsi get_data_info, create_correlation_heatmap (meski tak dipakai), dll.

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis") # Ganti palet warna agar lebih menarik

print("✅ Libraries loaded successfully")

# %% [markdown]
# ## 1. Load Data
# Memuat data yang sudah dibersihkan dan digabungkan.

# %%
# Load the processed data (versi imputed untuk visualisasi lebih mulus)
try:
    df = pd.read_csv(MERGED_DATA_IMPUTED)
    print("Dataset loaded!")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
except FileNotFoundError:
    print(f"❌ Error: File {MERGED_DATA_IMPUTED} tidak ditemukan.")
    print("Pastikan Anda sudah menjalankan 'dataprep.py' terlebih dahulu.")
    exit()


# %%
# Basic information
get_data_info(df, "Child Mortality Dataset Overview")

# %% [markdown]
# ## 2. Data Overview
# Melihat ringkasan data, termasuk data yang hilang.

# %%
# Check data types and missing values
print("Data Types:")
print(df.dtypes)

print("\n" + "="*60)
print("Missing Values (%):")
missing_pct = (df.isnull().sum() / len(df)) * 100
missing_df = pd.DataFrame({'Percentage': missing_pct})
print(missing_df[missing_df['Percentage'] > 0].sort_values('Percentage', ascending=False).round(1))

# %%
# Summary statistics
print("\n" + "="*60)
print("Summary Statistics (Numeric Data):")
print(df.describe().round(2))

# %% [markdown]
# ## 3. Bagaimana Sebaran Angka Kematian Balita?
# Melihat distribusi angka kematian balita (U5MR) di seluruh dunia.

# %%
# Analyze the target variable: under_five_mortality_rate
target_col = 'under_five_mortality_rate'

if target_col in df.columns:
    plt.figure(figsize=(10, 6))
    
    # --- VISUALISASI UTAMA: Histogram Sederhana ---
    sns.histplot(df[target_col].dropna(), bins=50, kde=True, color='skyblue')
    
    # Tambahkan garis rata-rata dan median
    mean_val = df[target_col].mean()
    median_val = df[target_col].median()
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Rata-rata: {mean_val:.1f}')
    plt.axvline(median_val, color='green', linestyle='-', linewidth=2, label=f'Median: {median_val:.1f}')
    
    plt.title('Distribusi Angka Kematian Balita Global (2000-2023)', fontsize=14, fontweight='bold')
    plt.xlabel('Angka Kematian Balita (per 1.000 kelahiran hidup)')
    plt.ylabel('Jumlah Data (Negara-Tahun)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'target_distribution_simple.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 {target_col} Statistics:")
    print(f"Mean: {mean_val:.2f}")
    print(f"Median: {median_val:.2f}")
    print(f"Std Dev: {df[target_col].std():.2f}")
    print(f"Min: {df[target_col].min():.2f}")
    print(f"Max: {df[target_col].max():.2f}")
    print("\nInsight: Kebanyakan data (negara-tahun) memiliki angka kematian rendah (di bawah 50),")
    print("         namun ada beberapa kasus dengan angka kematian sangat tinggi, membuat rata-rata (merah)")
    print("         tertarik ke kanan, lebih tinggi dari median (hijau).")

# %% [markdown]
# ## 4. Bagaimana Tren Angka Kematian Balita Global?
# Melihat perubahan rata-rata angka kematian balita dunia dari waktu ke waktu.

# %%
# Mortality trend over time
if 'year' in df.columns and target_col in df.columns:
    yearly_avg = df.groupby('year')[target_col].mean().reset_index()
    
    fig = px.line(yearly_avg, x='year', y=target_col,
                  title='<b>Tren Penurunan Angka Kematian Balita Global (2000-2023)</b>',
                  labels={'year': 'Tahun', target_col: 'Rata-rata Angka Kematian Balita'},
                  markers=True) # Tambahkan marker agar lebih jelas
    fig.update_traces(line_color='#e74c3c', line_width=3)
    fig.update_layout(hovermode='x unified', height=500, title_x=0.5)
    fig.write_html(FIGURES_DIR / 'temporal_trend_simple.html')
    fig.show()
    
    # Calculate change
    start_rate = yearly_avg[target_col].iloc[0]
    end_rate = yearly_avg[target_col].iloc[-1]
    change_pct = ((end_rate - start_rate) / start_rate) * 100
    
    print(f"\n📈 Tren Waktu:")
    print(f"Angka kematian tahun {yearly_avg['year'].iloc[0]}: {start_rate:.1f}")
    print(f"Angka kematian tahun {yearly_avg['year'].iloc[-1]}: {end_rate:.1f}")
    print(f"Perubahan: <b>{change_pct:.1f}%</b> (Menurun secara signifikan!)")

# %% [markdown]
# ## 5. Negara Mana yang Paling Berhasil & Perlu Perhatian?
# Membandingkan negara dengan angka kematian balita tertinggi dan terendah di tahun terakhir.

# %%
# Top and bottom countries by mortality rate (latest year)
if 'country' in df.columns and 'year' in df.columns:
    latest_year = df['year'].max()
    latest_data = df[df['year'] == latest_year].sort_values(target_col, ascending=False).dropna(subset=[target_col])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True) # Share X axis for comparison
    
    # Top 10 countries with highest mortality
    top_10 = latest_data.head(10)
    sns.barplot(x=target_col, y='country', data=top_10, ax=axes[0], palette='Reds_r')
    axes[0].set_xlabel('Angka Kematian Balita')
    axes[0].set_ylabel('Negara')
    axes[0].set_title(f'10 Negara Angka Kematian Tertinggi ({latest_year})', fontweight='bold')
    # Tambahkan label angka
    for index, value in enumerate(top_10[target_col]):
        axes[0].text(value, index, f' {value:.1f}', va='center')
    
    # Bottom 10 countries with lowest mortality
    bottom_10 = latest_data.tail(10).sort_values(target_col, ascending=True)
    sns.barplot(x=target_col, y='country', data=bottom_10, ax=axes[1], palette='Greens_r')
    axes[1].set_xlabel('Angka Kematian Balita')
    axes[1].set_ylabel('') # Hapus label Y duplikat
    axes[1].set_title(f'10 Negara Angka Kematian Terendah ({latest_year})', fontweight='bold')
    # Tambahkan label angka
    for index, value in enumerate(bottom_10[target_col]):
        axes[1].text(value, index, f' {value:.1f}', va='center')

    fig.suptitle(f'Perbandingan Angka Kematian Balita di Tahun {latest_year}', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout for suptitle
    plt.savefig(FIGURES_DIR / 'countries_comparison_simple.png', dpi=300, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## 6. Faktor Apa yang Berkaitan dengan Angka Kematian Balita?
# Melihat hubungan antara angka kematian balita dengan faktor lain seperti vaksinasi dan status gizi.

# %%
# Correlation with target variable
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
exclude_cols = ['year'] 
numeric_cols = [col for col in numeric_cols if col not in exclude_cols and col != target_col] # Exclude target too

target_corr = pd.Series(dtype=float) 
if target_col in df.columns and len(numeric_cols) > 0:
    target_corr = df[numeric_cols + [target_col]].corr()[target_col].sort_values()
    target_corr = target_corr[target_corr.index != target_col]
    
    if not target_corr.empty:
        plt.figure(figsize=(10, 7))
        
        # --- VISUALISASI UTAMA: Bar Chart Korelasi ---
        target_corr.plot(kind='barh', color=(target_corr > 0).map({True: 'salmon', False: 'lightgreen'}))
        plt.title('Faktor yang Berkaitan dengan Angka Kematian Balita', fontsize=14, fontweight='bold')
        plt.xlabel('Kekuatan Hubungan (Korelasi)\n(Negatif = Baik, Positif = Buruk)')
        plt.ylabel('Faktor')
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.axvline(0, color='black', linewidth=0.8) # Garis nol
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'target_correlations_simple.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n🎯 Faktor yang Berkaitan dengan Angka Kematian Balita:")
        print("   (Semakin ke kiri/negatif, semakin baik hubungannya untuk menurunkan kematian)")
        print("   (Semakin ke kanan/positif, semakin buruk hubungannya)")
        print(target_corr)

# %% [markdown]
# ## 7. Melihat Hubungan Lebih Dekat: Stunting & Vaksinasi
# Fokus pada dua faktor terkuat yang ditemukan: Stunting (positif) dan Vaksin DTP3 (negatif).

# %%
# Scatter plots for the two most important features
if target_col in df.columns and not target_corr.empty:
    
    # Tentukan 2 fitur terkuat (satu positif, satu negatif)
    strongest_pos_feat = target_corr[target_corr > 0].idxmax() if any(target_corr > 0) else None
    strongest_neg_feat = target_corr[target_corr < 0].idxmin() if any(target_corr < 0) else None
    
    plot_features = [feat for feat in [strongest_pos_feat, strongest_neg_feat] if feat] # Ambil yang valid

    if not plot_features:
        print("⚠️ Tidak ada korelasi signifikan untuk ditampilkan.")
    else:
        num_plots = len(plot_features)
        fig, axes = plt.subplots(1, num_plots, figsize=(8 * num_plots, 6), sharey=True)
        if num_plots == 1: axes = [axes] # Pastikan axes selalu list

        for idx, feature in enumerate(plot_features):
            if feature in df.columns:
                plot_data = df[[feature, target_col]].dropna()
                
                if plot_data.empty or len(plot_data) < 2:
                    axes[idx].text(0.5, 0.5, f"No valid data for\n{feature}", ha='center', va='center', transform=axes[idx].transAxes, color='red')
                    axes[idx].set_title(f'{feature.capitalize()} vs Kematian Balita', fontweight='bold')
                    continue

                # --- VISUALISASI UTAMA: Scatter Plot dengan Garis Tren ---
                sns.regplot(x=feature, y=target_col, data=plot_data, ax=axes[idx],
                            scatter_kws={'alpha': 0.2, 's': 15}, 
                            line_kws={'color': 'red', 'linestyle': '--'})
                
                corr_val = target_corr[feature]
                axes[idx].set_title(f'{feature.capitalize()} vs Kematian Balita\n(Hubungan: {"Positif Kuat" if corr_val > 0.5 else ("Negatif Kuat" if corr_val < -0.5 else ("Positif" if corr_val > 0 else "Negatif"))})', 
                                    fontweight='bold', fontsize=12)
                axes[idx].set_xlabel(f'Tingkat {feature.capitalize()} (%)')
                axes[idx].set_ylabel('Angka Kematian Balita' if idx == 0 else '') # Hanya label Y di plot pertama

        fig.suptitle('Hubungan Antara Status Gizi/Vaksinasi dengan Kematian Balita', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(FIGURES_DIR / 'scatter_key_relationships_simple.png', dpi=300, bbox_inches='tight')
        plt.show()

# %% [markdown]
# ## 8. Peta Sebaran Angka Kematian Balita Dunia
# Melihat visualisasi angka kematian balita di peta dunia pada tahun terakhir.

# %%
# World map visualization
if 'country' in df.columns and 'year' in df.columns:
    latest_year = df['year'].max()
    latest_data = df[df['year'] == latest_year].dropna(subset=[target_col])
    
    fig = px.choropleth(latest_data,
                        locations='country',
                        locationmode='country names',
                        color=target_col,
                        hover_name='country',
                        color_continuous_scale='Reds', # Skala warna merah, makin gelap makin tinggi
                        title=f'<b>Peta Angka Kematian Balita Dunia ({latest_year})</b>',
                        labels={target_col:'Angka Kematian Balita'})
    
    fig.update_layout(height=600, title_x=0.5)
    fig.write_html(FIGURES_DIR / 'world_map_simple.html')
    fig.show()

# %% [markdown]
# ## 9. Ringkasan Temuan Utama

# %%
print("\n" + "="*70)
print("📋 RINGKASAN TEMUAN UTAMA (Mudah Dipahami)")
print("="*70)

insights = []

# Target Variable
if target_col in df.columns:
    insights.append(f"1. 🌍 **Sebaran:** Angka kematian balita sangat bervariasi, dari sangat rendah ({df[target_col].min():.1f}) hingga sangat tinggi ({df[target_col].max():.1f}). Kebanyakan negara berada di angka rendah.")
    insights.append(f"2. 📉 **Tren Global:** Kabar baik! Rata-rata angka kematian balita dunia telah turun drastis sekitar {change_pct:.0f}% sejak tahun 2000.")

# Correlations
if not target_corr.empty:
    pos_corr = target_corr[target_corr > 0]
    neg_corr = target_corr[target_corr < 0]

    if not neg_corr.empty:
        top_neg_feat = neg_corr.idxmin()
        insights.append(f"3. 👍 **Faktor Penolong:** Cakupan <b>vaksinasi</b> (seperti {top_neg_feat.upper()}) sangat berkaitan dengan <b>menurunnya</b> angka kematian balita.")
    
    if not pos_corr.empty:
        top_pos_feat = pos_corr.idxmax()
        insights.append(f"4. 👎 **Faktor Risiko:** Tingginya angka <b>{top_pos_feat}</b> sangat berkaitan dengan <b>meningkatnya</b> angka kematian balita.")

# Geographic
insights.append(f"5. 🗺️ **Geografis:** Peta menunjukkan perbedaan besar antar wilayah. Beberapa wilayah (terutama di Afrika Sub-Sahara) masih menghadapi tantangan besar.")

# Time Period
insights.append(f"6. ⏳ **Periode Data:** Analisis ini mencakup data dari tahun {df['year'].min()} hingga {df['year'].max()}.")

for i, insight in enumerate(insights):
    print(insight)

print("\n" + "="*70)
print("✅ EDA COMPLETE! Insight siap dibagikan.")
print("="*70)

# %%