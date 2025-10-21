"""
🧠 Dashboard Analisis Kematian Balita
Aplikasi Streamlit interaktif untuk mengeksplorasi, memprediksi, dan meramal angka kematian balita.

Penulis: Raihan Aprilialdy Risanto
Institusi: Universitas Negeri Jakarta
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
import joblib
from pathlib import Path
import sys
import warnings
from typing import Tuple, Optional, List, Dict
warnings.filterwarnings('ignore')

# ==============================================================================
# KONFIGURASI & KONSTANTA
# ==============================================================================

st.set_page_config(
    page_title="Analisis Kematian Balita", 
    page_icon="👶", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Konstanta halaman
PAGES = {
    'summary': "🏠 Ringkasan Global",
    'geo': "🗺️ Analisis Geografis",
    'factors': "💡 Analisis Faktor",
    'forecast': "🔮 Peramalan & Simulasi",
    'diagnostic': "🔍 Diagnostic Dashboard",
    'calculator': "🧮 Kalkulator Intervensi",
    'ews': "🚨 Sistem Peringatan Dini"
}

# Konstanta intervensi
INTERVENTIONS = {
    'vaksinasi': {'name': 'Vaksinasi Universal', 'reduction': 0.25},
    'gizi': {'name': 'Program Gizi', 'reduction': 0.20},
    'wash': {'name': 'WASH (Air & Sanitasi)', 'reduction': 0.20},
    'komprehensif': {'name': 'Paket Komprehensif', 'reduction': 0.50}
}

# Color schemes
SEVERITY_COLORS = {
    '✅ Baik': '#00CC96',
    '⚠️ Perlu Perhatian': '#FFA15A',
    '🔴 Kritis': '#EF553B',
    '🆘 Darurat': '#8B0000'
}

ALERT_COLORS = {
    '🔴 KRITIS': '#8B0000',
    '🟠 PERINGATAN': '#FF8C00',
    '🟡 PERHATIAN': '#FFD700',
    '✅ BAIK': '#32CD32'
}

# ==============================================================================
# INISIALISASI KONFIGURASI
# ==============================================================================

try:
    BASE_DIR = Path(__file__).resolve().parent.parent
    SRC_DIR = BASE_DIR / 'src'
    sys.path.append(str(BASE_DIR))
    from src.config import *
except (ImportError, FileNotFoundError) as e:
    st.error(f"⚠️ Gagal memuat konfigurasi: {e}")
    st.info("Pastikan struktur folder dan file config.py sudah benar.")
    st.stop()

# ==============================================================================
# FUNGSI HELPER - DATA LOADING & CACHING
# ==============================================================================

# ==============================================================================
# FUNGSI HELPER - DATA LOADING & CACHING
# ==============================================================================

@st.cache_data
def run_prophet_forecast(
    _df: pd.DataFrame, 
    country: str, 
    years: int,
    target_col: str
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Peramalan Prophet dengan error handling."""
    df_country = _df[_df['country'] == country].copy()
    
    # Validasi data
    if df_country.empty or df_country[target_col].isnull().all():
        return None, "Data historis tidak tersedia."
    
    df_prophet = df_country[['year', target_col]].dropna().copy()
    
    if len(df_prophet) < 5:
        return None, f"Data terlalu sedikit ({len(df_prophet)} tahun)."
    
    # Persiapan Prophet
    df_prophet['ds'] = pd.to_datetime(df_prophet['year'], format='%Y')
    df_prophet.rename(columns={target_col: 'y'}, inplace=True)
    
    # Set bounds
    df_prophet['floor'] = 0.1
    cap = max(300, df_country[target_col].max() * 1.2)
    df_prophet['cap'] = cap
    
    try:
        # Training
        model = Prophet(
            yearly_seasonality=True, 
            interval_width=0.95, 
            growth='logistic'
        )
        model.fit(df_prophet)
        
        # Forecast
        future = model.make_future_dataframe(periods=years, freq='AS')
        future['floor'] = 0.1
        future['cap'] = cap
        
        forecast = model.predict(future)
        return forecast, None
    except Exception as e:
        return None, f"Error peramalan: {str(e)}"

@st.cache_resource
def load_model(path: Path) -> Tuple[Optional[object], Optional[List[str]]]:
    """Memuat model pipeline dan ekstrak fitur."""
    try:
        pipeline = joblib.load(path)
        preprocessor = pipeline.named_steps['preprocessor']
        
        num_features = preprocessor.transformers_[0][2]
        cat_features = preprocessor.transformers_[1][2]
        features = list(num_features) + list(cat_features)
        
        return pipeline, features
    except (FileNotFoundError, KeyError, AttributeError) as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None, None

@st.cache_data
def load_feature_importance(path: Path) -> Optional[pd.DataFrame]:
    """Memuat data feature importance."""
    try:
        if path.exists():
            return pd.read_csv(path)
        return None
    except Exception as e:
        st.warning(f"⚠️ Gagal memuat feature importance: {e}")
        return None

# ==============================================================================
# FUNGSI HELPER - ANALISIS & PREDIKSI
# ==============================================================================

# ==============================================================================
# FUNGSI HELPER - DATA LOADING & CACHING
# ==============================================================================

@st.cache_data(ttl=3600)
def load_data(path: Path) -> Optional[pd.DataFrame]:
    """Memuat dan mempersiapkan data dengan validasi."""
    try:
        df = pd.read_csv(path)
        
        # Data cleaning
        df['country'] = df['country'].replace('Niger', 'Nigeria')
        
        # --- PERBAIKAN LOGIKA AGREGRASI ---
        # Pisahkan kolom numerik dan non-numerik (kategorikal)
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        # Hapus 'year' karena itu bagian dari grouping key
        if 'year' in numeric_cols:
            numeric_cols.remove('year')
            
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        # Hapus 'country' karena itu bagian dari grouping key
        if 'country' in categorical_cols:
            categorical_cols.remove('country')

        # Buat dictionary untuk agregasi
        agg_dict = {}
        # Agregasi mean untuk kolom numerik
        for col in numeric_cols:
            agg_dict[col] = 'mean'
        # Agregasi 'first' (ambil nilai pertama) untuk kolom kategorikal
        for col in categorical_cols:
            agg_dict[col] = 'first'

        # Lakukan groupby dengan dictionary agregasi
        df = df.groupby(['country', 'year'], as_index=False).agg(agg_dict)
        # --- AKHIR PERBAIKAN ---

        df.rename(columns={'under_five_mortality_rate': TARGET_VARIABLE}, 
                 inplace=True, errors='ignore')
        
        # Validasi
        if df.empty or TARGET_VARIABLE not in df.columns:
            st.error("Data tidak valid atau kolom target tidak ditemukan.")
            return None
            
        return df
    except FileNotFoundError:
        st.error(f"❌ File tidak ditemukan: {path}")
        return None
    except Exception as e:
        st.error(f"❌ Error memuat data: {e}")
        return None

def make_prediction(
    pipeline: object, 
    input_data: pd.DataFrame
) -> Optional[float]:
    """Membuat prediksi dengan error handling."""
    try:
        prediction = pipeline.predict(input_data)
        return float(prediction[0])
    except Exception as e:
        st.error(f"❌ Gagal prediksi: {e}")
        return None

def calculate_trend(df: pd.DataFrame, window: int = 5) -> Optional[float]:
    """Hitung tren menggunakan regresi linear."""
    recent = df.tail(window)
    if len(recent) < 3:
        return None
    
    x = recent['year'].values
    y = recent[TARGET_VARIABLE].values
    
    try:
        slope = np.polyfit(x, y, 1)[0]
        return slope
    except:
        return None

# ==============================================================================
# FUNGSI HELPER - VISUALISASI
# ==============================================================================

def create_time_series_plot(
    df: pd.DataFrame, 
    x: str, 
    y: str, 
    title: str
) -> go.Figure:
    """Membuat plot time series standar."""
    fig = px.line(df, x=x, y=y, markers=True)
    fig.update_layout(
        title=title,
        hovermode='x unified',
        height=400
    )
    return fig

def create_choropleth_map(
    df: pd.DataFrame, 
    color_col: str,
    title: str,
    color_scale: str = 'Reds'
) -> go.Figure:
    """Membuat peta choropleth standar."""
    fig = px.choropleth(
        df,
        locations='country',
        locationmode='country names',
        color=color_col,
        hover_name='country',
        color_continuous_scale=color_scale
    )
    
    fig.update_layout(
        title_text=f'<b>{title}</b>',
        title_x=0.5,
        height=600,
        margin={"r":0, "t":50, "l":0, "b":0},
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='natural earth'
        )
    )
    
    return fig

# ==============================================================================
# FUNGSI HELPER - ANALISIS DIAGNOSTIC
# ==============================================================================

def identify_problems(
    country_data: pd.Series, 
    df_global: pd.DataFrame
) -> List[Dict]:
    """Identifikasi masalah kesehatan di negara."""
    problems = []
    
    # Stunting
    if 'stunting' in country_data and pd.notna(country_data['stunting']):
        if country_data['stunting'] > df_global['stunting'].mean() * 1.2:
            problems.append({
                'faktor': '🍽️ Stunting',
                'nilai': f"{country_data['stunting']:.1f}%",
                'rata_rata_global': f"{df_global['stunting'].mean():.1f}%",
                'severity': (country_data['stunting'] - df_global['stunting'].mean()) 
                           / df_global['stunting'].mean() * 100,
                'dampak': 'TINGGI',
                'deskripsi': "Kurang gizi kronis - prioritas utama intervensi."
            })
    
    # Wasting
    if 'wasting' in country_data and pd.notna(country_data['wasting']):
        if country_data['wasting'] > df_global['wasting'].mean() * 1.2:
            problems.append({
                'faktor': '⚠️ Wasting',
                'nilai': f"{country_data['wasting']:.1f}%",
                'rata_rata_global': f"{df_global['wasting'].mean():.1f}%",
                'severity': (country_data['wasting'] - df_global['wasting'].mean()) 
                           / df_global['wasting'].mean() * 100,
                'dampak': 'TINGGI',
                'deskripsi': "Gizi akut buruk - memerlukan respons cepat."
            })
    
    # Vaksinasi DTP3
    if 'dtp3' in country_data and pd.notna(country_data['dtp3']):
        if country_data['dtp3'] < 90:
            problems.append({
                'faktor': '💉 Vaksinasi DTP3',
                'nilai': f"{country_data['dtp3']:.1f}%",
                'rata_rata_global': "90% (target WHO)",
                'severity': (90 - country_data['dtp3']) / 90 * 100,
                'dampak': 'SEDANG-TINGGI',
                'deskripsi': "Cakupan rendah meningkatkan risiko penyakit."
            })
    
    # Air bersih
    if 'basic_water' in country_data and pd.notna(country_data['basic_water']):
        if country_data['basic_water'] < 80:
            problems.append({
                'faktor': '💧 Air Bersih',
                'nilai': f"{country_data['basic_water']:.1f}%",
                'rata_rata_global': "80%+",
                'severity': (80 - country_data['basic_water']) / 80 * 100,
                'dampak': 'SEDANG',
                'deskripsi': "Akses terbatas memicu penyakit infeksi."
            })
    
    # Sanitasi
    if 'basic_sanitation' in country_data and pd.notna(country_data['basic_sanitation']):
        if country_data['basic_sanitation'] < 75:
            problems.append({
                'faktor': '🚽 Sanitasi',
                'nilai': f"{country_data['basic_sanitation']:.1f}%",
                'rata_rata_global': "75%+",
                'severity': (75 - country_data['basic_sanitation']) / 75 * 100,
                'dampak': 'SEDANG',
                'deskripsi': "Sanitasi buruk memicu diare dan penyakit."
            })
    
    # GDP
    if 'gdp_per_capita' in country_data and pd.notna(country_data['gdp_per_capita']):
        median_gdp = df_global['gdp_per_capita'].median()
        if country_data['gdp_per_capita'] < median_gdp * 0.5:
            problems.append({
                'faktor': '💰 Kemiskinan',
                'nilai': f"${country_data['gdp_per_capita']:.0f}",
                'rata_rata_global': f"${median_gdp:.0f} (median)",
                'severity': (median_gdp - country_data['gdp_per_capita']) 
                           / median_gdp * 100,
                'dampak': 'TINGGI',
                'deskripsi': "Kemiskinan membatasi akses layanan kesehatan."
            })
    
    # Sort by severity
    problems.sort(key=lambda x: x['severity'], reverse=True)
    return problems

def generate_interventions(problems: List[Dict]) -> List[Dict]:
    """Generate daftar intervensi berdasarkan masalah."""
    interventions = []
    
    for prob in problems:
        if 'Stunting' in prob['faktor'] or 'Wasting' in prob['faktor']:
            interventions.append({
                'intervensi': '🍲 Program Gizi & Fortifikasi',
                'impact': 9,
                'effort': 6,
                'timeline': '2-5 tahun',
                'cost': 'Sedang-Tinggi',
                'detail': (
                    'Distribusi makanan bergizi',
                    'Fortifikasi makanan pokok',
                    'Edukasi gizi ibu & anak',
                    'Suplementasi vitamin A & zat besi'
                )
            })
        
        if 'Vaksinasi' in prob['faktor']:
            interventions.append({
                'intervensi': '💉 Kampanye Imunisasi Massal',
                'impact': 8,
                'effort': 4,
                'timeline': '6-12 bulan',
                'cost': 'Rendah-Sedang',
                'detail': (
                    'Mobile vaccination units',
                    'SMS reminder system',
                    'Insentif untuk orang tua',
                    'Pelatihan tenaga kesehatan'
                )
            })
        
        if 'Air Bersih' in prob['faktor'] or 'Sanitasi' in prob['faktor']:
            interventions.append({
                'intervensi': '🚰 Infrastruktur WASH',
                'impact': 7,
                'effort': 8,
                'timeline': '3-7 tahun',
                'cost': 'Tinggi',
                'detail': (
                    'Pembangunan sumur bor',
                    'Program jamban sehat',
                    'Kampanye cuci tangan',
                    'Water treatment plants'
                )
            })
        
        if 'Kemiskinan' in prob['faktor']:
            interventions.append({
                'intervensi': '💰 Program Perlindungan Sosial',
                'impact': 8,
                'effort': 7,
                'timeline': 'Berkelanjutan',
                'cost': 'Tinggi',
                'detail': (
                    'Conditional Cash Transfer',
                    'Asuransi kesehatan universal',
                    'Subsidi pangan & pendidikan',
                    'Pemberdayaan ekonomi ibu'
                )
            })
    
    # Remove duplicates
    unique_interventions = []
    seen = set()
    for i in interventions:
        key = i['intervensi']
        if key not in seen:
            seen.add(key)
            unique_interventions.append(i)
    
    return unique_interventions

# ==============================================================================
# FUNGSI HELPER - EARLY WARNING SYSTEM
# ==============================================================================

@st.cache_data
def get_ews_alerts(_df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Deteksi negara dengan peringatan."""
    alerts = []
    
    for country in _df['country'].unique():
        country_df = _df[_df['country'] == country].sort_values('year')
        
        if len(country_df) < 5:
            continue
        
        # Calculate metrics
        trend = calculate_trend(country_df)
        latest_akb = country_df.iloc[-1][target_col]
        previous_akb = country_df.iloc[-2][target_col]
        change = latest_akb - previous_akb
        
        # Determine alert level
        alert_level = None
        reasons = []
        
        if change > 0.1:
            alert_level = "🔴 KRITIS"
            reasons.append(f"AKB naik {change:.1f} poin")
        elif trend and trend > 0.2:
            alert_level = "🟠 PERINGATAN"
            reasons.append("Tren memburuk")
        elif abs(change) < 0.5 and latest_akb > 25:
            alert_level = "🟡 PERHATIAN"
            reasons.append("Stagnan di atas target SDG")
        
        if latest_akb > 75:
            if alert_level is None:
                alert_level = "🟠 PERINGATAN"
            reasons.append(f"AKB sangat tinggi ({latest_akb:.1f})")
        
        if alert_level:
            alerts.append({
                'negara': country,
                'level': alert_level,
                'akb_terkini': latest_akb,
                'perubahan_tahunan': change,
                'tren_5_tahun': trend if trend else 0,
                'alasan': '; '.join(reasons)
            })
    
    return pd.DataFrame(alerts)

# ==============================================================================
# UI STYLING
# ==============================================================================

def apply_custom_css():
    """Apply custom CSS styling."""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5em;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stAlert {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# INISIALISASI DATA
# ==============================================================================

apply_custom_css()

with st.spinner("🔄 Memuat data dan model..."):
    df = load_data(MERGED_DATA_IMPUTED)
    pipeline, FEATURES = load_model(SAVED_MODEL_PIPELINE)
    df_importance = load_feature_importance(REPORTS_DIR / 'feature_importance.csv')

# Validasi
if df is None or pipeline is None:
    st.error("❌ Gagal memuat data atau model. Aplikasi tidak dapat dilanjutkan.")
    st.stop()

target = TARGET_VARIABLE

# ==============================================================================
# SIDEBAR
# ==============================================================================

with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 1.5rem;">
            <span style="font-size: 3rem; margin-right: 1rem;">👶</span>
            <div>
                <h1 style="margin: 0; line-height: 1;">Analisis</h1>
                <h2 style="margin: 0; line-height: 1; font-weight: 400;">Kematian Balita</h2>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    page = st.radio(
        "📌 Navigasi Dashboard",
        list(PAGES.values()),
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Page descriptions
    page_descriptions = {
        PAGES['summary']: "Gambaran umum, statistik kunci, dan tren global AKB.",
        PAGES['geo']: "Sebaran geografis AKB di seluruh dunia.",
        PAGES['factors']: "Faktor-faktor yang paling berpengaruh terhadap AKB.",
        PAGES['forecast']: "Peramalan tren masa depan dan simulasi intervensi.",
        PAGES['diagnostic']: "Diagnosis akar masalah dan rekomendasi intervensi.",
        PAGES['calculator']: "Estimasi dampak intervensi dalam menyelamatkan nyawa.",
        PAGES['ews']: "Identifikasi negara dengan tren memburuk atau stagnan."
    }
    
    st.info(page_descriptions.get(page, ""))
    
    with st.expander("ℹ️ Tentang Dashboard"):
        st.markdown("""
        Dashboard ini menganalisis faktor yang mempengaruhi Angka Kematian Balita (U5MR) 
        menggunakan data historis dan machine learning.
        
        **Institusi:** Universitas Negeri Jakarta
        
        ---
        
        [![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github)](https://github.com/Falconxima)
        """)

# ==============================================================================
# HALAMAN: RINGKASAN GLOBAL
# ==============================================================================

if page == PAGES['summary']:
    st.markdown('<p class="main-header">📊 Ringkasan Global Angka Kematian Balita</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Dashboard ini menampilkan gambaran umum dan tren Angka Kematian Balita (U5MR) 
    dari tahun 2000 hingga 2023 berdasarkan data global.
    """)
    
    # Statistik kunci
    st.markdown("### 📈 Statistik Kunci (2000-2023)")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Rata-rata Global",
            f"{df[target].mean():.1f}",
            help="Rata-rata AKB per 1.000 kelahiran"
        )
    
    with col2:
        st.metric(
            "Median Global",
            f"{df[target].median():.1f}",
            help="Nilai tengah AKB"
        )
    
    with col3:
        max_idx = df[target].idxmax()
        st.metric(
            "AKB Tertinggi",
            f"{df[target].max():.1f}",
            help=f"{df.loc[max_idx]['country']}"
        )
    
    with col4:
        min_idx = df[target].idxmin()
        st.metric(
            "AKB Terendah",
            f"{df[target].min():.1f}",
            help=f"{df.loc[min_idx]['country']}"
        )
    
    st.caption("💡 AKB: Angka Kematian Balita (per 1.000 kelahiran hidup)")
    
    st.markdown("---")
    
    # Visualisasi
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.markdown("### 📉 Tren Penurunan Global")
        yearly_avg = df.groupby('year')[target].mean().reset_index()
        
        fig = create_time_series_plot(
            yearly_avg, 
            'year', 
            target,
            'Rata-rata Global AKB (2000-2023)'
        )
        fig.update_traces(
            hovertemplate="Tahun %{x}: %{y:.1f} per 1.000 kelahiran"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 Distribusi Nilai AKB")
        fig = px.histogram(
            df, 
            x=target, 
            nbins=50,
            labels={target: 'AKB'}
        )
        fig.update_layout(
            yaxis_title="Frekuensi",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# HALAMAN: ANALISIS GEOGRAFIS
# ==============================================================================

elif page == PAGES['geo']:
    st.markdown('<p class="main-header">🗺️ Analisis Geografis</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Eksplorasi sebaran geografis Angka Kematian Balita di seluruh dunia. 
    Gunakan slider untuk melihat perubahan dari tahun ke tahun.
    """)
    
    selected_year = st.slider(
        "📅 Pilih Tahun:",
        min_value=int(df['year'].min()),
        max_value=int(df['year'].max()),
        value=int(df['year'].max())
    )
    
    map_data = df[df['year'] == selected_year].dropna(subset=[target])
    
    if not map_data.empty:
        fig_map = create_choropleth_map(
            map_data,
            target,
            f'Sebaran AKB Global - {selected_year}'
        )
        fig_map.update_traces(
            hovertemplate="<b>%{hovertext}</b><br>AKB: %{z:.1f} per 1.000"
        )
        st.plotly_chart(fig_map, use_container_width=True)
        
        # Top & Bottom countries
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🔴 10 Negara AKB Tertinggi")
            top10 = map_data.nlargest(10, target)[['country', target]]
            top10['AKB'] = top10[target].apply(lambda x: f"{x:.1f}")
            st.dataframe(
                top10[['country', 'AKB']].reset_index(drop=True),
                hide_index=True,
                use_container_width=True
            )
        
        with col2:
            st.markdown("#### 🟢 10 Negara AKB Terendah")
            bottom10 = map_data.nsmallest(10, target)[['country', target]]
            bottom10['AKB'] = bottom10[target].apply(lambda x: f"{x:.1f}")
            st.dataframe(
                bottom10[['country', 'AKB']].reset_index(drop=True),
                hide_index=True,
                use_container_width=True
            )
    else:
        st.warning(f"⚠️ Tidak ada data untuk tahun {selected_year}")

# ==============================================================================
# HALAMAN: ANALISIS FAKTOR
# ==============================================================================

elif page == PAGES['factors']:
    st.markdown('<p class="main-header">💡 Analisis Faktor Risiko & Pelindung</p>', 
                unsafe_allow_html=True)
    
    if df_importance is not None:
        st.markdown("### 🎯 Faktor Paling Berpengaruh")
        
        # Prepare display
        df_imp_display = df_importance.copy()
        df_imp_display['feature_display'] = (
            df_imp_display['feature']
            .str.replace('sdgregion_', '', regex=False)
            .str.replace('_', ' ')
            .str.title()
        )
        
        # Bar chart
        fig_imp = px.bar(
            df_imp_display.head(10).sort_values('importance'),
            x='importance',
            y='feature_display',
            orientation='h',
            title='Top 10 Faktor Paling Berpengaruh',
            labels={'importance': 'Skor Pengaruh', 'feature_display': 'Faktor'}
        )
        fig_imp.update_layout(height=500)
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 📊 Hubungan Faktor Kunci dengan AKB")
        
        # Scatter plots
        col1, col2 = st.columns(2)
        
        with col1:
            if 'stunting' in df.columns:
                fig_stunting = px.scatter(
                    df.dropna(subset=['stunting', target]),
                    x='stunting',
                    y=target,
                    trendline='ols',
                    opacity=0.3,
                    title='Stunting vs AKB',
                    labels={'stunting': 'Stunting (%)', target: 'AKB'}
                )
                fig_stunting.update_traces(
                    hovertemplate="Stunting: %{x:.1f}%<br>AKB: %{y:.1f}"
                )
                st.plotly_chart(fig_stunting, use_container_width=True)
        
        with col2:
            if 'dtp3' in df.columns:
                fig_dtp3 = px.scatter(
                    df.dropna(subset=['dtp3', target]),
                    x='dtp3',
                    y=target,
                    trendline='ols',
                    opacity=0.3,
                    title='Vaksin DTP3 vs AKB',
                    labels={'dtp3': 'DTP3 (%)', target: 'AKB'}
                )
                fig_dtp3.update_traces(
                    hovertemplate="DTP3: %{x:.0f}%<br>AKB: %{y:.1f}"
                )
                st.plotly_chart(fig_dtp3, use_container_width=True)
        
        # Additional correlations
        st.markdown("### 🔗 Matriks Korelasi")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target in numeric_cols and len(numeric_cols) > 5:
            corr_matrix = df[numeric_cols].corr()
            target_corr = corr_matrix[target].sort_values(ascending=False)
            
            fig_corr = px.bar(
                x=target_corr.values[1:11],
                y=target_corr.index[1:11],
                orientation='h',
                title='Top 10 Korelasi dengan AKB',
                labels={'x': 'Korelasi', 'y': 'Faktor'}
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("⚠️ Data feature importance tidak tersedia.")

# ==============================================================================
# HALAMAN: PERAMALAN & SIMULASI
# ==============================================================================

elif page == PAGES['forecast']:
    st.markdown('<p class="main-header">🔮 Peramalan & Simulasi Interaktif</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Buat peramalan tren AKB untuk negara tertentu. Hasil peramalan akan tetap tersimpan 
    dan dapat digunakan untuk simulasi intervensi.
    """)
    
    # Initialize session state jika belum ada
    if 'forecast_history' not in st.session_state:
        st.session_state.forecast_history = []
    
    # --- BAGIAN INPUT ---
    st.markdown("### ⚙️ Pengaturan Peramalan Baru")
    col1, col2 = st.columns(2)
    
    with col1:
        country_list = sorted(df['country'].unique())
        default_idx = country_list.index('Indonesia') if 'Indonesia' in country_list else 0
        selected_country = st.selectbox(
            "🌍 Pilih Negara:",
            country_list,
            index=default_idx,
            key="forecast_country_select" # Memberi key untuk stabilitas
        )
    
    with col2:
        forecast_years = st.slider(
            "📅 Jangka Waktu Peramalan (tahun):",
            5, 50, 10,
            key="forecast_years_slider"
        )
    
    # --- PERBAIKAN LOGIKA 1: Tombol hanya untuk membuat dan menyimpan peramalan ---
    if st.button("▶️ Jalankan & Tambah Peramalan Baru", type="primary"):
        with st.spinner(f"🔄 Membuat peramalan untuk {selected_country}..."):
            forecast, error = run_prophet_forecast(
                df, selected_country, forecast_years, target
            )
        
        if error:
            st.error(f"❌ {error}")
        elif forecast is not None:
            st.success(f"✅ Peramalan untuk {selected_country} berhasil dibuat dan disimpan!")
            
            # Create visualization
            fig = go.Figure()
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines',
                line_color='rgba(0,100,80,0.2)', name='Batas Atas', showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines',
                line_color='rgba(0,100,80,0.2)', name='Interval Kepercayaan 95%'
            ))
            
            # Historical data
            hist_data = df[df['country'] == selected_country]
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(hist_data['year'], format='%Y'), y=hist_data[target],
                mode='markers', marker=dict(color='black', size=8), name='Data Historis'
            ))
            
            # Forecast line
            fig.add_trace(go.Scatter(
                x=forecast['ds'], y=forecast['yhat'], mode='lines',
                line=dict(color='red', width=3), name='Peramalan'
            ))
            
            # SDG target line
            fig.add_hline(
                y=25, line_dash="dash", line_color="green", annotation_text="Target SDG 2030"
            )
            
            fig.update_layout(
                title=f'Peramalan AKB - {selected_country} ({forecast_years} tahun)',
                xaxis_title='Tahun', yaxis_title='AKB (per 1.000 kelahiran)',
                hovermode='x unified', height=500
            )
            
            # Store everything needed in history
            st.session_state.forecast_history.append({
                'country': selected_country,
                'years': forecast_years,
                'data': forecast,
                'figure': fig # Simpan objek figure-nya!
            })

    # --- PERBAIKAN LOGIKA 2: Tampilkan hasil dari session_state di luar blok tombol ---
    if st.session_state.forecast_history:
        st.markdown("---")
        st.markdown("### 📜 Hasil Peramalan & Simulasi")
        
        # Opsi untuk membersihkan riwayat
        if st.button("🗑️ Hapus Semua Riwayat"):
            st.session_state.forecast_history = []
            st.rerun()
        
        # Ambil hasil peramalan TERBARU dari riwayat
        latest = st.session_state.forecast_history[-1]
        
        # --- TAMPILKAN GRAFIK YANG DISIMPAN ---
        st.markdown(f"### 1️⃣ Hasil Peramalan Tren - **{latest['country']}**")
        st.plotly_chart(latest['figure'], use_container_width=True)
        
        # --- LANJUTKAN KE BAGIAN SIMULASI ---
        st.markdown(f"### 2️⃣ Simulasi Dampak Intervensi - **{latest['country']}**")
        
        forecast_df = latest['data']
        future_years = forecast_df[
            forecast_df['ds'].dt.year > df['year'].max()
        ]['ds'].dt.year.unique()
        
        if len(future_years) > 0:
            sim_year = st.selectbox(
                "📅 Pilih Tahun Simulasi:",
                future_years
            )
            
            baseline_u5mr = forecast_df[
                forecast_df['ds'].dt.year == sim_year
            ]['yhat'].values[0]
            
            baseline_u5mr = max(0.1, baseline_u5mr)
            
            st.info(
                f"📊 Perkiraan AKB **tanpa intervensi** di tahun **{sim_year}**: "
                f"**{baseline_u5mr:.2f}** per 1.000 kelahiran"
            )
            
            # Get last known data
            country_data_series = df[df['country'] == latest['country']].sort_values('year')
            
            if not country_data_series.empty:
                last_data = country_data_series.iloc[-1]
                
                # Intervention parameters
                st.markdown("#### 🎛️ Atur Skenario Intervensi")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    last_dtp3 = last_data.get('dtp3', 90)
                    default_dtp3 = int(last_dtp3) if pd.notna(last_dtp3) else 90
                    
                    target_dtp3 = st.slider(
                        "💉 Target Cakupan DTP3 (%):",
                        0, 100, default_dtp3
                    )
                
                with col2:
                    last_stunting = last_data.get('stunting')
                    label = "🍽️ Target Penurunan Stunting (%):"
                    if pd.notna(last_stunting):
                        label += f" (Saat ini: {last_stunting:.1f}%)"
                    
                    reduction_stunting = st.slider(label, 0, 100, 10)
                
                if st.button("▶️ Jalankan Simulasi", type="secondary"):
                    with st.spinner("🔄 Menjalankan simulasi..."):
                        input_baseline = pd.DataFrame([last_data[FEATURES]])
                        pred_baseline = make_prediction(pipeline, input_baseline)
                        
                        if pred_baseline is not None:
                            input_scenario = input_baseline.copy()
                            input_scenario['dtp3'] = target_dtp3
                            
                            if 'stunting' in input_scenario.columns and pd.notna(last_stunting):
                                input_scenario['stunting'] = (
                                    last_stunting * (1 - reduction_stunting / 100)
                                )
                            
                            pred_scenario = make_prediction(pipeline, input_scenario)
                            
                            if pred_scenario is not None:
                                effect = (pred_scenario - pred_baseline) / pred_baseline if pred_baseline > 0.01 else 0
                                effect = min(0, effect)
                                
                                final_u5mr = max(0.1, baseline_u5mr * (1 + effect))
                                delta = final_u5mr - baseline_u5mr
                                
                                st.success("✅ Simulasi selesai!")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("AKB dengan Intervensi", f"{final_u5mr:.2f}", delta=f"{delta:.2f}")
                                with col2:
                                    lives_saved_per_year = abs(delta) / 1000 * 100000
                                    st.metric("Nyawa Diselamatkan", f"{lives_saved_per_year:,.0f}", help="Per 100.000 kelahiran/tahun")
                                with col3:
                                    st.metric("Efek Intervensi", f"{effect:.2%}")
                                
                                st.caption("💡 Estimasi berdasarkan model machine learning dan tren Prophet.")
                            else:
                                st.error("❌ Gagal membuat prediksi skenario.")
                        else:
                            st.error("❌ Gagal membuat prediksi baseline.")
            else:
                st.warning(f"⚠️ Data historis untuk {latest['country']} tidak tersedia.")
        else:
            st.warning("⚠️ Tidak ada tahun future untuk simulasi.")
    else:
        # Tampilan jika belum ada peramalan sama sekali
        st.info("💡 Jalankan peramalan terlebih dahulu untuk menampilkan hasil dan mengaktifkan simulasi.")
# ==============================================================================
# HALAMAN: DIAGNOSTIC DASHBOARD
# ==============================================================================

elif page == PAGES['diagnostic']:
    st.markdown('<p class="main-header">🔍 Diagnosis Masalah Kematian Balita</p>', 
                unsafe_allow_html=True)
    
    st.markdown("## 🚨 Identifikasi Negara Kritis")
    
    latest_year = df['year'].max()
    df_latest = df[df['year'] == latest_year].copy()
    
    # Categorize severity
    df_latest['severity'] = pd.cut(
        df_latest[target],
        bins=[0, 25, 50, 100, float('inf')],
        labels=['✅ Baik', '⚠️ Perlu Perhatian', '🔴 Kritis', '🆘 Darurat']
    )
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        critical_count = len(df_latest[
            df_latest['severity'].isin(['🔴 Kritis', '🆘 Darurat'])
        ])
        st.metric("Negara Status Kritis", critical_count)
    
    with col2:
        avg_u5mr = df_latest[target].mean()
        st.metric(
            "Rata-rata AKB Global",
            f"{avg_u5mr:.1f}",
            delta=f"{avg_u5mr - 25:.1f} dari target SDG"
        )
    
    with col3:
        worst_idx = df_latest[target].idxmax()
        worst = df_latest.loc[worst_idx]
        st.metric(
            "Negara Terburuk",
            worst['country'],
            f"{worst[target]:.1f}"
        )
    
    # Severity map
    fig_severity = px.choropleth(
        df_latest,
        locations='country',
        locationmode='country names',
        color='severity',
        color_discrete_map=SEVERITY_COLORS,
        title=f'Peta Severity Kematian Balita ({latest_year})'
    )
    fig_severity.update_layout(height=600)
    st.plotly_chart(fig_severity, use_container_width=True)
    
    st.markdown("---")
    st.markdown("## 🔬 Analisis Akar Masalah")
    
    selected_country = st.selectbox(
        "🌍 Pilih negara untuk diagnosis:",
        sorted(df['country'].unique())
    )
    
    country_series = df[df['country'] == selected_country]
    
    if not country_series.empty:
        country_data = country_series.sort_values('year').iloc[-1]
        
        # Country profile
        st.markdown(f"### 📊 Profil: **{selected_country}**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("AKB Saat Ini", f"{country_data[target]:.1f}")
        
        with col2:
            if 'year' in country_data:
                st.metric("Tahun Data", int(country_data['year']))
        
        with col3:
            global_avg = df[df['year'] == country_data['year']][target].mean()
            delta = country_data[target] - global_avg
            st.metric(
                "vs Rata-rata Global",
                f"{delta:+.1f}",
                delta_color="inverse"
            )
        
        # Identify problems
        st.markdown("### 🎯 Faktor Risiko Utama")
        
        problems = identify_problems(country_data, df)
        
        if problems:
            for i, prob in enumerate(problems[:5], 1):
                with st.expander(
                    f"**#{i} {prob['faktor']}** - Dampak: {prob['dampak']}",
                    expanded=(i <= 3)
                ):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Nilai Saat Ini", prob['nilai'])
                    
                    with col2:
                        st.metric("Target/Rata-rata", prob['rata_rata_global'])
                    
                    with col3:
                        st.metric("Severity Score", f"{prob['severity']:.0f}/100")
                    
                    st.info(prob['deskripsi'])
        else:
            st.success("✅ Tidak ada faktor risiko kritis teridentifikasi.")
        
        st.markdown("---")
        st.markdown("## 💊 Rekomendasi Intervensi")
        
        if problems:
            interventions = generate_interventions(problems)
            
            if interventions:
                st.markdown("### 🎯 Prioritas Intervensi (Impact vs Effort)")
                
                # Create priority matrix
                fig_priority = go.Figure()
                
                for interv in interventions:
                    fig_priority.add_trace(go.Scatter(
                        x=[interv['effort']],
                        y=[interv['impact']],
                        mode='markers+text',
                        marker=dict(
                            size=20,
                            color=interv['impact'] * 10,
                            colorscale='Viridis',
                            showscale=True
                        ),
                        text=[interv['intervensi']],
                        textposition='top center',
                        hovertemplate=(
                            f"<b>{interv['intervensi']}</b><br>"
                            f"Impact: {interv['impact']}/10<br>"
                            f"Effort: {interv['effort']}/10<extra></extra>"
                        )
                    ))
                
                # Add quadrants
                fig_priority.add_shape(
                    type="rect", x0=0, y0=6, x1=5, y1=10,
                    fillcolor="lightgreen", opacity=0.2,
                    layer="below", line_width=0
                )
                fig_priority.add_shape(
                    type="rect", x0=6, y0=6, x1=10, y1=10,
                    fillcolor="lightyellow", opacity=0.2,
                    layer="below", line_width=0
                )
                fig_priority.add_shape(
                    type="rect", x0=6, y0=0, x1=10, y1=5,
                    fillcolor="lightcoral", opacity=0.2,
                    layer="below", line_width=0
                )
                
                fig_priority.update_layout(
                    title="Impact vs Effort Matrix",
                    xaxis_title="Effort (1=Mudah, 10=Sulit)",
                    yaxis_title="Impact (1=Rendah, 10=Tinggi)",
                    showlegend=False,
                    height=500,
                    xaxis=dict(range=[0, 10]),
                    yaxis=dict(range=[0, 10])
                )
                
                st.plotly_chart(fig_priority, use_container_width=True)
                
                # Detailed action plan
                st.markdown("### 📋 Detail Rencana Aksi")
                
                # Sort by ROI
                interventions_sorted = sorted(
                    interventions,
                    key=lambda x: x['impact'] / x['effort'],
                    reverse=True
                )
                
                for i, interv in enumerate(interventions_sorted, 1):
                    roi = interv['impact'] / interv['effort']
                    
                    with st.expander(
                        f"**{i}. {interv['intervensi']}** (ROI: {roi:.2f})"
                    ):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Impact Score", f"{interv['impact']}/10")
                        
                        with col2:
                            st.metric("Timeline", interv['timeline'])
                        
                        with col3:
                            st.metric("Estimasi Biaya", interv['cost'])
                        
                        st.markdown("**Langkah Implementasi:**")
                        for step in interv['detail']:
                            st.markdown(f"• {step}")
        
        st.markdown("---")
        st.markdown("## 🏆 Belajar dari Negara Sukses")
        
        # Calculate progress
        country_progress = df.groupby('country').agg({
            target: ['first', 'last'],
            'year': ['min', 'max']
        })
        country_progress.columns = ['akb_first', 'akb_last', 'year_first', 'year_last']
        country_progress['reduction'] = (
            country_progress['akb_first'] - country_progress['akb_last']
        )
        country_progress['reduction_pct'] = (
            country_progress['reduction'] / country_progress['akb_first'] * 100
        )
        
        success_countries = country_progress.nlargest(10, 'reduction_pct')
        
        st.markdown("### 🌟 Top 10 Negara dengan Penurunan Terbesar")
        
        for idx, (country, data) in enumerate(success_countries.iterrows(), 1):
            with st.expander(
                f"**#{idx} {country}** - Penurunan {data['reduction_pct']:.1f}%"
            ):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "AKB Awal",
                        f"{data['akb_first']:.1f}",
                        f"Tahun {int(data['year_first'])}"
                    )
                
                with col2:
                    st.metric(
                        "AKB Akhir",
                        f"{data['akb_last']:.1f}",
                        f"Tahun {int(data['year_last'])}"
                    )
                
                with col3:
                    st.metric(
                        "Penurunan",
                        f"{data['reduction']:.1f}",
                        f"-{data['reduction_pct']:.1f}%"
                    )
                
                # Success factors
                country_factors_series = df[df['country'] == country]
                if not country_factors_series.empty:
                    factors = country_factors_series.sort_values('year').iloc[-1]
                    
                    st.markdown("**Faktor Keberhasilan:**")
                    
                    success_factors = []
                    
                    if 'dtp3' in factors and factors['dtp3'] > 90:
                        success_factors.append("✅ Cakupan vaksinasi tinggi")
                    
                    if 'stunting' in factors and factors['stunting'] < 20:
                        success_factors.append("✅ Stunting rendah")
                    
                    if 'basic_water' in factors and factors['basic_water'] > 90:
                        success_factors.append("✅ Akses air bersih universal")
                    
                    if 'gdp_per_capita' in factors:
                        if factors['gdp_per_capita'] > df['gdp_per_capita'].median():
                            success_factors.append("✅ Ekonomi kuat")
                    
                    if success_factors:
                        for factor in success_factors:
                            st.markdown(f"• {factor}")
                    else:
                        st.markdown("• Data tidak lengkap")
    else:
        st.warning(f"⚠️ Tidak ada data untuk {selected_country}")

# ==============================================================================
# HALAMAN: KALKULATOR INTERVENSI
# ==============================================================================

elif page == PAGES['calculator']:
    st.markdown('<p class="main-header">🧮 Kalkulator Dampak Intervensi</p>', 
                unsafe_allow_html=True)
    
    with st.expander("🎓 Panduan Memahami AKB", expanded=False):
        st.markdown("""
        ### Apa itu Angka Kematian Balita (U5MR)?
        
        **Definisi:** Jumlah kematian anak usia 0-5 tahun per 1.000 kelahiran 
        hidup dalam satu tahun.
        
        ### 🔍 Penyebab Utama:
        - **Penyakit Infeksi (40%):** Pneumonia, Diare, Malaria
        - **Komplikasi Kelahiran Prematur (18%)**
        - **Malnutrisi (45%):** Stunting, Wasting
        - **Faktor Lingkungan:** Air, sanitasi, polusi
        
        ### 💊 Intervensi Efektif:
        
        | Intervensi | Potensi Penurunan |
        |---|---|
        | Imunisasi lengkap | 20-30% |
        | ASI Eksklusif | 13% |
        | WASH | 15-25% |
        
        ### 🎯 Target SDG 3.2:
        **AKB ≤25 per 1.000 pada tahun 2030**
        """)
    
    st.markdown("## 🧮 Kalkulator Dampak Real-time")
    
    # Input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        population = st.number_input(
            "👶 Populasi Balita:",
            min_value=1000,
            value=1000000,
            step=10000,
            format="%d"
        )
        
        current_u5mr = st.number_input(
            "📊 AKB Saat Ini (per 1.000):",
            min_value=1.0,
            max_value=300.0,
            value=50.0,
            step=0.1
        )
    
    with col2:
        intervention = st.selectbox(
            "💊 Jenis Intervensi:",
            list(INTERVENTIONS.keys()),
            format_func=lambda x: INTERVENTIONS[x]['name']
        )
        
        coverage = st.slider(
            "🎯 Target Cakupan (%):",
            0, 100, 90
        )
    
    # Calculate impact
    reduction_potential = INTERVENTIONS[intervention]['reduction']
    actual_reduction = reduction_potential * (coverage / 100)
    new_u5mr = current_u5mr * (1 - actual_reduction)
    lives_saved = (current_u5mr - new_u5mr) / 1000 * population
    
    st.markdown("### 🎯 Estimasi Dampak")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "AKB Setelah Intervensi",
            f"{new_u5mr:.1f}",
            delta=f"{new_u5mr - current_u5mr:.2f}"
        )
    
    with col2:
        st.metric(
            "Nyawa Diselamatkan/Tahun",
            f"{lives_saved:,.0f}"
        )
    
    with col3:
        st.metric(
            "Penurunan Relatif",
            f"{actual_reduction * 100:.1f}%"
        )
    
    # Long-term projection
    st.markdown("### 📊 Proyeksi 10 Tahun")
    
    projection_data = []
    for year in range(1, 11):
        cumulative_lives = lives_saved * year
        projected_u5mr = current_u5mr * ((1 - actual_reduction) ** year)
        
        projection_data.append({
            'Tahun': year,
            'Nyawa Diselamatkan': cumulative_lives,
            'AKB': projected_u5mr
        })
    
    df_projection = pd.DataFrame(projection_data)
    
    # Dual-axis chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_projection['Tahun'],
        y=df_projection['Nyawa Diselamatkan'],
        name='Nyawa Diselamatkan (Kumulatif)',
        marker_color='lightgreen',
        yaxis='y2'
    ))
    
    fig.add_trace(go.Scatter(
        x=df_projection['Tahun'],
        y=df_projection['AKB'],
        name='Proyeksi AKB',
        line=dict(color='red', width=3),
        yaxis='y'
    ))
    
    fig.update_layout(
        title='Proyeksi Dampak Intervensi (10 Tahun)',
        xaxis_title='Tahun ke-',
        yaxis=dict(title='AKB (per 1.000)'),
        yaxis2=dict(
            title='Nyawa Diselamatkan (Kumulatif)',
            overlaying='y',
            side='right'
        ),
        hovermode='x unified',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    if lives_saved > 0:
        total_lives = df_projection['Nyawa Diselamatkan'].iloc[-1]
        st.success(
            f"💚 Dalam 10 tahun, intervensi ini dapat menyelamatkan "
            f"**{total_lives:,.0f} nyawa anak**!"
        )

# ==============================================================================
# HALAMAN: EARLY WARNING SYSTEM
# ==============================================================================

elif page == PAGES['ews']:
    st.markdown('<p class="main-header">🚨 Sistem Peringatan Dini</p>', 
                unsafe_allow_html=True)
    
    st.info(
        "Sistem ini mengidentifikasi negara yang menunjukkan tanda-tanda "
        "**memburuk** atau **stagnan** dalam upaya menurunkan AKB."
    )
    
    # Get alerts
    df_alerts = get_ews_alerts(df, target)
    
    # Filter controls
    st.markdown("### 🎚️ Filter Peringatan")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        show_critical = st.checkbox("🔴 KRITIS", value=True)
    
    with col2:
        show_warning = st.checkbox("🟠 PERINGATAN", value=True)
    
    with col3:
        show_attention = st.checkbox("🟡 PERHATIAN", value=True)
    
    with col4:
        st.metric("Total Peringatan", len(df_alerts))
    
    # Filter alerts
    levels_to_show = []
    if show_critical:
        levels_to_show.append("🔴 KRITIS")
    if show_warning:
        levels_to_show.append("🟠 PERINGATAN")
    if show_attention:
        levels_to_show.append("🟡 PERHATIAN")
    
    filtered_alerts = df_alerts[df_alerts['level'].isin(levels_to_show)]
    
    # Sort by severity
    severity_map = {"🔴 KRITIS": 3, "🟠 PERINGATAN": 2, "🟡 PERHATIAN": 1}
    filtered_alerts['severity_score'] = filtered_alerts['level'].map(severity_map)
    filtered_alerts = filtered_alerts.sort_values(
        ['severity_score', 'akb_terkini'],
        ascending=[False, False]
    )
    
    # Display alerts
    st.markdown(f"### 📋 Daftar Negara ({len(filtered_alerts)})")
    
    if not filtered_alerts.empty:
        for _, row in filtered_alerts.iterrows():
            with st.expander(
                f"{row['level']} - **{row['negara']}** (AKB: {row['akb_terkini']:.1f})"
            ):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("AKB Terkini", f"{row['akb_terkini']:.1f}")
                
                with col2:
                    st.metric(
                        "Perubahan Tahunan",
                        f"{row['perubahan_tahunan']:.2f}",
                        delta_color="inverse"
                    )
                
                with col3:
                    st.metric(
                        "Tren 5 Tahun",
                        f"{row['tren_5_tahun']:.2f}",
                        delta_color="inverse"
                    )
                
                st.warning(f"**Alasan:** {row['alasan']}")
                
                # Country trend
                country_data = df[df['country'] == row['negara']].sort_values('year')
                
                fig_trend = px.line(
                    country_data,
                    x='year',
                    y=target,
                    title=f"Tren AKB - {row['negara']}",
                    markers=True
                )
                
                fig_trend.add_hline(
                    y=25,
                    line_dash="dash",
                    line_color="green",
                    annotation_text="Target SDG 2030"
                )
                
                fig_trend.update_layout(height=300)
                st.plotly_chart(fig_trend, use_container_width=True)
                
                # Recommendations
                st.markdown("**🎯 Rekomendasi Aksi Cepat:**")
                
                if "KRITIS" in row['level'] or row['akb_terkini'] > 75:
                    st.markdown("""
                    - 🚨 **Emergency Response**: Mobilisasi bantuan internasional
                    - 💉 Kampanye vaksinasi massal segera
                    - 🍲 Distribusi makanan bergizi darurat
                    - 🏥 Penguatan sistem kesehatan primer
                    """)
                elif "PERINGATAN" in row['level']:
                    st.markdown("""
                    - 🔍 Investigasi mendalam penyebab peningkatan
                    - 📊 Audit program kesehatan ibu dan anak
                    - 🏥 Perkuat kapasitas fasilitas kesehatan
                    - 🤝 Kolaborasi dengan stakeholder lokal
                    """)
                else:
                    st.markdown("""
                    - 📈 Review dan evaluasi strategi saat ini
                    - 💡 Adopsi best practice dari negara sukses
                    - 🤝 Tingkatkan koordinasi antar-sektor
                    - 📊 Perbaiki sistem monitoring dan evaluasi
                    """)
    else:
        st.info("✅ Tidak ada negara yang memenuhi kriteria filter.")
    
    st.markdown("---")
    st.markdown("### 🗺️ Peta Peringatan Dini Global")
    
    # Create map data
    df_map_alert = df[df['year'] == df['year'].max()].copy()
    df_map_alert = df_map_alert.merge(
        df_alerts[['negara', 'level']],
        left_on='country',
        right_on='negara',
        how='left'
    )
    df_map_alert['level'].fillna('✅ BAIK', inplace=True)
    
    # Create choropleth
    fig_alert_map = px.choropleth(
        df_map_alert,
        locations='country',
        locationmode='country names',
        color='level',
        color_discrete_map=ALERT_COLORS,
        hover_data={target: ':.1f', 'level': True},
        title='Peta Status Peringatan Global'
    )
    
    fig_alert_map.update_layout(
        height=600,
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='natural earth'
        )
    )
    
    st.plotly_chart(fig_alert_map, use_container_width=True)
    
    # Summary statistics
    st.markdown("### 📊 Ringkasan Statistik")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        critical_pct = len(df_alerts[df_alerts['level'] == '🔴 KRITIS']) / len(df_alerts) * 100
        st.metric(
            "% Negara Kritis",
            f"{critical_pct:.1f}%",
            help="Persentase negara dengan status kritis"
        )
    
    with col2:
        avg_change = df_alerts['perubahan_tahunan'].mean()
        st.metric(
            "Rata-rata Perubahan",
            f"{avg_change:.2f}",
            delta_color="inverse"
        )
    
    with col3:
        worsening = len(df_alerts[df_alerts['perubahan_tahunan'] > 0])
        st.metric(
            "Negara Memburuk",
            worsening,
            help="AKB meningkat tahun-ke-tahun"
        )
    
    with col4:
        stagnant = len(df_alerts[
            (abs(df_alerts['perubahan_tahunan']) < 0.5) & 
            (df_alerts['akb_terkini'] > 25)
        ])
        st.metric(
            "Negara Stagnan",
            stagnant,
            help="Tidak ada perubahan signifikan"
        )

# ==============================================================================
# FOOTER
# ==============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>Dashboard Analisis Kematian Balita</strong></p>
    <p>Dikembangkan oleh Kelompok 19 Project Data and Analysis | Universitas Negeri Jakarta</p>
    <p style="font-size: 0.9em;">
        Data bersumber dari WHO, UNICEF, dan World Bank | 
        Model: Random Forest & Prophet
    </p>
</div>
""", unsafe_allow_html=True)