🧠 Analisis Kematian Balita Global & Dashboard Interaktif

Penulis: Kelompok 19 Proyek Data & Analisis

Institusi: Universitas Negeri Jakarta

📝 Deskripsi Proyek

Proyek ini bertujuan untuk menganalisis faktor-faktor kunci yang mempengaruhi Angka Kematian Balita (AKB) atau Under-five Mortality Rate (U5MR) di seluruh dunia. Dengan menggunakan data dari WHO, UNICEF, dan World Bank, proyek ini mencakup seluruh alur kerja data science: mulai dari pembersihan dan penggabungan data, analisis data eksploratif (EDA), pemodelan prediktif dengan Machine Learning, hingga peramalan tren masa depan.

Hasil akhir dari proyek ini adalah sebuah dashboard web interaktif yang dibangun menggunakan Streamlit. Dashboard ini memungkinkan pengguna untuk mengeksplorasi data, memahami faktor risiko, dan mensimulasikan dampak dari intervensi kebijakan (seperti peningkatan cakupan vaksinasi atau penurunan stunting) terhadap angka kematian balita.

✨ Fitur Utama

Analisis Data Komprehensif: Menggabungkan dan membersihkan data dari berbagai sumber (mortalitas, imunisasi, nutrisi).

Pemodelan Prediktif: Menggunakan model RandomForestRegressor untuk memprediksi U5MR dan mengidentifikasi faktor-faktor paling berpengaruh (R² > 0.92).

Peramalan Tren Jangka Panjang: Menggunakan model Prophet untuk meramal tren U5MR hingga 30-50 tahun ke depan untuk negara mana pun.

Simulasi Intervensi Hibrida: Menggabungkan kekuatan Prophet dan Random Forest untuk mensimulasikan dampak perubahan kebijakan (misalnya, menaikkan cakupan vaksin) pada U5MR di masa depan.

Dashboard Interaktif: Semua temuan disajikan dalam aplikasi Streamlit yang mudah digunakan, memungkinkan eksplorasi data, perbandingan negara, dan simulasi secara real-time.

🚀 Tampilan Dashboard

Berikut adalah tampilan halaman utama dari dashboard interaktif yang dihasilkan:

🛠️ Teknologi yang Digunakan

Bahasa: Python 3.10+

Analisis Data: Pandas, NumPy

Machine Learning: Scikit-learn (RandomForest, Pipeline), Prophet

Visualisasi: Plotly, Matplotlib, Seaborn

Dashboard: Streamlit

Manajemen Model: Joblib

📂 Struktur Proyek

child_mortality_analysis/
├── data/
│   ├── raw/          # File data asli (.xlsx)
│   └── processed/    # File data olahan (.csv)
├── outputs/
│   ├── figures/      # Gambar dan plot yang disimpan
│   ├── models/       # File model yang sudah dilatih (.joblib)
│   └── reports/      # Laporan metrik, insight, & rekomendasi (.csv, .txt)
├── src/
│   ├── config.py     # File konfigurasi path dan parameter
│   ├── dataprep.py   # Skrip untuk persiapan dan penggabungan data
│   ├── eda_notebook.py # Skrip untuk analisis data eksploratif
│   ├── modeling_notebook .py # Skrip untuk melatih model Random Forest
│   ├── evaluation_analysis.py # Skrip untuk evaluasi model mendalam
│   ├── forecasting.py  # Skrip untuk melatih model Prophet
│   └── utils.py      # Fungsi-fungsi bantuan
├── app.py              # Skrip utama untuk menjalankan dashboard
└── README.md           # File ini


⚙️ Instalasi & Cara Menjalankan

Untuk menjalankan proyek ini di komputer lokal Anda, ikuti langkah-langkah berikut:

1. Clone Repositori (Jika ada di Git)

git clone [URL_REPOSITORI_ANDA]
cd child_mortality_analysis


2. Buat Virtual Environment (Sangat Direkomendasikan)

python -m venv venv
venv\Scripts\activate  # Untuk Windows
# source venv/bin/activate  # Untuk macOS/Linux


3. Instal Dependensi
Pastikan Anda memiliki file requirements.txt yang berisi semua library yang dibutuhkan, lalu jalankan:

pip install pandas numpy scikit-learn prophet matplotlib seaborn plotly streamlit joblib openpyxl


(Jika belum ada requirements.txt, perintah di atas akan menginstal semua yang dibutuhkan).

4. Jalankan Pipeline Analisis (Secara Berurutan)

Jalankan skrip-skrip berikut dari terminal, pastikan Anda berada di direktori child_mortality_analysis:

# Langkah 1: Persiapan Data (Wajib)
# Menggabungkan semua file .xlsx menjadi file .csv yang bersih
python src/dataprep.py

# Langkah 2: Latih Model Prediktif (Wajib)
# Melatih model Random Forest dan menyimpannya
python "src/modeling_notebook .py"

# Langkah 3: Latih Model Peramalan (Opsional, tapi dibutuhkan untuk dashboard)
# Membuat plot peramalan per negara menggunakan Prophet
python src/forecasting.py


5. Jalankan Dashboard Streamlit

Setelah semua skrip di atas berhasil dijalankan, luncurkan aplikasi web interaktif:

streamlit run streamlit_app.py


Buka browser Anda dan akses alamat URL yang muncul di terminal (biasanya http://localhost:8501).

📊 Temuan Kunci

Tren Positif: Angka kematian balita global menunjukkan tren penurunan yang signifikan sejak tahun 2000.

Faktor Paling Berpengaruh: Model mengidentifikasi 3 faktor paling dominan yang mempengaruhi U5MR:

Region Geografis (terutama berada di Sub-Sahara Afrika).

Tingkat Stunting (korelasi positif kuat).

Cakupan Vaksin DTP3 (korelasi negatif kuat).

Dampak Intervensi: Simulasi menunjukkan bahwa peningkatan cakupan vaksin dan penurunan stunting secara signifikan dapat mengakselerasi penurunan angka kematian balita di masa depan.

📄 Sumber Data

Data yang digunakan dalam proyek ini bersumber dari organisasi internasional terkemuka:

UN IGME (UN Inter-agency Group for Child Mortality Estimation): Data Angka Kematian Balita.

WHO/UNICEF (WUENIC): Data cakupan imunisasi global.

JME (UNICEF/WHO/World Bank Group Joint Child Malnutrition Estimates): Data malnutrisi anak (stunting, overweight).
