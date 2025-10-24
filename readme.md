🧠 Dashboard Analisis Kematian Balita

Sebuah dashboard Streamlit interaktif yang dirancang untuk mengeksplorasi data global Angka Kematian Balita (U5MR), menganalisis faktor-faktor risiko, memprediksi tren masa depan, dan mensimulasikan dampak intervensi kebijakan.

Proyek ini dibuat oleh Kelompok 19 Proyek Data dan Analisis dari Universitas Negeri Jakarta.

🚀 Fitur Utama

Dashboard ini dibagi menjadi beberapa modul fungsional:

🏠 Ringkasan Global: Menampilkan statistik kunci (rata-rata, median, min/max) dan tren penurunan AKB global dari tahun 2000-2023.

🗺️ Analisis Geografis: Peta choropleth interaktif untuk melihat sebaran AKB di seluruh dunia, dilengkapi dengan slider tahun untuk melihat perubahan historis.

💡 Analisis Faktor: Menampilkan faktor-faktor yang paling berpengaruh terhadap AKB (seperti stunting dan vaksinasi DTP3) berdasarkan feature importance dari model Machine Learning.

🔮 Peramalan & Simulasi:

Peramalan: Memprediksi tren AKB masa depan untuk negara tertentu menggunakan model Prophet.

Simulasi: Menggunakan model Random Forest yang telah dilatih untuk menjalankan skenario "what-if". Pengguna dapat mengubah slider (misal: target cakupan vaksin atau angka stunting) untuk melihat estimasi dampaknya terhadap AKB.

🔍 Diagnostic Dashboard: Memberikan analisis mendalam per negara untuk mengidentifikasi "akar masalah" (misal: stunting tinggi, sanitasi buruk) dan memberikan rekomendasi intervensi.

🧮 Kalkulator Intervensi: Alat sederhana untuk menghitung estimasi nyawa yang diselamatkan berdasarkan populasi balita dan jenis intervensi yang dipilih.

🚨 Sistem Peringatan Dini (EWS): Mengidentifikasi dan menandai negara-negara yang tren AKB-nya memburuk atau stagnan (tidak ada perbaikan).

🛠️ Instalasi & Penggunaan

Untuk menjalankan dashboard ini di komputer lokal Anda, ikuti langkah-langkah berikut:

1. Clone Repositori

# Ganti dengan URL repo Anda
git clone [https://github.com/Falconxima/child-mortality-dashboard.git](https://github.com/Falconxima/child-mortality-dashboard.git)
cd child-mortality-dashboard


2. Buat Virtual Environment (Sangat Direkomendasikan)

# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate


3. Install Dependencies
Anda perlu membuat file requirements.txt terlebih dahulu. Berdasarkan skrip Anda, isinya kira-kira seperti ini:

(Buat file requirements.txt dan isi dengan ini)

streamlit
pandas
numpy
plotly
prophet
scikit-learn
joblib
streamlit-option-menu


Setelah file itu dibuat, jalankan:

pip install -r requirements.txt


4. Jalankan Aplikasi Streamlit
Pastikan Anda berada di direktori utama (tempat app.py berada), lalu jalankan:

streamlit run app.py


Aplikasi akan otomatis terbuka di browser Anda.

📂 Struktur Proyek

Proyek ini diatur dengan struktur folder yang memisahkan data, kode, dan output.

child-mortality-dashboard/
├── app.py           (Skrip Streamlit utama untuk dashboard)
├── data/
│   ├── raw/         (Data mentah .xlsx dari WHO, UNICEF, dll)
│   └── processed/   (Data bersih .csv hasil dari dataprep.py)
├── outputs/
│   ├── models/      (Model .joblib yang sudah dilatih)
│   └── reports/     (Hasil evaluasi, feature importance .csv)
├── src/
│   ├── config.py    (Konfigurasi path dan variabel global)
│   ├── utils.py     (Fungsi helper, misal: normalisasi nama negara)
│   ├── dataprep.py  (Skrip untuk membersihkan & menggabung data)
│   ├── modeling_notebook .py (Skrip untuk melatih & menyimpan model)
│   ├── forecasting.py (Skrip pengembangan model Prophet)
│   ├── ... (skrip analisis lainnya)
└── requirements.txt (Daftar library Python yang dibutuhkan)


🔄 Metodologi & Alur Kerja

Dashboard ini adalah hasil akhir dari pipeline data yang terdiri dari beberapa langkah:

Persiapan Data (dataprep.py): Tiga set data mentah (Excel) tentang mortalitas, imunisasi, dan nutrisi dibersihkan, diproses, dan digabungkan menjadi satu dataset bersih: merged_data_full_yearly_imputed.csv.

Pelatihan Model (modeling_notebook .py): Dataset bersih digunakan untuk melatih model regresi (Random Forest) untuk memprediksi under_five_mortality_rate berdasarkan fitur-fitur lain (vaksinasi, stunting, dll). Model dievaluasi (menggunakan R² dan RMSE) dan disimpan sebagai random_forest_u5mr_pipeline.joblib.

Pengembangan Analisis (forecasting.py, evaluation_notebook.py): Skrip terpisah digunakan untuk mengembangkan logika peramalan Prophet dan melakukan evaluasi model yang lebih mendalam.

Integrasi Dashboard (app.py): Skrip app.py memuat aset yang sudah jadi (.csv bersih dan model .joblib) dan mengintegrasikan semua logika analisis (termasuk Prophet dan simulasi) ke dalam antarmuka pengguna (UI) yang interaktif menggunakan Streamlit.
