# 🚀 Deployment Guide - Child Mortality Analysis Dashboard

Panduan lengkap untuk menjalankan proyek dari awal hingga deployment dashboard.

---

## 📋 Prerequisites

### 1. Software Requirements
- **Python 3.10 atau lebih baru**
- **Jupyter Notebook** (opsional untuk analisis interaktif)
- **Git** (untuk version control)

### 2. Install Dependencies

```bash
# Clone atau download project
cd child_mortality_analysis

# Buat virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install semua dependencies
pip install -r requirements.txt
```

---

## 🗂️ Persiapan Data

### 1. Download Dataset

Download ketiga file Excel berikut dan letakkan di folder `data/raw/`:

1. **Under-five_Mortality_Rates_2024.xlsx** - dari UNICEF/WHO
2. **wuenic2024rev_web-update.xlsx** - dari WHO WUENIC
3. **jme_database_country_model_2025.xlsx** - dari JME

### 2. Struktur Folder

Pastikan struktur folder seperti ini:

```
child_mortality_analysis/
├── data/
│   └── raw/
│       ├── Under-five_Mortality_Rates_2024.xlsx
│       ├── wuenic2024rev_web-update.xlsx
│       └── jme_database_country_model_2025.xlsx
├── src/
├── notebooks/
├── app.py
└── requirements.txt
```

---

## 🎯 Menjalankan Pipeline

#### Step 1: Data Preparation

```bash
python src/dataprep.py
```

Output:
- `data/processed/merged_data_full.csv`
- `data/processed/merged_data_full_yearly_imputed.csv`

#### Step 2: Exploratory Data Analysis

```bash
jupyter notebook notebooks/02_eda.py
```

Atau jalankan semua cells dalam notebook untuk:
- Analisis distribusi data
- Visualisasi trends
- Correlation analysis
- Output: `outputs/figures/` berbagai visualisasi

#### Step 3: Machine Learning Modeling

```bash
jupyter notebook notebooks/03_modeling.ipynb
```

Notebook ini akan:
- Train multiple models
- Hyperparameter tuning
- Save best model
- Output: `outputs/models/best_model.pkl`

#### Step 4: Model Evaluation

```bash
jupyter notebook notebooks/04_evaluation.ipynb
```

Untuk:
- Evaluate model performance
- Error analysis
- Generate insights
- Output: `outputs/reports/`

---

## 🌐 Menjalankan Dashboard

### Local Development

```bash
streamlit run app.py
```

Dashboard akan terbuka di: **http://localhost:8501**

### Features Dashboard:
- 🏠 **Home**: Overview dan metrics utama
- 📊 **Data Explorer**: Explore dan filter data
- 📈 **Trends Analysis**: Analisis temporal
- 🗺️ **Geographic Analysis**: Visualisasi peta
- 🤖 **Predictions**: Model predictions
- 💡 **Insights**: Rekomendasi kebijakan

---

## ☁️ Deployment ke Cloud

### Opsi 1: Streamlit Cloud (Gratis & Mudah)

1. **Push ke GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin <your-repo-url>
   git push -u origin main
   ```

2. **Deploy di Streamlit Cloud**
   - Buka https://streamlit.io/cloud
   - Sign in dengan GitHub
   - Click "New app"
   - Pilih repository Anda
   - Main file: `app.py`
   - Click "Deploy"!

3. **Catatan Penting:**
   - Upload processed data ke GitHub (jangan raw data Excel yang besar)
   - Atau gunakan file data yang sudah di-host online
   - Model files bisa di-upload atau retrain di cloud

### Opsi 2: Heroku

1. **Install Heroku CLI**
   ```bash
   # Download dari: https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Buat file tambahan:**

   `Procfile`:
   ```
   web: streamlit run app.py --server.port=$PORT
   ```

   `setup.sh`:
   ```bash
   mkdir -p ~/.streamlit/
   echo "[server]
   headless = true
   port = $PORT
   enableCORS = false
   " > ~/.streamlit/config.toml
   ```

3. **Deploy:**
   ```bash
   heroku login
   heroku create your-app-name
   git push heroku main
   heroku open
   ```

### Opsi 3: Railway / Render

Similar dengan Heroku tapi lebih mudah:

1. Push code ke GitHub
2. Connect repository ke Railway/Render
3. Set start command: `streamlit run app.py`
4. Deploy!

---

## 🔍 Troubleshooting

### Problem: "Module not found"
**Solution:**
```bash
pip install -r requirements.txt
```

### Problem: "Data file not found"
**Solution:**
- Pastikan file Excel ada di `data/raw/`
- Check nama file exactly seperti di config.py

### Problem: "Model not found" di dashboard
**Solution:**
- Dashboard tetap bisa jalan tanpa model (hanya untuk explorasi data)
- Untuk fitur predictions, run modeling notebook dulu

### Problem: Memory error saat modeling
**Solution:**
- Reduce data size atau sample data
- Gunakan less complex models
- Run di mesin dengan RAM lebih besar

### Problem: Streamlit terlalu lambat
**Solution:**
- Pastikan menggunakan `@st.cache_data` untuk loading data
- Reduce number of data points di visualisasi
- Optimize queries dan filters

---

## 📊 Output Files

Setelah menjalankan pipeline, Anda akan mendapatkan:

```
outputs/
├── figures/
│   ├── target_variable_analysis.png
│   ├── temporal_trend.html
│   ├── correlation_heatmap_full.png
│   ├── model_comparison.png
│   ├── prediction_vs_actual.png
│   ├── feature_importance.png
│   └── ... (dan lainnya)
│
├── reports/
│   ├── feature_summary.csv
│   ├── feature_importance.csv
│   ├── model_performance.csv
│   └── policy_recommendations.txt
│
└── models/
    ├── best_model.pkl
    ├── scaler.pkl
    └── feature_names.pkl
```

---

## 🎓 Tips untuk Presentasi/Demo

1. **Persiapan:**
   - Run dashboard sebelum presentasi untuk memastikan semuanya works
   - Bookmark halaman-halaman penting di dashboard
   - Siapkan beberapa contoh country untuk comparison

2. **Demo Flow:**
   - Start di Home page → show key metrics
   - Data Explorer → demonstrate filtering
   - Trends → show temporal changes
   - Geographic → highlight disparities
   - Predictions → demo model capabilities
   - Insights → present recommendations

3. **Backup Plan:**
   - Simpan screenshots penting
   - Export key visualizations as images
   - Siapkan PDF summary dari notebooks

---

## 🔐 Security & Best Practices

1. **Jangan commit data sensitif:**
   ```bash
   # Tambahkan ke .gitignore
   data/raw/*.xlsx
   *.pkl
   __pycache__/
   *.pyc
   .env
   ```

2. **Use environment variables untuk credentials:**
   ```python
   import os
   API_KEY = os.getenv('API_KEY')
   ```

3. **Version control untuk code, bukan data:**
   - Upload processed data ke cloud storage
   - Load data from URLs dalam production

---

## 🆘 Support

Jika ada masalah:

1. Check error messages carefully
2. Google error message + "streamlit" atau "pandas"
3. Check Streamlit documentation: https://docs.streamlit.io
4. Check project README.md untuk details

---

## ✅ Checklist Deployment

- [ ] All dependencies installed
- [ ] Data files in correct location
- [ ] Data preparation completed
- [ ] Models trained and saved
- [ ] Dashboard runs locally
- [ ] All visualizations working
- [ ] Tested all dashboard features
- [ ] Code pushed to GitHub
- [ ] Deployed to cloud platform
- [ ] Tested cloud deployment
- [ ] Documentation complete

---

## 🎉 Congratulations!

Jika Anda sampai di sini, dashboard Anda seharusnya sudah running!

**Dashboard URL:** `https://your-app-name.streamlit.app` (atau platform lain)

**Next Steps:**
- Share link dashboard dengan supervisor/stakeholders
- Gather feedback untuk improvements
- Keep updating data dan retrain models
- Expand analysis dengan features baru

---

**Happy Analyzing! 🚀📊**

*Created by: Raihan Aprilialdy Risanto*  
*Universitas Negeri Jakarta - Sistem dan Informasi Teknologi*