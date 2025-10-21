# ⚡ Quick Start Guide

Panduan cepat untuk menjalankan proyek dalam 5 menit!

---

## 🚀 Setup Cepat (5 Menit)

### 1️⃣ Install Dependencies (1 menit)
```bash
pip install -r requirements.txt
```

### 2️⃣ Letakkan Data (1 menit)
Taruh 3 file Excel di folder `data/raw/`:
- Under-five_Mortality_Rates_2024.xlsx
- wuenic2024rev_web-update.xlsx  
- jme_database_country_model_2025.xlsx

### 3️⃣ Run Data Preparation (2 menit)
```bash
python src/dataprep.py
```

### 4️⃣ Launch Dashboard (1 menit)
```bash
streamlit run app.py
```

**Done!** 🎉 Buka browser: http://localhost:8501

---

## 🎯 Command Cheat Sheet

### Data Processing
```bash
# Run data preparation
python src/dataprep.py

# Check processed data
ls data/processed/
```

### Jupyter Notebooks
```bash
# Start Jupyter
jupyter notebook

# Run specific notebook
jupyter nbconvert --execute --to notebook notebooks/03_modeling.ipynb
```

### Dashboard
```bash
# Local
streamlit run app.py

# Custom port
streamlit run app.py --server.port 8502

# Stop server
Ctrl + C
```

### Full Pipeline
```bash
# Run everything
python run_all.py
```

---

## 🔧 Common Commands

### Virtual Environment
```bash
# Create
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

# Deactivate
deactivate
```

### Package Management
```bash
# Install specific package
pip install pandas

# Update all packages
pip install --upgrade -r requirements.txt

# Check installed packages
pip list

# Generate requirements
pip freeze > requirements.txt
```

### Git
```bash
# Initialize
git init

# Add files
git add .

# Commit
git commit -m "Your message"

# Push
git push origin main
```

---

## 📊 Dashboard Pages Explained

| Page | Purpose | Key Features |
|------|---------|--------------|
| 🏠 **Home** | Overview | Key metrics, quick insights |
| 📊 **Data Explorer** | Browse data | Filters, downloads, correlations |
| 📈 **Trends** | Time analysis | Temporal trends, country comparison |
| 🗺️ **Geographic** | Maps | Choropleth, rankings, regional stats |
| 🤖 **Predictions** | ML models | Batch predictions, manual input |
| 💡 **Insights** | Recommendations | Key findings, policy advice |

---

## 🐛 Quick Fixes

### Dashboard tidak muncul?
```bash
# Check Streamlit is installed
streamlit --version

# Try different port
streamlit run app.py --server.port 8502
```

### Import error?
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Data not found?
```bash
# Check file location
ls data/raw/

# Re-run data prep
python src/dataprep.py
```

### Model not found in dashboard?
```bash
# Run modeling notebook first
jupyter notebook notebooks/03_modeling.ipynb

# Or dashboard works without model for data exploration
```

---

## 🎓 Learning Path

### Beginner (Hari 1)
1. ✅ Setup environment
2. ✅ Run data preparation
3. ✅ Launch dashboard
4. ✅ Explore data

### Intermediate (Hari 2-3)
1. ✅ Run EDA notebook
2. ✅ Understand correlations
3. ✅ Analyze trends
4. ✅ Run modeling

### Advanced (Hari 4-5)
1. ✅ Fine-tune models
2. ✅ Customize dashboard
3. ✅ Add new features
4. ✅ Deploy to cloud

---

## 💡 Pro Tips

1. **Always use virtual environment** → Prevents conflicts
2. **Run dataprep first** → Everything depends on it
3. **Dashboard works without models** → Good for EDA
4. **Use Jupyter for interactive analysis** → Better than scripts
5. **Cache data in Streamlit** → Faster performance
6. **Git commit often** → Track your progress
7. **Test locally before deploy** → Save time debugging
8. **Read error messages carefully** → They tell you what's wrong

---

## 📞 Need Help?

1. **Check README.md** → Detailed documentation
2. **Check DEPLOYMENT_GUIDE.md** → Deployment specifics
3. **Google the error** → Usually quick solution
4. **Streamlit docs** → https://docs.streamlit.io
5. **Pandas docs** → https://pandas.pydata.org/docs/

---

## ✅ Pre-Presentation Checklist

- [ ] Dashboard runs locally without errors
- [ ] All visualizations load properly
- [ ] Test filters and interactions
- [ ] Model predictions working
- [ ] Screenshots saved as backup
- [ ] Understand key insights
- [ ] Practice demo flow
- [ ] Internet connection stable (if using cloud data)
- [ ] Backup plan ready

---

## 🎉 Success Indicators

You're ready when:
- ✅ Dashboard loads in under 5 seconds
- ✅ All pages accessible
- ✅ Filters work smoothly
- ✅ Visualizations are clear
- ✅ Predictions are reasonable
- ✅ No error messages
- ✅ Data updates correctly

---

## 🚀 Next Level

After basic deployment:
1. Add authentication (Streamlit Auth)
2. Connect to live data API
3. Add more ML models
4. Create automated reports
5. Set up CI/CD pipeline
6. Add unit tests
7. Optimize performance
8. Create mobile-responsive design

---

**Ready to Start? Run:**
```bash
python run_all.py
```

**🎯 Good Luck!**

*Created with ❤️ for Data Science Excellence*