"""
Test script to verify all files are in place
Run this before launching the dashboard
"""
from pathlib import Path
import pandas as pd
import sys

def print_header(text):
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)

def check_file(file_path, required=True):
    """Check if file exists and return status"""
    if file_path.exists():
        if file_path.suffix == '.csv':
            try:
                df = pd.read_csv(file_path)
                print(f"   ✅ {file_path.name} ({len(df):,} rows, {len(df.columns)} columns)")
                return True, df
            except Exception as e:
                print(f"   ⚠️  {file_path.name} - Error reading: {e}")
                return False, None
        else:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"   ✅ {file_path.name} ({size_mb:.2f} MB)")
            return True, None
    else:
        status = "❌" if required else "⚠️"
        note = "REQUIRED" if required else "optional"
        print(f"   {status} {file_path.name} - NOT FOUND ({note})")
        return False, None

print_header("🔍 PROJECT SETUP VERIFICATION")

# Base directory
BASE_DIR = Path(__file__).parent

# Check data files
print_header("📊 DATA FILES")

raw_files = [
    (BASE_DIR / 'data' / 'raw' / 'Under-five_Mortality_Rates_2024.xlsx', True),
    (BASE_DIR / 'data' / 'raw' / 'wuenic2024rev_web-update.xlsx', True),
    (BASE_DIR / 'data' / 'raw' / 'jme_database_country_model_2025.xlsx', True),
]

processed_files = [
    (BASE_DIR / 'data' / 'processed' / 'merged_data_full.csv', False),
    (BASE_DIR / 'data' / 'processed' / 'merged_data_full_yearly_imputed.csv', True),
]

print("\n📁 Raw Data:")
raw_ok = all(check_file(f, req)[0] for f, req in raw_files)

print("\n📁 Processed Data:")
processed_ok = False
main_data = None
for file, required in processed_files:
    exists, df = check_file(file, required)
    if exists and df is not None and required:
        processed_ok = True
        main_data = df

# Check model files
print_header("🤖 MODEL FILES")

model_file = BASE_DIR / 'outputs' / 'models' / 'random_forest_u5mr_pipeline.joblib'
model_exists, _ = check_file(model_file, required=False)

alt_model = BASE_DIR / 'outputs' / 'models' / 'best_model.pkl'
if not model_exists:
    alt_exists, _ = check_file(alt_model, required=False)

# Check report files
print_header("📋 REPORT FILES")

report_files = [
    BASE_DIR / 'outputs' / 'reports' / 'feature_importance.csv',
    BASE_DIR / 'outputs' / 'reports' / 'model_performance.csv',
    BASE_DIR / 'outputs' / 'reports' / 'feature_summary.csv'
]

for file in report_files:
    check_file(file, required=False)

# Check source files
print_header("📝 SOURCE FILES")

src_files = [
    (BASE_DIR / 'src' / 'config.py', False),
    (BASE_DIR / 'src' / 'utils.py', False),
    (BASE_DIR / 'src' / 'dataprep.py', True),
    (BASE_DIR / 'app_config.py', True),
    (BASE_DIR / 'streamlit_app.py', True),
    (BASE_DIR / 'requirements.txt', True),
]

src_ok = all(check_file(f, req)[0] for f, req in src_files)

# Test data content
if main_data is not None:
    print_header("🧪 DATA VALIDATION")
    
    print("\n📊 Data Info:")
    print(f"   Shape: {main_data.shape}")
    print(f"   Columns: {len(main_data.columns)}")
    
    # Check for required columns
    print("\n🔍 Required Columns:")
    required_cols = ['country', 'year']
    for col in required_cols:
        if col in main_data.columns:
            unique = main_data[col].nunique()
            print(f"   ✅ {col} ({unique:,} unique values)")
        else:
            print(f"   ❌ {col} - NOT FOUND")
    
    # Check for target column
    print("\n🎯 Target Column:")
    target_candidates = ['under_five_mortality_rate', 'u5mr', 'mortality_rate']
    target_found = None
    
    for candidate in target_candidates:
        if candidate in main_data.columns:
            target_found = candidate
            stats = main_data[candidate].describe()
            print(f"   ✅ {candidate}")
            print(f"      Mean: {stats['mean']:.2f}")
            print(f"      Range: {stats['min']:.2f} - {stats['max']:.2f}")
            break
    
    if not target_found:
        mortality_cols = [c for c in main_data.columns if 'mortality' in c.lower()]
        if mortality_cols:
            print(f"   ⚠️  Standard target not found")
            print(f"      Available: {mortality_cols}")
        else:
            print(f"   ❌ No mortality column found!")
    
    # Check for feature columns
    print("\n🔢 Feature Columns:")
    exclude = ['country', 'iso3', 'region', 'year', target_found] if target_found else ['country', 'year']
    numeric_features = [c for c in main_data.select_dtypes(include=['int64', 'float64']).columns 
                       if c not in exclude]
    print(f"   Found {len(numeric_features)} numeric features")
    if len(numeric_features) > 0:
        print(f"   Examples: {numeric_features[:5]}")
    
    # Check for missing values
    print("\n❓ Missing Values:")
    missing = main_data.isnull().sum().sum()
    if missing == 0:
        print(f"   ✅ No missing values")
    else:
        missing_pct = (missing / (len(main_data) * len(main_data.columns))) * 100
        print(f"   ⚠️  {missing:,} missing values ({missing_pct:.2f}%)")

# Test model loading
if model_exists:
    print_header("🧪 MODEL VALIDATION")
    
    try:
        import joblib
        model = joblib.load(model_file)
        print(f"   ✅ Model loaded successfully")
        print(f"   Type: {type(model).__name__}")
        
        # Check if it's a pipeline
        if hasattr(model, 'named_steps'):
            print(f"   ✅ Model is a sklearn Pipeline")
            steps = list(model.named_steps.keys())
            print(f"   Steps: {steps}")
            
            # Try to get features
            if 'preprocessor' in model.named_steps:
                try:
                    preprocessor = model.named_steps['preprocessor']
                    if hasattr(preprocessor, 'transformers_'):
                        num_features = preprocessor.transformers_[0][2] if len(preprocessor.transformers_) > 0 else []
                        cat_features = preprocessor.transformers_[1][2] if len(preprocessor.transformers_) > 1 else []
                        all_features = list(num_features) + list(cat_features)
                        print(f"   ✅ Features extracted: {len(all_features)}")
                        if len(all_features) > 0:
                            print(f"      Examples: {all_features[:5]}")
                except Exception as e:
                    print(f"   ⚠️  Could not extract features: {e}")
        
    except Exception as e:
        print(f"   ❌ Error loading model: {e}")

# Final summary
print_header("📊 SUMMARY")

print("\n🎯 Status:")
if processed_ok and src_ok:
    print("   ✅ READY TO RUN DASHBOARD")
    print("   Command: streamlit run streamlit_app.py")
elif not processed_ok:
    print("   ❌ Data not ready")
    print("   Run: python src/dataprep.py")
elif not src_ok:
    print("   ❌ Source files missing")
    print("   Check: app_config.py and streamlit_app.py")

if model_exists:
    print("\n   ✅ Model available for predictions")
else:
    print("\n   ⚠️  Model not found (optional)")
    print("   To enable predictions, run: notebooks/03_modeling.ipynb")

# Next steps
print("\n🚀 Next Steps:")
if not processed_ok:
    print("   1. Place Excel files in data/raw/")
    print("   2. Run: python src/dataprep.py")
    print("   3. Run: python test_setup.py (this script)")
elif not model_exists:
    print("   1. ✅ Data ready")
    print("   2. (Optional) Train model: Open notebooks/03_modeling.ipynb")
    print("   3. Launch dashboard: streamlit run streamlit_app.py")
else:
    print("   ✅ Everything ready!")
    print("   Run: streamlit run streamlit_app.py")

print("\n" + "="*70)
print("  For troubleshooting, see: TROUBLESHOOTING.md")
print("="*70 + "\n")

# Exit with appropriate code
if processed_ok and src_ok:
    sys.exit(0)
else:
    sys.exit(1)