"""
Utility functions for Child Mortality Analysis Project
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_excel_file(file_path, sheet_name=0):
    """
    Load an Excel file with error handling
    
    Args:
        file_path: Path to Excel file
        sheet_name: Sheet name or index to load
    
    Returns:
        DataFrame or None if error
    """
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        print(f"✅ Loaded: {Path(file_path).name} - Sheet: {sheet_name}")
        print(f"   Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

def get_data_info(df, name="Dataset"):
    """
    Print comprehensive information about a DataFrame
    
    Args:
        df: pandas DataFrame
        name: Name of the dataset for display
    """
    print(f"\n{'='*60}")
    print(f"📊 {name} Information")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"\nColumns ({len(df.columns)}):")
    print(df.columns.tolist())
    print(f"\nData Types:")
    print(df.dtypes.value_counts())
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing': missing[missing > 0],
        'Percentage': missing_pct[missing > 0]
    }).sort_values('Missing', ascending=False)
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("No missing values!")
    print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

def normalize_country_names(df, country_col='Country'):
    """
    Normalize country names for consistency across datasets
    
    Args:
        df: DataFrame with country column
        country_col: Name of the country column
    
    Returns:
        DataFrame with normalized country names
    """
    df = df.copy()
    
    # Common country name mappings
    country_mapping = {
        'United States of America': 'United States',
        'USA': 'United States',
        'UK': 'United Kingdom',
        'Congo (Democratic Republic of the)': 'Democratic Republic of Congo',
        'Congo (Republic of the)': 'Republic of Congo',
        'Korea, Republic of': 'South Korea',
        'Korea (Democratic People\'s Republic of)': 'North Korea',
        'Iran (Islamic Republic of)': 'Iran',
        'Venezuela (Bolivarian Republic of)': 'Venezuela',
        'Bolivia (Plurinational State of)': 'Bolivia',
        'Tanzania, United Republic of': 'Tanzania',
        'Viet Nam': 'Vietnam',
        'Lao People\'s Democratic Republic': 'Laos',
        'Syrian Arab Republic': 'Syria',
        'Türkiye': 'Turkey',
    }
    
    if country_col in df.columns:
        # Strip whitespace
        df[country_col] = df[country_col].str.strip()
        
        # Apply mapping
        df[country_col] = df[country_col].replace(country_mapping)
    
    return df

def melt_year_columns(df, id_vars, value_name='Value'):
    """
    Melt year columns from wide to long format
    
    Args:
        df: DataFrame with year columns (e.g., 2000, 2001, ...)
        id_vars: List of identifier columns to keep
        value_name: Name for the value column
    
    Returns:
        DataFrame in long format
    """
    # Find year columns (numeric columns that could be years)
    year_cols = [col for col in df.columns if str(col).isdigit() and 1980 <= int(col) <= 2030]
    
    if not year_cols:
        print("⚠️ No year columns found")
        return df
    
    # Melt the dataframe
    df_long = df.melt(
        id_vars=id_vars,
        value_vars=year_cols,
        var_name='Year',
        value_name=value_name
    )
    
    # Convert Year to integer
    df_long['Year'] = df_long['Year'].astype(int)
    
    # Remove rows with missing values
    df_long = df_long.dropna(subset=[value_name])
    
    return df_long

def impute_missing_values(df, method='interpolate', group_by=None):
    """
    Impute missing values in the dataset
    
    Args:
        df: DataFrame with missing values
        method: Imputation method ('interpolate', 'forward_fill', 'backward_fill', 'mean')
        group_by: Column to group by before imputation (e.g., 'Country')
    
    Returns:
        DataFrame with imputed values
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if group_by and group_by in df.columns:
        # Group-wise imputation
        for col in numeric_cols:
            if method == 'interpolate':
                df[col] = df.groupby(group_by)[col].transform(
                    lambda x: x.interpolate(method='linear', limit_direction='both')
                )
            elif method == 'forward_fill':
                df[col] = df.groupby(group_by)[col].ffill()
            elif method == 'backward_fill':
                df[col] = df.groupby(group_by)[col].bfill()
            elif method == 'mean':
                df[col] = df.groupby(group_by)[col].transform(
                    lambda x: x.fillna(x.mean())
                )
    else:
        # Global imputation
        for col in numeric_cols:
            if method == 'interpolate':
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
            elif method == 'forward_fill':
                df[col] = df[col].ffill()
            elif method == 'backward_fill':
                df[col] = df[col].bfill()
            elif method == 'mean':
                df[col] = df[col].fillna(df[col].mean())
    
    return df

def plot_missing_data(df, title="Missing Data Analysis", figsize=(12, 6)):
    """
    Visualize missing data patterns
    
    Args:
        df: DataFrame to analyze
        title: Plot title
        figsize: Figure size
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if len(missing) == 0:
        print("✅ No missing data!")
        return
    
    fig, ax = plt.subplots(figsize=figsize)
    missing.plot(kind='barh', ax=ax, color='salmon')
    ax.set_xlabel('Number of Missing Values')
    ax.set_title(title)
    plt.tight_layout()
    return fig

def create_correlation_heatmap(df, title="Correlation Heatmap", figsize=(14, 10)):
    """
    Create a correlation heatmap for numeric columns
    
    Args:
        df: DataFrame
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib figure
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        print("⚠️ Not enough numeric columns for correlation")
        return None
    
    corr = numeric_df.corr()
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def save_dataframe(df, file_path, index=False):
    """
    Save DataFrame to CSV with confirmation
    
    Args:
        df: DataFrame to save
        file_path: Path to save file
        index: Whether to save index
    """
    try:
        df.to_csv(file_path, index=index)
        print(f"✅ Saved: {Path(file_path).name}")
        print(f"   Shape: {df.shape}")
        print(f"   Location: {file_path}")
    except Exception as e:
        print(f"❌ Error saving {file_path}: {e}")

def create_feature_summary(df):
    """
    Create a summary of all features in the dataset
    
    Args:
        df: DataFrame to summarize
    
    Returns:
        DataFrame with feature statistics
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    summary = pd.DataFrame({
        'dtype': df.dtypes,
        'missing': df.isnull().sum(),
        'missing_pct': (df.isnull().sum() / len(df)) * 100,
        'unique': df.nunique(),
    })
    
    # Add numeric statistics
    for col in numeric_df.columns:
        summary.loc[col, 'mean'] = numeric_df[col].mean()
        summary.loc[col, 'std'] = numeric_df[col].std()
        summary.loc[col, 'min'] = numeric_df[col].min()
        summary.loc[col, 'max'] = numeric_df[col].max()
    
    return summary.sort_values('missing', ascending=False)

print("✅ Utility functions loaded successfully")