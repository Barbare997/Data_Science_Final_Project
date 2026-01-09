"""
Data Processing Module for Urban Pulse Project

This module contains functions for cleaning, preprocessing, and feature engineering
of traffic volume data.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load traffic volume data from CSV file.
    
    Parameters
    ----------
    file_path : str
        Path to the CSV file containing traffic data
        
    Returns
    -------
    pd.DataFrame
        Raw traffic data
        
    Raises
    ------
    FileNotFoundError
        If the file path does not exist
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✓ Data loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}. Please check the path.")


def inspect_data(df: pd.DataFrame) -> dict:
    """
    Generate a data quality report.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to inspect
        
    Returns
    -------
    dict
        Dictionary containing data quality metrics
    """
    report = {
        'shape': df.shape,
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'dtypes': df.dtypes.to_dict(),
        'duplicates': df.duplicated().sum(),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2
    }
    
    print("=" * 60)
    print("DATA QUALITY REPORT")
    print("=" * 60)
    print(f"Shape: {report['shape'][0]} rows × {report['shape'][1]} columns")
    print(f"\nMissing Values:")
    for col, count in report['missing_values'].items():
        if count > 0:
            print(f"  {col}: {count} ({report['missing_percentage'][col]:.2f}%)")
    print(f"\nDuplicate Rows: {report['duplicates']}")
    print(f"Memory Usage: {report['memory_usage_mb']:.2f} MB")
    print("=" * 60)
    
    return report


def handle_missing_values(df: pd.DataFrame, 
                         strategy: str = 'forward_fill',
                         columns: Optional[list] = None) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    strategy : str, optional
        Strategy for handling missing values:
        - 'forward_fill': Forward fill (good for time series)
        - 'mean': Fill with mean (for numeric columns)
        - 'median': Fill with median (for numeric columns)
        - 'mode': Fill with mode (for categorical columns)
        - 'drop': Drop rows with missing values
    columns : list, optional
        Specific columns to process. If None, processes all columns.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled
        
    Raises
    ------
    ValueError
        If strategy is not recognized
    """
    df_clean = df.copy()
    
    if columns is None:
        columns = df_clean.columns.tolist()
    
    valid_strategies = ['forward_fill', 'mean', 'median', 'mode', 'drop']
    if strategy not in valid_strategies:
        raise ValueError(f"Strategy must be one of {valid_strategies}")
    
    print(f"\nHandling missing values using '{strategy}' strategy...")
    
    for col in columns:
        if df_clean[col].isnull().sum() > 0:
            before = df_clean[col].isnull().sum()
            
            # Special handling for holiday column: use "None" instead of forward fill
            # Holidays are single-day events, so forward fill incorrectly propagates them
            if col == 'holiday':
                df_clean[col].fillna('None', inplace=True)
            elif strategy == 'forward_fill':
                df_clean[col].ffill(inplace=True)
                df_clean[col].bfill(inplace=True)  # Fill remaining with backward fill
            elif strategy == 'mean':
                if df_clean[col].dtype in ['int64', 'float64']:
                    df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            elif strategy == 'median':
                if df_clean[col].dtype in ['int64', 'float64']:
                    df_clean[col].fillna(df_clean[col].median(), inplace=True)
            elif strategy == 'mode':
                mode_value = df_clean[col].mode()[0] if not df_clean[col].mode().empty else None
                if mode_value is not None:
                    df_clean[col].fillna(mode_value, inplace=True)
            elif strategy == 'drop':
                df_clean = df_clean.dropna(subset=[col])
            
            after = df_clean[col].isnull().sum()
            print(f"  {col}: {before} -> {after} missing values")
    
    return df_clean


def detect_outliers_iqr(df: pd.DataFrame, 
                       column: str, 
                       factor: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers using Interquartile Range (IQR) method.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    column : str
        Column name to check for outliers
    factor : float, optional
        IQR multiplier (default: 1.5)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with outlier flags added
    """
    df_out = df.copy()
    
    Q1 = df_out[column].quantile(0.25)
    Q3 = df_out[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    outlier_flag_col = f'{column}_outlier'
    df_out[outlier_flag_col] = (df_out[column] < lower_bound) | (df_out[column] > upper_bound)
    
    outlier_count = df_out[outlier_flag_col].sum()
    print(f"\nOutliers detected in '{column}': {outlier_count} ({outlier_count/len(df)*100:.2f}%)")
    print(f"  Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
    
    return df_out


def handle_outliers(df: pd.DataFrame, 
                   column: str, 
                   method: str = 'cap',
                   factor: float = 1.5) -> pd.DataFrame:
    """
    Handle outliers in a numeric column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    column : str
        Column name containing outliers
    method : str, optional
        Method to handle outliers:
        - 'cap': Cap outliers at IQR bounds
        - 'remove': Remove rows with outliers
    factor : float, optional
        IQR multiplier (default: 1.5)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with outliers handled
    """
    df_clean = df.copy()
    
    Q1 = df_clean[column].quantile(0.25)
    Q3 = df_clean[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    if method == 'cap':
        df_clean[column] = df_clean[column].clip(lower=lower_bound, upper=upper_bound)
        print(f"✓ Capped outliers in '{column}' at [{lower_bound:.2f}, {upper_bound:.2f}]")
    elif method == 'remove':
        before = len(df_clean)
        df_clean = df_clean[(df_clean[column] >= lower_bound) & (df_clean[column] <= upper_bound)]
        after = len(df_clean)
        print(f"✓ Removed {before - after} outlier rows from '{column}'")
    
    return df_clean


def parse_datetime(df: pd.DataFrame, 
                  date_column: str = 'date_time') -> pd.DataFrame:
    """
    Parse datetime column and extract temporal features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    date_column : str, optional
        Name of the datetime column (default: 'date_time')
        
    Returns
    -------
    pd.DataFrame
        DataFrame with parsed datetime and temporal features
    """
    df_parsed = df.copy()
    
    # Parse datetime
    if date_column in df_parsed.columns:
        df_parsed[date_column] = pd.to_datetime(df_parsed[date_column], errors='coerce')
        
        # Extract temporal features
        df_parsed['year'] = df_parsed[date_column].dt.year
        df_parsed['month'] = df_parsed[date_column].dt.month
        df_parsed['day'] = df_parsed[date_column].dt.day
        df_parsed['hour'] = df_parsed[date_column].dt.hour
        df_parsed['day_of_week'] = df_parsed[date_column].dt.dayofweek  # 0=Monday, 6=Sunday
        df_parsed['is_weekend'] = df_parsed['day_of_week'].isin([5, 6]).astype(int)
        
        print(f"✓ Parsed datetime column '{date_column}' and extracted temporal features")
    
    return df_parsed


def create_rush_hour_feature(df: pd.DataFrame, 
                            hour_column: str = 'hour') -> pd.DataFrame:
    """
    Create rush hour feature based on hour of day.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    hour_column : str, optional
        Name of the hour column (default: 'hour')
        
    Returns
    -------
    pd.DataFrame
        DataFrame with rush hour feature added
    """
    df_feat = df.copy()
    
    # Define rush hours: 7-9 AM (morning) and 5-7 PM (evening)
    df_feat['is_rush_hour'] = df_feat[hour_column].isin([7, 8, 9, 17, 18, 19]).astype(int)
    
    # More granular rush hour classification
    df_feat['rush_hour_type'] = 'normal'
    df_feat.loc[df_feat[hour_column].isin([7, 8, 9]), 'rush_hour_type'] = 'morning_rush'
    df_feat.loc[df_feat[hour_column].isin([17, 18, 19]), 'rush_hour_type'] = 'evening_rush'
    
    print("✓ Created rush hour features")
    
    return df_feat


def create_holiday_feature(df: pd.DataFrame,
                          holiday_column: str = 'holiday') -> pd.DataFrame:
    """
    Create binary holiday feature from holiday column.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    holiday_column : str, optional
        Name of the holiday column (default: 'holiday')
        
    Returns
    -------
    pd.DataFrame
        DataFrame with is_holiday binary feature added
    """
    df_feat = df.copy()
    
    # Create binary feature: 1 if holiday, 0 if None
    df_feat['is_holiday'] = (df_feat[holiday_column] != 'None').astype(int)
    
    holiday_count = df_feat['is_holiday'].sum()
    print(f"✓ Created is_holiday feature: {holiday_count} holiday observations")
    
    return df_feat


def create_traffic_stress_level(df: pd.DataFrame, 
                                volume_column: str = 'traffic_volume',
                                method: str = 'quantile') -> pd.DataFrame:
    """
    Create traffic stress level classification (Low/Medium/High).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    volume_column : str, optional
        Name of the traffic volume column (default: 'traffic_volume')
    method : str, optional
        Method for classification:
        - 'quantile': Use 33rd and 67th percentiles
        - 'fixed': Use fixed thresholds
        
    Returns
    -------
    pd.DataFrame
        DataFrame with traffic_stress_level and is_congested features
    """
    df_feat = df.copy()
    
    if method == 'quantile':
        low_threshold = df_feat[volume_column].quantile(0.33)
        high_threshold = df_feat[volume_column].quantile(0.67)
    else:  # fixed thresholds (example values, adjust based on your data)
        low_threshold = 2000
        high_threshold = 4000
    
    # Create 3-level classification
    df_feat['traffic_stress_level'] = 'Low'
    df_feat.loc[df_feat[volume_column] >= low_threshold, 'traffic_stress_level'] = 'Medium'
    df_feat.loc[df_feat[volume_column] >= high_threshold, 'traffic_stress_level'] = 'High'
    
    # Create binary classification (for ML models)
    df_feat['is_congested'] = (df_feat[volume_column] >= high_threshold).astype(int)
    
    print(f"✓ Created traffic stress levels:")
    print(f"  Low: < {low_threshold:.0f}")
    print(f"  Medium: {low_threshold:.0f} - {high_threshold:.0f}")
    print(f"  High: >= {high_threshold:.0f}")
    
    return df_feat


def encode_categorical_features(df: pd.DataFrame, 
                               columns: Optional[list] = None,
                               method: str = 'one_hot') -> pd.DataFrame:
    """
    Encode categorical features for machine learning.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    columns : list, optional
        List of categorical columns to encode. If None, auto-detect.
    method : str, optional
        Encoding method: 'one_hot' or 'label'
        
    Returns
    -------
    pd.DataFrame
        DataFrame with encoded categorical features
    """
    df_encoded = df.copy()
    
    if columns is None:
        # Auto-detect categorical columns
        columns = df_encoded.select_dtypes(include=['object']).columns.tolist()
        # Exclude datetime columns
        columns = [col for col in columns if 'date' not in col.lower()]
    
    print(f"\nEncoding {len(columns)} categorical columns using '{method}' method...")
    
    for col in columns:
        if col in df_encoded.columns:
            if method == 'one_hot':
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded.drop(columns=[col], inplace=True)
                print(f"  {col}: Created {len(dummies.columns)} one-hot encoded columns")
            elif method == 'label':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                print(f"  {col}: Label encoded")
    
    return df_encoded


def preprocess_pipeline(df: pd.DataFrame,
                       target_column: str = 'traffic_volume',
                       date_column: str = 'date_time',
                       missing_strategy: str = 'forward_fill',
                       outlier_method: str = 'cap') -> Tuple[pd.DataFrame, dict]:
    """
    Preprocessing pipeline for traffic data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame
    target_column : str, optional
        Name of the target variable column (default: 'traffic_volume')
    date_column : str, optional
        Name of the datetime column (default: 'date_time')
    missing_strategy : str, optional
        Strategy for handling missing values (default: 'forward_fill')
    outlier_method : str, optional
        Method for handling outliers (default: 'cap')
        
    Returns
    -------
    Tuple[pd.DataFrame, dict]
        Processed DataFrame and preprocessing report
    """
    print("=" * 60)
    print("STARTING DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Initial inspection
    initial_report = inspect_data(df)
    
    # Step 2: Parse datetime
    df_processed = parse_datetime(df, date_column=date_column)
    
    # Step 3: Handle missing values
    df_processed = handle_missing_values(df_processed, strategy=missing_strategy)
    
    # Step 4: Handle outliers in numeric columns
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        df_processed = handle_outliers(df_processed, target_column, method=outlier_method)
    
    # Step 5: Create derived features
    df_processed = create_rush_hour_feature(df_processed)
    df_processed = create_holiday_feature(df_processed)
    df_processed = create_traffic_stress_level(df_processed, volume_column=target_column)
    
    # Step 6: Final inspection
    final_report = inspect_data(df_processed)
    
    preprocessing_report = {
        'initial_shape': initial_report['shape'],
        'final_shape': final_report['shape'],
        'rows_removed': initial_report['shape'][0] - final_report['shape'][0],
        'missing_values_before': initial_report['missing_values'],
        'missing_values_after': final_report['missing_values'],
        'features_created': [
            'year', 'month', 'day', 'hour', 'day_of_week', 'is_weekend',
            'is_rush_hour', 'rush_hour_type', 'is_holiday', 
            'traffic_stress_level', 'is_congested'
        ]
    }
    
    print("\n" + "=" * 60)
    print("PREPROCESSING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Initial rows: {preprocessing_report['initial_shape'][0]}")
    print(f"Final rows: {preprocessing_report['final_shape'][0]}")
    print(f"Features created: {len(preprocessing_report['features_created'])}")
    print("=" * 60)
    
    return df_processed, preprocessing_report


def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Load and clean data in one step.
    
    Parameters
    ----------
    file_path : str
        Path to the raw data CSV file
        
    Returns
    -------
    pd.DataFrame
        Cleaned and preprocessed DataFrame
    """
    df_raw = load_data(file_path)
    df_clean, _ = preprocess_pipeline(df_raw)
    return df_clean

