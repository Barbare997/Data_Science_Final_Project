# Data Dictionary - Metro Interstate Traffic Volume Dataset

## Dataset Overview

**Dataset Name**: Metro Interstate Traffic Volume Dataset  
**Location**: `data/raw/Metro_Interstate_Traffic_Volume.csv`  
**Original Source**: UCI ML Repository / Kaggle  
**Description**: Hourly traffic volume data for I-94 Westbound traffic, including weather and temporal features.

## Column Descriptions

### Target Variable

| Column Name | Data Type | Description | Values |
|------------|-----------|-------------|--------|
| `traffic_volume` | int64 | Hourly traffic volume (vehicles per hour) | 0 - 7280 |
| `is_congested` | int64 | Binary classification (derived) | 0 = Not Congested, 1 = Congested |
| `traffic_stress_level` | object | Three-level classification (derived) | Low, Medium, High |

### Temporal Features

| Column Name | Data Type | Description | Values |
|------------|-----------|-------------|--------|
| `date_time` | datetime64 | Timestamp of observation | 2012-10-02 09:00:00 to 2018-09-30 23:00:00 |
| `year` | int64 | Year (derived) | 2012 - 2018 |
| `month` | int64 | Month (derived) | 1 - 12 |
| `day` | int64 | Day of month (derived) | 1 - 31 |
| `hour` | int64 | Hour of day (derived) | 0 - 23 |
| `day_of_week` | int64 | Day of week (derived) | 0 = Monday, 6 = Sunday |
| `is_weekend` | int64 | Weekend flag (derived) | 0 = Weekday, 1 = Weekend |

### Weather Features

| Column Name | Data Type | Description | Values |
|------------|-----------|-------------|--------|
| `temp` | float64 | Temperature in Kelvin | ~240 - 310 K |
| `rain_1h` | float64 | Rainfall in the last hour (mm) | 0.0 - 55.63 |
| `snow_1h` | float64 | Snowfall in the last hour (mm) | 0.0 - 0.51 |
| `clouds_all` | int64 | Cloud cover percentage | 0 - 100 |
| `weather_main` | object | Main weather condition | Clear, Clouds, Rain, Drizzle, Mist, etc. |
| `weather_description` | object | Detailed weather description | Various descriptions |

### Derived Features

| Column Name | Data Type | Description | Values |
|------------|-----------|-------------|--------|
| `is_rush_hour` | int64 | Rush hour flag (derived) | 0 = No, 1 = Yes (7-9 AM, 5-7 PM) |
| `rush_hour_type` | object | Rush hour classification (derived) | normal, morning_rush, evening_rush |

### Other Features

| Column Name | Data Type | Description | Values |
|------------|-----------|-------------|--------|
| `holiday` | object | Holiday indicator | None, Columbus Day, etc. |

## Data Quality Notes

### Missing Values
- Most columns have no missing values
- Some weather columns may have occasional missing values
- Missing values are handled using forward-fill strategy (appropriate for time series)

### Outliers
- Traffic volume outliers detected using IQR method
- Outliers are capped at IQR bounds rather than removed (to preserve data)

### Data Transformations
1. **Datetime Parsing**: `date_time` column parsed to extract temporal features
2. **Feature Engineering**: Created 10+ derived features
3. **Encoding**: Categorical variables encoded for ML models

## Usage Notes

- **Time Series Data**: Data is chronological, so temporal features are important
- **Seasonality**: Consider seasonal patterns (month, day_of_week)
- **Rush Hours**: Strong predictor - 7-9 AM and 5-7 PM show highest traffic
- **Weather Impact**: Adverse weather (rain, snow) typically reduces traffic volume

## References

- Dataset location: `data/raw/Metro_Interstate_Traffic_Volume.csv`
- Original sources: [UCI ML Repository](https://archive.ics.uci.edu/ml/index.php) or [Kaggle](https://www.kaggle.com/datasets)

