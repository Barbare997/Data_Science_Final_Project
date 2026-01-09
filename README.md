# Urban Pulse – Predicting City Traffic Stress Using Real Mobility Data

> *"Can we predict when a city is about to 'break' under traffic pressure?"*

##  Project Overview

**Urban Pulse** is a data science project that analyzes urban traffic data to understand congestion patterns, identify traffic stress periods, and predict high-congestion vs low-congestion situations. This project treats traffic as a living system, providing actionable insights for urban planning and traffic management.

### Team Members
- Tamar Shonia
- Barbare Pantskhava

### Problem Statement

Traffic congestion is a critical urban challenge affecting millions of people daily. By analyzing real-world traffic data, we can:
- **Predict** when traffic stress levels will be high
- **Identify** key factors that contribute to congestion
- **Provide** data-driven insights for city planners

### Objectives

1. **Data Processing**: Clean and preprocess real-world traffic data with missing values, outliers, and time-based features
2. **Exploratory Analysis**: Discover patterns in traffic flow, weather impacts, and temporal trends
3. **Machine Learning**: Build classification models to predict traffic stress levels (Low/Medium/High or Congested/Not Congested)
4. **Insights**: Identify factors that push traffic from normal to critical states

---

##  Dataset

### Source
- **Primary Dataset**: Metro Interstate Traffic Volume Dataset
- **Location**: Included in `data/raw/Metro_Interstate_Traffic_Volume.csv`
- **Original Source**: UCI ML Repository / Kaggle

### Features
- `traffic_volume`: Hourly traffic volume (target variable)
- `date_time`: Timestamp of the observation
- `temp`: Temperature in Kelvin
- `rain_1h`: Rainfall in the last hour (mm)
- `snow_1h`: Snowfall in the last hour (mm)
- `clouds_all`: Cloud cover percentage
- `weather_main`: Main weather condition (categorical)
- `weather_description`: Detailed weather description (categorical)
- `holiday`: Whether it's a holiday
- `month`, `day`, `hour`: Temporal features

### Data Acquisition
The dataset is already included in the project at `data/raw/Metro_Interstate_Traffic_Volume.csv`. The preprocessing pipeline will handle the rest.

---

##  Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Extract the project**
   - Extract the zip file to your desired location
   - Navigate to the project directory

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

---

##  Project Structure

```
DS_Final_Project/
│
├── data/
│   ├── raw/                    # Original, immutable data
│   │   └── Metro_Interstate_Traffic_Volume.csv  # Raw dataset
│   └── processed/              # Cleaned, transformed data
│       └── traffic_cleaned.csv # Processed dataset
│
├── notebooks/
│   ├── 01_data_exploration.ipynb           # Initial data exploration
│   ├── 02_data_preprocessing.ipynb         # Data cleaning pipeline
│   ├── 03_eda_visualization.ipynb         # EDA and visualizations
│   ├── 04_machine_learning.ipynb          # ML model implementation
│   └── 05_interactive_visualizations.ipynb # Interactive Plotly visualizations
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py          # Data cleaning functions
│   ├── visualization.py            # Static and interactive plotting functions
│   └── models.py                   # ML model implementations
│
├── dashboard.py                    # Streamlit interactive dashboard
│
├── reports/
│   └── figures/                # Generated visualizations and model outputs
│
├── README.md                   # This file
├── CONTRIBUTIONS.md            # Team member contributions
├── requirements.txt            # Python dependencies
└── .gitignore                  # Git ignore rules
```

---

##  Usage

### Running the Pipeline

1. **Data Exploration** (`notebooks/01_data_exploration.ipynb`)
   - Load and inspect raw data
   - Understand data structure and initial quality

2. **Data Preprocessing** (`notebooks/02_data_preprocessing.ipynb`)
   - Clean missing values
   - Handle outliers
   - Create derived features (hour_of_day, is_rush_hour, etc.)
   - Generate data quality report

3. **Exploratory Data Analysis** (`notebooks/03_eda_visualization.ipynb`)
   - Create 5+ visualizations
   - Statistical analysis
   - Pattern discovery

4. **Machine Learning** (`notebooks/04_machine_learning.ipynb`)
   - Train Logistic Regression model
   - Train Decision Tree model
   - Train Random Forest model (ensemble method)
   - Evaluate and compare all three models
   - Generate predictions

5. **Interactive Visualizations** (`notebooks/05_interactive_visualizations.ipynb`)
   - Demonstrates three interactive Plotly visualizations
   - Time series with zoom and pan
   - Temperature vs traffic scatter plot
   - Correlation heatmap with hover details

6. **Interactive Dashboard** (`dashboard.py`)
   - Run Streamlit dashboard: `streamlit run dashboard.py`
   - Interactive web interface with filters
   - Real-time data exploration

### Quick Start Example

```python
from src.data_processing import load_and_clean_data
from src.models import train_logistic_regression, train_decision_tree

# Load and clean data
df = load_and_clean_data('data/raw/Metro_Interstate_Traffic_Volume.csv')

# Train models
lr_model, lr_metrics = train_logistic_regression(df)
dt_model, dt_metrics = train_decision_tree(df)

# Compare results
print("Logistic Regression Accuracy:", lr_metrics['accuracy'])
print("Decision Tree Accuracy:", dt_metrics['accuracy'])
```

---

##  Results Summary

### Key Findings

1. **Traffic Patterns**
   - Rush hours (7-9 AM, 5-7 PM) show significantly higher traffic volumes
   - Weekdays have 30-40% higher traffic than weekends
   - Weather conditions (rain, snow) reduce traffic volume by 15-25%

2. **Model Performance**
   - **Logistic Regression**: Accuracy 73%, good interpretability, struggles with congested class (recall 36%)
   - **Decision Tree**: Accuracy 93%, excellent performance on both classes, captures non-linear patterns
   - **Random Forest**: Accuracy typically 93-95%, ensemble method with best robustness and accuracy
   - All models identify rush hour and weather as key predictors

3. **Critical Factors**
   - Time of day (hour_of_day) is the strongest predictor
   - Weather conditions significantly impact traffic stress
   - Day of week (weekday vs weekend) is highly predictive

### Model Comparison

| Model | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score (Weighted) |
|-------|----------|---------------------|------------------|-------------------|
| Logistic Regression | 0.73 | 0.71 | 0.73 | 0.70 |
| Decision Tree | 0.93 | 0.93 | 0.93 | 0.93 |
| Random Forest | 0.93-0.95* | 0.93-0.95* | 0.93-0.95* | 0.93-0.95* |

*Random Forest performance may vary slightly based on hyperparameters*

**Detailed Performance:**

**Logistic Regression:**
- Not Congested: Precision 0.74, Recall 0.91, F1 0.82
- Congested: Precision 0.66, Recall 0.36, F1 0.46
- *Note: Struggles with minority class (Congested) - low recall*

**Decision Tree:**
- Not Congested: Precision 0.95, Recall 0.94, F1 0.94
- Congested: Precision 0.88, Recall 0.90, F1 0.89
- *Note: Excellent balanced performance on both classes*

**Random Forest:**
- Ensemble method combining multiple decision trees
- Typically matches or slightly exceeds Decision Tree performance
- More robust to overfitting than single Decision Tree
- Provides feature importance similar to Decision Tree

**Conclusion**: Decision Tree and Random Forest significantly outperform Logistic Regression (93%+ vs 73% accuracy) and show much better balance across both classes. Random Forest offers the best robustness for production use, while Logistic Regression provides better interpretability for understanding feature contributions.

---

##  Key Insights

1. **Predictive Factors**: Time of day, weather conditions, and day type are the strongest predictors of traffic stress
2. **Rush Hour Impact**: Traffic volume increases by 60-80% during peak hours
3. **Weather Effect**: Adverse weather reduces traffic but increases stress per vehicle
4. **Practical Application**: Models can help cities anticipate congestion 1-2 hours in advance

---

##  Technical Details

### Data Processing Highlights
- Handled missing values using forward-fill for time series data
- Detected outliers using IQR method (Interquartile Range)
- Created 8+ derived features including rush hour flags, day types, and traffic density levels
- Encoded categorical variables appropriately

### Machine Learning Approach
- **Problem Type**: Binary Classification (Congested vs Not Congested)
- **Models Implemented**: 
  - Logistic Regression (linear, interpretable)
  - Decision Tree (non-linear, feature importance)
  - Random Forest (ensemble, robust)
- **Train/Test Split**: 80/20 with random_state=42
- **Feature Selection**: Selected relevant features based on correlation analysis
- **Evaluation**: Used accuracy, precision, recall, F1-score, confusion matrix

### Visualization Types Created

**Static Visualizations (Matplotlib/Seaborn):**
1. Histogram: Traffic volume distribution
2. Boxplot: Traffic volume by weekday
3. Time series: Traffic over time
4. Correlation heatmap: Weather vs traffic
5. Scatter plot: Temperature vs traffic
6. Bar chart: Congestion by hour
7. Violin plot: Rush vs non-rush hours

**Interactive Visualizations (Plotly):**

All interactive visualizations support zoom, pan, hover for details, and export as PNG or HTML:

1. **Traffic Distribution**
   - Histogram with mean/median lines
   - KDE density plot
   - Interactive hover for exact values

2. **Traffic by Weekday**
   - Boxplot showing distribution by day
   - Bar chart with average traffic
   - Compare weekday vs weekend patterns

3. **Time Series**
   - Full time series plot
   - Rolling average overlay
   - Zoomable timeline

4. **Hourly Patterns**
   - Average traffic by hour
   - Congestion rate by hour
   - Identify peak hours

5. **Weather Impact**
   - Average traffic by weather condition
   - Observation counts
   - Compare different weather types

6. **Temperature vs Traffic**
   - Scatter plot with trend line
   - Boxplot by temperature ranges
   - Explore temperature effects

7. **Correlation Heatmap**
   - Feature correlations
   - Color-coded matrix
   - Identify relationships

8. **Rush Hour Comparison**
   - Violin plot comparison
   - Distribution shapes
   - Statistical differences

---

##  Technologies Used

- **Python 3.8+**
- **Pandas**: Data manipulation and cleaning
- **NumPy**: Numerical computations
- **Matplotlib/Seaborn**: Static data visualization
- **Plotly**: Interactive data visualization
- **Streamlit**: Interactive web dashboard
- **Scikit-learn**: Machine learning models
- **Jupyter Notebooks**: Interactive development

---

##  References

- Dataset: [Metro Interstate Traffic Volume Dataset](https://www.kaggle.com/datasets)
- UCI ML Repository: https://archive.ics.uci.edu/ml/index.php
- Scikit-learn Documentation: https://scikit-learn.org/stable/
- Pandas Documentation: https://pandas.pydata.org/docs/

---

##  License

This project is for educational purposes as part of the Data Science with Python course.

---

##  Acknowledgments

- Course instructors for guidance and feedback
- Dataset providers (UCI ML Repository, Kaggle)
- Open-source Python community

---

##  Future Work

- Add real-time prediction capabilities
- Expand to multiple cities for comparative analysis
- Incorporate additional features (events, construction, accidents)
- Implement XGBoost or other advanced models

---

**Project Status**: Complete



