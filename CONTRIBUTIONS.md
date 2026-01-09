# Team Contributions

## Project: Urban Pulse – Predicting City Traffic Stress

### Team Members
- Tamar Shonia
- Barbare Pantskhava

---

## Contribution Breakdown

### Tamar Shonia
**Role**: Data Engineer / Preprocessing Specialist

**Contributions**:
- Set up project structure
- Implemented data preprocessing pipeline (`src/data_processing.py`)
- Created data cleaning functions with error handling for file loading and input validation
- Developed feature engineering logic (rush hour detection, day type classification, traffic stress levels)
- Handled missing values and outlier detection using IQR method
- Created data quality reports and documentation
- Created `notebooks/01_data_exploration.ipynb`
- Created `notebooks/02_data_preprocessing.ipynb`
- Wrote data dictionary documentation (`data/DATA_DICTIONARY.md`)
- Created static visualization functions (`src/visualization.py`)
- Created `notebooks/03_eda_visualization.ipynb`
- Performed statistical analysis and correlation studies
- Identified key patterns and insights from exploratory data analysis

**Files Modified/Created**:
- `src/data_processing.py`
- `src/visualization.py`
- `notebooks/01_data_exploration.ipynb`
- `notebooks/02_data_preprocessing.ipynb`
- `notebooks/03_eda_visualization.ipynb`
- `data/DATA_DICTIONARY.md`
- `README.md` (data processing and dataset sections)
- `reports/figures/` (static visualization outputs)

---

### Barbare Pantskhava
**Role**: ML Engineer / Visualization Specialist

**Contributions**:
- Added interactive visualization functions to visualization module (Plotly)
- Created `notebooks/05_interactive_visualizations.ipynb`
- Implemented Logistic Regression model
- Implemented Decision Tree model
- Implemented Random Forest model (bonus third model)
- Conducted model evaluation and comparison
- Created model utility functions (`src/models.py`)
- Created `notebooks/04_machine_learning.ipynb`
- Generated model performance reports and metrics
- Developed Streamlit interactive dashboard (`dashboard.py`)
- Created README.md with results and methodology
- Ensured code quality and PEP 8 compliance

**Files Modified/Created**:
- `src/visualization.py` (added interactive functions)
- `src/models.py`
- `dashboard.py`
- `notebooks/04_machine_learning.ipynb`
- `notebooks/05_interactive_visualizations.ipynb`
- `README.md` (main documentation)
- `reports/figures/` (interactive visualization outputs and model outputs)

---

## Collaboration Notes

### Communication
- **Meetings**: Regular team meetings to coordinate work and review progress
- **Tools**: Regular communication for integration and code sharing
- **Timeline**: Followed project timeline as outlined in guidelines

### Code Review Process
- All code was reviewed by the other team member before finalizing
- Ensured consistency in coding style and documentation
- Verified integration between data processing and analysis components

### Workflow
- **Phase 1**: Tamar Shonia focused on data exploration and preprocessing pipeline
- **Phase 2**: Tamar Shonia created static visualizations while Barbare Pantskhava began ML model development
- **Phase 3**: Barbare Pantskhava completed ML models, interactive visualizations, and dashboard
- **Integration**: Both members collaborated on final documentation and project integration

### Challenges Overcome
- Coordinated data preprocessing output format for compatibility with visualization and modeling functions
- Integrated static and interactive visualizations in same module
- Ensured consistent feature engineering between preprocessing and model training

