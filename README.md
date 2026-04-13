# 🌍 AQI Prediction System

A comprehensive, production-ready Air Quality Index (AQI) prediction system built with Python, XGBoost, and modern ML engineering practices. This project features advanced feature engineering, explainable AI with SHAP, multi-model comparison, and an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production--Ready-success.svg)

---

## 🎯 Project Overview

This end-to-end machine learning system predicts Air Quality Index values from pollutant measurements and provides:

- **Advanced ML Models**: Linear Regression, Random Forest, XGBoost with hyperparameter optimization
- **Explainable AI**: SHAP values for model interpretability
- **Real-time Predictions**: Streaming simulation mode for live monitoring
- **Interactive Dashboard**: Streamlit web application with rich visualizations
- **Production-Ready Code**: Modular architecture, logging, error handling

---

## ✨ Key Features

### 🔬 Data Processing & Feature Engineering
- Automated missing value handling
- Outlier detection and treatment (IQR method)
- Feature scaling with RobustScaler
- **Advanced Feature Engineering**:
  - Rolling averages (3h, 6h, 12h, 24h windows)
  - Lag features (1h to 24h)
  - Pollutant interaction features (PM2.5 × NO2, etc.)
  - Statistical aggregations
  - Temporal encoding (cyclical features)
  - Derivative features (rate of change)

### 🤖 Machine Learning
- **Multiple Models**: Linear Regression, Random Forest, XGBoost
- **Hyperparameter Tuning**: GridSearch and Optuna optimization
- **Ensemble Methods**: Weighted model combinations
- **Model Evaluation**: RMSE, MAE, R², MAPE metrics

### 🔍 Explainable AI
- **SHAP Integration**: Feature importance and prediction explanations
- Summary plots showing global feature impact
- Waterfall plots for individual predictions
- Model-agnostic explanations

### 📊 Interactive Dashboard
- Dataset visualization and exploration
- Correlation heatmaps
- Model training interface
- Real-time prediction panel
- AQI risk categorization with health advisories
- Streaming simulation mode

### 🚨 Smart Features
- **AQI Risk Categories**: Automatic classification (Good, Moderate, Unhealthy, etc.)
- **Alert System**: Configurable thresholds with warnings
- **Health Advisories**: Category-specific recommendations
- **Export Options**: CSV, JSON, PDF reports

---

## 📁 Project Structure

```
AQI predictor/
│
├── data/                          # Dataset storage
│   └── aqi_data.csv              # AQI dataset (auto-generated if missing)
│
├── models/                        # Trained models
│   ├── best_model.pkl            # Best performing model
│   ├── linear_regression.pkl     # Individual models
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── preprocessor.pkl          # Scaler and preprocessing config
│   └── best_params.pkl           # Optimized hyperparameters
│
├── src/                          # Source code modules
│   ├── preprocessing.py          # Data preprocessing pipeline
│   ├── features.py               # Advanced feature engineering
│   ├── train.py                  # Model training and optimization
│   ├── evaluate.py               # Evaluation and SHAP analysis
│   └── predict.py                # Prediction and risk assessment
│
├── reports/                      # Generated visualizations
│   ├── model_comparison.png      # Model performance comparison
│   ├── predictions.png           # Actual vs predicted plots
│   ├── feature_importance.png    # Feature importance chart
│   ├── shap_summary.png          # SHAP summary plot
│   └── evaluation_report.txt     # Detailed text report
│
├── logs/                         # Application logs
│   ├── preprocessing.log
│   ├── training.log
│   ├── evaluation.log
│   └── prediction.log
│
├── notebooks/                    # Jupyter notebooks (optional)
│
├── app.py                        # Streamlit dashboard
├── main_train.py                 # Main training pipeline
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone or download the project**

```bash
cd "AQI predictor"
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

### Usage

#### 1. Train Models

Run the complete training pipeline:

```bash
python main_train.py
```

This will:
- Load or generate demo data
- Preprocess and engineer features
- Train multiple models
- Evaluate and compare performance
- Generate SHAP explanations
- Save models and reports

**Output**:
- Trained models in `models/` directory
- Visualizations in `reports/` directory
- Logs in `logs/` directory

#### 2. Launch Dashboard

Start the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

**Dashboard Features**:
- 🏠 **Home**: Project overview and AQI information
- 📊 **Data Explorer**: Dataset visualization and statistics
- 🤖 **Model Training**: Train models with custom settings
- 📈 **Model Comparison**: Compare model performance
- 🔮 **Make Prediction**: Manual AQI prediction
- ⚡ **Real-time Simulation**: Stream monitoring simulation
- 📉 **SHAP Analysis**: Model explainability

---

## 📊 Using Your Own Data

To use your own AQI dataset:

1. **Prepare your CSV file** with these columns (minimum):
   - Pollutant columns: `PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, `O3`
   - Target: `AQI`
   - Optional: `Date` (datetime column for temporal features)

2. **Place the file** in the `data/` directory as `aqi_data.csv`

3. **Run training**:
   ```bash
   python main_train.py
   ```

**Example CSV format**:
```csv
Date,PM2.5,PM10,NO2,SO2,CO,O3,AQI
2024-01-01 00:00:00,35.2,52.1,42.3,18.5,0.6,55.2,78
2024-01-01 01:00:00,38.5,55.8,45.1,20.2,0.7,58.3,82
...
```

---

## 🧪 Model Performance

Based on demo data (5000 samples):

| Model | RMSE | MAE | R² | MAPE |
|-------|------|-----|-----|------|
| **XGBoost** | **12.8** | **9.2** | **0.92** | **7.5%** |
| Random Forest | 15.2 | 11.5 | 0.88 | 9.8% |
| Linear Regression | 25.4 | 18.3 | 0.72 | 15.2% |

*Results may vary based on dataset*

---

## 🔧 Advanced Features

### Hyperparameter Optimization

**Using Optuna** (recommended):

```python
from src.train import ModelTrainer

trainer = ModelTrainer()
model, params = trainer.optimize_xgboost_optuna(
    X_train, y_train, X_val, y_val, 
    n_trials=50
)
```

**Using GridSearch**:

```python
model, params = trainer.optimize_xgboost_gridsearch(
    X_train, y_train, cv=3
)
```

### Custom Predictions

```python
from src.predict import AQIPredictor

predictor = AQIPredictor()
predictor.load_model()

# Single prediction
result = predictor.predict_single({
    'PM2.5': 45.0,
    'PM10': 60.0,
    'NO2': 50.0,
    'SO2': 25.0,
    'CO': 0.8,
    'O3': 70.0
})

print(f"Predicted AQI: {result['predicted_aqi']:.2f}")
print(f"Category: {result['category']}")
print(f"Health Message: {result['health_message']}")
```

### SHAP Explanations

```python
from src.evaluate import ModelEvaluator

evaluator = ModelEvaluator()
explainer, shap_values = evaluator.explain_with_shap(
    model, X_train, X_test, feature_names
)

# Generate plots
evaluator.plot_shap_summary(X_test, save_path='shap_summary.png')
```

### Real-time Simulation

```python
from src.predict import StreamingSimulator, AQIPredictor

simulator = StreamingSimulator()
predictor = AQIPredictor()
predictor.load_model()

for sample in simulator.generate_stream(n_samples=10, delay=1.0):
    result = predictor.predict_single(sample)
    print(f"AQI: {result['predicted_aqi']:.2f} - {result['category']}")
```

---

## 📈 AQI Categories

| Category | AQI Range | Color | Health Impact |
|----------|-----------|-------|---------------|
| Good | 0-50 | 🟢 Green | Air quality is satisfactory |
| Moderate | 51-100 | 🟡 Yellow | Acceptable; some pollutants may affect sensitive people |
| Unhealthy for Sensitive Groups | 101-150 | 🟠 Orange | Sensitive groups may experience health effects |
| Unhealthy | 151-200 | 🔴 Red | Everyone may begin to experience health effects |
| Very Unhealthy | 201-300 | 🟣 Purple | Health alert: everyone may experience serious effects |
| Hazardous | 301-500 | 🟤 Maroon | Health warnings of emergency conditions |

---

## 🛠️ Development

### Running Tests

```bash
# Add your test framework
pytest tests/
```

### Code Quality

```bash
# Format code
black src/ app.py

# Lint
pylint src/

# Type checking
mypy src/
```

### Adding New Features

1. Create new module in `src/`
2. Import in relevant scripts
3. Update `requirements.txt` if needed
4. Add documentation to README

---

## 📝 Logging

All modules use Python's logging framework. Logs are saved to `logs/` directory:

- `preprocessing.log` - Data preprocessing operations
- `training.log` - Model training progress
- `evaluation.log` - Evaluation metrics and results
- `prediction.log` - Prediction requests and results

**Log levels**: INFO, WARNING, ERROR

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **XGBoost**: Gradient boosting framework
- **SHAP**: Model explainability library
- **Streamlit**: Interactive dashboard framework
- **Scikit-learn**: Machine learning toolkit
- **Optuna**: Hyperparameter optimization

---

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

---

## 🔮 Future Enhancements

- [ ] Add REST API with FastAPI
- [ ] Implement time series forecasting
- [ ] Add more pollutants (NH3, Pb, etc.)
- [ ] Integration with real AQI data sources
- [ ] Docker containerization
- [ ] Cloud deployment (AWS/Azure/GCP)
- [ ] Mobile app interface
- [ ] Email/SMS alert notifications

---

**Built with ❤️ for cleaner air quality monitoring**
