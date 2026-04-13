# 📝 Quick Reference Card - AQI Prediction System

## 🚀 Essential Commands

### Setup & Installation
```bash
# Automated (Recommended)
./run.sh                    # macOS/Linux
run.bat                     # Windows

# Manual
pip install -r requirements.txt
python main_train.py
```

### Launch Applications
```bash
streamlit run app.py        # Dashboard (Port 8501)
uvicorn api:app --reload    # API Server (Port 8000)
```

---

## 📂 File Locations

| What | Where |
|------|-------|
| Data | `data/aqi_data.csv` |
| Models | `models/best_model.pkl` |
| Reports | `reports/*.png` |
| Logs | `logs/*.log` |

---

## 🐍 Python Quick Start

### Train Models
```python
python main_train.py
```

### Make Prediction
```python
from src.predict import AQIPredictor

predictor = AQIPredictor()
predictor.load_model()

result = predictor.predict_single({
    'PM2.5': 45.0, 'PM10': 60.0,
    'NO2': 50.0, 'SO2': 25.0,
    'CO': 0.8, 'O3': 70.0
})

print(f"AQI: {result['predicted_aqi']:.2f}")
print(f"Category: {result['category']}")
```

### Load Data
```python
from src.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
data = preprocessor.load_data('data/aqi_data.csv')
```

### Train Custom Model
```python
from src.train import ModelTrainer

trainer = ModelTrainer()
models = trainer.train_all_models(X_train, y_train)
```

---

## 🌐 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |
| `/aqi/categories` | GET | AQI categories |
| `/alert/check` | POST | Check alerts |
| `/health` | GET | API health |

### Example Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"PM25":45,"PM10":60,"NO2":50,"SO2":25,"CO":0.8,"O3":70}'
```

---

## 📊 Dashboard Pages

1. 🏠 **Home** - Overview
2. 📊 **Data Explorer** - Visualizations
3. 🤖 **Model Training** - Train models
4. 📈 **Model Comparison** - Compare performance
5. 🔮 **Make Prediction** - Predict AQI
6. ⚡ **Real-time Simulation** - Stream data
7. 📉 **SHAP Analysis** - Explainability

---

## 🎯 AQI Categories

| Range | Category | Action |
|-------|----------|--------|
| 0-50 | Good 🟢 | Normal activity |
| 51-100 | Moderate 🟡 | Sensitive groups limit |
| 101-150 | Unhealthy (SG) 🟠 | Reduce outdoor |
| 151-200 | Unhealthy 🔴 | Avoid outdoor |
| 201-300 | Very Unhealthy 🟣 | Stay indoors |
| 301-500 | Hazardous 🟤 | Emergency |

---

## 🔧 Common Tasks

### Add Your Data
```bash
# 1. Save CSV as data/aqi_data.csv
# 2. Run training
python main_train.py
```

### Optimize Model
```python
from src.train import ModelTrainer

trainer = ModelTrainer()
model, params = trainer.optimize_xgboost_optuna(
    X_train, y_train, X_val, y_val, n_trials=50
)
```

### Generate SHAP Plot
```python
from src.evaluate import ModelEvaluator

evaluator = ModelEvaluator()
explainer, shap_values = evaluator.explain_with_shap(
    model, X_train, X_test, feature_names
)
evaluator.plot_shap_summary(X_test, save_path='shap.png')
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Module not found | `pip install -r requirements.txt` |
| Model not found | `python main_train.py` |
| Port in use | Use different port: `--server.port 8502` |
| Memory error | Reduce data size or features |

---

## 📞 Support

- Documentation: `README.md`
- Setup Guide: `SETUP_GUIDE.md`
- Project Summary: `PROJECT_SUMMARY.md`
- Logs: `logs/` directory

---

**Keep this card handy for quick reference! 📌**
