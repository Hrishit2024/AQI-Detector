# 🌍 AQI PREDICTION SYSTEM - COMPLETE PROJECT INDEX

## 📚 Documentation Files

### Getting Started
- **[README.md](README.md)** - Main project documentation (comprehensive)
- **[SETUP_GUIDE.md](SETUP_GUIDE.md)** - Step-by-step setup instructions
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick command reference
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)** - Complete feature overview

### Configuration
- **[requirements.txt](requirements.txt)** - Python dependencies
- **[.gitignore](.gitignore)** - Git ignore rules

---

## 💻 Application Files

### Main Applications
- **[app.py](app.py)** - Streamlit dashboard (800+ lines)
  - 7 interactive pages
  - Data visualization
  - Model training interface
  - Prediction panel
  - Real-time simulation
  - SHAP analysis

- **[api.py](api.py)** - FastAPI REST API (400+ lines)
  - RESTful endpoints
  - Swagger documentation
  - Request validation
  - Error handling

- **[main_train.py](main_train.py)** - Complete training pipeline (200+ lines)
  - End-to-end workflow
  - Automated execution
  - Model evaluation
  - Report generation

### Setup Scripts
- **[run.sh](run.sh)** - Automated setup (macOS/Linux)
- **[run.bat](run.bat)** - Automated setup (Windows)

---

## 🧠 Core Modules (src/)

### 1. [src/preprocessing.py](src/preprocessing.py)
**Purpose:** Data preprocessing and cleaning

**Key Classes:**
- `DataPreprocessor`

**Features:**
- Load CSV data
- Handle missing values (median, mean, mode, drop)
- Detect outliers (IQR, Z-score)
- Extract datetime features (hour, day, month, season)
- Feature scaling (RobustScaler)
- Train-test splitting
- Save/load preprocessor

**Lines:** ~400

---

### 2. [src/features.py](src/features.py)
**Purpose:** Advanced feature engineering

**Key Classes:**
- `AdvancedFeatureEngineer`

**Features:**
- Rolling averages (3h, 6h, 12h, 24h)
- Lag features (1h to 24h)
- Interaction features (pollutant combinations)
- Statistical features (mean, max, min, std)
- Exponential transformations (log, sqrt, square)
- Temporal aggregations (rush hour, night, season)
- Derivative features (rate of change)

**Lines:** ~450

---

### 3. [src/train.py](src/train.py)
**Purpose:** Model training and optimization

**Key Classes:**
- `ModelTrainer`
- `EnsembleModel`

**Features:**
- Multiple models (Linear Regression, Random Forest, XGBoost)
- GridSearch optimization
- Optuna optimization
- Cross-validation
- Feature importance
- Model persistence
- Ensemble methods

**Lines:** ~450

---

### 4. [src/evaluate.py](src/evaluate.py)
**Purpose:** Model evaluation and explainability

**Key Classes:**
- `ModelEvaluator`

**Features:**
- Calculate metrics (RMSE, MAE, R², MAPE)
- Multi-model comparison
- Prediction plots
- Residual analysis
- SHAP explanations
- Feature importance plots
- Report generation

**Lines:** ~500

---

### 5. [src/predict.py](src/predict.py)
**Purpose:** Predictions and risk assessment

**Key Classes:**
- `AQIPredictor`
- `StreamingSimulator`
- `PredictionExporter`

**Features:**
- Single predictions
- Batch predictions
- AQI categorization (6 EPA levels)
- Health advisories
- Alert system
- Streaming simulation
- Export to CSV/JSON/PDF

**Lines:** ~450

---

### 6. [src/__init__.py](src/__init__.py)
**Purpose:** Package initialization

**Features:**
- Export main classes
- Version info

**Lines:** ~20

---

## 📊 Data & Models

### Data Directory (data/)
- **Purpose:** Store datasets
- **Expected file:** `aqi_data.csv`
- **Format:** CSV with pollutant columns + AQI
- **Auto-generated:** Demo data created if missing

### Models Directory (models/)
- **Purpose:** Store trained models
- **Files generated:**
  - `best_model.pkl` - Best performing model
  - `linear_regression.pkl`
  - `random_forest.pkl`
  - `xgboost.pkl`
  - `preprocessor.pkl` - Scaler configuration
  - `best_params.pkl` - Optimized hyperparameters

### Reports Directory (reports/)
- **Purpose:** Store visualizations
- **Files generated:**
  - `model_comparison.png`
  - `predictions.png`
  - `feature_importance.png`
  - `shap_summary.png`
  - `evaluation_report.txt`

### Logs Directory (logs/)
- **Purpose:** Store application logs
- **Files generated:**
  - `preprocessing.log`
  - `training.log`
  - `evaluation.log`
  - `prediction.log`
  - `main_training.log`

---

## 📓 Notebooks

### [notebooks/AQI_Analysis.md](notebooks/AQI_Analysis.md)
**Purpose:** Jupyter notebook tutorial

**Contents:**
- Complete workflow demonstration
- Data exploration examples
- Model training walkthrough
- Visualization examples
- Code snippets

---

## 🎯 Quick Start Guide

### For Complete Beginners

1. **Open Terminal/Command Prompt**

2. **Navigate to project:**
   ```bash
   cd "AQI predictor"
   ```

3. **Run automated setup:**
   ```bash
   # macOS/Linux
   ./run.sh
   
   # Windows
   run.bat
   ```

4. **Wait for training to complete** (~2-5 minutes)

5. **Dashboard launches automatically** or run:
   ```bash
   streamlit run app.py
   ```

### For Developers

1. **Manual setup:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

2. **Train models:**
   ```bash
   python main_train.py
   ```

3. **Launch applications:**
   ```bash
   # Dashboard
   streamlit run app.py
   
   # API
   uvicorn api:app --reload
   ```

---

## 📖 Learning Path

### Level 1: Basic Usage
1. Read [README.md](README.md)
2. Run `./run.sh` or `run.bat`
3. Explore Streamlit dashboard
4. Try making predictions

### Level 2: Customization
1. Read [SETUP_GUIDE.md](SETUP_GUIDE.md)
2. Add your own data
3. Adjust feature engineering
4. Experiment with hyperparameters

### Level 3: Development
1. Study source code in `src/`
2. Review [notebooks/AQI_Analysis.md](notebooks/AQI_Analysis.md)
3. Modify models and features
4. Deploy API to production

---

## 🔍 File Usage Guide

### When You Want To...

**Set up the project:**
→ Use `run.sh` or `run.bat`

**Understand the system:**
→ Read `README.md`

**Follow step-by-step setup:**
→ Read `SETUP_GUIDE.md`

**Quick command reference:**
→ Check `QUICK_REFERENCE.md`

**See all features:**
→ Review `PROJECT_SUMMARY.md`

**Train models:**
→ Run `main_train.py`

**Use the dashboard:**
→ Run `streamlit run app.py`

**Use the API:**
→ Run `uvicorn api:app --reload`

**Add your data:**
→ Save to `data/aqi_data.csv`

**Check results:**
→ Look in `reports/` directory

**Debug issues:**
→ Check `logs/` directory

**Learn by example:**
→ Study `notebooks/AQI_Analysis.md`

**Modify preprocessing:**
→ Edit `src/preprocessing.py`

**Add features:**
→ Edit `src/features.py`

**Change models:**
→ Edit `src/train.py`

**Customize predictions:**
→ Edit `src/predict.py`

---

## 📊 Project Statistics

| Metric | Count |
|--------|-------|
| **Total Files** | 20+ |
| **Python Modules** | 6 |
| **Documentation Files** | 5 |
| **Total Code Lines** | 3500+ |
| **Features Engineered** | 100+ |
| **Models Supported** | 3 |
| **Dashboard Pages** | 7 |
| **API Endpoints** | 6 |
| **AQI Categories** | 6 |

---

## 🏆 Key Innovations

1. **Advanced Feature Engineering** - 100+ features from 10 base features
2. **SHAP Integration** - Full explainability support
3. **Multi-Model Pipeline** - Compare 3 algorithms automatically
4. **Hyperparameter Optimization** - Optuna integration
5. **Real-time Simulation** - Streaming data support
6. **Production API** - FastAPI with documentation
7. **Interactive Dashboard** - 7-page Streamlit app
8. **Alert System** - Configurable AQI thresholds
9. **Export Capabilities** - CSV, JSON, PDF
10. **Complete Automation** - One-click setup

---

## 🎓 Educational Value

### Learn About:
- Data preprocessing techniques
- Feature engineering strategies
- Machine learning pipelines
- Model optimization
- Model evaluation metrics
- Explainable AI (SHAP)
- Web dashboard development (Streamlit)
- REST API development (FastAPI)
- Production-ready code practices
- Logging and error handling

---

## 🚀 Deployment Options

### Local Deployment
- ✅ Already configured
- ✅ Run with `./run.sh`

### Cloud Deployment (Future)
- AWS/Azure/GCP compatible
- Docker-ready structure
- API for easy integration
- Streamlit cloud compatible

---

## 📞 Support Resources

1. **README.md** - Full documentation
2. **SETUP_GUIDE.md** - Setup help
3. **QUICK_REFERENCE.md** - Command cheat sheet
4. **Logs** - Detailed error information
5. **Code comments** - Inline documentation

---

## ✅ Verification Checklist

Before using, ensure:
- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Directories created (data, models, reports, logs)
- [ ] Models trained (`python main_train.py`)
- [ ] Dashboard works (`streamlit run app.py`)

---

## 🎯 Next Steps

1. **Explore** the dashboard
2. **Try** making predictions
3. **Add** your own data
4. **Optimize** models
5. **Deploy** the API
6. **Share** your results

---

**Start exploring from [README.md](README.md) or run `./run.sh` to begin!**

---

*Last updated: February 2026*
*Project: AQI Prediction System v1.0*
*Status: Production Ready ✅*
