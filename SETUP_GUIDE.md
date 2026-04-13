# 🚀 AQI Prediction System - Complete Setup Guide

This guide will walk you through setting up and running the complete AQI Prediction System.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Running the System](#running-the-system)
4. [Using Your Own Data](#using-your-own-data)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

- **Python 3.8 or higher** ([Download](https://www.python.org/downloads/))
- **pip** (usually comes with Python)
- **Git** (optional, for cloning repositories)

### Check Your Python Installation

```bash
python --version
# or
python3 --version
```

You should see something like `Python 3.8.x` or higher.

---

## Installation

### Method 1: Automated Setup (Recommended)

#### On macOS/Linux:

```bash
# Navigate to project directory
cd "AQI predictor"

# Make script executable
chmod +x run.sh

# Run setup and training
./run.sh
```

#### On Windows:

```cmd
REM Navigate to project directory
cd "AQI predictor"

REM Run setup and training
run.bat
```

This will:
- Create a virtual environment
- Install all dependencies
- Run the complete training pipeline
- Optionally launch the dashboard

### Method 2: Manual Setup

#### Step 1: Create Virtual Environment

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**
```cmd
python -m venv venv
venv\Scripts\activate
```

#### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

#### Step 3: Create Directories

```bash
# macOS/Linux
mkdir -p data models reports logs

# Windows
mkdir data models reports logs
```

#### Step 4: Run Training Pipeline

```bash
python main_train.py
```

This will:
- Generate demo data (if no data exists)
- Preprocess the data
- Engineer features
- Train multiple models (Linear Regression, Random Forest, XGBoost)
- Evaluate and compare models
- Generate SHAP explanations
- Save models and reports

**Expected Output:**
```
============================================================
STARTING AQI PREDICTION TRAINING PIPELINE
============================================================

[STEP 1] Loading data...
[STEP 2] Preprocessing data...
[STEP 3] Engineering advanced features...
[STEP 4] Splitting data...
[STEP 5] Training multiple models...
[STEP 6] Evaluating models...
[STEP 7] Generating visualizations...
[STEP 8] Generating SHAP explanations...
[STEP 9] Saving models...

============================================================
TRAINING PIPELINE COMPLETED SUCCESSFULLY!
============================================================

Best Model: XGBoost
R² Score: 0.9200
RMSE: 12.80
MAE: 9.20
```

---

## Running the System

### 1. Launch Interactive Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

**Dashboard Features:**
- 🏠 **Home**: Project overview
- 📊 **Data Explorer**: Visualizations and statistics
- 🤖 **Model Training**: Train models with custom settings
- 📈 **Model Comparison**: Compare performance metrics
- 🔮 **Make Prediction**: Manual AQI prediction
- ⚡ **Real-time Simulation**: Streaming data simulation
- 📉 **SHAP Analysis**: Model explainability

### 2. Make Predictions Programmatically

```python
from src.predict import AQIPredictor

# Initialize predictor
predictor = AQIPredictor()
predictor.load_model()

# Make prediction
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

### 3. Launch REST API (Bonus Feature)

```bash
# Start API server
uvicorn api:app --reload --host 0.0.0.0 --port 8000
```

**Access API:**
- API docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

**Example API Request:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "PM25": 45.0,
    "PM10": 60.0,
    "NO2": 50.0,
    "SO2": 25.0,
    "CO": 0.8,
    "O3": 70.0
  }'
```

---

## Using Your Own Data

### Data Format Requirements

Your CSV file should have these columns:

**Required:**
- `PM2.5` - PM2.5 concentration (μg/m³)
- `PM10` - PM10 concentration (μg/m³)
- `NO2` - Nitrogen dioxide (ppb)
- `SO2` - Sulfur dioxide (ppb)
- `CO` - Carbon monoxide (ppm)
- `O3` - Ozone (ppb)
- `AQI` - Air Quality Index (target variable)

**Optional:**
- `Date` - Datetime column for temporal features
- `hour`, `day`, `month` - Pre-extracted time features

### Example CSV Structure

```csv
Date,PM2.5,PM10,NO2,SO2,CO,O3,AQI
2024-01-01 00:00:00,35.2,52.1,42.3,18.5,0.6,55.2,78
2024-01-01 01:00:00,38.5,55.8,45.1,20.2,0.7,58.3,82
2024-01-01 02:00:00,32.1,48.9,38.7,16.8,0.5,52.1,72
```

### Steps to Use Your Data

1. **Prepare your CSV file** according to the format above

2. **Save it as** `data/aqi_data.csv`

3. **Run training pipeline:**
   ```bash
   python main_train.py
   ```

4. The system will automatically:
   - Load your data
   - Preprocess it
   - Engineer features
   - Train models
   - Generate predictions

---

## Advanced Usage

### Hyperparameter Optimization

#### Using Optuna (Recommended)

```python
from src.train import ModelTrainer
from src.preprocessing import DataPreprocessor

# Load and prepare data
preprocessor = DataPreprocessor()
# ... prepare X_train, y_train, X_val, y_val

# Optimize
trainer = ModelTrainer()
best_model, best_params = trainer.optimize_xgboost_optuna(
    X_train, y_train, X_val, y_val,
    n_trials=50  # More trials = better results, but slower
)

print(f"Best parameters: {best_params}")
```

### Custom Feature Engineering

```python
from src.features import AdvancedFeatureEngineer

engineer = AdvancedFeatureEngineer()

# Customize feature engineering
data_engineered = engineer.engineer_all_features(
    data,
    include_rolling=True,      # Rolling averages
    include_lag=True,          # Lag features
    include_interaction=True,  # Pollutant interactions
    include_statistical=True,  # Statistical features
    include_exponential=False, # Exponential transforms (optional)
    include_temporal=True,     # Temporal features
    include_derivative=False   # Rate of change (optional)
)
```

### SHAP Explanations

```python
from src.evaluate import ModelEvaluator

evaluator = ModelEvaluator()

# Generate SHAP explanations
explainer, shap_values = evaluator.explain_with_shap(
    model, X_train, X_test, feature_names
)

# Create visualizations
evaluator.plot_shap_summary(X_test, save_path='shap_summary.png')
evaluator.plot_shap_waterfall(X_test, idx=0, save_path='shap_waterfall.png')
```

### Real-time Streaming Simulation

```python
from src.predict import StreamingSimulator, AQIPredictor

# Initialize
simulator = StreamingSimulator()
predictor = AQIPredictor()
predictor.load_model()

# Stream predictions
for sample in simulator.generate_stream(n_samples=10, delay=1.0):
    result = predictor.predict_single(sample)
    print(f"AQI: {result['predicted_aqi']:.2f} - {result['category']}")
    
    # Check for alerts
    if result['predicted_aqi'] > 150:
        print("⚠️ ALERT: High AQI detected!")
```

---

## Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError

**Problem:** `ModuleNotFoundError: No module named 'xxx'`

**Solution:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

#### 2. Model Not Found

**Problem:** `FileNotFoundError: Model file not found`

**Solution:**
```bash
# Train models first
python main_train.py
```

#### 3. Memory Error During Training

**Problem:** `MemoryError` or system freezing

**Solution:**
- Reduce dataset size
- Reduce number of features:
  ```python
  # In main_train.py, modify feature engineering
  data_engineered = engineer.engineer_all_features(
      data_processed,
      include_rolling=False,  # Disable to reduce features
      include_lag=False,
      include_interaction=True
  )
  ```
- Use fewer optimization trials:
  ```python
  n_trials=10  # Instead of 50
  ```

#### 4. Streamlit Port Already in Use

**Problem:** `Address already in use`

**Solution:**
```bash
# Use different port
streamlit run app.py --server.port 8502
```

#### 5. SHAP Errors

**Problem:** SHAP visualization fails

**Solution:**
- SHAP is optional. The system will continue without it
- Try installing specific version:
  ```bash
  pip install shap==0.42.0
  ```

### Getting Help

If you encounter issues:

1. **Check logs** in `logs/` directory
2. **Review error messages** carefully
3. **Verify Python version** (3.8+)
4. **Ensure all dependencies** are installed
5. **Check data format** matches requirements

---

## Performance Tips

### For Faster Training

1. **Reduce dataset size** for testing:
   ```python
   data_sample = data.sample(n=1000)  # Use 1000 samples
   ```

2. **Skip optimization** for quick testing:
   ```python
   # Just train default models
   trainer.train_all_models(X_train, y_train)
   ```

3. **Reduce features**:
   ```python
   # Use fewer feature engineering steps
   include_rolling=False
   ```

### For Better Accuracy

1. **Use more data** (5000+ samples recommended)

2. **Enable all feature engineering**:
   ```python
   engineer.engineer_all_features(
       data,
       include_rolling=True,
       include_lag=True,
       include_interaction=True,
       include_statistical=True,
       include_temporal=True
   )
   ```

3. **Optimize hyperparameters**:
   ```python
   trainer.optimize_xgboost_optuna(X_train, y_train, X_val, y_val, n_trials=100)
   ```

4. **Use ensemble methods**:
   ```python
   from src.train import EnsembleModel
   
   ensemble = EnsembleModel([model1, model2, model3])
   predictions = ensemble.predict(X_test)
   ```

---

## Next Steps

After successful setup:

1. ✅ **Explore the dashboard** - Run `streamlit run app.py`
2. ✅ **Try predictions** - Use the prediction panel
3. ✅ **Analyze results** - Check SHAP explanations
4. ✅ **Use your own data** - Replace demo data
5. ✅ **Optimize models** - Try hyperparameter tuning
6. ✅ **Deploy API** - Use FastAPI for production

---

## Support

For additional support:
- Check the [README.md](README.md) for detailed documentation
- Review code documentation in each module
- Examine example notebooks in `notebooks/`

---

**Happy Predicting! 🌍**
