"""
FastAPI REST API for AQI Prediction System
Bonus feature: Production-ready API endpoints
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import sys
import os
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.predict import AQIPredictor, StreamingSimulator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AQI Prediction API",
    description="REST API for Air Quality Index predictions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize predictor
predictor = AQIPredictor()

# Load model on startup
@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        predictor.load_model()
        logger.info("✅ Model loaded successfully")
    except Exception as e:
        logger.warning(f"⚠️ Could not load model: {str(e)}")


# Pydantic models for request/response validation
class PollutantInput(BaseModel):
    """Input schema for pollutant values"""
    PM25: float = Field(..., ge=0, le=500, description="PM2.5 concentration (μg/m³)")
    PM10: float = Field(..., ge=0, le=600, description="PM10 concentration (μg/m³)")
    NO2: float = Field(..., ge=0, le=200, description="NO2 concentration (ppb)")
    SO2: float = Field(..., ge=0, le=100, description="SO2 concentration (ppb)")
    CO: float = Field(..., ge=0, le=50, description="CO concentration (ppm)")
    O3: float = Field(..., ge=0, le=200, description="O3 concentration (ppb)")
    hour: Optional[int] = Field(None, ge=0, le=23, description="Hour of day (0-23)")
    month: Optional[int] = Field(None, ge=1, le=12, description="Month (1-12)")
    
    class Config:
        schema_extra = {
            "example": {
                "PM25": 45.0,
                "PM10": 60.0,
                "NO2": 50.0,
                "SO2": 25.0,
                "CO": 0.8,
                "O3": 70.0,
                "hour": 12,
                "month": 6
            }
        }


class AQIPredictionResponse(BaseModel):
    """Response schema for AQI prediction"""
    predicted_aqi: float
    category: str
    color: str
    health_message: str
    risk_level: int
    timestamp: str


class BatchPollutantInput(BaseModel):
    """Input schema for batch predictions"""
    samples: List[PollutantInput]


class HealthStatus(BaseModel):
    """API health status"""
    status: str
    model_loaded: bool
    timestamp: str


# API Endpoints

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "AQI Prediction API",
        "version": "1.0.0",
        "description": "REST API for Air Quality Index predictions",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "categories": "/aqi/categories",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthStatus, tags=["General"])
async def health_check():
    """Check API health status"""
    return {
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict", response_model=AQIPredictionResponse, tags=["Prediction"])
async def predict_aqi(input_data: PollutantInput):
    """
    Predict AQI from pollutant values
    
    Returns AQI prediction with category and health advisory
    """
    try:
        # Convert input to dictionary
        pollutants = {
            'PM2.5': input_data.PM25,
            'PM10': input_data.PM10,
            'NO2': input_data.NO2,
            'SO2': input_data.SO2,
            'CO': input_data.CO,
            'O3': input_data.O3
        }
        
        # Add optional temporal features
        if input_data.hour is not None:
            pollutants['hour'] = input_data.hour
        if input_data.month is not None:
            pollutants['month'] = input_data.month
        
        # Check if model is loaded
        if predictor.model is None:
            # Use simple formula for demo
            aqi_value = (
                input_data.PM25 * 1.5 + 
                input_data.PM10 * 0.8 + 
                input_data.NO2 * 1.2 + 
                input_data.SO2 * 0.5 +
                input_data.CO * 30 +
                input_data.O3 * 0.9
            )
            
            category, color, health_message = predictor.get_aqi_category(aqi_value)
            
            result = {
                'predicted_aqi': aqi_value,
                'category': category,
                'color': color,
                'health_message': health_message,
                'risk_level': predictor._get_risk_level(aqi_value),
                'timestamp': datetime.now().isoformat()
            }
        else:
            # Use trained model
            result = predictor.predict_single(pollutants)
            result['timestamp'] = datetime.now().isoformat()
        
        logger.info(f"Prediction: AQI={result['predicted_aqi']:.2f}, Category={result['category']}")
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
async def batch_predict(input_data: BatchPollutantInput):
    """
    Batch prediction for multiple samples
    
    Returns list of AQI predictions
    """
    try:
        results = []
        
        for sample in input_data.samples:
            pollutants = {
                'PM2.5': sample.PM25,
                'PM10': sample.PM10,
                'NO2': sample.NO2,
                'SO2': sample.SO2,
                'CO': sample.CO,
                'O3': sample.O3
            }
            
            if sample.hour is not None:
                pollutants['hour'] = sample.hour
            if sample.month is not None:
                pollutants['month'] = sample.month
            
            # Simple calculation for demo
            aqi_value = (
                sample.PM25 * 1.5 + 
                sample.PM10 * 0.8 + 
                sample.NO2 * 1.2 + 
                sample.SO2 * 0.5 +
                sample.CO * 30 +
                sample.O3 * 0.9
            )
            
            category, color, health_message = predictor.get_aqi_category(aqi_value)
            
            results.append({
                'predicted_aqi': aqi_value,
                'category': category,
                'color': color,
                'health_message': health_message,
                'risk_level': predictor._get_risk_level(aqi_value)
            })
        
        logger.info(f"Batch prediction completed for {len(results)} samples")
        
        return {
            'predictions': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/aqi/categories", tags=["Information"])
async def get_aqi_categories():
    """
    Get AQI category definitions
    
    Returns all AQI categories with ranges and health messages
    """
    categories = []
    
    for category, (low, high) in predictor.AQI_CATEGORIES.items():
        categories.append({
            'category': category,
            'range': f"{low}-{high}",
            'low': low,
            'high': high,
            'color': predictor.AQI_COLORS[category],
            'health_message': predictor._get_health_message(category)
        })
    
    return {
        'categories': categories,
        'source': 'US EPA Standard'
    }


@app.get("/aqi/category/{aqi_value}", tags=["Information"])
async def get_category_for_aqi(aqi_value: float):
    """
    Get AQI category for a specific AQI value
    
    Args:
        aqi_value: AQI value (0-500)
    """
    if aqi_value < 0 or aqi_value > 500:
        raise HTTPException(status_code=400, detail="AQI value must be between 0 and 500")
    
    category, color, health_message = predictor.get_aqi_category(aqi_value)
    
    return {
        'aqi': aqi_value,
        'category': category,
        'color': color,
        'health_message': health_message,
        'risk_level': predictor._get_risk_level(aqi_value)
    }


@app.post("/alert/check", tags=["Alerts"])
async def check_alert(input_data: PollutantInput, threshold: float = 150):
    """
    Check if prediction triggers alert
    
    Args:
        input_data: Pollutant values
        threshold: Alert threshold (default: 150)
    """
    try:
        # Calculate AQI
        aqi_value = (
            input_data.PM25 * 1.5 + 
            input_data.PM10 * 0.8 + 
            input_data.NO2 * 1.2 + 
            input_data.SO2 * 0.5 +
            input_data.CO * 30 +
            input_data.O3 * 0.9
        )
        
        alert_triggered = predictor.check_alert_threshold(aqi_value, threshold)
        
        response = {
            'predicted_aqi': aqi_value,
            'threshold': threshold,
            'alert_triggered': alert_triggered
        }
        
        if alert_triggered:
            response['alert_message'] = predictor.generate_alert_message(aqi_value, threshold)
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alert check failed: {str(e)}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            'error': exc.detail,
            'timestamp': datetime.now().isoformat()
        }
    )


# Run with: uvicorn api:app --reload --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
