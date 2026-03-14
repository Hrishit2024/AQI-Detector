"""
Streamlit Dashboard for AQI Prediction System
Interactive web application with visualizations and predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
import joblib
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessing import DataPreprocessor
from src.features import AdvancedFeatureEngineer
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator
from src.predict import AQIPredictor, StreamingSimulator

# Page configuration
st.set_page_config(
    page_title="AQI Prediction System",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        border-radius: 0.25rem;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        border-radius: 0.25rem;
    }
    .danger-box {
        padding: 1rem;
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


def load_demo_data():
    """Load Kaggle city_day dataset"""
    if not os.path.exists('data/city_day.csv'):
        st.error("data/city_day.csv not found. Please place the Kaggle dataset in the data/ directory.")
        st.stop()
    df = pd.read_csv('data/city_day.csv')
    df = df.dropna(subset=['AQI'])
    df['AQI'] = df['AQI'].clip(upper=500)
    return df


def main():
    """Main application"""
    
    # Header
    st.markdown('<p class="main-header">🌍 AQI Prediction System</p>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["🏠 Home", "📊 Data Explorer", "🤖 Model Training", "🔮 AQI Predictor"]
    )
    
    # =======================
    # HOME PAGE
    # =======================
    if page == "🏠 Home":
        st.title("Welcome to AQI Prediction System")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**🎯 Objective**")
            st.write("Predict Air Quality Index using advanced ML techniques")
        
        with col2:
            st.info("**🧠 Models**")
            st.write("Linear Regression, Random Forest, XGBoost with optimization")
        
        with col3:
            st.info("**✨ Features**")
            st.write("SHAP explanations, Real-time predictions, Model comparison")
        
        st.markdown("---")
        
        st.subheader("📋 Project Overview")
        
        features_list = """
        - **Data Preprocessing**: Handle missing values, outliers, feature scaling
        - **Feature Engineering**: Rolling averages, lag features, interactions
        - **Multiple Models**: Linear Regression, Random Forest, XGBoost
        - **Hyperparameter Tuning**: GridSearch and Optuna optimization
        - **Explainable AI**: SHAP values for model interpretability
        - **Real-time Predictions**: Streaming simulation mode
        - **AQI Risk Categories**: Automatic health advisory messages
        """
        
        st.markdown(features_list)
        
        st.markdown("---")
        
        # AQI Information
        st.subheader("📚 AQI Categories")
        
        aqi_info = pd.DataFrame({
            'Category': ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 'Unhealthy', 'Very Unhealthy', 'Hazardous'],
            'Range': ['0-50', '51-100', '101-150', '151-200', '201-300', '301-500'],
            'Color': ['🟢 Green', '🟡 Yellow', '🟠 Orange', '🔴 Red', '🟣 Purple', '🟤 Maroon']
        })
        
        st.table(aqi_info)
    
    # =======================
    # DATA EXPLORER
    # =======================
    elif page == "📊 Data Explorer":
        st.title("Data Exploration & Visualization")
        
        # Load data
        data = load_demo_data()
        
        st.subheader("📁 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Samples", len(data))
        col2.metric("Features", len(data.columns) - 1)
        col3.metric("Mean AQI", f"{data['AQI'].mean():.2f}")
        col4.metric("Max AQI", f"{data['AQI'].max():.2f}")
        
        # Show raw data
        if st.checkbox("Show Raw Data"):
            st.dataframe(data.head(100))
        
        st.markdown("---")
        
        # Statistical summary
        st.subheader("📈 Statistical Summary")
        st.dataframe(data.describe())
        
        st.markdown("---")
        
        # Visualizations
        st.subheader("📊 Visualizations")
        
        tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Correlation", "Time Series", "Pollutant Comparison"])
        
        with tab1:
            # AQI Distribution
            fig = px.histogram(data, x='AQI', nbins=50, 
                             title='AQI Distribution',
                             color_discrete_sequence=['#1f77b4'])
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Pollutant distributions
            pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
            pollutants = [p for p in pollutants if p in data.columns]
            
            if pollutants:
                selected_pollutant = st.selectbox("Select Pollutant", pollutants)
                fig = px.histogram(data, x=selected_pollutant, nbins=50,
                                 title=f'{selected_pollutant} Distribution')
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Correlation heatmap
            st.write("### Correlation Heatmap")
            
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            corr_data = data[numeric_cols].corr()
            
            fig = px.imshow(corr_data, 
                          text_auto='.2f',
                          aspect='auto',
                          color_continuous_scale='RdBu_r',
                          title='Feature Correlation Matrix')
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            # Time series (if applicable)
            st.write("### AQI over Time")

            if 'Date' in data.columns:
                ts = data.copy()
                ts['Date'] = pd.to_datetime(ts['Date'])
                ts = ts.sort_values('Date')

                # City filter
                if 'City' in ts.columns:
                    cities = sorted(ts['City'].dropna().unique().tolist())
                    selected_city = st.selectbox("Filter by City", ["All Cities"] + cities)
                    if selected_city != "All Cities":
                        ts = ts[ts['City'] == selected_city]

                monthly_avg = ts.groupby(ts['Date'].dt.to_period('M'))['AQI'].mean().reset_index()
                monthly_avg['Date'] = monthly_avg['Date'].astype(str)
                fig = px.line(monthly_avg, x='Date', y='AQI',
                            title='Monthly Average AQI Over Time', markers=False)
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)

                yearly_avg = ts.groupby(ts['Date'].dt.year)['AQI'].mean().reset_index()
                yearly_avg.columns = ['Year', 'AQI']
                fig = px.bar(yearly_avg, x='Year', y='AQI', title='Average AQI by Year')
                st.plotly_chart(fig, use_container_width=True)

            elif 'month' in data.columns:
                monthly_avg = data.groupby('month')['AQI'].mean().reset_index()
                fig = px.bar(monthly_avg, x='month', y='AQI',
                           title='Average AQI by Month')
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No time-based column found in the dataset.")
        
        with tab4:
            # Pollutant comparison
            st.write("### Pollutant Comparison")
            
            pollutants = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
            pollutants = [p for p in pollutants if p in data.columns]
            
            if pollutants:
                pollutant_means = data[pollutants].mean()
                
                fig = go.Figure(data=[
                    go.Bar(x=pollutant_means.index, y=pollutant_means.values,
                          marker_color='indianred')
                ])
                fig.update_layout(title='Average Pollutant Levels',
                                xaxis_title='Pollutant',
                                yaxis_title='Average Concentration')
                st.plotly_chart(fig, use_container_width=True)

            # City-wise AQI comparison (Kaggle dataset)
            if 'City' in data.columns:
                st.write("### City-wise Average AQI")
                city_avg = data.groupby('City')['AQI'].mean().sort_values(ascending=False).reset_index()
                fig = px.bar(city_avg, x='City', y='AQI',
                             title='Average AQI by City',
                             color='AQI', color_continuous_scale='RdYlGn_r')
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
    
    # =======================
    # MODEL TRAINING
    # =======================
    elif page == "🤖 Model Training":
        st.title("Model Training & Optimization")
        
        st.info("This section allows you to train models on your dataset")
        
        # Load data
        data = load_demo_data()
        
        # Training options
        st.subheader("⚙️ Training Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
            include_feature_engineering = st.checkbox("Advanced Feature Engineering", value=True)
        
        with col2:
            optimize_model = st.checkbox("Hyperparameter Optimization (Optuna)", value=False)
            n_trials = st.number_input("Optimization Trials", 10, 100, 30) if optimize_model else 30
        
        if st.button("🚀 Start Training", type="primary"):
            with st.spinner("Training in progress..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Preprocessing
                    status_text.text("Step 1/5: Preprocessing data...")
                    progress_bar.progress(20)

                    # Drop metadata/non-feature columns from Kaggle dataset
                    train_data = data.copy()
                    drop_cols = [c for c in ['City', 'AQI_Bucket'] if c in train_data.columns]
                    train_data = train_data.drop(columns=drop_cols)
                    # Cap AQI at 500 (already done in load_demo_data but ensure)
                    train_data['AQI'] = train_data['AQI'].clip(upper=500)

                    preprocessor = DataPreprocessor()
                    df_processed = preprocessor.preprocess_pipeline(train_data, target_col='AQI', fit=True)
                    
                    # Step 2: Feature Engineering
                    if include_feature_engineering:
                        status_text.text("Step 2/5: Engineering features...")
                        progress_bar.progress(40)
                        
                        engineer = AdvancedFeatureEngineer()
                        df_processed = engineer.engineer_all_features(
                            df_processed,
                            include_rolling=True,
                            include_lag=True,
                            include_interaction=True,
                            include_statistical=True
                        )
                    
                    # Step 3: Split data
                    status_text.text("Step 3/5: Splitting data...")
                    progress_bar.progress(60)
                    
                    X_train, X_test, y_train, y_test = preprocessor.split_data(
                        df_processed, test_size=test_size
                    )
                    
                    # Step 4: Train models
                    status_text.text("Step 4/5: Training models...")
                    progress_bar.progress(80)
                    
                    trainer = ModelTrainer()
                    
                    if optimize_model:
                        # Create validation set from training data
                        X_train_df = X_train.copy()
                        X_train_df['AQI'] = y_train.values
                        X_train_opt, X_val, y_train_opt, y_val = preprocessor.split_data(
                            X_train_df, target_col='AQI', test_size=0.2
                        )
                        
                        model, params = trainer.optimize_xgboost_optuna(
                            X_train_opt, y_train_opt, X_val, y_val, n_trials=n_trials
                        )
                        st.success(f"✅ Optimization completed! Best params: {params}")
                    else:
                        models = trainer.train_all_models(X_train, y_train)
                    
                    # Step 5: Evaluate
                    status_text.text("Step 5/5: Evaluating models...")
                    progress_bar.progress(100)
                    
                    evaluator = ModelEvaluator()
                    
                    if optimize_model:
                        y_pred = model.predict(X_test)
                        metrics = evaluator.calculate_metrics(y_test, y_pred, 'Optimized XGBoost')
                    else:
                        results_df = evaluator.evaluate_multiple_models(trainer.models, X_test, y_test)
                        st.subheader("📊 Model Comparison Results")
                        st.dataframe(results_df, use_container_width=True)
                        # Save metrics for Model Comparison tab
                        os.makedirs('reports', exist_ok=True)
                        results_df.to_csv('reports/model_metrics.csv', index=False)
                    
                    status_text.text("✅ Training completed!")
                    st.success("Training completed successfully!")
                    
                    # Save models
                    trainer.save_all_models()
                    preprocessor.save_preprocessor()
                    
                    st.info("💾 Models saved to 'models/' directory")
                    
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")

    # =======================
    # AQI PREDICTOR
    # =======================
    elif page == "🔮 AQI Predictor":
        st.title("AQI Prediction Panel")

        st.subheader("📝 Enter Pollutant Values")

        col1, col2, col3 = st.columns(3)

        with col1:
            pm25 = st.number_input("PM2.5 (μg/m³)", 0.0, 500.0, 5.0, 0.1)
            pm10 = st.number_input("PM10 (μg/m³)", 0.0, 600.0, 10.0, 0.1)

        with col2:
            no2 = st.number_input("NO2 (ppb)", 0.0, 200.0, 8.0, 0.1)
            so2 = st.number_input("SO2 (ppb)", 0.0, 100.0, 3.0, 0.1)

        with col3:
            co = st.number_input("CO (ppm)", 0.0, 50.0, 0.2, 0.01)
            o3 = st.number_input("O3 (ppb)", 0.0, 200.0, 15.0, 0.1)

        st.markdown("---")

        if st.button("🔮 Predict AQI", type="primary"):
            try:
                available_model_path = next(
                    (
                        path for path in [
                            'models/best_model.pkl',
                            'models/xgboost.pkl',
                            'models/random_forest.pkl',
                            'models/linear_regression.pkl'
                        ]
                        if os.path.exists(path)
                    ),
                    None
                )
                predictor = AQIPredictor(model_path=available_model_path or 'models/best_model.pkl')

                if available_model_path is None:
                    aqi_value = (
                        pm25 * 1.5 + pm10 * 0.8 + no2 * 1.2 +
                        so2 * 0.5 + co * 30 + o3 * 0.9
                    )

                    category, color, health_message = predictor.get_aqi_category(aqi_value)

                    result = {
                        'predicted_aqi': aqi_value,
                        'category': category,
                        'color': color,
                        'health_message': health_message,
                        'risk_level': predictor._get_risk_level(aqi_value)
                    }
                else:
                    input_data = {
                        'PM2.5': pm25,
                        'PM10': pm10,
                        'NO2': no2,
                        'SO2': so2,
                        'CO': co,
                        'O3': o3
                    }
                    result = predictor.predict_single(input_data)
                    st.caption(f"Using trained model: {os.path.basename(available_model_path)}")

                st.markdown("---")
                st.subheader("🎯 Prediction Results")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Predicted AQI", f"{result['predicted_aqi']:.2f}")

                with col2:
                    st.metric("Category", result['category'])

                with col3:
                    st.metric("Risk Level", f"{result['risk_level']}/5")

                if result['risk_level'] <= 1:
                    st.markdown(
                        f'<div class="success-box">{result["health_message"]}</div>',
                        unsafe_allow_html=True
                    )
                elif result['risk_level'] <= 3:
                    st.markdown(
                        f'<div class="warning-box">{result["health_message"]}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="danger-box">⚠️ {result["health_message"]}</div>',
                        unsafe_allow_html=True
                    )

                alert_threshold = st.slider("Alert Threshold", 50, 300, 150)
                alert_msg = predictor.generate_alert_message(result['predicted_aqi'], alert_threshold)

                if alert_msg:
                    st.error(alert_msg)

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
    

if __name__ == "__main__":
    main()
