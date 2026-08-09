import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from xgboost_model import FloodSeverityXGBoostModel

class FloodDataProcessor:
    """Enhanced flood data processor with XGBoost prediction capabilities"""
    
    def __init__(self, model_path='xgboost_flood_model.pkl'):
        self.sample_data = None
        self.xgboost_model = FloodSeverityXGBoostModel()
        self.model_path = model_path
        
        # Try to load pre-trained model
        if os.path.exists(model_path):
            try:
                self.xgboost_model.load_model(model_path)
                print(f"Loaded pre-trained model from {model_path}")
            except Exception as e:
                print(f"Could not load model: {e}")
    
    def load_sample_data(self, data_path='flood_training_data.csv'):
        """Load flood data from CSV"""
        if self.sample_data is None:
            try:
                self.sample_data = pd.read_csv(data_path)
                
                # Convert date column if exists
                if 'date' in self.sample_data.columns:
                    self.sample_data['date'] = pd.to_datetime(self.sample_data['date'])
                
                print(f"Loaded {len(self.sample_data)} records from {data_path}")
            except FileNotFoundError:
                print(f"Data file not found: {data_path}")
                self.sample_data = pd.DataFrame()
        
        return self.sample_data
    
    def predict_severity(self, data):
        """Predict flood severity using XGBoost model"""
        
        if not self.xgboost_model.is_trained:
            raise ValueError("XGBoost model is not trained. Please train the model first.")
        
        # Make predictions
        predictions, probabilities = self.xgboost_model.predict(data)
        
        # Add predictions to dataframe
        result_df = data.copy()
        result_df['predicted_severity'] = predictions
        
        # Add confidence scores
        max_proba = probabilities.max(axis=1)
        result_df['prediction_confidence'] = max_proba
        
        # Add probability for each class
        for i, class_name in enumerate(self.xgboost_model.label_encoder.classes_):
            result_df[f'prob_{class_name.lower()}'] = probabilities[:, i]
        
        return result_df
    
    def predict_single_location(self, features):
        """Predict severity for a single location"""
        
        if not self.xgboost_model.is_trained:
            raise ValueError("XGBoost model is not trained. Please train the model first.")
        
        return self.xgboost_model.predict_single(features)
    
    def calculate_risk_metrics(self, df):
        """Calculate comprehensive risk metrics from flood data"""
        
        if df is None or df.empty:
            return {
                'total_locations': 0,
                'high_risk_zones': 0,
                'medium_risk_zones': 0,
                'low_risk_zones': 0,
                'critical_zones': 0,
                'total_population_at_risk': 0,
                'avg_water_depth': 0,
                'max_water_depth': 0
            }
        
        # Determine which severity column to use
        severity_col = 'predicted_severity' if 'predicted_severity' in df.columns else 'severity_level'
        
        metrics = {
            'total_locations': len(df),
            'high_risk_zones': len(df[df[severity_col].isin(['High', 'Critical'])]),
            'medium_risk_zones': len(df[df[severity_col] == 'Medium']),
            'low_risk_zones': len(df[df[severity_col] == 'Low']),
            'critical_zones': len(df[df[severity_col] == 'Critical']),
            'total_population_at_risk': int(df['population_affected'].sum()) if 'population_affected' in df.columns else 0,
            'avg_water_depth': float(df['water_depth_m'].mean()) if 'water_depth_m' in df.columns else 0,
            'max_water_depth': float(df['water_depth_m'].max()) if 'water_depth_m' in df.columns else 0
        }
        
        return metrics
    
    def filter_by_severity(self, df, severity_levels):
        """Filter data by severity levels"""
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Determine which severity column to use
        severity_col = 'predicted_severity' if 'predicted_severity' in df.columns else 'severity_level'
        
        return df[df[severity_col].isin(severity_levels)]
    
    def get_statistics(self, df):
        """Get statistical summary of flood data"""
        
        if df is None or df.empty:
            return None
        
        stats = {}
        
        if 'water_depth_m' in df.columns:
            stats['water_depth'] = df['water_depth_m'].describe()
        
        if 'population_affected' in df.columns:
            stats['population'] = df['population_affected'].describe()
        
        # Use appropriate severity column
        severity_col = 'predicted_severity' if 'predicted_severity' in df.columns else 'severity_level'
        if severity_col in df.columns:
            stats['severity_distribution'] = df[severity_col].value_counts()
        
        if 'rainfall_mm' in df.columns:
            stats['rainfall'] = df['rainfall_mm'].describe()
        
        if 'river_level_m' in df.columns:
            stats['river_level'] = df['river_level_m'].describe()
        
        return stats
    
    def validate_data(self, df):
        """Validate flood data format and content"""
        
        required_columns = [
            'location', 'latitude', 'longitude'
        ]
        
        errors = []
        warnings = []
        
        # Check for required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Check optional but important columns
        important_columns = ['water_depth_m', 'population_affected', 'rainfall_mm']
        missing_important = [col for col in important_columns if col not in df.columns]
        if missing_important:
            warnings.append(f"Missing important columns: {', '.join(missing_important)}")
        
        # Validate data types and ranges
        if 'latitude' in df.columns:
            if not df['latitude'].between(-90, 90).all():
                errors.append("Latitude values must be between -90 and 90")
        
        if 'longitude' in df.columns:
            if not df['longitude'].between(-180, 180).all():
                errors.append("Longitude values must be between -180 and 180")
        
        if 'water_depth_m' in df.columns:
            if (df['water_depth_m'] < 0).any():
                errors.append("Water depth cannot be negative")
        
        if 'rainfall_mm' in df.columns:
            if (df['rainfall_mm'] < 0).any():
                errors.append("Rainfall cannot be negative")
        
        return len(errors) == 0, errors, warnings
    
    def prepare_for_prediction(self, df):
        """Prepare data for XGBoost prediction"""
        
        # Ensure all required features are present
        required_features = [
            'rainfall_mm', 'river_level_m', 'soil_moisture_percent',
            'terrain_elevation_m', 'distance_to_river_km', 'drainage_capacity_percent',
            'historical_flood_count', 'population_density', 'urban_area_percent',
            'water_depth_m'
        ]
        
        missing_features = [f for f in required_features if f not in df.columns]
        
        if missing_features:
            raise ValueError(f"Missing required features for prediction: {', '.join(missing_features)}")
        
        return df
    
    def get_top_risk_locations(self, df, n=10):
        """Get top N locations by risk"""
        
        if df is None or df.empty:
            return pd.DataFrame()
        
        # Use risk_score if available, otherwise use water_depth and population
        if 'risk_score' in df.columns:
            return df.nlargest(n, 'risk_score')
        elif 'water_depth_m' in df.columns and 'population_affected' in df.columns:
            df_copy = df.copy()
            df_copy['combined_risk'] = df_copy['water_depth_m'] * np.log1p(df_copy['population_affected'])
            return df_copy.nlargest(n, 'combined_risk')
        else:
            return df.head(n)
    
    def get_state_summary(self, df):
        """Get summary statistics by state"""
        
        if df is None or df.empty or 'state' not in df.columns:
            return pd.DataFrame()
        
        severity_col = 'predicted_severity' if 'predicted_severity' in df.columns else 'severity_level'
        
        # Aggregate by state
        state_summary = df.groupby('state').agg({
            'location': 'count',
            'population_affected': 'sum' if 'population_affected' in df.columns else 'count',
            'water_depth_m': 'mean' if 'water_depth_m' in df.columns else 'count',
            severity_col: lambda x: x.value_counts().index[0] if len(x) > 0 else 'Unknown'
        }).reset_index()
        
        state_summary.columns = ['State', 'Affected Locations', 'Total Population at Risk', 
                                 'Avg Water Depth (m)', 'Dominant Severity']
        
        state_summary = state_summary.sort_values('Total Population at Risk', ascending=False)
        
        return state_summary
    
    def add_temporal_features(self, df):
        """Add temporal features for time-series analysis"""
        
        if 'date' not in df.columns:
            return df
        
        df_copy = df.copy()
        df_copy['date'] = pd.to_datetime(df_copy['date'])
        
        df_copy['year'] = df_copy['date'].dt.year
        df_copy['month'] = df_copy['date'].dt.month
        df_copy['day_of_year'] = df_copy['date'].dt.dayofyear
        df_copy['week_of_year'] = df_copy['date'].dt.isocalendar().week
        df_copy['is_monsoon'] = df_copy['month'].isin([6, 7, 8, 9])
        
        return df_copy
    
    def get_model_info(self):
        """Get information about the loaded XGBoost model"""
        
        if not self.xgboost_model.is_trained:
            return None
        
        feature_importance = self.xgboost_model.get_feature_importance()
        
        return {
            'is_trained': self.xgboost_model.is_trained,
            'num_features': len(self.xgboost_model.feature_names),
            'features': self.xgboost_model.feature_names,
            'classes': list(self.xgboost_model.label_encoder.classes_),
            'feature_importance': feature_importance
        }
