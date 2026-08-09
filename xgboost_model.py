import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import pickle
import os

class FloodSeverityXGBoostModel:
    """XGBoost classifier for flood severity prediction"""
    
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.is_trained = False
        
    def prepare_features(self, df):
        """Prepare features for training/prediction"""
        
        # Define feature columns
        feature_columns = [
            'rainfall_mm',
            'prev_24h_rainfall_mm',
            'prev_48h_rainfall_mm',
            'river_level_m',
            'soil_moisture_percent',
            'terrain_elevation_m',
            'distance_to_river_km',
            'drainage_capacity_percent',
            'historical_flood_count',
            'population_density',
            'urban_area_percent',
            'water_depth_m'
        ]
        
        # Select only available features
        available_features = [col for col in feature_columns if col in df.columns]
        X = df[available_features].copy()
        
        # Feature engineering
        if 'rainfall_mm' in X.columns and 'prev_24h_rainfall_mm' in X.columns:
            X['rainfall_increase_rate'] = X['rainfall_mm'] - X['prev_24h_rainfall_mm']
        
        if 'river_level_m' in X.columns and 'terrain_elevation_m' in X.columns:
            X['flood_risk_ratio'] = X['river_level_m'] / (X['terrain_elevation_m'] + 1)
        
        if 'soil_moisture_percent' in X.columns and 'drainage_capacity_percent' in X.columns:
            X['saturation_index'] = X['soil_moisture_percent'] / (X['drainage_capacity_percent'] + 1)
        
        if 'population_density' in X.columns and 'urban_area_percent' in X.columns:
            X['urban_population_factor'] = (X['population_density'] * X['urban_area_percent']) / 100
        
        # Handle missing values
        X = X.fillna(0)
        
        self.feature_names = X.columns.tolist()
        
        return X
    
    def train(self, data_path='flood_training_data.csv', test_size=0.2, random_state=42):
        """Train the XGBoost model"""
        
        print("Loading training data...")
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} records")
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Encode target labels
        y = self.label_encoder.fit_transform(df['severity_level'])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTraining set: {len(X_train)} records")
        print(f"Test set: {len(X_test)} records")
        
        # Define XGBoost parameters
        params = {
            'objective': 'multi:softmax',
            'num_class': len(self.label_encoder.classes_),
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_state,
            'eval_metric': 'mlogloss'
        }
        
        # Train model
        print("\nTraining XGBoost model...")
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate model
        print("\nEvaluating model...")
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"\nTraining Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
        # Detailed metrics
        metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'precision': precision_score(y_test, y_test_pred, average='weighted'),
            'recall': recall_score(y_test, y_test_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_test_pred, average='weighted')
        }
        
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(
            y_test, y_test_pred,
            target_names=self.label_encoder.classes_
        ))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_test_pred)
        print(cm)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        self.is_trained = True
        
        return metrics, feature_importance
    
    def predict(self, X):
        """Predict severity level for new data"""
        
        if not self.is_trained or self.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        # Prepare features
        X_prepared = self.prepare_features(X)
        
        # Make predictions
        y_pred = self.model.predict(X_prepared)
        
        # Decode labels
        severity_predictions = self.label_encoder.inverse_transform(y_pred)
        
        # Get prediction probabilities
        y_proba = self.model.predict_proba(X_prepared)
        
        return severity_predictions, y_proba
    
    def predict_single(self, features_dict):
        """Predict severity for a single location"""
        
        # Convert to DataFrame
        df = pd.DataFrame([features_dict])
        
        # Predict
        predictions, probabilities = self.predict(df)
        
        # Get confidence scores for each class
        confidence_scores = {
            self.label_encoder.classes_[i]: float(probabilities[0][i])
            for i in range(len(self.label_encoder.classes_))
        }
        
        return {
            'predicted_severity': predictions[0],
            'confidence_scores': confidence_scores,
            'max_confidence': float(probabilities[0].max())
        }
    
    def save_model(self, model_path='xgboost_flood_model.pkl'):
        """Save trained model to disk"""
        
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }
        1
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self, model_path='xgboost_flood_model.pkl'):
        """Load trained model from disk"""
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.is_trained = True
        
        print(f"Model loaded from {model_path}")
    
    def get_feature_importance(self):
        """Get feature importance as DataFrame"""
        
        if not self.is_trained:
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


def train_and_save_model():
    """Helper function to train and save the model"""
    
    model = FloodSeverityXGBoostModel()
    metrics, feature_importance = model.train('flood_training_data.csv')
    model.save_model('xgboost_flood_model.pkl')
    
    return model, metrics, feature_importance


if __name__ == "__main__":
    # Train and save model
    print("="*60)
    print("Training Flood Severity XGBoost Classifier")
    print("="*60)
    
    model, metrics, feature_importance = train_and_save_model()
    
    print("\n" + "="*60)
    print("Model training complete!")
    print("="*60)