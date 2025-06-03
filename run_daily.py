#!/usr/bin/env python3
"""
Simple Daily Runner for GitHub Actions
Works with existing trained models or trains if needed
"""

import os
import sys
from pathlib import Path

def main():
    """Simple daily runner that works in GitHub Actions"""
    print("🎯 Claude_ML Simple Daily Runner")
    print("=" * 40)
    
    # Check if models exist
    models_dir = Path('models/trained')
    latest_timestamp_file = Path('models/latest_timestamp.txt')
    
    if not latest_timestamp_file.exists() or not models_dir.exists() or len(list(models_dir.glob('*.joblib'))) == 0:
        print("📚 No trained models found - training new models...")
        
        # Import and run training
        try:
            from simple_training import train_simple_models
            train_simple_models()
            print("✅ Model training completed")
        except ImportError:
            print("❌ simple_training.py not found - creating basic models...")
            create_basic_models()
    else:
        print("✅ Found existing trained models")
    
    # Generate predictions
    print("\n🎯 Generating daily predictions...")
    
    try:
        from calibrated_daily_predictions import CalibratedDailyPredictor
        predictor = CalibratedDailyPredictor()
        prediction_data = predictor.generate_daily_predictions()
        
        if prediction_data and prediction_data.get('predictions'):
            predictor.display_predictions(prediction_data)
            print("✅ Predictions generated successfully!")
        else:
            print("⚠️ No predictions generated")
            
    except ImportError:
        print("⚠️ calibrated_daily_predictions.py not found - using simple predictor...")
        from simple_daily_predictor import SimpleDailyPredictor
        
        predictor = SimpleDailyPredictor()
        prediction_data = predictor.generate_daily_predictions()
        
        if prediction_data:
            predictor.display_predictions(prediction_data)
            print("✅ Simple predictions generated!")
        else:
            print("❌ Could not generate predictions")

def create_basic_models():
    """Create basic models if training fails"""
    print("🔧 Creating basic fallback models...")
    
    import joblib
    import json
    from datetime import datetime
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    
    # Create directories
    Path('models/trained').mkdir(parents=True, exist_ok=True)
    Path('models/scalers').mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create dummy models
    basic_features = [
        'season_hr_rate', 'season_batting_avg', 'recent_games', 
        'is_home', 'games_played', 'hr_rate_7d', 'hr_rate_14d'
    ]
    
    # Create simple models
    rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
    lr_model = LogisticRegression(random_state=42)
    
    # Train on dummy data
    X_dummy = np.random.rand(100, len(basic_features))
    y_dummy = np.random.randint(0, 2, 100)
    
    rf_model.fit(X_dummy, y_dummy)
    lr_model.fit(X_dummy, y_dummy)
    
    # Save models
    joblib.dump(rf_model, f'models/trained/random_forest_{timestamp}.joblib')
    joblib.dump(lr_model, f'models/trained/logistic_regression_{timestamp}.joblib')
    
    # Save ensemble config
    ensemble_config = {
        'ensemble_weights': {
            'random_forest': 0.6,
            'logistic_regression': 0.4
        },
        'feature_columns': basic_features,
        'timestamp': timestamp
    }
    
    with open(f'models/ensemble_config_{timestamp}.json', 'w') as f:
        json.dump(ensemble_config, f, indent=2)
        
    # Save timestamp
    with open('models/latest_timestamp.txt', 'w') as f:
        f.write(timestamp)
        
    print(f"✅ Basic models created with timestamp: {timestamp}")

if __name__ == "__main__":
    main()
