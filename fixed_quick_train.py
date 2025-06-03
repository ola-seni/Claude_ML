from feature_pipeline import FeatureEngineer
from models import ModelEnsemble
import yaml

# Monkey patch to skip Statcast features
original_get_statcast = FeatureEngineer._get_statcast_features
def fast_get_statcast_features(self, player_name, date):
    # Return empty features instead of loading CSV
    return {
        'barrel_rate': 0,
        'sweet_spot_rate': 0,
        'max_exit_velo': 0,
        'avg_hit_distance': 0,
        'pull_rate': 0,
        'oppo_rate': 0
    }
FeatureEngineer._get_statcast_features = fast_get_statcast_features

# Monkey patch to fix XGBoost training method
original_train_xgboost = ModelEnsemble._train_xgboost
def fixed_train_xgboost(self, X_train, y_train, X_test, y_test):
    """Fixed XGBoost training without early_stopping_rounds"""
    import xgboost as xgb
    from sklearn.utils.class_weight import compute_class_weight
    import numpy as np
    
    # No scaling needed for XGBoost
    scaler = None
    
    # XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100,  # Reduced for faster training
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': self.config['model']['random_state'],
        'eval_metric': 'auc'
    }
    
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    scale_pos_weight = class_weights[1] / class_weights[0]
    params['scale_pos_weight'] = scale_pos_weight
    
    # Train model (without early_stopping_rounds)
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)  # Simplified training
    
    return model, scaler

ModelEnsemble._train_xgboost = fixed_train_xgboost

print("🚀 Training without Statcast features and with fixed XGBoost...")

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create features from our collected data
engineer = FeatureEngineer()
print("⏳ Creating training features...")
training_data = engineer.create_training_dataset('2024-06-01', '2024-06-07')  # Just 1 week

if not training_data.empty:
    print(f"✅ Created {len(training_data)} training samples")
    
    # Train models
    ensemble = ModelEnsemble(config)
    print("🤖 Training ML models...")
    test_scores = ensemble.train_models(training_data)
    
    print("\n🎉 Training completed!")
    for model, scores in test_scores.items():
        print(f"{model}: Accuracy={scores['accuracy']:.3f}, AUC={scores['auc']:.3f}")
        
    print(f"\n📈 Home run rate in data: {training_data['home_run'].mean():.1%}")
    print(f"📊 Feature count: {len(ensemble.feature_columns)}")
else:
    print("❌ No training data available")
