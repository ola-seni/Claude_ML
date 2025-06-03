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

# Monkey patch to fix all model training methods
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

def fixed_train_neural_network(self, X_train, y_train, X_test, y_test):
    """Fixed Neural Network training without sample_weight"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    
    # Neural networks need scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Neural network parameters (simplified)
    params = {
        'hidden_layer_sizes': (50, 25),  # Smaller for faster training
        'activation': 'relu',
        'solver': 'adam',
        'alpha': 0.001,
        'learning_rate': 'adaptive',
        'max_iter': 200,  # Reduced for faster training
        'random_state': self.config['model']['random_state'],
        'early_stopping': True,
        'validation_fraction': 0.1
    }
    
    model = MLPClassifier(**params)
    model.fit(X_train_scaled, y_train)  # Without sample_weight
    
    return model, scaler

def fixed_train_random_forest(self, X_train, y_train, X_test, y_test):
    """Fixed Random Forest training"""
    from sklearn.ensemble import RandomForestClassifier
    
    # No scaling needed for Random Forest
    scaler = None
    
    # Random Forest parameters
    params = {
        'n_estimators': 100,  # Reduced for faster training
        'max_depth': 8,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'max_features': 'sqrt',
        'random_state': self.config['model']['random_state'],
        'class_weight': 'balanced',
        'n_jobs': -1
    }
    
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    return model, scaler

def fixed_train_logistic_regression(self, X_train, y_train, X_test, y_test):
    """Fixed Logistic Regression training"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    
    # Logistic regression needs scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Logistic regression parameters
    params = {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'liblinear',
        'random_state': self.config['model']['random_state'],
        'class_weight': 'balanced',
        'max_iter': 1000
    }
    
    model = LogisticRegression(**params)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

# Apply all patches
ModelEnsemble._train_xgboost = fixed_train_xgboost
ModelEnsemble._train_neural_network = fixed_train_neural_network
ModelEnsemble._train_random_forest = fixed_train_random_forest
ModelEnsemble._train_logistic_regression = fixed_train_logistic_regression

print("🚀 Training with all compatibility fixes applied...")

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
    
    # Test a quick prediction
    print("\n🔮 Testing predictions...")
    try:
        sample_features = training_data.head(5)
        probabilities = ensemble.predict_probability(sample_features)
        print(f"Sample predictions: {probabilities}")
        print("✅ Prediction system working!")
    except Exception as e:
        print(f"⚠️  Prediction test failed: {e}")
else:
    print("❌ No training data available")
