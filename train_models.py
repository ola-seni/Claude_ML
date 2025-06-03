from feature_pipeline import FeatureEngineer
from models import ModelEnsemble
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Create features from our collected data
engineer = FeatureEngineer()
print("Creating training features from collected data...")
training_data = engineer.create_training_dataset('2024-04-01', '2024-07-01')

if not training_data.empty:
    print(f"✅ Created {len(training_data)} training samples")
    
    # Train models
    ensemble = ModelEnsemble(config)
    print("🤖 Training ML models...")
    test_scores = ensemble.train_models(training_data)
    
    print("\n🎉 Training completed!")
    for model, scores in test_scores.items():
        print(f"{model}: Accuracy={scores['accuracy']:.3f}, AUC={scores['auc']:.3f}")
else:
    print("❌ No training data available")
