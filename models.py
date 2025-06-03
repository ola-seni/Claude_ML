"""
ML Models for Claude_ML Home Run Prediction
Trains and manages ensemble of models
"""

import pandas as pd
import numpy as np
import sqlite3
import joblib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
import xgboost as xgb
from sklearn.neural_network import MLPClassifier

# For handling class imbalance
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

logger = logging.getLogger('ModelTrainer')


class ModelEnsemble:
    """Ensemble of ML models for home run prediction"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.ensemble_weights = {}
        self.is_trained = False
        
        # Create model directory
        Path('models/trained').mkdir(parents=True, exist_ok=True)
        Path('models/scalers').mkdir(parents=True, exist_ok=True)
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        logger.info("Preparing training data")
        
        # Separate features and target
        X = df[self.feature_columns].copy()
        y = df['home_run'].copy()
        
        # Handle any remaining NaN values
        X = X.fillna(0)
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        logger.info(f"Training data shape: {X.shape}")
        logger.info(f"Positive class ratio: {y.mean():.3%}")
        
        return X.values, y.values
        
    def train_models(self, training_data: pd.DataFrame):
        """Train all models in the ensemble"""
        logger.info("Starting model training")
        
        # Load feature columns
        with open('data/processed/features/feature_columns.json', 'r') as f:
            self.feature_columns = json.load(f)
            
        # Prepare data
        X, y = self.prepare_data(training_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=1-self.config['model']['train_test_split'],
            random_state=self.config['model']['random_state'],
            stratify=y
        )
        
        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=self.config['model']['random_state'])
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        
        logger.info(f"After SMOTE - Training samples: {len(X_train_balanced)}, Positive ratio: {y_train_balanced.mean():.3%}")
        
        # Train each model
        models_to_train = self.config['model']['algorithms']
        test_scores = {}
        
        for model_name in models_to_train:
            logger.info(f"Training {model_name}")
            
            if model_name == 'xgboost':
                model, scaler = self._train_xgboost(X_train_balanced, y_train_balanced, X_test, y_test)
            elif model_name == 'random_forest':
                model, scaler = self._train_random_forest(X_train_balanced, y_train_balanced, X_test, y_test)
            elif model_name == 'neural_network':
                model, scaler = self._train_neural_network(X_train_balanced, y_train_balanced, X_test, y_test)
            elif model_name == 'logistic_regression':
                model, scaler = self._train_logistic_regression(X_train_balanced, y_train_balanced, X_test, y_test)
            else:
                logger.warning(f"Unknown model: {model_name}")
                continue
                
            # Store model and scaler
            self.models[model_name] = model
            self.scalers[model_name] = scaler
            
            # Test performance
            X_test_scaled = scaler.transform(X_test) if scaler else X_test
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            test_scores[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'auc': auc
            }
            
            logger.info(f"{model_name} - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, AUC: {auc:.3f}")
            
        # Calculate ensemble weights based on AUC scores
        self._calculate_ensemble_weights(test_scores)
        
        # Save models
        self._save_models()
        
        self.is_trained = True
        logger.info("Model training completed")
        
        return test_scores
        
    def _train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model"""
        # No scaling needed for XGBoost
        scaler = None
        
        # XGBoost parameters
        params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.config['model']['random_state'],
            'eval_metric': 'auc'
        }
        
        # Calculate class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        scale_pos_weight = class_weights[1] / class_weights[0]
        params['scale_pos_weight'] = scale_pos_weight
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        return model, scaler
        
    def _train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model"""
        # No scaling needed for Random Forest
        scaler = None
        
        # Random Forest parameters
        params = {
            'n_estimators': 200,
            'max_depth': 10,
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
        
    def _train_neural_network(self, X_train, y_train, X_test, y_test):
        """Train Neural Network model"""
        # Neural networks need scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Calculate class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        # Neural network parameters
        params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,
            'learning_rate': 'adaptive',
            'max_iter': 500,
            'random_state': self.config['model']['random_state'],
            'early_stopping': True,
            'validation_fraction': 0.1
        }
        
        model = MLPClassifier(**params)
        
        # Apply class weights manually through sample weights
        sample_weights = np.array([class_weight_dict[label] for label in y_train])
        model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
        
        return model, scaler
        
    def _train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """Train Logistic Regression model"""
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
        
    def _calculate_ensemble_weights(self, test_scores: Dict):
        """Calculate ensemble weights based on model performance"""
        if self.config['model']['ensemble_method'] == 'weighted_average':
            # Weight by AUC score
            total_auc = sum(scores['auc'] for scores in test_scores.values())
            
            for model_name, scores in test_scores.items():
                self.ensemble_weights[model_name] = scores['auc'] / total_auc
                
        else:  # Equal weights
            num_models = len(test_scores)
            for model_name in test_scores.keys():
                self.ensemble_weights[model_name] = 1.0 / num_models
                
        logger.info(f"Ensemble weights: {self.ensemble_weights}")
        
    def predict_probability(self, features: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions"""
        if not self.is_trained:
            self.load_models()
            
        # Prepare features
        X = features[self.feature_columns].fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Get predictions from each model
        ensemble_probs = np.zeros(len(X))
        
        for model_name, model in self.models.items():
            # Scale features if needed
            if self.scalers[model_name] is not None:
                X_scaled = self.scalers[model_name].transform(X)
            else:
                X_scaled = X.values
                
            # Get probability predictions
            probs = model.predict_proba(X_scaled)[:, 1]
            
            # Add to ensemble with weight
            weight = self.ensemble_weights[model_name]
            ensemble_probs += weight * probs
            
        return ensemble_probs
        
    def predict(self, features: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Make binary predictions"""
        probs = self.predict_probability(features)
        return (probs >= threshold).astype(int)
        
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from tree-based models"""
        importance_df = pd.DataFrame({'feature': self.feature_columns})
        
        # XGBoost importance
        if 'xgboost' in self.models:
            xgb_importance = self.models['xgboost'].feature_importances_
            importance_df['xgboost'] = xgb_importance
            
        # Random Forest importance
        if 'random_forest' in self.models:
            rf_importance = self.models['random_forest'].feature_importances_
            importance_df['random_forest'] = rf_importance
            
        # Average importance
        numeric_cols = [col for col in importance_df.columns if col != 'feature']
        if numeric_cols:
            importance_df['average'] = importance_df[numeric_cols].mean(axis=1)
            importance_df = importance_df.sort_values('average', ascending=False)
            
        return importance_df
        
    def _save_models(self):
        """Save trained models and scalers"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for model_name, model in self.models.items():
            # Save model
            model_path = f'models/trained/{model_name}_{timestamp}.joblib'
            joblib.dump(model, model_path)
            
            # Save scaler if exists
            if self.scalers[model_name] is not None:
                scaler_path = f'models/scalers/{model_name}_scaler_{timestamp}.joblib'
                joblib.dump(self.scalers[model_name], scaler_path)
                
        # Save ensemble configuration
        ensemble_config = {
            'ensemble_weights': self.ensemble_weights,
            'feature_columns': self.feature_columns,
            'timestamp': timestamp
        }
        
        with open(f'models/ensemble_config_{timestamp}.json', 'w') as f:
            json.dump(ensemble_config, f, indent=2)
            
        # Save "latest" links
        with open('models/latest_timestamp.txt', 'w') as f:
            f.write(timestamp)
            
        logger.info(f"Models saved with timestamp: {timestamp}")
        
    def load_models(self):
        """Load latest trained models"""
        try:
            # Get latest timestamp
            with open('models/latest_timestamp.txt', 'r') as f:
                timestamp = f.read().strip()
                
            # Load ensemble config
            with open(f'models/ensemble_config_{timestamp}.json', 'r') as f:
                config = json.load(f)
                
            self.ensemble_weights = config['ensemble_weights']
            self.feature_columns = config['feature_columns']
            
            # Load models
            for model_name in self.ensemble_weights.keys():
                model_path = f'models/trained/{model_name}_{timestamp}.joblib'
                self.models[model_name] = joblib.load(model_path)
                
                # Load scaler if exists
                scaler_path = f'models/scalers/{model_name}_scaler_{timestamp}.joblib'
                if Path(scaler_path).exists():
                    self.scalers[model_name] = joblib.load(scaler_path)
                else:
                    self.scalers[model_name] = None
                    
            self.is_trained = True
            logger.info(f"Models loaded from timestamp: {timestamp}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
            
    def cross_validate(self, training_data: pd.DataFrame, cv_folds: int = 5):
        """Perform cross-validation"""
        logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        X, y = self.prepare_data(training_data)
        cv_results = {}
        
        for model_name in self.config['model']['algorithms']:
            logger.info(f"Cross-validating {model_name}")
            
            if model_name == 'xgboost':
                model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    max_depth=6,
                    learning_rate=0.1,
                    n_estimators=100,  # Reduced for CV
                    random_state=self.config['model']['random_state']
                )
                X_cv = X
                
            elif model_name == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=100,  # Reduced for CV
                    max_depth=10,
                    random_state=self.config['model']['random_state'],
                    class_weight='balanced'
                )
                X_cv = X
                
            elif model_name == 'neural_network':
                model = MLPClassifier(
                    hidden_layer_sizes=(50,),  # Simplified for CV
                    random_state=self.config['model']['random_state'],
                    max_iter=200
                )
                scaler = StandardScaler()
                X_cv = scaler.fit_transform(X)
                
            elif model_name == 'logistic_regression':
                model = LogisticRegression(
                    random_state=self.config['model']['random_state'],
                    class_weight='balanced'
                )
                scaler = StandardScaler()
                X_cv = scaler.fit_transform(X)
                
            # Perform cross-validation
            cv_scores = cross_val_score(model, X_cv, y, cv=cv_folds, scoring='roc_auc')
            
            cv_results[model_name] = {
                'mean_auc': cv_scores.mean(),
                'std_auc': cv_scores.std(),
                'scores': cv_scores.tolist()
            }
            
            logger.info(f"{model_name} CV AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
            
        return cv_results


def main():
    """Train models from command line"""
    import yaml
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # Load training data
    training_data = pd.read_csv('data/processed/training/training_data.csv')
    
    # Initialize and train models
    ensemble = ModelEnsemble(config)
    
    # Cross-validation first
    cv_results = ensemble.cross_validate(training_data)
    print("\nCross-validation results:")
    for model, results in cv_results.items():
        print(f"{model}: {results['mean_auc']:.3f} (+/- {results['std_auc']*2:.3f})")
        
    # Train final models
    test_scores = ensemble.train_models(training_data)
    
    # Display results
    print("\nFinal test scores:")
    for model, scores in test_scores.items():
        print(f"{model}: Accuracy={scores['accuracy']:.3f}, AUC={scores['auc']:.3f}")
        
    # Feature importance
    feature_importance = ensemble.get_feature_importance()
    print(f"\nTop 10 most important features:")
    print(feature_importance.head(10)[['feature', 'average']])


if __name__ == "__main__":
    main()