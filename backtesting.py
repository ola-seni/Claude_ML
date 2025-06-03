"""
Backtesting Module for Claude_ML
Evaluates model performance and ROI
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from models import ModelEnsemble
from feature_pipeline import FeatureEngineer
from prediction_engine import PredictionEngine

logger = logging.getLogger('Backtesting')


class Backtester:
    """Comprehensive backtesting and performance analysis"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.db_path = config.get('database', {}).get('path', 'data/mlb_predictions.db')
        
        # Create results directory
        Path('data/backtesting').mkdir(parents=True, exist_ok=True)
        
    def run_historical_backtest(self, start_date: str, end_date: str, 
                               retrain_frequency: str = 'weekly') -> Dict:
        """Run complete historical backtest with periodic retraining"""
        logger.info(f"Running historical backtest from {start_date} to {end_date}")
        
        results = {
            'daily_results': [],
            'overall_metrics': {},
            'tier_performance': {},
            'monthly_performance': {},
            'feature_importance_evolution': [],
            'prediction_calibration': {}
        }
        
        # Initialize components
        feature_engineer = FeatureEngineer(self.db_path)
        
        # Date range processing
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Initial training period (first 30 days)
        training_end = current_date + timedelta(days=30)
        
        # Train initial model
        logger.info("Training initial model...")
        initial_training_data = feature_engineer.create_training_dataset(
            start_date, 
            training_end.strftime('%Y-%m-%d')
        )
        
        ensemble = ModelEnsemble(self.config)
        ensemble.train_models(initial_training_data)
        
        # Move to prediction phase
        current_date = training_end + timedelta(days=1)
        last_retrain = training_end
        
        while current_date <= end_date_obj:
            date_str = current_date.strftime('%Y-%m-%d')
            
            # Check if we need to retrain
            days_since_retrain = (current_date - last_retrain).days
            
            if retrain_frequency == 'weekly' and days_since_retrain >= 7:
                logger.info(f"Retraining model for {date_str}")
                
                # Retrain with expanding window
                retrain_data = feature_engineer.create_training_dataset(
                    start_date,
                    current_date.strftime('%Y-%m-%d')
                )
                
                if not retrain_data.empty:
                    ensemble.train_models(retrain_data)
                    last_retrain = current_date
                    
            elif retrain_frequency == 'monthly' and days_since_retrain >= 30:
                logger.info(f"Retraining model for {date_str}")
                
                retrain_data = feature_engineer.create_training_dataset(
                    start_date,
                    current_date.strftime('%Y-%m-%d')
                )
                
                if not retrain_data.empty:
                    ensemble.train_models(retrain_data)
                    last_retrain = current_date
                    
            # Generate predictions for this date
            try:
                daily_result = self._backtest_single_day(
                    ensemble, 
                    feature_engineer, 
                    date_str
                )
                
                if daily_result:
                    results['daily_results'].append(daily_result)
                    
            except Exception as e:
                logger.warning(f"Failed to backtest {date_str}: {e}")
                
            current_date += timedelta(days=1)
            
        # Analyze results
        if results['daily_results']:
            results['overall_metrics'] = self._calculate_overall_metrics(results['daily_results'])
            results['tier_performance'] = self._analyze_tier_performance(results['daily_results'])
            results['monthly_performance'] = self._analyze_monthly_performance(results['daily_results'])
            results['prediction_calibration'] = self._analyze_prediction_calibration(results['daily_results'])
            
        # Save results
        self._save_backtest_results(results, start_date, end_date)
        
        return results
        
    def _backtest_single_day(self, ensemble: ModelEnsemble, 
                            feature_engineer: FeatureEngineer, 
                            date: str) -> Optional[Dict]:
        """Backtest a single day"""
        
        # Create features for this date
        features_df = feature_engineer.create_features_for_date(date)
        
        if features_df.empty:
            return None
            
        # Apply same filters as real predictions
        prediction_engine = PredictionEngine(self.config)
        prediction_engine.ensemble = ensemble
        prediction_engine.feature_engineer = feature_engineer
        
        filtered_df = prediction_engine._apply_prediction_filters(features_df)
        
        if filtered_df.empty:
            return None
            
        # Generate predictions
        probabilities = ensemble.predict_probability(filtered_df)
        
        # Create predictions dataframe
        predictions_df = filtered_df[['player_name', 'game_id', 'date']].copy()
        predictions_df['hr_probability'] = probabilities
        predictions_df['actual_hr'] = filtered_df['home_run'].astype(int)
        
        # Assign tiers
        predictions_df = prediction_engine._assign_confidence_tiers(predictions_df)
        
        # Select top predictions
        final_predictions = prediction_engine._select_final_predictions(predictions_df)
        
        # Calculate daily metrics
        daily_result = {
            'date': date,
            'total_predictions': len(final_predictions),
            'total_hrs': final_predictions['actual_hr'].sum(),
            'accuracy': final_predictions['actual_hr'].mean(),
            'avg_probability': final_predictions['hr_probability'].mean(),
            'max_probability': final_predictions['hr_probability'].max(),
            'predictions': final_predictions.to_dict('records')
        }
        
        # Tier-specific metrics
        for tier in ['premium', 'standard', 'value']:
            tier_preds = final_predictions[final_predictions['confidence_tier'] == tier]
            if not tier_preds.empty:
                daily_result[f'{tier}_count'] = len(tier_preds)
                daily_result[f'{tier}_hits'] = tier_preds['actual_hr'].sum()
                daily_result[f'{tier}_accuracy'] = tier_preds['actual_hr'].mean()
                daily_result[f'{tier}_avg_prob'] = tier_preds['hr_probability'].mean()
                
        return daily_result
        
    def _calculate_overall_metrics(self, daily_results: List[Dict]) -> Dict:
        """Calculate overall performance metrics"""
        
        # Combine all predictions
        all_predictions = []
        for day in daily_results:
            all_predictions.extend(day['predictions'])
            
        if not all_predictions:
            return {}
            
        df = pd.DataFrame(all_predictions)
        
        metrics = {
            'total_days': len(daily_results),
            'total_predictions': len(df),
            'total_home_runs': df['actual_hr'].sum(),
            'overall_accuracy': df['actual_hr'].mean(),
            'avg_daily_predictions': len(df) / len(daily_results),
            'avg_daily_hrs': df['actual_hr'].sum() / len(daily_results),
            'home_run_rate': df['actual_hr'].mean(),
            'avg_prediction_probability': df['hr_probability'].mean()
        }
        
        # Probability calibration
        prob_bins = np.arange(0, 1.1, 0.1)
        metrics['calibration'] = {}
        
        for i in range(len(prob_bins) - 1):
            bin_mask = (df['hr_probability'] >= prob_bins[i]) & (df['hr_probability'] < prob_bins[i+1])
            bin_predictions = df[bin_mask]
            
            if len(bin_predictions) > 0:
                metrics['calibration'][f'{prob_bins[i]:.1f}-{prob_bins[i+1]:.1f}'] = {
                    'predicted_prob': bin_predictions['hr_probability'].mean(),
                    'actual_rate': bin_predictions['actual_hr'].mean(),
                    'count': len(bin_predictions)
                }
                
        # ROI Analysis (hypothetical betting)
        metrics['roi_analysis'] = self._calculate_roi_metrics(df)
        
        return metrics
        
    def _analyze_tier_performance(self, daily_results: List[Dict]) -> Dict:
        """Analyze performance by confidence tier"""
        
        tier_metrics = {}
        
        for tier in ['premium', 'standard', 'value']:
            total_predictions = sum(day.get(f'{tier}_count', 0) for day in daily_results)
            total_hits = sum(day.get(f'{tier}_hits', 0) for day in daily_results)
            
            if total_predictions > 0:
                tier_metrics[tier] = {
                    'total_predictions': total_predictions,
                    'total_hits': total_hits,
                    'accuracy': total_hits / total_predictions,
                    'hit_rate': total_hits / len([d for d in daily_results if d.get(f'{tier}_count', 0) > 0]),
                    'avg_daily_predictions': total_predictions / len(daily_results)
                }
                
        return tier_metrics
        
    def _analyze_monthly_performance(self, daily_results: List[Dict]) -> Dict:
        """Analyze performance by month"""
        
        monthly_data = {}
        
        for day in daily_results:
            date_obj = datetime.strptime(day['date'], '%Y-%m-%d')
            month_key = date_obj.strftime('%Y-%m')
            
            if month_key not in monthly_data:
                monthly_data[month_key] = {
                    'days': 0,
                    'predictions': 0,
                    'hits': 0,
                    'total_probability': 0
                }
                
            monthly_data[month_key]['days'] += 1
            monthly_data[month_key]['predictions'] += day['total_predictions']
            monthly_data[month_key]['hits'] += day['total_hrs']
            monthly_data[month_key]['total_probability'] += day['avg_probability'] * day['total_predictions']
            
        # Calculate monthly metrics
        monthly_metrics = {}
        for month, data in monthly_data.items():
            if data['predictions'] > 0:
                monthly_metrics[month] = {
                    'days': data['days'],
                    'total_predictions': data['predictions'],
                    'total_hits': data['hits'],
                    'accuracy': data['hits'] / data['predictions'],
                    'avg_daily_predictions': data['predictions'] / data['days'],
                    'avg_prediction_probability': data['total_probability'] / data['predictions']
                }
                
        return monthly_metrics
        
    def _analyze_prediction_calibration(self, daily_results: List[Dict]) -> Dict:
        """Analyze how well predicted probabilities match actual outcomes"""
        
        all_predictions = []
        for day in daily_results:
            all_predictions.extend(day['predictions'])
            
        if not all_predictions:
            return {}
            
        df = pd.DataFrame(all_predictions)
        
        # Bin predictions by probability
        df['prob_bin'] = pd.cut(df['hr_probability'], bins=10, include_lowest=True)
        
        calibration_data = df.groupby('prob_bin').agg({
            'hr_probability': ['mean', 'count'],
            'actual_hr': 'mean'
        }).round(3)
        
        calibration_data.columns = ['avg_predicted_prob', 'count', 'actual_rate']
        
        # Calculate calibration metrics
        calibration_metrics = {
            'calibration_curve': calibration_data.to_dict('index'),
            'brier_score': np.mean((df['hr_probability'] - df['actual_hr']) ** 2),
            'log_loss': -np.mean(df['actual_hr'] * np.log(df['hr_probability'] + 1e-15) + 
                               (1 - df['actual_hr']) * np.log(1 - df['hr_probability'] + 1e-15))
        }
        
        return calibration_metrics
        
    def _calculate_roi_metrics(self, predictions_df: pd.DataFrame) -> Dict:
        """Calculate ROI metrics for hypothetical betting"""
        
        # Simple ROI calculation
        # Assume we bet $10 on each prediction
        # Payout odds roughly = 1 / probability
        
        bet_amount = 10
        total_bet = len(predictions_df) * bet_amount
        total_winnings = 0
        
        for _, pred in predictions_df.iterrows():
            if pred['actual_hr'] == 1:  # Won the bet
                # Simple payout calculation
                payout_odds = max(1.5, 1 / pred['hr_probability'])  # Minimum 1.5x odds
                total_winnings += bet_amount * payout_odds
                
        roi = (total_winnings - total_bet) / total_bet if total_bet > 0 else 0
        
        roi_metrics = {
            'total_bets': len(predictions_df),
            'total_bet_amount': total_bet,
            'total_winnings': total_winnings,
            'net_profit': total_winnings - total_bet,
            'roi_percentage': roi * 100,
            'break_even_rate': 1 / np.mean([max(1.5, 1/p) for p in predictions_df['hr_probability']])
        }
        
        return roi_metrics
        
    def _save_backtest_results(self, results: Dict, start_date: str, end_date: str):
        """Save backtest results"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = Path('data/backtesting')
        
        # Save detailed results as JSON
        import json
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
                
        # Clean results for JSON
        json_results = json.loads(json.dumps(results, default=convert_numpy))
        
        with open(results_dir / f'backtest_results_{start_date}_{end_date}_{timestamp}.json', 'w') as f:
            json.dump(json_results, f, indent=2)
            
        # Create summary report
        self._create_summary_report(results, start_date, end_date, timestamp)
        
        # Create visualizations
        self._create_performance_charts(results, start_date, end_date, timestamp)
        
        logger.info(f"Backtest results saved with timestamp: {timestamp}")
        
    def _create_summary_report(self, results: Dict, start_date: str, 
                             end_date: str, timestamp: str):
        """Create human-readable summary report"""
        
        report = []
        report.append("CLAUDE_ML BACKTESTING REPORT")
        report.append("=" * 50)
        report.append(f"Period: {start_date} to {end_date}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall metrics
        overall = results.get('overall_metrics', {})
        if overall:
            report.append("OVERALL PERFORMANCE")
            report.append("-" * 20)
            report.append(f"Total Days: {overall.get('total_days', 0)}")
            report.append(f"Total Predictions: {overall.get('total_predictions', 0)}")
            report.append(f"Total Home Runs: {overall.get('total_home_runs', 0)}")
            report.append(f"Overall Accuracy: {overall.get('overall_accuracy', 0):.1%}")
            report.append(f"Avg Daily Predictions: {overall.get('avg_daily_predictions', 0):.1f}")
            report.append(f"Home Run Rate: {overall.get('home_run_rate', 0):.1%}")
            report.append("")
            
        # Tier performance
        tier_perf = results.get('tier_performance', {})
        if tier_perf:
            report.append("TIER PERFORMANCE")
            report.append("-" * 20)
            for tier, metrics in tier_perf.items():
                report.append(f"{tier.upper()}:")
                report.append(f"  Predictions: {metrics.get('total_predictions', 0)}")
                report.append(f"  Hits: {metrics.get('total_hits', 0)}")
                report.append(f"  Accuracy: {metrics.get('accuracy', 0):.1%}")
                report.append("")
                
        # ROI Analysis
        roi = overall.get('roi_analysis', {})
        if roi:
            report.append("ROI ANALYSIS (Hypothetical)")
            report.append("-" * 25)
            report.append(f"Total Bets: {roi.get('total_bets', 0)}")
            report.append(f"Net Profit: ${roi.get('net_profit', 0):.2f}")
            report.append(f"ROI: {roi.get('roi_percentage', 0):.1f}%")
            report.append(f"Break-even Rate: {roi.get('break_even_rate', 0):.1%}")
            report.append("")
            
        # Top monthly performance
        monthly = results.get('monthly_performance', {})
        if monthly:
            report.append("MONTHLY PERFORMANCE (Top 3)")
            report.append("-" * 30)
            
            # Sort by accuracy
            sorted_months = sorted(
                monthly.items(), 
                key=lambda x: x[1].get('accuracy', 0), 
                reverse=True
            )
            
            for month, metrics in sorted_months[:3]:
                report.append(f"{month}: {metrics.get('accuracy', 0):.1%} accuracy "
                           f"({metrics.get('total_hits', 0)}/{metrics.get('total_predictions', 0)})")
                
        # Save report
        report_path = Path('data/backtesting') / f'summary_report_{start_date}_{end_date}_{timestamp}.txt'
        with open(report_path, 'w') as f:
            f.write('\n'.join(report))
            
    def _create_performance_charts(self, results: Dict, start_date: str, 
                                 end_date: str, timestamp: str):
        """Create performance visualization charts"""
        
        try:
            daily_results = results.get('daily_results', [])
            if not daily_results:
                return
                
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Claude_ML Backtest Performance ({start_date} to {end_date})', fontsize=16)
            
            # 1. Daily accuracy over time
            dates = [datetime.strptime(d['date'], '%Y-%m-%d') for d in daily_results]
            accuracies = [d['accuracy'] for d in daily_results]
            
            axes[0, 0].plot(dates, accuracies, linewidth=1, alpha=0.7)
            axes[0, 0].set_title('Daily Accuracy Over Time')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Add trend line
            if len(dates) > 1:
                z = np.polyfit(range(len(dates)), accuracies, 1)
                p = np.poly1d(z)
                axes[0, 0].plot(dates, p(range(len(dates))), "r--", alpha=0.8, linewidth=2)
                
            # 2. Monthly performance
            monthly = results.get('monthly_performance', {})
            if monthly:
                months = list(monthly.keys())
                monthly_acc = [monthly[m]['accuracy'] for m in months]
                
                axes[0, 1].bar(months, monthly_acc)
                axes[0, 1].set_title('Monthly Accuracy')
                axes[0, 1].set_ylabel('Accuracy')
                axes[0, 1].tick_params(axis='x', rotation=45)
                
            # 3. Tier performance comparison
            tier_perf = results.get('tier_performance', {})
            if tier_perf:
                tiers = list(tier_perf.keys())
                tier_acc = [tier_perf[t]['accuracy'] for t in tiers]
                
                bars = axes[1, 0].bar(tiers, tier_acc, color=['gold', 'silver', 'orange'])
                axes[1, 0].set_title('Accuracy by Confidence Tier')
                axes[1, 0].set_ylabel('Accuracy')
                
                # Add value labels on bars
                for bar, acc in zip(bars, tier_acc):
                    height = bar.get_height()
                    axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                   f'{acc:.1%}', ha='center', va='bottom')
                    
            # 4. Prediction calibration
            calibration = results.get('prediction_calibration', {})
            calib_curve = calibration.get('calibration_curve', {})
            
            if calib_curve:
                bin_centers = []
                actual_rates = []
                predicted_probs = []
                
                for bin_name, data in calib_curve.items():
                    if isinstance(data, dict) and data.get('count', 0) > 5:  # Only bins with enough samples
                        predicted_probs.append(data['avg_predicted_prob'])
                        actual_rates.append(data['actual_rate'])
                        
                if predicted_probs and actual_rates:
                    axes[1, 1].scatter(predicted_probs, actual_rates, alpha=0.7, s=50)
                    axes[1, 1].plot([0, 1], [0, 1], 'r--', alpha=0.8)  # Perfect calibration line
                    axes[1, 1].set_title('Prediction Calibration')
                    axes[1, 1].set_xlabel('Predicted Probability')
                    axes[1, 1].set_ylabel('Actual Rate')
                    axes[1, 1].set_xlim(0, 1)
                    axes[1, 1].set_ylim(0, 1)
                    
            plt.tight_layout()
            
            # Save chart
            chart_path = Path('data/backtesting') / f'performance_charts_{start_date}_{end_date}_{timestamp}.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Performance charts saved to {chart_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create performance charts: {e}")


def main():
    """Run backtesting from command line"""
    import yaml
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # Initialize backtester
    backtester = Backtester(config)
    
    # Run backtest for last 30 days
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    print(f"Running backtest from {start_date} to {end_date}")
    
    results = backtester.run_historical_backtest(
        start_date, 
        end_date, 
        retrain_frequency='weekly'
    )
    
    # Print summary
    overall = results.get('overall_metrics', {})
    if overall:
        print(f"\nBacktest Results:")
        print(f"Total Predictions: {overall.get('total_predictions', 0)}")
        print(f"Overall Accuracy: {overall.get('overall_accuracy', 0):.1%}")
        print(f"ROI: {overall.get('roi_analysis', {}).get('roi_percentage', 0):.1f}%")


if __name__ == "__main__":
    main()