"""
Prediction Engine for Claude_ML
Generates daily home run predictions
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json

from models import ModelEnsemble
from feature_pipeline import FeatureEngineer

logger = logging.getLogger('PredictionEngine')


class PredictionEngine:
    """Generates and manages daily home run predictions"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.ensemble = ModelEnsemble(config)
        self.feature_engineer = FeatureEngineer()
        
        # Load trained models if they exist
        try:
            self.ensemble.load_models()
            logger.info("Loaded trained models successfully")
        except FileNotFoundError:
            logger.warning("No trained models found - need to train first")
            # Don't raise error, just continue without models
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
            
    def generate_daily_predictions(self, date: str = None) -> pd.DataFrame:
        """Generate predictions for a specific date (default: today)"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
            
        logger.info(f"Generating predictions for {date}")
        
        # Create features for today's games
        features_df = self.feature_engineer.create_features_for_date(date)
        
        if features_df.empty:
            logger.warning(f"No games found for {date}")
            return pd.DataFrame()
            
        # Apply filters
        filtered_df = self._apply_prediction_filters(features_df)
        
        if filtered_df.empty:
            logger.warning(f"No players passed filters for {date}")
            return pd.DataFrame()
            
        # Generate probability predictions
        probabilities = self.ensemble.predict_probability(filtered_df)
        
        # Create predictions dataframe
        predictions_df = filtered_df[['player_name', 'game_id', 'date']].copy()
        predictions_df['hr_probability'] = probabilities
        
        # Add confidence tiers
        predictions_df = self._assign_confidence_tiers(predictions_df)
        
        # Add game context
        predictions_df = self._add_game_context(predictions_df)
        
        # Sort by probability
        predictions_df = predictions_df.sort_values('hr_probability', ascending=False)
        
        # Select final predictions
        final_predictions = self._select_final_predictions(predictions_df)
        
        # Save predictions
        self._save_predictions(final_predictions, date)
        
        logger.info(f"Generated {len(final_predictions)} predictions for {date}")
        return final_predictions
        
    def _apply_prediction_filters(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Apply filters to select eligible players"""
        filtered_df = features_df.copy()
        
        # Filter 1: Minimum probability threshold
        min_prob = self.config['predictions']['min_probability']
        
        # Get initial probabilities to filter
        initial_probs = self.ensemble.predict_probability(filtered_df)
        filtered_df['temp_prob'] = initial_probs
        filtered_df = filtered_df[filtered_df['temp_prob'] >= min_prob]
        filtered_df = filtered_df.drop('temp_prob', axis=1)
        
        if filtered_df.empty:
            return filtered_df
            
        # Filter 2: Exclude pitchers if configured
        if self.config['predictions']['exclude_pitchers']:
            # This would need position data - for now we'll skip
            pass
            
        # Filter 3: Minimum season home runs
        min_season_hrs = self.config['predictions']['min_season_hrs']
        if 'season_hrs' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['season_hrs'] >= min_season_hrs]
            
        # Filter 4: Must have recent at-bats (active players)
        if 'games_played_7d' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['games_played_7d'] > 0]
            
        logger.info(f"Filtered from {len(features_df)} to {len(filtered_df)} eligible players")
        return filtered_df
        
    def _assign_confidence_tiers(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Assign confidence tiers to predictions"""
        df = predictions_df.copy()
        
        # Sort by probability
        df = df.sort_values('hr_probability', ascending=False)
        
        # Get tier counts from config
        premium_count = self.config['predictions']['tiers']['premium']
        standard_count = self.config['predictions']['tiers']['standard']
        value_count = self.config['predictions']['tiers']['value']
        
        # Assign tiers
        df['confidence_tier'] = 'low'
        df.iloc[:premium_count, df.columns.get_loc('confidence_tier')] = 'premium'
        df.iloc[premium_count:premium_count+standard_count, df.columns.get_loc('confidence_tier')] = 'standard'
        df.iloc[premium_count+standard_count:premium_count+standard_count+value_count, df.columns.get_loc('confidence_tier')] = 'value'
        
        return df
        
    def _add_game_context(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Add game context information"""
        df = predictions_df.copy()
        
        conn = sqlite3.connect(self.feature_engineer.db_path)
        
        # Get game information
        game_info_query = '''
            SELECT 
                g.game_id,
                g.home_team,
                g.away_team,
                g.venue,
                g.start_time,
                g.weather_temp,
                g.weather_conditions
            FROM games g
            WHERE g.game_id IN ({})
        '''.format(','.join(['?' for _ in df['game_id'].unique()]))
        
        game_info = pd.read_sql_query(
            game_info_query, 
            conn, 
            params=df['game_id'].unique().tolist()
        )
        
        # Get pitcher information
        pitcher_query = '''
            SELECT 
                pp.game_id,
                pp.pitcher_name,
                pp.era,
                pp.whip,
                pp.home_runs_allowed
            FROM pitcher_performance pp
            WHERE pp.game_id IN ({})
        '''.format(','.join(['?' for _ in df['game_id'].unique()]))
        
        pitcher_info = pd.read_sql_query(
            pitcher_query,
            conn,
            params=df['game_id'].unique().tolist()
        )
        
        conn.close()
        
        # Merge game context
        df = df.merge(game_info, on='game_id', how='left')
        df = df.merge(pitcher_info, on='game_id', how='left')
        
        # Add readable game description
        df['matchup'] = df.apply(self._create_matchup_description, axis=1)
        
        return df
        
    def _create_matchup_description(self, row) -> str:
        """Create human-readable matchup description"""
        try:
            home_team = row.get('home_team', 'Unknown')
            away_team = row.get('away_team', 'Unknown')
            venue = row.get('venue', 'Unknown Venue')
            pitcher = row.get('pitcher_name', 'Unknown Pitcher')
            temp = row.get('weather_temp', 'Unknown')
            
            return f"{away_team} @ {home_team} ({venue}) vs {pitcher}, {temp}°F"
        except:
            return "Game details unavailable"
            
    def _select_final_predictions(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Select final predictions based on configuration"""
        total_predictions = self.config['predictions']['total_predictions']
        
        # Take top predictions
        final_df = predictions_df.head(total_predictions).copy()
        
        # Add prediction metadata
        final_df['prediction_id'] = range(1, len(final_df) + 1)
        final_df['prediction_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        final_df['model_version'] = self._get_model_version()
        
        return final_df
        
    def _get_model_version(self) -> str:
        """Get current model version"""
        try:
            with open('models/latest_timestamp.txt', 'r') as f:
                return f.read().strip()
        except:
            return 'unknown'
            
    def _save_predictions(self, predictions_df: pd.DataFrame, date: str):
        """Save predictions to database"""
        conn = sqlite3.connect(self.feature_engineer.db_path)
        
        # Create table if not exists
        create_table_query = '''
            CREATE TABLE IF NOT EXISTS daily_predictions (
                prediction_id INTEGER,
                prediction_date TEXT,
                game_date TEXT,
                player_name TEXT,
                game_id TEXT,
                hr_probability REAL,
                confidence_tier TEXT,
                matchup TEXT,
                venue TEXT,
                pitcher_name TEXT,
                model_version TEXT,
                actual_result INTEGER DEFAULT NULL,
                PRIMARY KEY (prediction_date, game_date, player_name)
            )
        '''
        
        conn.execute(create_table_query)
        
        # Save predictions
        predictions_to_save = predictions_df[[
            'prediction_id', 'prediction_date', 'date', 'player_name', 
            'game_id', 'hr_probability', 'confidence_tier', 'matchup',
            'venue', 'pitcher_name', 'model_version'
        ]].copy()
        
        predictions_to_save = predictions_to_save.rename(columns={'date': 'game_date'})
        
        predictions_to_save.to_sql(
            'daily_predictions', 
            conn, 
            if_exists='append', 
            index=False
        )
        
        conn.close()
        
        # Also save as CSV for easy access
        csv_path = f'data/predictions/predictions_{date}.csv'
        predictions_df.to_csv(csv_path, index=False)
        
        logger.info(f"Saved predictions to database and {csv_path}")
        
    def update_prediction_results(self, date: str):
        """Update predictions with actual results"""
        logger.info(f"Updating prediction results for {date}")
        
        conn = sqlite3.connect(self.feature_engineer.db_path)
        
        # Get predictions for this date
        predictions_query = '''
            SELECT player_name, game_id
            FROM daily_predictions
            WHERE game_date = ? AND actual_result IS NULL
        '''
        
        predictions = pd.read_sql_query(
            predictions_query, 
            conn, 
            params=(date,)
        )
        
        if predictions.empty:
            logger.info("No predictions to update")
            conn.close()
            return
            
        # Get actual results
        results_query = '''
            SELECT player_name, game_id, home_runs
            FROM player_performance
            WHERE date = ?
        '''
        
        results = pd.read_sql_query(
            results_query,
            conn,
            params=(date,)
        )
        
        # Merge predictions with results
        merged = predictions.merge(
            results, 
            on=['player_name', 'game_id'], 
            how='left'
        )
        
        # Update database
        for _, row in merged.iterrows():
            actual_result = 1 if (row['home_runs'] > 0) else 0
            
            update_query = '''
                UPDATE daily_predictions
                SET actual_result = ?
                WHERE game_date = ? AND player_name = ? AND game_id = ?
            '''
            
            conn.execute(
                update_query,
                (actual_result, date, row['player_name'], row['game_id'])
            )
            
        conn.commit()
        conn.close()
        
        logger.info(f"Updated {len(merged)} prediction results")
        
    def get_prediction_accuracy(self, start_date: str, end_date: str) -> Dict:
        """Calculate prediction accuracy over date range"""
        conn = sqlite3.connect(self.feature_engineer.db_path)
        
        query = '''
            SELECT 
                confidence_tier,
                COUNT(*) as total_predictions,
                SUM(actual_result) as correct_predictions,
                AVG(hr_probability) as avg_probability,
                AVG(actual_result) as actual_rate
            FROM daily_predictions
            WHERE game_date >= ? AND game_date <= ?
            AND actual_result IS NOT NULL
            GROUP BY confidence_tier
        '''
        
        results = pd.read_sql_query(
            query,
            conn,
            params=(start_date, end_date)
        )
        
        conn.close()
        
        # Calculate accuracy metrics
        accuracy_metrics = {}
        
        for _, row in results.iterrows():
            tier = row['confidence_tier']
            total = row['total_predictions']
            correct = row['correct_predictions']
            
            accuracy_metrics[tier] = {
                'total_predictions': total,
                'correct_predictions': correct,
                'accuracy': correct / total if total > 0 else 0,
                'avg_probability': row['avg_probability'],
                'actual_rate': row['actual_rate']
            }
            
        # Overall accuracy
        overall_query = '''
            SELECT 
                COUNT(*) as total,
                SUM(actual_result) as correct,
                AVG(hr_probability) as avg_prob,
                AVG(actual_result) as actual_rate
            FROM daily_predictions
            WHERE game_date >= ? AND game_date <= ?
            AND actual_result IS NOT NULL
        '''
        
        conn = sqlite3.connect(self.feature_engineer.db_path)
        overall = conn.execute(overall_query, (start_date, end_date)).fetchone()
        conn.close()
        
        if overall[0] > 0:
            accuracy_metrics['overall'] = {
                'total_predictions': overall[0],
                'correct_predictions': overall[1],
                'accuracy': overall[1] / overall[0],
                'avg_probability': overall[2],
                'actual_rate': overall[3]
            }
            
        return accuracy_metrics
        
    def format_predictions_for_output(self, predictions_df: pd.DataFrame) -> str:
        """Format predictions for human-readable output"""
        if predictions_df.empty:
            return "No predictions generated for today."
            
        output = []
        output.append("🏟️ **CLAUDE_ML HOME RUN PREDICTIONS** ⚾")
        output.append(f"📅 Date: {predictions_df['date'].iloc[0]}")
        output.append(f"🎯 Total Predictions: {len(predictions_df)}\n")
        
        # Group by confidence tier
        tiers = {
            'premium': '💎 PREMIUM PICKS',
            'standard': '⭐ STANDARD PICKS', 
            'value': '💰 VALUE PICKS'
        }
        
        for tier_key, tier_name in tiers.items():
            tier_picks = predictions_df[predictions_df['confidence_tier'] == tier_key]
            
            if not tier_picks.empty:
                output.append(f"\n{tier_name}")
                output.append("=" * len(tier_name))
                
                for i, (_, pick) in enumerate(tier_picks.iterrows(), 1):
                    prob = pick['hr_probability']
                    player = pick['player_name']
                    matchup = pick.get('matchup', 'Game details TBA')
                    
                    output.append(f"{i}. **{player}** ({prob:.1%})")
                    output.append(f"   {matchup}")
                    
                    # Add key factors if available
                    factors = self._get_key_factors(pick)
                    if factors:
                        output.append(f"   🔑 {factors}")
                    output.append("")
                    
        # Add disclaimer
        output.append("\n⚠️ **DISCLAIMER**")
        output.append("These are AI-generated predictions for entertainment purposes.")
        output.append("Past performance does not guarantee future results.")
        
        return "\n".join(output)
        
    def _get_key_factors(self, prediction_row) -> str:
        """Extract key factors driving the prediction"""
        factors = []
        
        # Hot streak
        if prediction_row.get('hot_streak', 0) == 1:
            factors.append("🔥 Hot streak")
            
        # Good matchup
        if prediction_row.get('pitcher_era', 5.0) > 4.5:
            factors.append("🎯 Struggling pitcher")
            
        # Park factor
        if prediction_row.get('park_factor', 1.0) > 1.1:
            factors.append("🏟️ Hitter-friendly park")
            
        # Weather
        if prediction_row.get('temperature', 70) > 80:
            factors.append("☀️ Hot weather")
            
        # Recent power
        if prediction_row.get('avg_exit_velo_7d', 0) > 90:
            factors.append("💪 Hard contact recently")
            
        return " | ".join(factors[:3])  # Max 3 factors


def main():
    """Generate predictions from command line"""
    import yaml
    from datetime import datetime
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # Initialize prediction engine
    engine = PredictionEngine(config)
    
    # Generate today's predictions
    today = datetime.now().strftime('%Y-%m-%d')
    predictions = engine.generate_daily_predictions(today)
    
    if not predictions.empty:
        # Print formatted predictions
        formatted_output = engine.format_predictions_for_output(predictions)
        print(formatted_output)
        
        # Save formatted output
        with open(f'data/predictions/formatted_predictions_{today}.txt', 'w') as f:
            f.write(formatted_output)
            
    else:
        print("No predictions generated for today.")


if __name__ == "__main__":
    main()
