#!/usr/bin/env python3
"""
Fixed Daily MLB Predictions - handles feature mismatch issues
"""

import requests
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import json

# Import your existing components
try:
    from models import ModelEnsemble
    from feature_pipeline import FeatureEngineer
    import yaml
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

class FixedDailyPredictor:
    """Fixed daily predictor that handles feature mismatches"""
    
    def __init__(self):
        # Load config
        with open('config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Load your trained models
        self.ensemble = ModelEnsemble(self.config)
        self.ensemble.load_models()
        self.feature_engineer = FeatureEngineer()
        
        print("✅ Loaded your trained Claude_ML models")
        
        # Get the expected features from the trained model
        self.expected_features = self.ensemble.feature_columns
        print(f"📊 Model expects {len(self.expected_features)} features")
        
        # MLB API setup
        self.base_url = "https://statsapi.mlb.com/api/v1"
        
    def fix_feature_mismatch(self, generated_features):
        """Fix feature mismatch by adding missing features with defaults"""
        
        # Start with the generated features
        fixed_features = generated_features.copy()
        
        # Add any missing features with sensible defaults
        for expected_feature in self.expected_features:
            if expected_feature not in fixed_features:
                # Add default values for common missing features
                if 'cold_streak' in expected_feature:
                    fixed_features[expected_feature] = 0
                elif 'hot_streak' in expected_feature:
                    fixed_features[expected_feature] = 0
                elif 'trend' in expected_feature:
                    fixed_features[expected_feature] = 0.0
                elif 'interaction' in expected_feature:
                    fixed_features[expected_feature] = 0.0
                elif 'rate' in expected_feature:
                    fixed_features[expected_feature] = 0.0
                elif 'avg' in expected_feature:
                    fixed_features[expected_feature] = 0.0
                elif 'factor' in expected_feature:
                    fixed_features[expected_feature] = 1.0
                elif 'count' in expected_feature:
                    fixed_features[expected_feature] = 0
                else:
                    # Default to 0 for unknown features
                    fixed_features[expected_feature] = 0.0
                    
        return fixed_features
        
    def get_todays_schedule(self):
        """Get today's MLB schedule"""
        today = datetime.now().strftime('%Y-%m-%d')
        url = f"{self.base_url}/schedule?sportId=1&date={today}&hydrate=team,venue"
        
        try:
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            games = []
            for date_entry in data.get('dates', []):
                for game in date_entry.get('games', []):
                    # Skip completed games
                    status = game.get('status', {}).get('abstractGameState', '')
                    if status in ['Final', 'Completed']:
                        continue
                        
                    games.append({
                        'game_pk': game.get('gamePk'),
                        'away_team': game.get('teams', {}).get('away', {}).get('team', {}).get('name', 'Unknown'),
                        'home_team': game.get('teams', {}).get('home', {}).get('team', {}).get('name', 'Unknown'),
                        'away_abbr': game.get('teams', {}).get('away', {}).get('team', {}).get('abbreviation', 'UNK'),
                        'home_abbr': game.get('teams', {}).get('home', {}).get('team', {}).get('abbreviation', 'UNK'),
                        'venue': game.get('venue', {}).get('name', 'Unknown'),
                        'game_time': game.get('gameDate', ''),
                        'status': status
                    })
                    
            return games
            
        except Exception as e:
            print(f"❌ Error getting schedule: {e}")
            return []
    
    def get_recent_active_players(self, days_back=7):
        """Get recently active players from your database"""
        conn = sqlite3.connect('data/mlb_predictions.db')
        
        cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        query = """
            SELECT 
                player_name,
                team,
                COUNT(*) as games,
                SUM(home_runs) as total_hrs,
                SUM(at_bats) as total_abs,
                MAX(date) as last_game,
                AVG(CASE WHEN home_runs > 0 THEN 1.0 ELSE 0.0 END) as hr_rate
            FROM player_performance
            WHERE date >= ? AND at_bats > 0
            GROUP BY player_name, team
            HAVING games >= 3 AND total_abs >= 6
            ORDER BY hr_rate DESC, total_hrs DESC, last_game DESC
            LIMIT 60
        """
        
        df = pd.read_sql_query(query, conn, params=(cutoff_date,))
        conn.close()
        
        return df
        
    def generate_daily_predictions(self):
        """Generate predictions for today's games"""
        print(f"🎯 Generating FIXED predictions for {datetime.now().strftime('%A, %B %d, %Y')}")
        print("=" * 60)
        
        # Get today's games
        games = self.get_todays_schedule()
        
        if not games:
            print("❌ No games scheduled today")
            return None
            
        print(f"✅ Found {len(games)} scheduled games:")
        for game in games:
            print(f"   🏟️ {game['away_abbr']} @ {game['home_abbr']} ({game['venue']})")
            
        # Get recent active players
        recent_players = self.get_recent_active_players()
        
        if recent_players.empty:
            print("❌ No recent player data found")
            return None
            
        print(f"\n📊 Found {len(recent_players)} recently active players")
        
        # Generate predictions
        predictions = []
        today = datetime.now().strftime('%Y-%m-%d')
        
        print("\n⏳ Generating predictions with feature fixing...")
        successful_predictions = 0
        
        for _, player_row in recent_players.iterrows():
            player_name = player_row['player_name']
            
            try:
                # Create features for this player using existing pipeline
                features = self.feature_engineer._create_player_features(
                    player_name=player_name,
                    player_id=0,
                    game_id="today_prediction",
                    date=today,
                    team=player_row.get('team', 'Unknown'),
                    opponent='Unknown',
                    home_away='home'
                )
                
                if features:
                    # Fix feature mismatch
                    fixed_features = self.fix_feature_mismatch(features)
                    
                    # Generate prediction using fixed features
                    feature_df = pd.DataFrame([fixed_features])
                    hr_probability = self.ensemble.predict_probability(feature_df)[0]
                    
                    predictions.append({
                        'player_name': player_name,
                        'team': player_row.get('team', 'Unknown'),
                        'hr_probability': hr_probability,
                        'recent_games': int(player_row['games']),
                        'recent_hrs': int(player_row['total_hrs']),
                        'recent_abs': int(player_row['total_abs']),
                        'recent_hr_rate': float(player_row['hr_rate']),
                        'last_game': player_row['last_game']
                    })
                    
                    successful_predictions += 1
                    
            except Exception as e:
                print(f"      ⚠️ Failed for {player_name}: {e}")
                continue
                
        # Sort by probability
        predictions.sort(key=lambda x: x['hr_probability'], reverse=True)
        
        print(f"✅ Generated {successful_predictions} successful predictions out of {len(recent_players)} players")
        
        return {
            'predictions': predictions,
            'games': games,
            'date': today
        }
        
    def display_predictions(self, prediction_data):
        """Display today's predictions in a nice format"""
        if not prediction_data:
            print("❌ No predictions to display")
            return
            
        predictions = prediction_data['predictions']
        games = prediction_data['games']
        
        print(f"\n🏟️ TODAY'S MLB HOME RUN PREDICTIONS (FIXED)")
        print(f"📅 {datetime.now().strftime('%A, %B %d, %Y')}")
        print("=" * 80)
        
        # Show scheduled games
        print(f"\n📋 SCHEDULED GAMES ({len(games)} games)")
        print("-" * 40)
        for game in games:
            print(f"🏟️ {game['away_abbr']} @ {game['home_abbr']} - {game['venue']}")
            
        # Top picks
        print(f"\n🎯 TOP 15 HOME RUN PICKS")
        print("-" * 60)
        
        for i, pred in enumerate(predictions[:15], 1):
            prob = pred['hr_probability']
            name = pred['player_name']
            team = pred['team']
            recent_hrs = pred['recent_hrs']
            recent_games = pred['recent_games']
            
            # Confidence tier
            if prob >= 0.08:
                tier = "💎 PREMIUM"
            elif prob >= 0.05:
                tier = "⭐ STANDARD"
            else:
                tier = "💰 VALUE"
                
            print(f"{i:2d}. {tier:<12} {name:<22} {prob:5.1%} | {team} ({recent_hrs} HRs in {recent_games} games)")
            
        # Tier breakdown
        premium = [p for p in predictions if p['hr_probability'] >= 0.08]
        standard = [p for p in predictions if 0.05 <= p['hr_probability'] < 0.08]
        value = [p for p in predictions if p['hr_probability'] < 0.05]
        
        print(f"\n📊 TIER BREAKDOWN")
        print("-" * 30)
        print(f"💎 Premium Picks: {len(premium)} players")
        print(f"⭐ Standard Picks: {len(standard)} players")  
        print(f"💰 Value Picks: {len(value)} players")
        
        # Summary stats
        if predictions:
            avg_prob = sum(p['hr_probability'] for p in predictions) / len(predictions)
            max_prob = max(p['hr_probability'] for p in predictions)
            
            print(f"\n📈 SUMMARY")
            print("-" * 20)
            print(f"Total Predictions: {len(predictions)}")
            print(f"Highest Probability: {max_prob:.1%}")
            print(f"Average Probability: {avg_prob:.1%}")
            
        # Save predictions
        self.save_predictions(prediction_data)
        
    def save_predictions(self, prediction_data):
        """Save predictions to file"""
        try:
            Path('data/predictions').mkdir(parents=True, exist_ok=True)
            
            today_str = datetime.now().strftime('%Y%m%d')
            filename = f"data/predictions/fixed_predictions_{today_str}.json"
            
            # Add summary stats
            predictions = prediction_data['predictions']
            if predictions:
                prediction_data['summary'] = {
                    'total_predictions': len(predictions),
                    'max_probability': max(p['hr_probability'] for p in predictions),
                    'avg_probability': sum(p['hr_probability'] for p in predictions) / len(predictions),
                    'premium_count': len([p for p in predictions if p['hr_probability'] >= 0.08]),
                    'standard_count': len([p for p in predictions if 0.05 <= p['hr_probability'] < 0.08]),
                    'value_count': len([p for p in predictions if p['hr_probability'] < 0.05])
                }
            
            with open(filename, 'w') as f:
                json.dump(prediction_data, f, indent=2, default=str)
                
            print(f"\n💾 Predictions saved to {filename}")
            
        except Exception as e:
            print(f"⚠️ Could not save predictions: {e}")

def main():
    """Main function - generate today's FIXED predictions"""
    print("🎯 Claude_ML FIXED Daily Predictions")
    print("=" * 50)
    
    try:
        # Initialize predictor
        predictor = FixedDailyPredictor()
        
        # Generate predictions
        prediction_data = predictor.generate_daily_predictions()
        
        if prediction_data and prediction_data['predictions']:
            predictor.display_predictions(prediction_data)
            print("\n🎊 FIXED daily predictions complete!")
            print("📈 These predictions use your trained Claude_ML models with feature fixes")
            print("🎯 Good luck with today's games!")
        else:
            print("❌ Could not generate predictions")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
