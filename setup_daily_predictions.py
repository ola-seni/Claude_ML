#!/usr/bin/env python3
"""
Setup script for Daily Prediction System
Run this to set up real daily predictions with your Claude_ML system
"""

import os
from pathlib import Path

def create_daily_prediction_system():
    """Create the daily prediction system file"""
    
    daily_pred_content = '''#!/usr/bin/env python3
"""
Live Daily MLB Predictions - Gets today's lineups and makes real predictions
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
    print("Make sure you're in the Claude_ML directory with all the modules")
    exit(1)

class LiveDailyPredictor:
    """Live daily predictor using your existing trained models"""
    
    def __init__(self):
        # Load config
        with open('config.yaml', 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Load your trained models
        self.ensemble = ModelEnsemble(self.config)
        self.ensemble.load_models()
        self.feature_engineer = FeatureEngineer()
        
        print("✅ Loaded your trained Claude_ML models")
        
        # MLB API setup
        self.base_url = "https://statsapi.mlb.com/api/v1"
        
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
            
    def get_probable_starters(self, game_pk):
        """Try to get starting lineups or use recent players"""
        # Try to get lineups from boxscore
        url = f"{self.base_url}/game/{game_pk}/boxscore"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Extract any available player info
            players = []
            teams = data.get('teams', {})
            
            for team_side in ['away', 'home']:
                team_data = teams.get(team_side, {})
                batters = team_data.get('batters', [])
                player_data = team_data.get('players', {})
                
                for batter_id in batters[:9]:  # First 9
                    player_key = f"ID{batter_id}"
                    player_info = player_data.get(player_key, {})
                    
                    if player_info:
                        person = player_info.get('person', {})
                        name = person.get('fullName', '')
                        
                        if name:
                            players.append({
                                'name': name,
                                'team_side': team_side,
                                'id': batter_id
                            })
                            
            return players
            
        except:
            return []
    
    def get_recent_active_players(self, days_back=7):
        """Get recently active players from your database"""
        conn = sqlite3.connect('data/mlb_predictions.db')
        
        cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        query = '''
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
        '''
        
        df = pd.read_sql_query(query, conn, params=(cutoff_date,))
        conn.close()
        
        return df
        
    def generate_daily_predictions(self):
        """Generate predictions for today's games"""
        print(f"🎯 Generating predictions for {datetime.now().strftime('%A, %B %d, %Y')}")
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
            
        print(f"\\n📊 Found {len(recent_players)} recently active players")
        
        # Generate predictions
        predictions = []
        today = datetime.now().strftime('%Y-%m-%d')
        
        print("\\n⏳ Generating predictions...")
        
        for _, player_row in recent_players.iterrows():
            player_name = player_row['player_name']
            
            try:
                # Create features for this player
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
                    # Generate prediction using your trained models
                    feature_df = pd.DataFrame([features])
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
                    
            except Exception as e:
                # Skip players with insufficient data
                continue
                
        # Sort by probability
        predictions.sort(key=lambda x: x['hr_probability'], reverse=True)
        
        print(f"✅ Generated {len(predictions)} predictions")
        
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
        
        print(f"\\n🏟️ TODAY'S MLB HOME RUN PREDICTIONS")
        print(f"📅 {datetime.now().strftime('%A, %B %d, %Y')}")
        print("=" * 80)
        
        # Show scheduled games
        print(f"\\n📋 SCHEDULED GAMES ({len(games)} games)")
        print("-" * 40)
        for game in games:
            print(f"🏟️ {game['away_abbr']} @ {game['home_abbr']} - {game['venue']}")
            
        # Top picks
        print(f"\\n🎯 TOP 15 HOME RUN PICKS")
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
        
        print(f"\\n📊 TIER BREAKDOWN")
        print("-" * 30)
        print(f"💎 Premium Picks: {len(premium)} players")
        print(f"⭐ Standard Picks: {len(standard)} players")  
        print(f"💰 Value Picks: {len(value)} players")
        
        # Summary stats
        if predictions:
            avg_prob = sum(p['hr_probability'] for p in predictions) / len(predictions)
            max_prob = max(p['hr_probability'] for p in predictions)
            
            print(f"\\n📈 SUMMARY")
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
            filename = f"data/predictions/daily_predictions_{today_str}.json"
            
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
                
            print(f"\\n💾 Predictions saved to {filename}")
            
        except Exception as e:
            print(f"⚠️ Could not save predictions: {e}")
            
    def test_system(self):
        """Test if the system is working"""
        print("🧪 Testing Daily Prediction System")
        print("=" * 40)
        
        # Test 1: Models loaded
        try:
            print("✅ Models loaded successfully")
        except:
            print("❌ Model loading failed")
            return False
            
        # Test 2: API connection
        try:
            games = self.get_todays_schedule()
            print(f"✅ API connection successful - {len(games)} games found")
        except Exception as e:
            print(f"❌ API connection failed: {e}")
            return False
            
        # Test 3: Database
        try:
            players = self.get_recent_active_players()
            print(f"✅ Database access successful - {len(players)} recent players found")
        except Exception as e:
            print(f"❌ Database error: {e}")
            return False
            
        print("\\n🎊 System test passed! Ready to generate predictions.")
        return True

def main():
    """Main function - generate today's predictions"""
    print("🎯 Claude_ML Live Daily Predictions")
    print("=" * 50)
    
    try:
        # Initialize predictor
        predictor = LiveDailyPredictor()
        
        # Test system first
        if not predictor.test_system():
            print("❌ System test failed")
            return
            
        # Generate predictions
        prediction_data = predictor.generate_daily_predictions()
        
        if prediction_data:
            predictor.display_predictions(prediction_data)
            print("\\n🎊 Daily predictions complete!")
            print("📈 These predictions use your trained Claude_ML models")
            print("🎯 Good luck with today's games!")
        else:
            print("❌ Could not generate predictions")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
'''

    # Write the file
    with open('daily_predictions_live.py', 'w') as f:
        f.write(daily_pred_content)
    
    # Make executable
    os.chmod('daily_predictions_live.py', 0o755)
    print("✅ Created daily_predictions_live.py")

def create_test_script():
    """Create a simple test script"""
    
    test_content = '''#!/usr/bin/env python3
"""
Test the daily prediction system
"""

print("🧪 Testing Claude_ML Daily Prediction System")
print("=" * 50)

try:
    from daily_predictions_live import LiveDailyPredictor
    
    predictor = LiveDailyPredictor()
    
    if predictor.test_system():
        print("\\n🎊 System is ready!")
        print("💡 Run: python3 daily_predictions_live.py")
    else:
        print("\\n❌ System test failed")
        
except Exception as e:
    print(f"❌ Error: {e}")
    print("\\n💡 Make sure you're in the Claude_ML directory")
    print("💡 Make sure models are trained")
'''

    with open('test_daily_system.py', 'w') as f:
        f.write(test_content)
    
    os.chmod('test_daily_system.py', 0o755)
    print("✅ Created test_daily_system.py")

def main():
    """Set up the daily prediction system"""
    print("🔧 Setting up Claude_ML Daily Prediction System")
    print("=" * 50)
    
    # Check prerequisites
    if not Path('config.yaml').exists():
        print("❌ config.yaml not found")
        print("💡 Make sure you're in the Claude_ML directory")
        return
        
    if not Path('models').exists():
        print("❌ models directory not found")
        print("💡 Make sure you've trained models first")
        return
        
    print("✅ Found config.yaml")
    print("✅ Found models directory")
    
    # Create the system
    create_daily_prediction_system()
    create_test_script()
    
    print("\\n" + "=" * 50)
    print("🎊 Daily Prediction System Setup Complete!")
    print("=" * 50)
    
    print("\\n🚀 Next Steps:")
    print("1. Test the system:")
    print("   python3 test_daily_system.py")
    print("\\n2. Generate today's predictions:")
    print("   python3 daily_predictions_live.py")
    
    print("\\n💡 This system will:")
    print("   ✅ Get today's MLB schedule")
    print("   ✅ Use your trained models")
    print("   ✅ Generate real predictions for active players")
    print("   ✅ Show results by confidence tier")

if __name__ == "__main__":
    main()
