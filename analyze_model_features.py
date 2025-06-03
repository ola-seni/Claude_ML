#!/usr/bin/env python3
"""
Analyze what features the trained model actually uses
"""

import pandas as pd
import sqlite3
from datetime import datetime, timedelta

try:
    from models import ModelEnsemble
    from feature_pipeline import FeatureEngineer
    import yaml
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit(1)

def analyze_model_features():
    """Analyze the features and their importance"""
    print("🔍 ANALYZING YOUR MODEL'S FEATURE USAGE")
    print("=" * 60)
    
    # Load the model
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    ensemble = ModelEnsemble(config)
    ensemble.load_models()
    
    print(f"📊 Your model uses {len(ensemble.feature_columns)} features")
    
    # Get feature importance if available
    try:
        feature_importance = ensemble.get_feature_importance()
        
        if not feature_importance.empty:
            print(f"\n🎯 TOP 20 MOST IMPORTANT FEATURES:")
            print("-" * 50)
            
            for i, (_, row) in enumerate(feature_importance.head(20).iterrows(), 1):
                feature_name = row['feature']
                avg_importance = row.get('average', 0)
                
                # Categorize the feature
                category = categorize_feature(feature_name)
                print(f"{i:2d}. {feature_name:<25} | {category}")
                
        print(f"\n📋 FEATURE CATEGORIES IN YOUR MODEL:")
        print("-" * 40)
        
        # Categorize all features
        categories = {}
        for feature in ensemble.feature_columns:
            category = categorize_feature(feature)
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
            
        for category, count in sorted(categories.items()):
            print(f"{category:<20}: {count:3d} features")
            
    except Exception as e:
        print(f"⚠️ Could not get feature importance: {e}")
        
        # Just categorize features
        print(f"\n📋 FEATURE CATEGORIES IN YOUR MODEL:")
        print("-" * 40)
        
        categories = {}
        for feature in ensemble.feature_columns:
            category = categorize_feature(feature)
            if category not in categories:
                categories[category] = 0
            categories[category] += 1
            
        for category, count in sorted(categories.items()):
            print(f"{category:<20}: {count:3d} features")

def categorize_feature(feature_name):
    """Categorize a feature by its name"""
    feature = feature_name.lower()
    
    if 'hr' in feature and ('rate' in feature or 'per' in feature):
        return "🏠 HR Rates"
    elif 'hr' in feature and any(x in feature for x in ['total', 'max', 'count']):
        return "🏠 HR Counts"
    elif any(x in feature for x in ['exit_velo', 'launch_angle', 'barrel', 'sweet_spot']):
        return "⚡ Statcast Quality"
    elif any(x in feature for x in ['batting_avg', 'hits', 'abs', 'contact']):
        return "🎯 Contact Ability"
    elif any(x in feature for x in ['pitcher', 'matchup', 'vs']):
        return "🤝 Matchup Analysis"
    elif any(x in feature for x in ['park', 'venue', 'factor']):
        return "🏟️ Park Effects"
    elif any(x in feature for x in ['weather', 'temp', 'wind']):
        return "🌤️ Weather"
    elif any(x in feature for x in ['trend', 'improving', 'hot', 'cold', 'streak']):
        return "📈 Form/Trends"
    elif any(x in feature for x in ['season', 'career', 'historical']):
        return "📊 Long-term Stats"
    elif any(x in feature for x in ['games', 'days', 'recent']):
        return "⏱️ Recent Activity"
    elif any(x in feature for x in ['home', 'away', 'day', 'month']):
        return "📍 Situational"
    elif any(x in feature for x in ['interaction', 'boost', 'advantage']):
        return "🔄 Interactions"
    else:
        return "❓ Other"

def test_non_hr_prediction():
    """Test if model can identify good candidates who haven't hit HRs recently"""
    print(f"\n🧪 TESTING: Can model find good candidates WITHOUT recent HRs?")
    print("-" * 60)
    
    conn = sqlite3.connect('data/mlb_predictions.db')
    
    # Find players with good contact but no recent HRs
    cutoff_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    
    query = """
        SELECT 
            player_name,
            COUNT(*) as games,
            SUM(home_runs) as total_hrs,
            SUM(hits) as total_hits,
            SUM(at_bats) as total_abs,
            CAST(SUM(hits) AS REAL) / SUM(at_bats) as batting_avg
        FROM player_performance
        WHERE date >= ? AND at_bats > 0
        GROUP BY player_name
        HAVING games >= 4 AND total_abs >= 10 AND total_hrs = 0 AND batting_avg >= 0.300
        ORDER BY batting_avg DESC
        LIMIT 5
    """
    
    good_contact_no_hrs = pd.read_sql_query(query, conn, params=(cutoff_date,))
    conn.close()
    
    if not good_contact_no_hrs.empty:
        print("📊 Players with good contact but NO recent HRs:")
        for _, player in good_contact_no_hrs.iterrows():
            avg = player['batting_avg']
            games = int(player['games'])
            abs = int(player['total_abs'])
            print(f"   {player['player_name']:<20}: .{avg:.3f}[3:] avg, {abs} ABs, 0 HRs in {games} games")
            
        print(f"\n💡 A good model should still give these players reasonable probabilities")
        print(f"   because they're making good contact (high batting average)")
        
    else:
        print("📊 No qualifying players found (good contact, no recent HRs)")

if __name__ == "__main__":
    analyze_model_features()
    test_non_hr_prediction()
