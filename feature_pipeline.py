"""
Feature Engineering Pipeline for Claude_ML
Transforms raw data into ML-ready features
"""

import pandas as pd
import numpy as np
import sqlite3
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging
from scipy import stats
import json

logger = logging.getLogger('FeaturePipeline')


class FeatureEngineer:
    """Creates ML features from raw data"""
    
    def __init__(self, db_path: str = 'data/mlb_predictions.db'):
        self.db_path = db_path
        self.feature_columns = []
        
    def create_features_for_date(self, date: str) -> pd.DataFrame:
        """Create all features for a specific date"""
        logger.info(f"Creating features for {date}")
        
        # Get all players who played on this date
        players = self._get_players_for_date(date)
        
        if players.empty:
            logger.warning(f"No players found for {date}")
            return pd.DataFrame()
            
        # Create features for each player
        features_list = []
        
        for _, player in players.iterrows():
            features = self._create_player_features(
                player['player_name'],
                player['player_id'],
                player['game_id'],
                date,
                player['team'],
                player['opponent_team'],
                player['home_away']
            )
            
            if features:
                features['home_run'] = player['home_runs'] > 0
                features_list.append(features)
                
        # Combine all features
        if features_list:
            df = pd.DataFrame(features_list)
            self.feature_columns = [col for col in df.columns if col not in ['player_name', 'game_id', 'date', 'home_run']]
            return df
        
        return pd.DataFrame()
        
    def _create_player_features(self, player_name: str, player_id: int, 
                               game_id: str, date: str, team: str, 
                               opponent: str, home_away: str) -> Dict:
        """Create comprehensive feature set for a player"""
        features = {
            'player_name': player_name,
            'game_id': game_id,
            'date': date
        }
        
        # 1. Recent Performance Features (most predictive)
        features.update(self._get_recent_performance_features(player_name, date))
        
        # 2. Season Statistics
        features.update(self._get_season_stats_features(player_name, date))
        
        # 3. Statcast Quality Metrics
        features.update(self._get_statcast_features(player_name, date))
        
        # 4. Matchup Features
        features.update(self._get_matchup_features(player_name, game_id))
        
        # 5. Park and Weather Features
        features.update(self._get_environmental_features(game_id, team, home_away))
        
        # 6. Situational Features
        features.update(self._get_situational_features(player_name, date, team))
        
        # 7. Trend Features
        features.update(self._get_trend_features(player_name, date))
        
        # 8. Interaction Features (combinations that matter)
        features.update(self._create_interaction_features(features))
        
        return features
        
    def _get_recent_performance_features(self, player_name: str, date: str) -> Dict:
        """Get recent performance metrics"""
        features = {}
        
        # Define rolling windows
        windows = [3, 7, 14, 30]
        
        conn = sqlite3.connect(self.db_path)
        
        for window in windows:
            # Calculate date range
            end_date = datetime.strptime(date, '%Y-%m-%d')
            start_date = end_date - timedelta(days=window)
            
            # Query recent performance
            query = '''
                SELECT 
                    COUNT(*) as games,
                    SUM(home_runs) as total_hrs,
                    SUM(hits) as total_hits,
                    SUM(at_bats) as total_abs,
                    AVG(exit_velocity) as avg_exit_velo,
                    AVG(launch_angle) as avg_launch_angle,
                    AVG(hard_hit_rate) as avg_hard_hit_rate,
                    MAX(home_runs) as max_hrs_game
                FROM player_performance
                WHERE player_name = ? 
                AND date >= ? AND date < ?
            '''
            
            result = conn.execute(query, (player_name, start_date.strftime('%Y-%m-%d'), date)).fetchone()
            
            if result and result[0] > 0:  # Has games in window
                games = result[0]
                hrs = result[1] or 0
                hits = result[2] or 0
                abs = result[3] or 0
                
                # Basic rates
                features[f'hr_rate_{window}d'] = hrs / games if games > 0 else 0
                features[f'hr_per_ab_{window}d'] = hrs / abs if abs > 0 else 0
                features[f'batting_avg_{window}d'] = hits / abs if abs > 0 else 0
                features[f'games_played_{window}d'] = games
                
                # Quality metrics
                features[f'avg_exit_velo_{window}d'] = result[4] or 0
                features[f'avg_launch_angle_{window}d'] = result[5] or 0
                features[f'avg_hard_hit_rate_{window}d'] = result[6] or 0
                features[f'max_hrs_single_game_{window}d'] = result[7] or 0
                
                # Hot/Cold indicator
                if window == 7:
                    # Compare to season average
                    if hrs / games > 0.15:  # More than 0.15 HR/game is hot
                        features['hot_streak'] = 1
                    elif hrs == 0:
                        features['cold_streak'] = 1
                    else:
                        features['hot_streak'] = 0
                        features['cold_streak'] = 0
            else:
                # No recent games - set defaults
                for metric in ['hr_rate', 'hr_per_ab', 'batting_avg', 'games_played',
                             'avg_exit_velo', 'avg_launch_angle', 'avg_hard_hit_rate', 
                             'max_hrs_single_game']:
                    features[f'{metric}_{window}d'] = 0
                    
        conn.close()
        
        # Weighted recent performance (more recent = higher weight)
        if all(f'hr_rate_{w}d' in features for w in windows):
            weights = [0.4, 0.3, 0.2, 0.1]  # 3-day gets 40%, 7-day gets 30%, etc.
            weighted_hr_rate = sum(features[f'hr_rate_{w}d'] * weight 
                                  for w, weight in zip(windows, weights))
            features['weighted_recent_hr_rate'] = weighted_hr_rate
            
        return features
        
    def _get_season_stats_features(self, player_name: str, date: str) -> Dict:
        """Get season-long statistics"""
        features = {}
        
        conn = sqlite3.connect(self.db_path)
        
        # Get season stats up to this date
        season_year = datetime.strptime(date, '%Y-%m-%d').year
        
        query = '''
            SELECT 
                COUNT(*) as games,
                SUM(home_runs) as total_hrs,
                SUM(hits) as total_hits,
                SUM(at_bats) as total_abs,
                AVG(exit_velocity) as avg_exit_velo,
                AVG(launch_angle) as avg_launch_angle,
                0 as hr_variance
            FROM player_performance
            WHERE player_name = ? 
            AND date >= ? AND date < ?
        '''
        
        season_start = f"{season_year}-03-01"
        result = conn.execute(query, (player_name, season_start, date)).fetchone()
        
        if result and result[0] > 0:
            games = result[0]
            hrs = result[1] or 0
            hits = result[2] or 0
            abs = result[3] or 0
            
            features['season_games'] = games
            features['season_hrs'] = hrs
            features['season_hr_rate'] = hrs / games if games > 0 else 0
            features['season_hr_per_ab'] = hrs / abs if abs > 0 else 0
            features['season_batting_avg'] = hits / abs if abs > 0 else 0
            features['season_avg_exit_velo'] = result[4] or 0
            features['season_avg_launch_angle'] = result[5] or 0
            features['season_hr_consistency'] = 1 / (1 + (result[6] or 0))  # Lower variance = more consistent
            
            # ISO (Isolated Power) - slugging minus batting average
            # We'll estimate it based on HR rate
            features['season_iso_estimate'] = features['season_hr_per_ab'] * 4  # Simplified
            
        else:
            # Set defaults
            for metric in ['season_games', 'season_hrs', 'season_hr_rate', 
                          'season_hr_per_ab', 'season_batting_avg', 
                          'season_avg_exit_velo', 'season_avg_launch_angle',
                          'season_hr_consistency', 'season_iso_estimate']:
                features[metric] = 0
                
        conn.close()
        return features
        
    def _get_statcast_features(self, player_name: str, date: str) -> Dict:
        """Get Statcast quality metrics"""
        features = {}
        
        # Load Statcast data if available
        try:
            statcast_df = pd.read_csv('data/raw/stats/statcast_data.csv')
            
            # Filter to player and recent timeframe
            player_data = statcast_df[statcast_df['player_name'] == player_name]
            
            if not player_data.empty:
                # Quality metrics
                features['barrel_rate'] = (player_data['launch_speed'] >= 98).mean()
                features['sweet_spot_rate'] = ((player_data['launch_angle'] >= 8) & 
                                              (player_data['launch_angle'] <= 32)).mean()
                features['max_exit_velo'] = player_data['launch_speed'].max()
                features['avg_hit_distance'] = player_data['hit_distance_sc'].mean()
                
                # Pull tendency
                features['pull_rate'] = (player_data['hc_x'] < 125).mean()  # Simplified
                features['oppo_rate'] = (player_data['hc_x'] > 225).mean()  # Simplified
                
            else:
                # Set defaults
                for metric in ['barrel_rate', 'sweet_spot_rate', 'max_exit_velo',
                             'avg_hit_distance', 'pull_rate', 'oppo_rate']:
                    features[metric] = 0
                    
        except Exception as e:
            logger.warning(f"Could not load Statcast features: {e}")
            # Set defaults
            for metric in ['barrel_rate', 'sweet_spot_rate', 'max_exit_velo',
                         'avg_hit_distance', 'pull_rate', 'oppo_rate']:
                features[metric] = 0
                
        return features
        
    def _get_matchup_features(self, player_name: str, game_id: str) -> Dict:
        """Get batter vs pitcher matchup features"""
        features = {}
        
        conn = sqlite3.connect(self.db_path)
        
        # Get pitcher for this game
        query = '''
            SELECT pitcher_name, pitcher_id
            FROM pitcher_performance
            WHERE game_id = ?
            LIMIT 1
        '''
        
        result = conn.execute(query, (game_id,)).fetchone()
        
        if result:
            pitcher_name = result[0]
            
            # Get historical matchup
            history_query = '''
                SELECT 
                    COUNT(*) as matchups,
                    SUM(pp.home_runs) as hrs_vs_pitcher
                FROM player_performance pp
                JOIN pitcher_performance pitch ON pp.game_id = pitch.game_id
                WHERE pp.player_name = ? AND pitch.pitcher_name = ?
            '''
            
            history = conn.execute(history_query, (player_name, pitcher_name)).fetchone()
            
            if history and history[0] > 0:
                features['career_matchups'] = history[0]
                features['career_hrs_vs_pitcher'] = history[1] or 0
                features['career_hr_rate_vs_pitcher'] = features['career_hrs_vs_pitcher'] / features['career_matchups']
            else:
                features['career_matchups'] = 0
                features['career_hrs_vs_pitcher'] = 0
                features['career_hr_rate_vs_pitcher'] = 0
                
            # Get pitcher tendencies
            pitcher_query = '''
                SELECT 
                    AVG(home_runs_allowed) as avg_hrs_allowed,
                    AVG(era) as avg_era,
                    AVG(whip) as avg_whip
                FROM pitcher_performance
                WHERE pitcher_name = ?
            '''
            
            pitcher_stats = conn.execute(pitcher_query, (pitcher_name,)).fetchone()
            
            if pitcher_stats:
                features['pitcher_avg_hrs_allowed'] = pitcher_stats[0] or 0
                features['pitcher_era'] = pitcher_stats[1] or 4.5
                features['pitcher_whip'] = pitcher_stats[2] or 1.3
            else:
                features['pitcher_avg_hrs_allowed'] = 0.15  # League average
                features['pitcher_era'] = 4.5
                features['pitcher_whip'] = 1.3
                
        else:
            # Set defaults
            features['career_matchups'] = 0
            features['career_hrs_vs_pitcher'] = 0
            features['career_hr_rate_vs_pitcher'] = 0
            features['pitcher_avg_hrs_allowed'] = 0.15
            features['pitcher_era'] = 4.5
            features['pitcher_whip'] = 1.3
            
        conn.close()
        return features
        
    def _get_environmental_features(self, game_id: str, team: str, home_away: str) -> Dict:
        """Get park and weather features"""
        features = {}
        
        conn = sqlite3.connect(self.db_path)
        
        # Get game info
        query = '''
            SELECT venue, weather_temp, weather_wind_speed, weather_wind_dir
            FROM games
            WHERE game_id = ?
        '''
        
        result = conn.execute(query, (game_id,)).fetchone()
        
        if result:
            venue = result[0]
            
            # Park factors (from config or defaults)
            park_factors = {
                'Coors Field': 1.35,
                'Great American Ball Park': 1.18,
                'Globe Life Field': 1.15,
                'Yankee Stadium': 1.15,
                'Guaranteed Rate Field': 1.12,
                'Citizens Bank Park': 1.10,
                'American Family Field': 1.08,
                'Wrigley Field': 1.08,
                'Oracle Park': 0.90,
                'Oakland Coliseum': 0.90,
                'loanDepot park': 0.87
            }
            
            features['park_factor'] = park_factors.get(venue, 1.0)
            features['is_home'] = 1 if home_away == 'home' else 0
            
            # Weather features
            temp = result[1] or 72
            wind_speed = result[2] or 5
            wind_dir = result[3] or 0
            
            features['temperature'] = temp
            features['wind_speed'] = wind_speed
            
            # Temperature effect (optimal around 80-85)
            features['temp_factor'] = 1 + (temp - 70) * 0.01 if temp > 70 else 1 - (70 - temp) * 0.005
            
            # Wind effect (simplified - would need park orientation for accuracy)
            features['wind_factor'] = 1 + (wind_speed * 0.02) if wind_speed > 5 else 1.0
            
        else:
            # Set defaults
            features['park_factor'] = 1.0
            features['is_home'] = 0
            features['temperature'] = 72
            features['wind_speed'] = 5
            features['temp_factor'] = 1.0
            features['wind_factor'] = 1.0
            
        conn.close()
        return features
        
    def _get_situational_features(self, player_name: str, date: str, team: str) -> Dict:
        """Get situational context features"""
        features = {}
        
        # Day of week (some players perform better on certain days)
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        features['day_of_week'] = date_obj.weekday()
        features['is_weekend'] = 1 if date_obj.weekday() >= 5 else 0
        
        # Month (seasonal effects)
        features['month'] = date_obj.month
        features['is_summer'] = 1 if date_obj.month in [6, 7, 8] else 0
        
        # Days since last HR (due factor)
        conn = sqlite3.connect(self.db_path)
        
        last_hr_query = '''
            SELECT MAX(date) as last_hr_date
            FROM player_performance
            WHERE player_name = ? AND home_runs > 0 AND date < ?
        '''
        
        result = conn.execute(last_hr_query, (player_name, date)).fetchone()
        
        if result and result[0]:
            last_hr_date = datetime.strptime(result[0], '%Y-%m-%d')
            days_since = (date_obj - last_hr_date).days
            features['days_since_last_hr'] = min(days_since, 30)  # Cap at 30
        else:
            features['days_since_last_hr'] = 30  # Never hit HR or very long ago
            
        conn.close()
        return features
        
    def _get_trend_features(self, player_name: str, date: str) -> Dict:
        """Calculate trend features"""
        features = {}
        
        # Compare recent performance to season average
        if 'hr_rate_7d' in features and 'season_hr_rate' in features:
            features['hr_rate_trend'] = (
                features['hr_rate_7d'] - features['season_hr_rate']
            ) / (features['season_hr_rate'] + 0.01)  # Avoid division by zero
            
        # Exit velocity trend
        if 'avg_exit_velo_7d' in features and 'season_avg_exit_velo' in features:
            features['exit_velo_trend'] = (
                features['avg_exit_velo_7d'] - features['season_avg_exit_velo']
            )
            
        # Form indicator
        if 'hr_rate_3d' in features and 'hr_rate_14d' in features:
            features['improving_form'] = 1 if features['hr_rate_3d'] > features['hr_rate_14d'] else 0
            
        return features
        
    def _create_interaction_features(self, features: Dict) -> Dict:
        """Create interaction features that capture important combinations"""
        interaction_features = {}
        
        # Power + Contact quality
        if 'season_hr_rate' in features and 'avg_exit_velo_7d' in features:
            interaction_features['power_quality_score'] = (
                features['season_hr_rate'] * features['avg_exit_velo_7d'] / 100
            )
            
        # Hot streak + Good matchup
        if 'hot_streak' in features and 'pitcher_avg_hrs_allowed' in features:
            interaction_features['hot_vs_vulnerable'] = (
                features.get('hot_streak', 0) * features['pitcher_avg_hrs_allowed']
            )
            
        # Park factor + Pull tendency
        if 'park_factor' in features and 'pull_rate' in features:
            interaction_features['park_pull_advantage'] = (
                features['park_factor'] * features['pull_rate']
            )
            
        # Weather + Exit velocity
        if 'temp_factor' in features and 'avg_exit_velo_7d' in features:
            interaction_features['weather_power_boost'] = (
                features['temp_factor'] * (features['avg_exit_velo_7d'] / 100)
            )
            
        return interaction_features
        
    def _get_players_for_date(self, date: str) -> pd.DataFrame:
        """Get all players who played on a specific date"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT DISTINCT
                pp.player_name,
                pp.player_id,
                pp.game_id,
                pp.team,
                pp.home_away,
                pp.home_runs,
                CASE 
                    WHEN pp.home_away = 'home' THEN g.away_team
                    ELSE g.home_team
                END as opponent_team
            FROM player_performance pp
            JOIN games g ON pp.game_id = g.game_id
            WHERE pp.date = ? AND pp.at_bats > 0
        '''
        
        df = pd.read_sql_query(query, conn, params=(date,))
        conn.close()
        
        return df
        
    def create_training_dataset(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Create full training dataset for date range"""
        logger.info(f"Creating training dataset from {start_date} to {end_date}")
        
        all_features = []
        
        # Process each date
        current = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current <= end:
            date_str = current.strftime('%Y-%m-%d')
            
            # Create features for this date
            date_features = self.create_features_for_date(date_str)
            
            if not date_features.empty:
                all_features.append(date_features)
                
            current += timedelta(days=1)
            
        # Combine all features
        if all_features:
            final_df = pd.concat(all_features, ignore_index=True)
            
            # Remove any infinite values
            final_df = final_df.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values with appropriate defaults
            final_df = final_df.fillna(0)
            
            # Save to processed data
            final_df.to_csv('data/processed/training/training_data.csv', index=False)
            logger.info(f"Created training dataset with {len(final_df)} samples and {len(self.feature_columns)} features")
            
            # Save feature column names for later use
            with open('data/processed/features/feature_columns.json', 'w') as f:
                json.dump(self.feature_columns, f)
                
            return final_df
        
        return pd.DataFrame()


def main():
    """Create training dataset from historical data"""
    # Initialize feature engineer
    engineer = FeatureEngineer()
    
    # Create training dataset
    training_data = engineer.create_training_dataset('2024-04-01', '2024-07-01')
    
    if not training_data.empty:
        # Display feature importance preview
        print(f"\nCreated {len(training_data)} training samples")
        print(f"Features: {len(engineer.feature_columns)}")
        print(f"Home run rate: {training_data['home_run'].mean():.3%}")
        
        # Show top features by correlation with home runs
        correlations = training_data[engineer.feature_columns].corrwith(training_data['home_run']).abs().sort_values(ascending=False)
        print("\nTop 10 features by correlation:")
        print(correlations.head(10))
    

if __name__ == "__main__":
    main()
