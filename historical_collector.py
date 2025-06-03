#!/usr/bin/env python3
"""
Historical Data Collector for Claude_ML
Collects 2-3 months of historical MLB data for model training
"""

import os
import sys
import json
import time
import sqlite3
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import yaml
import requests
from bs4 import BeautifulSoup

# MLB Stats API
import statsapi

# Pybaseball for advanced stats
from pybaseball import (
    batting_stats, 
    pitching_stats,
    statcast,
    playerid_lookup,
    batting_stats_range,
    pitching_stats_range
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('HistoricalCollector')


class HistoricalDataCollector:
    """Collects historical MLB data for training"""
    
    def __init__(self, config=None, config_path: str = 'config.yaml'):
        """Initialize the collector with configuration"""
        if isinstance(config, dict):
            # If config dict is passed, use it directly
            self.config = config
        else:
            # If config_path is passed (or None), load from file
            actual_path = config if isinstance(config, str) else config_path
            self.config = self._load_config(actual_path)
        
        self.db_path = self.config['database']['path']
        self._setup_directories()
        self._setup_database()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _setup_directories(self):
        """Create necessary directories"""
        directories = [
            'data/raw/games',
            'data/raw/lineups', 
            'data/raw/weather',
            'data/raw/stats',
            'data/processed/training',
            'data/processed/features',
            'data/predictions',
            'logs'
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    def _setup_database(self):
        """Initialize SQLite database with tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Games table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                date TEXT,
                home_team TEXT,
                away_team TEXT,
                home_score INTEGER,
                away_score INTEGER,
                venue TEXT,
                weather_temp REAL,
                weather_wind_speed REAL,
                weather_wind_dir INTEGER,
                processed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Player performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS player_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT,
                date TEXT,
                player_name TEXT,
                player_id INTEGER,
                team TEXT,
                home_away TEXT,
                batting_order INTEGER,
                home_runs INTEGER,
                hits INTEGER,
                at_bats INTEGER,
                exit_velocity REAL,
                launch_angle REAL,
                hard_hit_rate REAL,
                FOREIGN KEY (game_id) REFERENCES games (game_id)
            )
        ''')
        
        # Pitcher performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pitcher_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT,
                date TEXT,
                pitcher_name TEXT,
                pitcher_id INTEGER,
                team TEXT,
                innings_pitched REAL,
                home_runs_allowed INTEGER,
                era REAL,
                whip REAL,
                strikeouts INTEGER,
                FOREIGN KEY (game_id) REFERENCES games (game_id)
            )
        ''')
        
        # Features table for ML
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT,
                date TEXT,
                player_name TEXT,
                feature_vector TEXT,  -- JSON string of features
                home_run BOOLEAN,
                FOREIGN KEY (game_id) REFERENCES games (game_id)
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def collect_historical_data(self, start_date: str, end_date: str):
        """Main method to collect all historical data"""
        logger.info(f"Starting historical data collection from {start_date} to {end_date}")
        
        # Convert dates
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        current = start
        
        total_days = (end - start).days
        processed_days = 0
        
        # Process each day
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            logger.info(f"Processing {date_str} ({processed_days}/{total_days})")
            
            try:
                # 1. Get games for this date
                games = self._collect_games(date_str)
                
                if games:
                    # 2. Get lineups for each game
                    lineups = self._collect_lineups(date_str, games)
                    
                    # 3. Get game results and player performance
                    self._collect_game_results(date_str, games)
                    
                    # 4. Get weather data
                    self._collect_weather(date_str, games)
                    
                    # 5. Save to database
                    self._save_daily_data(date_str, games, lineups)
                    
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing {date_str}: {e}")
                
            current += timedelta(days=1)
            processed_days += 1
            
        # After collecting all daily data, get season stats
        logger.info("Collecting season-wide statistics...")
        self._collect_season_stats(start_date, end_date)
        
        # Collect Statcast data in chunks
        logger.info("Collecting Statcast data...")
        self._collect_statcast_data(start_date, end_date)
        
        logger.info("Historical data collection complete!")
        
    def _collect_games(self, date_str: str) -> List[Dict]:
        """Collect games for a specific date"""
        try:
            games = statsapi.schedule(date=date_str)
            logger.info(f"Found {len(games)} games on {date_str}")
            
            # Filter to completed games only
            completed_games = [g for g in games if g['status'] == 'Final']
            
            return completed_games
            
        except Exception as e:
            logger.error(f"Error fetching games for {date_str}: {e}")
            return []
            
    def _collect_lineups(self, date_str: str, games: List[Dict]) -> Dict:
        """Collect lineups from Rotowire"""
        lineups = {}
        
        # Try Rotowire first
        try:
            url_date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%m-%d-%Y")
            url = f"https://www.rotowire.com/baseball/daily-lineups.php?date={url_date}"
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # Parse lineups (using your existing Rotowire parser logic)
            # ... (simplified for brevity)
            
        except Exception as e:
            logger.warning(f"Rotowire lineup fetch failed for {date_str}: {e}")
            
        # Fallback to MLB API for actual lineups from completed games
        for game in games:
            game_id = game['game_id']
            try:
                # Get box score for actual lineups
                boxscore = statsapi.boxscore_data(game_id)
                
                home_lineup = []
                away_lineup = []
                
                # Extract lineups from boxscore
                # ... (implementation details)
                
                lineups[game_id] = {
                    'home': home_lineup,
                    'away': away_lineup
                }
                
            except Exception as e:
                logger.error(f"Error getting lineup for game {game_id}: {e}")
                
        return lineups
        
    def _collect_game_results(self, date_str: str, games: List[Dict]):
        """Collect detailed game results and player performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for game in games:
            game_id = game['game_id']
            
            try:
                # Get detailed box score
                boxscore = statsapi.boxscore_data(game_id)
                
                # Process home team players
                home_players = boxscore.get('home', {}).get('players', {})
                for player_id, player_data in home_players.items():
                    if 'stats' in player_data and 'batting' in player_data['stats']:
                        batting = player_data['stats']['batting']
                        
                        cursor.execute('''
                            INSERT INTO player_performance 
                            (game_id, date, player_name, player_id, team, home_away,
                             home_runs, hits, at_bats)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            game_id, date_str,
                            player_data['person']['fullName'],
                            int(player_id.replace('ID', '')),
                            game['home_name'], 'home',
                            batting.get('homeRuns', 0),
                            batting.get('hits', 0),
                            batting.get('atBats', 0)
                        ))
                
                # Process away team players
                away_players = boxscore.get('away', {}).get('players', {})
                for player_id, player_data in away_players.items():
                    if 'stats' in player_data and 'batting' in player_data['stats']:
                        batting = player_data['stats']['batting']
                        
                        cursor.execute('''
                            INSERT INTO player_performance 
                            (game_id, date, player_name, player_id, team, home_away,
                             home_runs, hits, at_bats)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            game_id, date_str,
                            player_data['person']['fullName'],
                            int(player_id.replace('ID', '')),
                            game['away_name'], 'away',
                            batting.get('homeRuns', 0),
                            batting.get('hits', 0),
                            batting.get('atBats', 0)
                        ))
                
                conn.commit()
                
            except Exception as e:
                logger.error(f"Error processing game {game_id}: {e}")
                
        conn.close()
        
    def _collect_weather(self, date_str: str, games: List[Dict]):
        """Collect weather data for games"""
        # For historical data, we might need to use a historical weather API
        # or estimate based on typical conditions
        # For now, we'll use placeholder data
        pass
        
    def _collect_season_stats(self, start_date: str, end_date: str):
        """Collect season-wide statistics from FanGraphs via pybaseball"""
        try:
            # Get batting stats
            logger.info("Fetching FanGraphs batting statistics...")
            batting_data = batting_stats(2024, qual=50)  # 2024 season, min 50 PA
            
            # Save to CSV for reference
            batting_data.to_csv('data/raw/stats/fangraphs_batting_2024.csv', index=False)
            
            # Get pitching stats
            logger.info("Fetching FanGraphs pitching statistics...")
            pitching_data = pitching_stats(2024, qual=20)  # 2024 season, min 20 IP
            
            # Save to CSV
            pitching_data.to_csv('data/raw/stats/fangraphs_pitching_2024.csv', index=False)
            
            logger.info("FanGraphs data collection complete")
            
        except Exception as e:
            logger.error(f"Error collecting FanGraphs data: {e}")
            
    def _collect_statcast_data(self, start_date: str, end_date: str):
        """Collect Statcast data in chunks"""
        try:
            chunk_size = self.config['data_sources']['statcast']['chunk_size']
            
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            
            current = start
            all_data = []
            
            while current <= end:
                chunk_end = min(current + timedelta(days=chunk_size-1), end)
                
                logger.info(f"Fetching Statcast data from {current.strftime('%Y-%m-%d')} to {chunk_end.strftime('%Y-%m-%d')}")
                
                # Get Statcast data
                data = statcast(
                    start_dt=current.strftime('%Y-%m-%d'),
                    end_dt=chunk_end.strftime('%Y-%m-%d')
                )
                
                if data is not None and not data.empty:
                    all_data.append(data)
                    logger.info(f"Retrieved {len(data)} Statcast events")
                
                # Rate limiting
                time.sleep(5)
                
                current = chunk_end + timedelta(days=1)
                
            # Combine all chunks
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                
                # Save to CSV
                combined_data.to_csv('data/raw/stats/statcast_data.csv', index=False)
                logger.info(f"Saved {len(combined_data)} total Statcast events")
                
                # Process and update player performance with exit velo, launch angle, etc.
                self._update_player_metrics_from_statcast(combined_data)
                
        except Exception as e:
            logger.error(f"Error collecting Statcast data: {e}")
            
    def _update_player_metrics_from_statcast(self, statcast_data: pd.DataFrame):
        """Update player performance with Statcast metrics"""
        conn = sqlite3.connect(self.db_path)
        
        # Group by player and calculate aggregates
        player_metrics = statcast_data.groupby('player_name').agg({
            'launch_speed': ['mean', 'max'],
            'launch_angle': 'mean',
            'hit_distance_sc': 'mean'
        }).reset_index()
        
        # Flatten column names
        player_metrics.columns = ['player_name', 'avg_exit_velo', 'max_exit_velo', 
                                  'avg_launch_angle', 'avg_distance']
        
        # Calculate hard hit rate (95+ mph)
        for _, player in player_metrics.iterrows():
            player_data = statcast_data[statcast_data['player_name'] == player['player_name']]
            hard_hits = player_data[player_data['launch_speed'] >= 95]
            hard_hit_rate = len(hard_hits) / len(player_data) if len(player_data) > 0 else 0
            
            # Update database
            conn.execute('''
                UPDATE player_performance 
                SET exit_velocity = ?, launch_angle = ?, hard_hit_rate = ?
                WHERE player_name = ?
            ''', (
                player['avg_exit_velo'],
                player['avg_launch_angle'],
                hard_hit_rate,
                player['player_name']
            ))
            
        conn.commit()
        conn.close()
        
    def _save_daily_data(self, date_str: str, games: List[Dict], lineups: Dict):
        """Save daily data to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for game in games:
            game_id = game['game_id']
            
            # Insert game record
            cursor.execute('''
                INSERT OR IGNORE INTO games 
                (game_id, date, home_team, away_team, home_score, away_score, venue)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                game_id, date_str,
                game['home_name'], game['away_name'],
                game['home_score'], game['away_score'],
                game['venue_name']
            ))
            
        conn.commit()
        conn.close()
        
        logger.info(f"Saved {len(games)} games for {date_str}")


def main():
    """Main execution function"""
    # Initialize collector
    collector = HistoricalDataCollector()
    
    # Get dates from config
    config = collector.config
    start_date = config['data_collection']['start_date']
    end_date = config['data_collection']['end_date']
    
    # Collect historical data
    collector.collect_historical_data(start_date, end_date)
    
    logger.info("Historical data collection complete!")
    

if __name__ == "__main__":
    main()
