#!/usr/bin/env python3
"""
Fixed 2025 MLB Data Collector
Works with existing database schema
"""

import requests
import pandas as pd
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys

class Fixed2025DataCollector:
    """Collects actual 2025 MLB data with correct schema"""
    
    def __init__(self, db_path='data/mlb_predictions.db'):
        self.db_path = db_path
        self.base_url = "https://statsapi.mlb.com/api/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Claude_ML_Baseball_Predictor/1.0'
        })
        
    def collect_2025_data(self, start_date: str, end_date: str):
        """Collect 2025 MLB data for date range"""
        print(f"🏟️ Collecting REAL 2025 MLB data from {start_date} to {end_date}")
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Collect data day by day
        current_date = start_dt
        total_games = 0
        total_players = 0
        
        while current_date <= end_dt:
            date_str = current_date.strftime('%Y-%m-%d')
            print(f"📅 Collecting data for {date_str}...")
            
            try:
                # Get games for this date
                games = self._get_games_for_date(date_str)
                
                if games:
                    print(f"   Found {len(games)} games")
                    
                    # Process each game
                    for game in games:
                        try:
                            # Get detailed game data
                            game_data = self._get_game_details(game['gamePk'])
                            
                            if game_data:
                                # Save to database
                                self._save_game_data(game_data, date_str)
                                total_games += 1
                                
                                # Get player stats for this game
                                player_count = self._save_player_stats(game_data, date_str)
                                total_players += player_count
                                
                        except Exception as e:
                            print(f"      Error processing game {game.get('gamePk', 'unknown')}: {e}")
                            continue
                            
                else:
                    print(f"   No games found for {date_str}")
                    
                # Rate limiting
                time.sleep(0.3)
                
            except Exception as e:
                print(f"   Error collecting data for {date_str}: {e}")
                continue
                
            current_date += timedelta(days=1)
            
        print(f"\n✅ Collection complete!")
        print(f"   Games processed: {total_games}")
        print(f"   Player performances: {total_players}")
        
        return total_games, total_players
        
    def _get_games_for_date(self, date_str: str):
        """Get all MLB games for a specific date"""
        url = f"{self.base_url}/schedule?sportId=1&date={date_str}"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract games from the schedule
            games = []
            for date_entry in data.get('dates', []):
                games.extend(date_entry.get('games', []))
                
            return games
            
        except Exception as e:
            print(f"      Error fetching games for {date_str}: {e}")
            return []
            
    def _get_game_details(self, game_pk: int):
        """Get detailed game data including box score"""
        url = f"{self.base_url}/game/{game_pk}/boxscore"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except Exception as e:
            print(f"      Error fetching game details for {game_pk}: {e}")
            return None
            
    def _save_game_data(self, game_data, date_str):
        """Save game information to database (fixed schema)"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Extract game info
            game_pk = str(game_data.get('game', {}).get('pk', ''))
            
            teams = game_data.get('teams', {})
            away_team = teams.get('away', {}).get('team', {}).get('name', 'Unknown')
            home_team = teams.get('home', {}).get('team', {}).get('name', 'Unknown')
            
            # Get venue info
            venue = "Unknown Venue"
            
            # Use existing schema (no start_time column)
            conn.execute('''
                INSERT OR REPLACE INTO games 
                (game_id, date, home_team, away_team, venue)
                VALUES (?, ?, ?, ?, ?)
            ''', (game_pk, date_str, home_team, away_team, venue))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"      Error saving game data: {e}")
            
    def _save_player_stats(self, game_data, date_str):
        """Extract and save player statistics from game"""
        player_count = 0
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Extract game ID
            game_pk = str(game_data.get('game', {}).get('pk', ''))
            
            # Process both teams
            teams = game_data.get('teams', {})
            
            for team_side in ['away', 'home']:
                team_data = teams.get(team_side, {})
                team_name = team_data.get('team', {}).get('name', 'Unknown')
                
                # Get batters
                batters = team_data.get('batters', [])
                batting_stats = team_data.get('players', {})
                
                for batter_id in batters:
                    batter_key = f"ID{batter_id}"
                    player_info = batting_stats.get(batter_key, {})
                    
                    if player_info:
                        # Extract player data
                        person = player_info.get('person', {})
                        player_name = person.get('fullName', 'Unknown Player')
                        
                        # Extract batting stats
                        stats = player_info.get('stats', {})
                        batting = stats.get('batting', {})
                        
                        if batting:
                            at_bats = batting.get('atBats', 0)
                            hits = batting.get('hits', 0)
                            home_runs = batting.get('homeRuns', 0)
                            
                            # Only save if player had at-bats
                            if at_bats > 0:
                                conn.execute('''
                                    INSERT OR REPLACE INTO player_performance
                                    (player_id, player_name, game_id, date, team, home_away,
                                     at_bats, hits, home_runs, exit_velocity, launch_angle, hard_hit_rate)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                ''', (
                                    batter_id, player_name, game_pk, date_str, team_name, team_side,
                                    at_bats, hits, home_runs, 0, 0, 0  # Statcast data filled later
                                ))
                                
                                player_count += 1
                                
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"      Error saving player stats: {e}")
            
        return player_count
        
    def test_api_connection(self):
        """Test if we can connect to MLB Stats API"""
        print("🔗 Testing MLB Stats API connection...")
        
        try:
            # Get today's games
            today = datetime.now().strftime('%Y-%m-%d')
            url = f"{self.base_url}/schedule?sportId=1&date={today}"
            
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Count games
            total_games = 0
            for date_entry in data.get('dates', []):
                total_games += len(date_entry.get('games', []))
                
            print(f"✅ API connection successful!")
            print(f"   Found {total_games} games scheduled for {today}")
            
            return True
            
        except Exception as e:
            print(f"❌ API connection failed: {e}")
            return False


def main():
    """Main function to collect 2025 data"""
    collector = Fixed2025DataCollector()
    
    # Test API connection first
    if not collector.test_api_connection():
        print("❌ Cannot connect to MLB API. Check internet connection.")
        return
        
    # Collect 2025 data (last 60 days)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        
    print(f"\n🎯 Collecting 2025 MLB data from {start_date} to {end_date}")
    
    # Collect the data
    games, players = collector.collect_2025_data(start_date, end_date)
    
    if games > 0:
        print(f"\n🎉 Successfully collected 2025 data!")
        print(f"📊 Run 'python3 quick_predictions.py' to generate 2025 predictions")
    else:
        print(f"\n⚠️  No data collected. Check date range and API availability.")


if __name__ == "__main__":
    main()
