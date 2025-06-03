#!/usr/bin/env python3
"""
Automatic Daily Data Updater for Claude_ML
Ensures fresh data every day to avoid stale predictions
"""

import requests
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import time
from pathlib import Path
import json

class DailyDataUpdater:
    """Automatically updates MLB data for accurate predictions"""
    
    def __init__(self, db_path='data/mlb_predictions.db'):
        self.db_path = db_path
        self.base_url = "https://statsapi.mlb.com/api/v1"
        
    def check_data_freshness(self):
        """Check if we have fresh data for recent days"""
        print("🔍 Checking data freshness...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Check last 3 days
        dates_to_check = []
        for days_back in range(3):
            date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            dates_to_check.append(date)
        
        freshness_report = {}
        
        for date in dates_to_check:
            # Count games and HRs for this date
            query = """
                SELECT 
                    COUNT(DISTINCT game_id) as games,
                    COUNT(*) as player_records,
                    SUM(home_runs) as total_hrs
                FROM player_performance 
                WHERE date = ?
            """
            
            result = conn.execute(query, (date,)).fetchone()
            
            if result:
                games, records, hrs = result
                freshness_report[date] = {
                    'games': games or 0,
                    'records': records or 0,
                    'hrs': hrs or 0,
                    'fresh': self._is_data_fresh(games, records, hrs, date)
                }
            else:
                freshness_report[date] = {
                    'games': 0, 'records': 0, 'hrs': 0, 'fresh': False
                }
        
        conn.close()
        
        # Report freshness
        for date, data in freshness_report.items():
            status = "✅ Fresh" if data['fresh'] else "❌ Stale"
            print(f"   {date}: {data['games']} games, {data['records']} records, {data['hrs']} HRs | {status}")
        
        return freshness_report
        
    def _is_data_fresh(self, games, records, hrs, date):
        """Determine if data is fresh based on expected values"""
        date_obj = datetime.strptime(date, '%Y-%m-%d')
        
        # Don't expect data for future dates
        if date_obj > datetime.now():
            return True
            
        # For today, data might be partial
        if date_obj.date() == datetime.now().date():
            return games > 0  # Any data is good for today
            
        # For past dates, expect reasonable numbers
        if games == 0:
            return False  # No games = bad data
            
        # Expect reasonable HR rate (roughly 1-2 HRs per game on average)
        if games > 5 and hrs < games * 0.5:  # Very low HR rate
            return False
            
        # Expect reasonable number of player records
        if games > 0 and records < games * 15:  # Less than 15 players per game
            return False
            
        return True
        
    def update_stale_data(self, dates_to_update):
        """Update data for specific dates"""
        print(f"🔄 Updating stale data for {len(dates_to_update)} dates...")
        
        for date in dates_to_update:
            print(f"\n📅 Updating {date}...")
            
            # Clear existing data for this date
            self._clear_date_data(date)
            
            # Collect fresh data
            success = self._collect_date_data(date)
            
            if success:
                print(f"✅ Successfully updated {date}")
            else:
                print(f"❌ Failed to update {date}")
                
    def _clear_date_data(self, date):
        """Clear existing data for a specific date"""
        conn = sqlite3.connect(self.db_path)
        
        # Count before deletion
        count_before = conn.execute(
            "SELECT COUNT(*) FROM player_performance WHERE date = ?", 
            (date,)
        ).fetchone()[0]
        
        # Delete
        conn.execute("DELETE FROM player_performance WHERE date = ?", (date,))
        conn.execute("DELETE FROM games WHERE date = ?", (date,))
        conn.commit()
        conn.close()
        
        print(f"   Cleared {count_before} records for {date}")
        
    def _collect_date_data(self, date):
        """Collect complete data for a specific date"""
        url = f"{self.base_url}/schedule?sportId=1&date={date}&hydrate=boxscore"
        
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            data = response.json()
            
            total_games = 0
            total_records = 0
            total_hrs = 0
            
            conn = sqlite3.connect(self.db_path)
            
            for date_entry in data.get('dates', []):
                games = date_entry.get('games', [])
                
                for game in games:
                    game_pk = game.get('gamePk')
                    status = game.get('status', {}).get('abstractGameState', '')
                    
                    # Only process completed games
                    if status not in ['Final', 'Completed']:
                        continue
                        
                    total_games += 1
                    
                    # Save basic game info
                    self._save_game_info(conn, game, date)
                    
                    # Get detailed player stats
                    records, hrs = self._collect_game_player_stats(conn, game_pk, date)
                    total_records += records
                    total_hrs += hrs
                    
                    # Rate limiting
                    time.sleep(0.1)
            
            conn.commit()
            conn.close()
            
            print(f"   Collected: {total_games} games, {total_records} records, {total_hrs} HRs")
            
            # Verify collection was successful
            return total_games > 0 and total_records > 0
            
        except Exception as e:
            print(f"   ❌ Error collecting data for {date}: {e}")
            return False
            
    def _save_game_info(self, conn, game, date):
        """Save basic game information"""
        try:
            game_pk = game.get('gamePk')
            teams = game.get('teams', {})
            venue = game.get('venue', {}).get('name', 'Unknown Venue')
            
            away_team = teams.get('away', {}).get('team', {}).get('name', 'Unknown')
            home_team = teams.get('home', {}).get('team', {}).get('name', 'Unknown')
            
            conn.execute('''
                INSERT OR REPLACE INTO games 
                (game_id, date, home_team, away_team, venue, start_time)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (str(game_pk), date, home_team, away_team, venue, "19:00"))
            
        except Exception as e:
            print(f"      Warning: Could not save game info: {e}")
            
    def _collect_game_player_stats(self, conn, game_pk, date):
        """Collect player stats for a specific game"""
        url = f"{self.base_url}/game/{game_pk}/boxscore"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            box_data = response.json()
            
            records = 0
            hrs = 0
            
            teams = box_data.get('teams', {})
            
            for team_side in ['away', 'home']:
                team_data = teams.get(team_side, {})
                team_name = team_data.get('team', {}).get('name', 'Unknown')
                batters = team_data.get('batters', [])
                players = team_data.get('players', {})
                
                for batter_id in batters:
                    player_key = f"ID{batter_id}"
                    player_info = players.get(player_key, {})
                    
                    if player_info:
                        person = player_info.get('person', {})
                        name = person.get('fullName', 'Unknown')
                        
                        stats = player_info.get('stats', {}).get('batting', {})
                        
                        if stats:
                            at_bats = stats.get('atBats', 0)
                            hits = stats.get('hits', 0)
                            player_hrs = stats.get('homeRuns', 0)
                            
                            if at_bats > 0:  # Only save if player actually batted
                                conn.execute('''
                                    INSERT OR REPLACE INTO player_performance
                                    (player_id, player_name, game_id, date, team, home_away,
                                     at_bats, hits, home_runs, exit_velocity, launch_angle, hard_hit_rate)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                ''', (
                                    batter_id, name, str(game_pk), date, team_name, team_side,
                                    at_bats, hits, player_hrs, 0, 0, 0
                                ))
                                
                                records += 1
                                hrs += player_hrs
                                
                                # Log multi-HR games
                                if player_hrs >= 2:
                                    print(f"      🔥 {name}: {player_hrs} HRs")
            
            return records, hrs
            
        except Exception as e:
            print(f"      Warning: Could not get boxscore for game {game_pk}: {e}")
            return 0, 0
            
    def run_daily_update(self):
        """Run the complete daily update process"""
        print("🔄 DAILY DATA UPDATE PROCESS")
        print("=" * 40)
        
        # Step 1: Check freshness
        freshness_report = self.check_data_freshness()
        
        # Step 2: Identify stale dates
        stale_dates = [
            date for date, data in freshness_report.items() 
            if not data['fresh']
        ]
        
        if not stale_dates:
            print("\n✅ All data is fresh! No updates needed.")
            return True
            
        print(f"\n⚠️  Found {len(stale_dates)} dates with stale data: {stale_dates}")
        
        # Step 3: Update stale data
        self.update_stale_data(stale_dates)
        
        # Step 4: Verify updates
        print(f"\n🔍 Verifying updates...")
        final_freshness = self.check_data_freshness()
        
        # Check if all data is now fresh
        all_fresh = all(data['fresh'] for data in final_freshness.values())
        
        if all_fresh:
            print("\n🎊 Daily data update complete! All data is now fresh.")
            return True
        else:
            print("\n⚠️ Some data may still be stale. Check API availability.")
            return False
            
    def create_update_summary(self):
        """Create a summary of the data update for logging"""
        conn = sqlite3.connect(self.db_path)
        
        # Get recent data summary
        today = datetime.now().strftime('%Y-%m-%d')
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        summary = {}
        
        for date in [yesterday, today]:
            query = """
                SELECT 
                    COUNT(DISTINCT game_id) as games,
                    COUNT(*) as records,
                    SUM(home_runs) as hrs,
                    COUNT(DISTINCT player_name) as unique_players
                FROM player_performance 
                WHERE date = ?
            """
            
            result = conn.execute(query, (date,)).fetchone()
            
            if result:
                summary[date] = {
                    'games': result[0] or 0,
                    'records': result[1] or 0,
                    'hrs': result[2] or 0,
                    'players': result[3] or 0
                }
        
        conn.close()
        return summary

def main():
    """Run daily data update"""
    print("🔄 Claude_ML Daily Data Updater")
    print("=" * 40)
    
    updater = DailyDataUpdater()
    
    # Run the update
    success = updater.run_daily_update()
    
    if success:
        # Create summary
        summary = updater.create_update_summary()
        
        print(f"\n📊 DATA SUMMARY:")
        for date, data in summary.items():
            print(f"   {date}: {data['games']} games, {data['records']} records, {data['hrs']} HRs")
            
        print(f"\n✅ Daily data update completed successfully!")
        print(f"🎯 Ready for accurate predictions!")
        
    else:
        print(f"\n❌ Daily data update had issues")
        print(f"💡 Predictions may be less accurate")

if __name__ == "__main__":
    main()
