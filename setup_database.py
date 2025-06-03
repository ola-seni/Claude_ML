#!/usr/bin/env python3
import sqlite3

conn = sqlite3.connect('data/mlb_predictions.db')

conn.execute('''CREATE TABLE IF NOT EXISTS games (game_id TEXT PRIMARY KEY, date TEXT, home_team TEXT, away_team TEXT, venue TEXT, start_time TEXT, weather_temp REAL, weather_wind_speed REAL, weather_wind_dir REAL, weather_conditions TEXT)''')

conn.execute('''CREATE TABLE IF NOT EXISTS player_performance (id INTEGER PRIMARY KEY AUTOINCREMENT, player_id INTEGER, player_name TEXT, game_id TEXT, date TEXT, team TEXT, home_away TEXT, at_bats INTEGER, hits INTEGER, home_runs INTEGER, exit_velocity REAL, launch_angle REAL, hard_hit_rate REAL)''')

conn.execute('''CREATE TABLE IF NOT EXISTS daily_predictions (prediction_id INTEGER, prediction_date TEXT, game_date TEXT, player_name TEXT, game_id TEXT, hr_probability REAL, confidence_tier TEXT, matchup TEXT, venue TEXT, pitcher_name TEXT, model_version TEXT, actual_result INTEGER DEFAULT NULL, PRIMARY KEY (prediction_date, game_date, player_name))''')

conn.commit()
conn.close()
print('✅ Database initialized')
