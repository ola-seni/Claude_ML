"""
Setup Script for Claude_ML Home Run Prediction System
Handles installation, configuration, and initial setup
"""

import os
import sys
import subprocess
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timedelta
import yaml

logger = logging.getLogger('Setup')


class ClaudeMLSetup:
    """Setup and installation manager"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.python_executable = sys.executable
        
    def check_python_version(self):
        """Check if Python version is compatible"""
        python_version = sys.version_info
        
        if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
            print("❌ Python 3.8 or higher is required")
            print(f"   Current version: {python_version.major}.{python_version.minor}.{python_version.micro}")
            return False
            
        print(f"✅ Python version {python_version.major}.{python_version.minor}.{python_version.micro} is compatible")
        return True
        
    def create_directory_structure(self):
        """Create necessary directories"""
        directories = [
            'data/raw/lineups',
            'data/raw/stats', 
            'data/raw/weather',
            'data/processed/features',
            'data/processed/training',
            'data/predictions',
            'data/backtesting',
            'models/trained',
            'models/scalers',
            'logs'
        ]
        
        print("📁 Creating directory structure...")
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            
        print(f"✅ Created {len(directories)} directories")
        
    def install_dependencies(self):
        """Install required Python packages"""
        print("📦 Installing Python dependencies...")
        
        requirements_file = self.project_root / 'requirements.txt'
        
        if not requirements_file.exists():
            print("❌ requirements.txt not found")
            return False
            
        try:
            # Install packages
            subprocess.check_call([
                self.python_executable, '-m', 'pip', 'install', '-r', str(requirements_file)
            ])
            
            print("✅ Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False
            
    def setup_configuration(self):
        """Setup initial configuration"""
        print("⚙️  Setting up configuration...")
        
        config_file = self.project_root / 'config.yaml'
        
        if config_file.exists():
            overwrite = input("config.yaml already exists. Overwrite? (y/n): ").lower().strip()
            if overwrite != 'y':
                print("⏭️  Skipping configuration setup")
                return True
                
        # Create default config
        default_config = {
            'data_collection': {
                'historical_days': 90,
                'start_date': (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d'),
                'end_date': datetime.now().strftime('%Y-%m-%d')
            },
            'data_sources': {
                'rotowire': {'enabled': True, 'url': 'https://www.rotowire.com/baseball/daily-lineups.php'},
                'mlb_stats_api': {'enabled': True, 'rate_limit_delay': 0.5},
                'statcast': {'enabled': True, 'chunk_size': 7},
                'fangraphs': {'enabled': True, 'via_pybaseball': True},
                'weather': {'enabled': True, 'api_key': 'YOUR_OPENWEATHER_API_KEY'}
            },
            'features': {
                'rolling_windows': [3, 7, 14, 30],
                'min_plate_appearances': 50,
                'min_innings_pitched': 20,
                'include_groups': [
                    'batting_stats', 'power_metrics', 'recent_form',
                    'platoon_splits', 'park_factors', 'weather_impact',
                    'pitcher_tendencies', 'historical_matchups'
                ]
            },
            'model': {
                'algorithms': ['xgboost', 'random_forest', 'neural_network', 'logistic_regression'],
                'ensemble_method': 'weighted_average',
                'train_test_split': 0.8,
                'validation_split': 0.2,
                'random_state': 42,
                'retrain_frequency': 'weekly',
                'min_data_points': 1000
            },
            'predictions': {
                'run_times': {'morning': '10:00', 'evening': '17:00'},
                'total_predictions': 10,
                'tiers': {'premium': 3, 'standard': 4, 'value': 3},
                'min_probability': 0.04,
                'exclude_pitchers': True,
                'min_season_hrs': 5
            },
            'backtesting': {
                'enabled': True,
                'metrics': ['accuracy', 'precision', 'recall', 'roi']
            },
            'database': {
                'path': 'data/mlb_predictions.db',
                'backup_frequency': 'daily'
            },
            'logging': {
                'level': 'INFO',
                'file': 'logs/claude_ml.log',
                'max_size': '10MB',
                'backup_count': 5
            },
            'telegram': {
                'enabled': True,
                'bot_token': 'YOUR_BOT_TOKEN',
                'chat_id': 'YOUR_CHAT_ID'
            },
            'error_handling': {
                'max_retries': 3,
                'retry_delay': 60,
                'alert_on_failure': True
            },
            'park_factors': {
                'COL': 1.35, 'CIN': 1.18, 'TEX': 1.15, 'NYY': 1.15,
                'CWS': 1.12, 'PHI': 1.10, 'MIL': 1.08, 'CHC': 1.08,
                'BAL': 1.05, 'ATL': 1.05, 'HOU': 1.05, 'LAA': 1.05,
                'SF': 0.90, 'OAK': 0.90, 'MIA': 0.87
            },
            'feature_tracking': {
                'track_importance': True,
                'top_features_to_show': 20,
                'update_frequency': 'weekly'
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
            
        print(f"✅ Configuration saved to {config_file}")
        return True
        
    def setup_credentials(self):
        """Setup credentials"""
        print("🔐 Setting up credentials...")
        
        # Import and run credentials setup
        try:
            from credentials import setup_credentials
            setup_credentials()
            return True
        except ImportError:
            print("❌ credentials.py not found")
            return False
        except Exception as e:
            print(f"❌ Credentials setup failed: {e}")
            return False
            
    def initialize_database(self):
        """Initialize the SQLite database"""
        print("🗃️  Initializing database...")
        
        db_path = self.project_root / 'data' / 'mlb_predictions.db'
        
        try:
            conn = sqlite3.connect(db_path)
            
            # Create main tables
            tables = [
                '''
                CREATE TABLE IF NOT EXISTS games (
                    game_id TEXT PRIMARY KEY,
                    date TEXT,
                    home_team TEXT,
                    away_team TEXT,
                    venue TEXT,
                    start_time TEXT,
                    weather_temp REAL,
                    weather_wind_speed REAL,
                    weather_wind_dir REAL,
                    weather_conditions TEXT
                )
                ''',
                '''
                CREATE TABLE IF NOT EXISTS player_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    player_id INTEGER,
                    player_name TEXT,
                    game_id TEXT,
                    date TEXT,
                    team TEXT,
                    home_away TEXT,
                    at_bats INTEGER,
                    hits INTEGER,
                    home_runs INTEGER,
                    exit_velocity REAL,
                    launch_angle REAL,
                    hard_hit_rate REAL,
                    FOREIGN KEY (game_id) REFERENCES games (game_id)
                )
                ''',
                '''
                CREATE TABLE IF NOT EXISTS pitcher_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pitcher_id INTEGER,
                    pitcher_name TEXT,
                    game_id TEXT,
                    date TEXT,
                    team TEXT,
                    innings_pitched REAL,
                    earned_runs INTEGER,
                    home_runs_allowed INTEGER,
                    strikeouts INTEGER,
                    walks INTEGER,
                    era REAL,
                    whip REAL,
                    FOREIGN KEY (game_id) REFERENCES games (game_id)
                )
                ''',
                '''
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
                ''',
                '''
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    component TEXT,
                    action TEXT,
                    status TEXT,
                    message TEXT,
                    details TEXT
                )
                '''
            ]
            
            for table_sql in tables:
                conn.execute(table_sql)
                
            conn.commit()
            conn.close()
            
            print(f"✅ Database initialized at {db_path}")
            return True
            
        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            return False
            
    def run_initial_data_collection(self):
        """Run initial data collection"""
        print("📊 Running initial data collection...")
        
        collect = input("Collect historical data now? This may take 10-30 minutes (y/n): ").lower().strip()
        
        if collect != 'y':
            print("⏭️  Skipping initial data collection")
            print("💡 You can run it later with: python historical_collector.py")
            return True
            
        try:
            # Import and run data collection
            from historical_collector import main as collect_data
            collect_data()
            print("✅ Initial data collection completed")
            return True
            
        except Exception as e:
            print(f"❌ Data collection failed: {e}")
            print("💡 You can try again later with: python historical_collector.py")
            return False
            
    def run_initial_training(self):
        """Run initial model training"""
        print("🤖 Training initial models...")
        
        train = input("Train ML models now? This may take 5-15 minutes (y/n): ").lower().strip()
        
        if train != 'y':
            print("⏭️  Skipping initial training")
            print("💡 You can train later with: python main.py train")
            return True
            
        try:
            # Check if training data exists
            training_data_path = self.project_root / 'data' / 'processed' / 'training' / 'training_data.csv'
            
            if not training_data_path.exists():
                print("❌ No training data found. Run data collection first.")
                return False
                
            # Import and run training
            from models import main as train_models
            train_models()
            print("✅ Initial model training completed")
            return True
            
        except Exception as e:
            print(f"❌ Model training failed: {e}")
            print("💡 You can try again later with: python main.py train")
            return False
            
    def create_quick_start_guide(self):
        """Create a quick start guide"""
        guide_content = """# Claude_ML Quick Start Guide

## 🎯 Daily Usage

1. **Generate Predictions:**
   ```bash
   python main.py predict
   ```

2. **Run Telegram Bot:**
   ```bash
   python main.py telegram
   ```

3. **Start Scheduled Operations:**
   ```bash
   python main.py schedule
   ```

## 📊 Analysis & Backtesting

1. **Run Backtesting:**
   ```bash
   python main.py backtest 30  # Last 30 days
   ```

2. **Check System Status:**
   ```bash
   python main.py status
   ```

## 🔧 Maintenance

1. **Retrain Models:**
   ```bash
   python main.py train
   ```

2. **Update Historical Data:**
   ```bash
   python historical_collector.py
   ```

3. **Validate Credentials:**
   ```bash
   python credentials.py validate
   ```

## 📱 Telegram Commands

- `/start` - Welcome message and setup
- `/predictions` or `/today` - Get today's predictions
- `/yesterday` - See yesterday's results
- `/stats` - View recent performance
- `/backtest` - Run quick analysis

## 📁 Important Files

- `config.yaml` - Main configuration
- `.env` - API keys and credentials (keep secure!)
- `data/mlb_predictions.db` - Main database
- `logs/claude_ml.log` - System logs

## 🚨 Troubleshooting

1. **No predictions generated:**
   - Check if games are scheduled
   - Verify data collection is working
   - Review logs in `logs/claude_ml.log`

2. **Telegram bot not working:**
   - Validate credentials: `python credentials.py validate`
   - Check bot token and chat ID

3. **Model errors:**
   - Retrain models: `python main.py train`
   - Check if training data exists

## 📈 Performance Monitoring

Check `data/backtesting/` folder for:
- Performance reports
- Accuracy charts
- ROI analysis

Happy predicting! 🏟️⚾
"""
        
        guide_path = self.project_root / 'QUICK_START.md'
        with open(guide_path, 'w') as f:
            f.write(guide_content)
            
        print(f"✅ Quick start guide created: {guide_path}")
        
    def run_full_setup(self):
        """Run complete setup process"""
        print("🏟️ Claude_ML Setup Wizard ⚾")
        print("=" * 50)
        
        steps = [
            ("Checking Python version", self.check_python_version),
            ("Creating directories", self.create_directory_structure),
            ("Installing dependencies", self.install_dependencies),
            ("Setting up configuration", self.setup_configuration),
            ("Setting up credentials", self.setup_credentials),
            ("Initializing database", self.initialize_database),
            ("Collecting initial data", self.run_initial_data_collection),
            ("Training initial models", self.run_initial_training),
            ("Creating quick start guide", self.create_quick_start_guide)
        ]
        
        failed_steps = []
        
        for step_name, step_function in steps:
            print(f"\n🔄 {step_name}...")
            
            try:
                success = step_function()
                if not success:
                    failed_steps.append(step_name)
            except Exception as e:
                print(f"❌ {step_name} failed: {e}")
                failed_steps.append(step_name)
                
        # Summary
        print("\n" + "=" * 50)
        print("🎊 Setup Complete!")
        
        if failed_steps:
            print(f"⚠️  {len(failed_steps)} steps had issues:")
            for step in failed_steps:
                print(f"   - {step}")
            print("\n💡 Check the output above for troubleshooting steps")
        else:
            print("✅ All steps completed successfully!")
            
        print("\n🚀 Next Steps:")
        print("1. Review QUICK_START.md for usage instructions")
        print("2. Test with: python main.py predict")
        print("3. Start Telegram bot: python main.py telegram")
        
        return len(failed_steps) == 0


def main():
    """Command line interface for setup"""
    import sys
    
    setup = ClaudeMLSetup()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "full":
            setup.run_full_setup()
        elif command == "deps":
            setup.install_dependencies()
        elif command == "config":
            setup.setup_configuration()
        elif command == "credentials":
            setup.setup_credentials()
        elif command == "database":
            setup.initialize_database()
        elif command == "directories":
            setup.create_directory_structure()
        else:
            print("Available commands: full, deps, config, credentials, database, directories")
    else:
        # Interactive mode
        print("🏟️ Claude_ML Setup ⚾")
        print("\nChoose setup option:")
        print("1. Full setup (recommended for first time)")
        print("2. Install dependencies only")
        print("3. Configure credentials only")
        print("4. Initialize database only")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        if choice == "1":
            setup.run_full_setup()
        elif choice == "2":
            setup.install_dependencies()
        elif choice == "3":
            setup.setup_credentials()
        elif choice == "4":
            setup.initialize_database()
        else:
            print("Invalid choice")


if __name__ == "__main__":
    main()