"""
Main Orchestrator for Claude_ML
Coordinates all system components for daily operations
"""

import asyncio
import logging
import sys
import schedule
import time
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional
import sqlite3
import traceback

# Import our modules
from historical_collector import HistoricalDataCollector
from feature_pipeline import FeatureEngineer
from models import ModelEnsemble
from prediction_engine import PredictionEngine
from backtesting import Backtester
from telegram_bot import ClaudeMLBot

# Setup logging
def setup_logging(config: Dict):
    """Setup logging configuration"""
    log_config = config.get('logging', {})
    
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_config.get('file', 'logs/claude_ml.log')),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    return logging.getLogger('Claude_ML_Main')


class Claude_ML_System:
    """Main system orchestrator"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Setup logging
        self.logger = setup_logging(self.config)
        
        # Initialize components
        self.data_collector = None
        self.feature_engineer = None
        self.model_ensemble = None
        self.prediction_engine = None
        self.backtester = None
        self.telegram_bot = None
        
        # System status
        self.last_update = None
        self.last_training = None
        self.system_health = True
        
        self.logger.info("Claude_ML System initialized")
        
    def initialize_components(self):
        """Initialize all system components"""
        try:
            self.logger.info("Initializing system components...")
            
            # Data collector
            self.data_collector = HistoricalDataCollector(self.config)
            
            # Feature engineer
            self.feature_engineer = FeatureEngineer()
            
            # Model ensemble
            self.model_ensemble = ModelEnsemble(self.config)
            
            # Prediction engine
            self.prediction_engine = PredictionEngine(self.config)
            
            # Backtester
            self.backtester = Backtester(self.config)
            
            # Telegram bot (if enabled)
            if self.config.get('telegram', {}).get('enabled', False):
                self.telegram_bot = ClaudeMLBot(self.config)
                
            # Create necessary directories
            self._create_directories()
            
            # Initialize database
            self._initialize_database()
            
            self.logger.info("All components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            self.system_health = False
            raise
            
    def _create_directories(self):
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
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            
    def _initialize_database(self):
        """Initialize SQLite database with required tables"""
        db_path = self.config.get('database', {}).get('path', 'data/mlb_predictions.db')
        
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
        
        self.logger.info("Database initialized")
        
    async def daily_workflow(self):
        """Execute daily workflow"""
        self.logger.info("Starting daily workflow")
        
        try:
            # 1. Update data
            await self._update_daily_data()
            
            # 2. Check if model retraining is needed
            await self._check_model_retraining()
            
            # 3. Generate predictions
            await self._generate_daily_predictions()
            
            # 4. Update previous day results
            await self._update_previous_results()
            
            # 5. Send predictions via Telegram
            if self.telegram_bot:
                await self._send_predictions()
                
            # 6. Log system status
            self._log_system_status("daily_workflow", "success", "Daily workflow completed")
            
            self.logger.info("Daily workflow completed successfully")
            
        except Exception as e:
            self.logger.error(f"Daily workflow failed: {e}")
            self._log_system_status("daily_workflow", "error", str(e))
            
            # Send error notification if Telegram is enabled
            if self.telegram_bot:
                await self._send_error_notification(str(e))
                
            raise
            
    async def _update_daily_data(self):
        """Update daily data from all sources"""
        self.logger.info("Updating daily data...")
        
        today = datetime.now().strftime('%Y-%m-%d')
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Collect yesterday's final data and today's lineups
        await self.data_collector.collect_daily_data(yesterday)
        await self.data_collector.collect_daily_data(today)
        
        self.last_update = datetime.now()
        self.logger.info("Daily data update completed")
        
    async def _check_model_retraining(self):
        """Check if model retraining is needed"""
        retrain_freq = self.config.get('model', {}).get('retrain_frequency', 'weekly')
        
        # Check last training date
        try:
            with open('models/latest_timestamp.txt', 'r') as f:
                last_training_timestamp = f.read().strip()
                last_training_date = datetime.strptime(last_training_timestamp, '%Y%m%d_%H%M%S')
                
            days_since_training = (datetime.now() - last_training_date).days
            
            should_retrain = False
            if retrain_freq == 'daily':
                should_retrain = days_since_training >= 1
            elif retrain_freq == 'weekly':
                should_retrain = days_since_training >= 7
            elif retrain_freq == 'monthly':
                should_retrain = days_since_training >= 30
                
            if should_retrain:
                self.logger.info(f"Retraining models (last training: {days_since_training} days ago)")
                await self._retrain_models()
                
        except FileNotFoundError:
            # No previous training found - train models
            self.logger.info("No previous model training found - training new models")
            await self._retrain_models()
            
    async def _retrain_models(self):
        """Retrain models with latest data"""
        self.logger.info("Starting model retraining...")
        
        try:
            # Create training dataset
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
            
            training_data = self.feature_engineer.create_training_dataset(start_date, end_date)
            
            if not training_data.empty:
                # Train models
                test_scores = self.model_ensemble.train_models(training_data)
                
                # Log training results
                self._log_system_status(
                    "model_training", 
                    "success", 
                    f"Models retrained with {len(training_data)} samples",
                    str(test_scores)
                )
                
                self.last_training = datetime.now()
                self.logger.info("Model retraining completed")
                
            else:
                self.logger.warning("No training data available for retraining")
                
        except Exception as e:
            self.logger.error(f"Model retraining failed: {e}")
            self._log_system_status("model_training", "error", str(e))
            raise
            
    async def _generate_daily_predictions(self):
        """Generate predictions for today"""
        self.logger.info("Generating daily predictions...")
        
        today = datetime.now().strftime('%Y-%m-%d')
        
        try:
            predictions = self.prediction_engine.generate_daily_predictions(today)
            
            if not predictions.empty:
                self.logger.info(f"Generated {len(predictions)} predictions for {today}")
                
                # Log prediction summary
                summary = f"Premium: {len(predictions[predictions['confidence_tier'] == 'premium'])}, "
                summary += f"Standard: {len(predictions[predictions['confidence_tier'] == 'standard'])}, "
                summary += f"Value: {len(predictions[predictions['confidence_tier'] == 'value'])}"
                
                self._log_system_status("prediction_generation", "success", summary)
                
            else:
                self.logger.warning(f"No predictions generated for {today}")
                self._log_system_status("prediction_generation", "warning", "No predictions generated")
                
        except Exception as e:
            self.logger.error(f"Prediction generation failed: {e}")
            self._log_system_status("prediction_generation", "error", str(e))
            raise
            
    async def _update_previous_results(self):
        """Update results for previous day's predictions"""
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        try:
            self.prediction_engine.update_prediction_results(yesterday)
            self.logger.info(f"Updated prediction results for {yesterday}")
            
        except Exception as e:
            self.logger.warning(f"Failed to update results for {yesterday}: {e}")
            
    async def _send_predictions(self):
        """Send predictions via Telegram"""
        try:
            await self.telegram_bot.send_scheduled_predictions()
            self.logger.info("Predictions sent via Telegram")
            
        except Exception as e:
            self.logger.error(f"Failed to send Telegram predictions: {e}")
            raise
            
    async def _send_error_notification(self, error_message: str):
        """Send error notification via Telegram"""
        try:
            if self.telegram_bot and self.telegram_bot.application:
                error_text = f"🚨 **Claude_ML System Error** 🚨\n\n{error_message}\n\nTimestamp: {datetime.now()}"
                await self.telegram_bot.application.bot.send_message(
                    chat_id=self.config['telegram']['chat_id'],
                    text=error_text,
                    parse_mode='Markdown'
                )
        except:
            pass  # Don't let notification errors crash the system
            
    def _log_system_status(self, component: str, status: str, message: str, details: str = None):
        """Log system status to database"""
        try:
            conn = sqlite3.connect(self.config.get('database', {}).get('path', 'data/mlb_predictions.db'))
            
            conn.execute('''
                INSERT INTO system_logs (timestamp, component, action, status, message, details)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().isoformat(),
                component,
                'workflow',
                status,
                message,
                details
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"Failed to log system status: {e}")
            
    def run_backtest(self, days: int = 30):
        """Run backtesting analysis"""
        self.logger.info(f"Running {days}-day backtest")
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        results = self.backtester.run_historical_backtest(start_date, end_date)
        
        # Print summary
        overall = results.get('overall_metrics', {})
        if overall:
            print(f"\nBacktest Results ({start_date} to {end_date}):")
            print(f"Total Predictions: {overall.get('total_predictions', 0)}")
            print(f"Overall Accuracy: {overall.get('overall_accuracy', 0):.1%}")
            print(f"Home Run Rate: {overall.get('home_run_rate', 0):.1%}")
            
            roi = overall.get('roi_analysis', {})
            if roi:
                print(f"Hypothetical ROI: {roi.get('roi_percentage', 0):.1f}%")
                
        return results
        
    def schedule_jobs(self):
        """Schedule recurring jobs"""
        run_times = self.config.get('predictions', {}).get('run_times', {})
        
        # Morning predictions
        morning_time = run_times.get('morning', '10:00')
        schedule.every().day.at(morning_time).do(lambda: asyncio.run(self.daily_workflow()))
        
        # Evening predictions (optional second run)
        evening_time = run_times.get('evening', '17:00')
        if evening_time:
            schedule.every().day.at(evening_time).do(lambda: asyncio.run(self.daily_workflow()))
            
        # Weekly backtest
        schedule.every().sunday.at("22:00").do(lambda: self.run_backtest(7))
        
        self.logger.info(f"Jobs scheduled: {morning_time}, {evening_time}, weekly backtest")
        
    def run_scheduler(self):
        """Run the scheduler"""
        self.logger.info("Starting scheduler...")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("Scheduler stopped by user")
        except Exception as e:
            self.logger.error(f"Scheduler crashed: {e}")
            raise
            
    async def run_telegram_bot(self):
        """Run Telegram bot"""
        if self.telegram_bot:
            await self.telegram_bot.run_bot()
        else:
            self.logger.warning("Telegram bot not enabled")


async def main():
    """Main entry point"""
    print("🏟️ Starting Claude_ML System ⚾")
    
    # Initialize system
    system = Claude_ML_System()
    system.initialize_components()
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "predict":
            # Generate today's predictions
            await system.daily_workflow()
            
        elif command == "backtest":
            # Run backtesting
            days = int(sys.argv[2]) if len(sys.argv) > 2 else 30
            system.run_backtest(days)
            
        elif command == "train":
            # Train models
            await system._retrain_models()
            
        elif command == "telegram":
            # Run Telegram bot
            await system.run_telegram_bot()
            
        elif command == "schedule":
            # Run scheduler
            system.schedule_jobs()
            system.run_scheduler()
            
        elif command == "status":
            # System status check
            print(f"Last update: {system.last_update}")
            print(f"Last training: {system.last_training}")
            print(f"System health: {system.system_health}")
            
        else:
            print(f"Unknown command: {command}")
            print("Available commands: predict, backtest, train, telegram, schedule, status")
            
    else:
        # Default: run daily workflow
        await system.daily_workflow()


if __name__ == "__main__":
    asyncio.run(main())
