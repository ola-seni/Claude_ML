#!/usr/bin/env python3
"""
Fixed Daily MLB Prediction Orchestrator
Complete automated system for daily predictions with Telegram delivery
"""
from dotenv import load_dotenv
load_dotenv()  # Load .env file immediately

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Load environment variables IMMEDIATELY

load_dotenv()

print("🔧 Environment variables loaded")  # Debug line


def run_data_update():
    """Run the daily data update"""
    print("🔄 STEP 1: Updating Data")
    print("=" * 30)
    
    try:
        from real_2025_collector import Real2025DataCollector
        
        # Get recent data (last 7 days)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        collector = Real2025DataCollector()
        games, players = collector.collect_2025_data(start_date, end_date)
        
        if games > 0 or players > 0:
            print(f"✅ Data update completed: {games} games, {players} players")
            return True
        else:
            print("⚠️ No new data collected but continuing...")
            return True
            
    except Exception as e:
        print(f"❌ Data update failed: {e}")
        print("⚠️ Continuing with existing data...")
        return False

def run_predictions():
    """Generate daily predictions"""
    print("\n🎯 STEP 2: Generating Predictions")
    print("=" * 35)
    
    try:
        # Try the calibrated predictor first
        try:
            from calibrated_daily_predictions import CalibratedDailyPredictor
            predictor = CalibratedDailyPredictor()
        except:
            # Fall back to fixed predictor
            from fixed_daily_predictions import FixedDailyPredictor
            predictor = FixedDailyPredictor()
        
        prediction_data = predictor.generate_daily_predictions()
        
        if prediction_data and prediction_data['predictions']:
            predictor.display_predictions(prediction_data)
            print("\n✅ Predictions generated successfully")
            return True, prediction_data
        else:
            print("❌ No predictions could be generated")
            return False, None
            
    except Exception as e:
        print(f"❌ Prediction generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def send_telegram_predictions(bot_token, chat_id):
    """Send predictions via Telegram"""
    print("\n📱 STEP 3: Sending to Telegram")
    print("=" * 30)
    
    if not bot_token or not chat_id:
        print("⚠️ Telegram credentials not provided - skipping")
        return False
        
    try:
        from telegram_integration import send_daily_predictions_to_telegram
        
        success = send_daily_predictions_to_telegram(bot_token, chat_id)
        
        if success:
            print("✅ Predictions sent to Telegram successfully")
            return True
        else:
            print("❌ Failed to send predictions to Telegram")
            return False
            
    except Exception as e:
        print(f"❌ Telegram delivery failed: {e}")
        return False

def load_credentials():
    """Load credentials from environment variables"""
    # Environment variables are already loaded by load_dotenv() at the top
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    return bot_token, chat_id

def create_credentials_template():
    """Create a template credentials file"""
    template = {
        "telegram_bot_token": "YOUR_BOT_TOKEN_FROM_BOTFATHER",
        "telegram_chat_id": "YOUR_CHAT_ID_FROM_USERINFOBOT",
        "instructions": {
            "telegram_bot_token": "Get from @BotFather on Telegram",
            "telegram_chat_id": "Get from @userinfobot on Telegram"
        }
    }
    
    with open('credentials.json', 'w') as f:
        json.dump(template, f, indent=2)
        
    print("✅ Created credentials.json template")
    print("📝 Please edit credentials.json with your Telegram credentials")

def check_system_health():
    """Check if all system components are available"""
    print("🔍 SYSTEM HEALTH CHECK")
    print("=" * 25)
    
    issues = []
    
    # Check database
    db_path = Path('data/mlb_predictions.db')
    if not db_path.exists():
        issues.append("❌ Database not found")
    else:
        print("✅ Database found")
    
    # Check models
    models_path = Path('models/latest_timestamp.txt')
    if not models_path.exists():
        issues.append("❌ Trained models not found")
    else:
        print("✅ Trained models found")
    
    # Check config
    config_path = Path('config.yaml')
    if not config_path.exists():
        issues.append("❌ Config file not found")
    else:
        print("✅ Config file found")
    
    # Check credentials (with proper loading)
    bot_token, chat_id = load_credentials()
    if not bot_token or not chat_id:
        issues.append("⚠️ Telegram credentials not configured")
    else:
        print("✅ Telegram credentials found")
        print(f"   Bot token: ***{bot_token[-10:] if len(bot_token) > 10 else '***'}")
        print(f"   Chat ID: {chat_id}")
    
    if issues:
        print(f"\n⚠️ ISSUES FOUND:")
        for issue in issues:
            print(f"   {issue}")
        return False
    else:
        print(f"\n✅ System health check passed!")
        return True

def test_telegram_connection():
    """Test Telegram connection"""
    print("🧪 Testing Telegram Connection")
    print("=" * 30)
    
    bot_token, chat_id = load_credentials()
    
    if not bot_token or not chat_id:
        print("❌ Telegram credentials not found")
        print("💡 Check your .env file contains:")
        print("   TELEGRAM_BOT_TOKEN=your_token")
        print("   TELEGRAM_CHAT_ID=your_chat_id")
        return False
        
    try:
        from telegram_integration import TelegramBot
        bot = TelegramBot(bot_token, chat_id)
        
        if bot.test_connection():
            print("✅ Telegram connection successful!")
            return True
        else:
            print("❌ Telegram connection failed")
            return False
            
    except Exception as e:
        print(f"❌ Telegram test error: {e}")
        return False

def run_daily_workflow(args):
    """Run the complete daily workflow"""
    print("🏟️ CLAUDE_ML DAILY PREDICTION SYSTEM ⚾")
    print("=" * 50)
    print(f"📅 {datetime.now().strftime('%A, %B %d, %Y')}")
    print("")
    
    # Health check
    if not args.skip_health_check:
        if not check_system_health():
            print("\n❌ System health check failed")
            if not args.force:
                print("💡 Use --force to continue anyway")
                return False
    
    # Step 1: Update data (unless skipped)
    if not args.skip_data_update:
        data_success = run_data_update()
        if not data_success and not args.force:
            print("❌ Data update failed. Use --force to continue anyway")
            return False
    else:
        print("⏭️ Skipping data update (--skip-data-update)")
    
    # Step 2: Generate predictions
    pred_success, prediction_data = run_predictions()
    if not pred_success:
        print("❌ Prediction generation failed")
        return False
    
    # Step 3: Send to Telegram (unless skipped)
    if not args.skip_telegram:
        bot_token, chat_id = load_credentials()
        
        if bot_token and chat_id:
            telegram_success = send_telegram_predictions(bot_token, chat_id)
        else:
            print("⚠️ Telegram credentials not found - skipping delivery")
            telegram_success = False
    else:
        print("⏭️ Skipping Telegram delivery (--skip-telegram)")
        telegram_success = False
    
    # Summary
    print(f"\n🎊 DAILY WORKFLOW COMPLETE!")
    print("=" * 30)
    print(f"✅ Predictions generated: {pred_success}")
    print(f"📱 Telegram delivered: {telegram_success}")
    
    if pred_success:
        # Show quick summary
        if prediction_data and prediction_data.get('predictions'):
            predictions = prediction_data['predictions']
            top_pick = predictions[0]
            total_predictions = len(predictions)
            
            print(f"\n🎯 QUICK SUMMARY:")
            print(f"   Top pick: {top_pick['player_name']} ({top_pick['hr_probability']:.1%})")
            print(f"   Total predictions: {total_predictions}")
            
            # Show saved file
            today_str = datetime.now().strftime('%Y%m%d')
            pred_files = [
                f"data/predictions/calibrated_predictions_{today_str}.json",
                f"data/predictions/fixed_predictions_{today_str}.json",
                f"data/predictions/daily_predictions_{today_str}.json"
            ]
            
            for pred_file in pred_files:
                if Path(pred_file).exists():
                    print(f"   Saved to: {pred_file}")
                    break
    
    return pred_success

def main():
    """Main entry point with command line arguments"""
    parser = argparse.ArgumentParser(description='Claude_ML Daily Prediction System')
    
    parser.add_argument('--skip-data-update', action='store_true',
                       help='Skip the data update step')
    parser.add_argument('--skip-telegram', action='store_true',
                       help='Skip Telegram delivery')
    parser.add_argument('--skip-health-check', action='store_true',
                       help='Skip system health check')
    parser.add_argument('--force', action='store_true',
                       help='Continue even if some steps fail')
    parser.add_argument('--setup-credentials', action='store_true',
                       help='Create credentials template file')
    parser.add_argument('--test-telegram', action='store_true',
                       help='Test Telegram connection only')
    parser.add_argument('--health-check', action='store_true',
                       help='Run health check only')
    
    args = parser.parse_args()
    
    # Handle special commands
    if args.setup_credentials:
        create_credentials_template()
        return
        
    if args.health_check:
        check_system_health()
        return
        
    if args.test_telegram:
        test_telegram_connection()
        return
    
    # Run main workflow
    success = run_daily_workflow(args)
    
    if success:
        print(f"\n🎉 Success! Your daily predictions are ready!")
    else:
        print(f"\n❌ Workflow completed with issues")
        sys.exit(1)

if __name__ == "__main__":
    main()
