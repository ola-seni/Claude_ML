#!/usr/bin/env python3
"""
Complete Claude_ML Setup & Integration Script
Sets up the entire automated daily prediction system
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

class ClaudeMLIntegrator:
    """Complete system integrator"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.setup_complete = False
        
    def check_prerequisites(self):
        """Check if basic files exist"""
        print("🔍 Checking Prerequisites...")
        
        required_files = [
            'config.yaml',
            'models.py', 
            'feature_pipeline.py',
            'requirements.txt'
        ]
        
        missing_files = []
        for file in required_files:
            if not Path(file).exists():
                missing_files.append(file)
                
        if missing_files:
            print(f"❌ Missing required files: {missing_files}")
            return False
            
        print("✅ All prerequisite files found")
        return True
        
    def setup_telegram_credentials(self):
        """Interactive Telegram setup"""
        print("\n🤖 Telegram Bot Setup")
        print("=" * 30)
        
        env_file = Path('.env')
        if env_file.exists():
            print("✅ .env file already exists")
            return True
            
        print("Setting up Telegram bot credentials...")
        print("\n📝 Instructions:")
        print("1. Message @BotFather on Telegram")
        print("2. Send /newbot")
        print("3. Choose a name like 'Claude_ML_Baseball_Bot'")
        print("4. Copy the bot token")
        print("\n5. Message @userinfobot on Telegram") 
        print("6. Send /start")
        print("7. Copy your chat ID (the number)")
        
        # Get credentials from user
        bot_token = input("\nEnter your bot token: ").strip()
        chat_id = input("Enter your chat ID: ").strip()
        
        if not bot_token or not chat_id:
            print("❌ Invalid credentials")
            return False
            
        # Create .env file
        env_content = f"""# Claude_ML Environment Variables
TELEGRAM_BOT_TOKEN={bot_token}
TELEGRAM_CHAT_ID={chat_id}
WEATHER_API_KEY=optional
DATABASE_PATH=data/mlb_predictions.db
"""
        
        with open('.env', 'w') as f:
            f.write(env_content)
            
        print("✅ Telegram credentials saved to .env")
        return True
        
    def setup_gitignore(self):
        """Create .gitignore file"""
        gitignore_content = """# Credentials and sensitive data
.env
credentials.json
*.key

# Data files (too large for GitHub)
data/
models/trained/
logs/

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
.pytest_cache/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
        
        with open('.gitignore', 'w') as f:
            f.write(gitignore_content)
            
        print("✅ Created .gitignore file")
        
    def test_system_components(self):
        """Test each major component"""
        print("\n🧪 Testing System Components")
        print("=" * 35)
        
        # Test 1: Database
        try:
            import sqlite3
            db_path = 'data/mlb_predictions.db'
            if Path(db_path).exists():
                conn = sqlite3.connect(db_path)
                count = conn.execute("SELECT COUNT(*) FROM player_performance").fetchone()[0]
                conn.close()
                print(f"✅ Database: {count} player performance records")
            else:
                print("⚠️ Database not found - will need data collection")
        except Exception as e:
            print(f"❌ Database test failed: {e}")
            
        # Test 2: Models
        try:
            models_path = Path('models/latest_timestamp.txt')
            if models_path.exists():
                print("✅ Trained models found")
            else:
                print("⚠️ No trained models - will need training")
        except Exception as e:
            print(f"❌ Model check failed: {e}")
            
        # Test 3: Telegram
        try:
            from telegram_integration import TelegramBot
            
            # Load credentials
            bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
            chat_id = os.getenv('TELEGRAM_CHAT_ID')
            
            if bot_token and chat_id:
                bot = TelegramBot(bot_token, chat_id)
                if bot.test_connection():
                    print("✅ Telegram bot working")
                else:
                    print("❌ Telegram bot test failed")
            else:
                print("⚠️ Telegram credentials not found")
                
        except Exception as e:
            print(f"❌ Telegram test failed: {e}")
            
    def create_daily_runner(self):
        """Create simple daily runner script"""
        runner_content = '''#!/usr/bin/env python3
"""
Simple Daily Runner for Claude_ML
Run this script each day for predictions
"""

import os
import sys
from pathlib import Path

def main():
    print("🏟️ Claude_ML Daily Runner ⚾")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not Path('config.yaml').exists():
        print("❌ Not in Claude_ML directory")
        sys.exit(1)
        
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run the orchestrator
    try:
        from daily_orchestrator import run_daily_workflow
        import argparse
        
        # Create simple args
        args = argparse.Namespace(
            skip_data_update=False,
            skip_telegram=False, 
            skip_health_check=False,
            force=False
        )
        
        success = run_daily_workflow(args)
        
        if success:
            print("\\n🎉 Daily predictions complete!")
        else:
            print("\\n❌ Daily workflow had issues")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
        
        with open('run_daily.py', 'w') as f:
            f.write(runner_content)
            
        os.chmod('run_daily.py', 0o755)
        print("✅ Created run_daily.py script")
        
    def create_quick_start_guide(self):
        """Create a quick start guide"""
        guide_content = """# Claude_ML Quick Start Guide

## 🚀 Daily Usage (After Setup)

### Simple Daily Run
```bash
python3 run_daily.py
```

### Manual Steps
```bash
# 1. Update data
python3 real_2025_collector.py

# 2. Generate predictions  
python3 calibrated_daily_predictions.py

# 3. Send to Telegram
python3 telegram_integration.py
```

## 🔧 Troubleshooting

### No Predictions Generated
```bash
# Check system health
python3 daily_orchestrator.py --health-check

# Force data update
python3 real_2025_collector.py $(date -d '7 days ago' +%Y-%m-%d) $(date +%Y-%m-%d)

# Retrain if needed
python3 improved_training.py
```

### Telegram Not Working
```bash
# Test connection
python3 daily_orchestrator.py --test-telegram

# Update credentials in .env file
```

## 📅 Automation

### Option 1: Cron Job
```bash
# Run at 10 AM daily
0 10 * * * cd /path/to/claude_ml && python3 run_daily.py >> logs/daily.log 2>&1
```

### Option 2: Manual
- Run `python3 run_daily.py` each morning
- Check Telegram for predictions
- Save predictions are in `data/predictions/`

## 🎯 What You Get

- **Daily home run predictions** for all MLB games
- **Confidence tiers**: Premium (💎), Standard (⭐), Value (💰)  
- **Telegram delivery** with formatted predictions
- **Historical tracking** of prediction accuracy
- **Automated data updates** to avoid stale data issues

## 📊 Understanding Predictions

- **Premium picks**: Highest confidence (8%+ probability)
- **Standard picks**: Good confidence (5-8% probability) 
- **Value picks**: Lower confidence but potential upside (<5%)
- **Probabilities**: Based on 50+ features and ensemble ML models

Remember: These are predictions for entertainment. Baseball is unpredictable!
"""
        
        with open('QUICK_START.md', 'w') as f:
            f.write(guide_content)
            
        print("✅ Created QUICK_START.md guide")
        
    def run_complete_setup(self):
        """Run the complete setup process"""
        print("🔧 Claude_ML Complete Setup & Integration")
        print("=" * 50)
        
        # Step 1: Prerequisites
        if not self.check_prerequisites():
            print("❌ Prerequisites not met")
            return False
            
        # Step 2: Telegram setup
        if not self.setup_telegram_credentials():
            print("❌ Telegram setup failed")
            return False
            
        # Step 3: Git setup
        self.setup_gitignore()
        
        # Step 4: Create helper scripts
        self.create_daily_runner()
        self.create_quick_start_guide()
        
        # Step 5: Test components
        self.test_system_components()
        
        print("\n" + "=" * 50)
        print("🎊 SETUP COMPLETE!")
        print("=" * 50)
        
        print("\n🚀 Next Steps:")
        print("1. Read QUICK_START.md for usage instructions")
        print("2. Test the system:")
        print("   python3 daily_orchestrator.py --health-check")
        print("3. Run daily predictions:")
        print("   python3 run_daily.py")
        print("4. Set up GitHub (optional):")
        print("   git init && git add . && git commit -m 'Initial setup'")
        
        return True

def main():
    """Main setup function"""
    integrator = ClaudeMLIntegrator()
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Test mode - just check components
        integrator.test_system_components()
    else:
        # Full setup
        integrator.run_complete_setup()

if __name__ == "__main__":
    main()
