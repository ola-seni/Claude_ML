#!/usr/bin/env python3
"""
Repository Cleanup Script
Helps identify which files to include in your GitHub repository
"""

import os
from pathlib import Path
import shutil

def analyze_files():
    """Analyze current directory files"""
    print("🔍 Analyzing Claude_ML Files")
    print("=" * 40)
    
    # Essential files that MUST be included
    essential_files = {
        'config.yaml': 'System configuration',
        'models.py': 'ML model training & ensemble', 
        'feature_pipeline.py': 'Feature engineering pipeline',
        'telegram_integration.py': 'Telegram functions',
        'daily_orchestrator.py': 'Main workflow orchestrator',
        'real_2025_collector.py': 'Data collection from MLB API',
        'run_daily.py': 'Simple daily runner',
        'requirements.txt': 'Python dependencies',
        'complete_setup.py': 'Setup script'
    }
    
    # Optional but useful files
    optional_files = {
        'prediction_engine.py': 'Prediction generation',
        'backtesting.py': 'Performance evaluation', 
        'telegram_bot.py': 'Telegram bot class',
        'main.py': 'System orchestrator',
        'QUICK_START.md': 'Usage guide',
        'README.md': 'Documentation'
    }
    
    # Files to exclude (development/debug)
    exclude_patterns = [
        'debug_', 'test_', 'fix_', 'safe_', 'simple_', 'quick_',
        'working_', 'calibrated_', 'improved_', 'fixed_',
        '_test.py', '_debug.py'
    ]
    
    current_files = list(Path('.').glob('*.py')) + list(Path('.').glob('*.md')) + list(Path('.').glob('*.yaml')) + list(Path('.').glob('*.txt'))
    
    print("📋 FILE ANALYSIS")
    print("=" * 20)
    
    # Check essential files
    print("\n✅ ESSENTIAL FILES (Must include):")
    missing_essential = []
    for file, description in essential_files.items():
        if Path(file).exists():
            print(f"   ✅ {file:<25} - {description}")
        else:
            print(f"   ❌ {file:<25} - {description} (MISSING)")
            missing_essential.append(file)
    
    # Check optional files  
    print("\n⭐ OPTIONAL FILES (Good to include):")
    for file, description in optional_files.items():
        if Path(file).exists():
            print(f"   ✅ {file:<25} - {description}")
        else:
            print(f"   ⚪ {file:<25} - {description} (not found)")
    
    # Identify files to exclude
    print("\n🗑️  FILES TO EXCLUDE (Development/debug):")
    exclude_files = []
    for file_path in current_files:
        file_name = file_path.name
        should_exclude = any(pattern in file_name for pattern in exclude_patterns)
        
        if should_exclude and file_name not in essential_files and file_name not in optional_files:
            exclude_files.append(file_name)
            print(f"   🗑️  {file_name}")
    
    # Check for sensitive files
    print("\n🔐 SENSITIVE FILES (Never commit):")
    sensitive_files = ['.env', 'credentials.json']
    for file in sensitive_files:
        if Path(file).exists():
            print(f"   🔐 {file} - Add to .gitignore!")
        else:
            print(f"   ⚪ {file} - Not found")
    
    return essential_files, optional_files, exclude_files, missing_essential

def create_gitignore():
    """Create a proper .gitignore file"""
    gitignore_content = """# Credentials and sensitive data
.env
credentials.json
*.key
config_local.yaml

# Data files (too large for GitHub)
data/*.db
data/*.csv
data/raw/
data/processed/
data/predictions/
data/backtesting/

# Model files (too large for GitHub)  
models/trained/
models/scalers/
*.joblib
*.pkl

# Logs
logs/
*.log

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.so
.pytest_cache/
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Development files
debug_*.py
test_*.py
*_test.py
fix_*.py
safe_*.py
simple_*.py
quick_*.py
working_*.py
calibrated_*.py
improved_*.py
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("✅ Created .gitignore file")

def create_env_example():
    """Create example .env file"""
    env_example = """# Claude_ML Environment Variables
# Copy this file to .env and fill in your actual values

# Telegram Bot Credentials (required)
TELEGRAM_BOT_TOKEN=your_bot_token_from_botfather
TELEGRAM_CHAT_ID=your_chat_id_from_userinfobot

# OpenWeather API (optional)
WEATHER_API_KEY=your_openweather_api_key

# Database
DATABASE_PATH=data/mlb_predictions.db
"""
    
    with open('.env.example', 'w') as f:
        f.write(env_example)
    
    print("✅ Created .env.example file")

def create_simple_readme():
    """Create a simple README for manual running"""
    readme_content = """# Claude_ML Baseball Predictions

AI-powered MLB home run prediction system with 144k+ player performance records.

## 🚀 Quick Setup

1. **Clone and install:**
   ```bash
   git clone https://github.com/yourusername/claude-ml-baseball.git
   cd claude-ml-baseball
   pip install -r requirements.txt
   ```

2. **Set up credentials:**
   ```bash
   cp .env.example .env
   # Edit .env with your Telegram bot credentials
   ```

3. **Initialize system:**
   ```bash
   python3 complete_setup.py
   ```

## 📱 Daily Usage

**Generate today's predictions:**
```bash
python3 run_daily.py
```

**Manual workflow:**
```bash
# Check system health
python3 daily_orchestrator.py --health-check

# Test Telegram
python3 daily_orchestrator.py --test-telegram

# Full workflow  
python3 daily_orchestrator.py
```

## 🎯 What You Get

Daily Telegram messages with MLB home run predictions:
- 💎 Premium picks (highest confidence)
- ⭐ Standard picks (good confidence) 
- 💰 Value picks (potential upside)

## 📊 Features

- **144k+ player performance records**
- **50+ ML features** (recent form, park factors, weather, matchups)
- **Ensemble ML models** (XGBoost, Random Forest, Neural Network)
- **Automated data collection** from MLB APIs
- **Telegram delivery** with formatted predictions
- **Performance tracking** and backtesting

Built with ❤️ for baseball analytics.
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    
    print("✅ Created README.md file")

def main():
    """Main cleanup function"""
    print("🧹 Claude_ML Repository Cleanup")
    print("=" * 40)
    
    # Analyze current files
    essential, optional, exclude, missing = analyze_files()
    
    if missing:
        print(f"\n⚠️  WARNING: Missing essential files: {missing}")
        print("   Make sure these exist before creating repository")
    
    # Create repository files
    print(f"\n🔧 Creating repository files...")
    create_gitignore()
    create_env_example()
    
    if not Path('README.md').exists():
        create_simple_readme()
    else:
        print("⚪ README.md already exists (not overwriting)")
    
    # Summary and recommendations
    print(f"\n📋 SUMMARY & NEXT STEPS")
    print("=" * 30)
    print(f"✅ Essential files: {len([f for f in essential if Path(f).exists()])}/{len(essential)}")
    print(f"⭐ Optional files: {len([f for f in optional if Path(f).exists()])}")
    print(f"🗑️  Files to exclude: {len(exclude)}")
    
    print(f"\n🚀 Ready to create Git repository:")
    print(f"   git init")
    print(f"   git add .")
    print(f"   git commit -m 'Initial Claude_ML setup'")
    print(f"   # Then push to GitHub")
    
    print(f"\n💡 Repository will be ready for manual running!")

if __name__ == "__main__":
    main()
