<<<<<<< HEAD
# Claude_ML Baseball Predictions

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
=======
# Claude_ML
>>>>>>> origin/main
