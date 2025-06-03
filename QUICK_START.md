# Claude_ML Quick Start Guide

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
