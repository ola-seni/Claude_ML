#!/usr/bin/env python3
"""
FIXED Telegram Bot Integration for Claude_ML Daily Predictions
Sends daily predictions and handles user commands
"""

import asyncio
import requests
import json
from datetime import datetime, timedelta
from pathlib import Path

class TelegramBot:
    """Simple Telegram bot for sending predictions - FIXED VERSION"""
    
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
        
    def send_message(self, text, parse_mode='Markdown'):
        """Send a message to Telegram - FIXED VERSION"""
        url = f"{self.base_url}/sendMessage"
        
        # FIXED: Use proper data format
        data = {
            'chat_id': self.chat_id,
            'text': text
        }
        
        # Add parse_mode only if specified
        if parse_mode:
            data['parse_mode'] = parse_mode
        
        # Split long messages (Telegram has 4096 char limit)
        if len(text) > 4000:
            messages = self._split_message(text, 4000)
            for i, msg in enumerate(messages):
                try:
                    msg_data = {
                        'chat_id': self.chat_id,
                        'text': msg
                    }
                    if parse_mode:
                        msg_data['parse_mode'] = parse_mode
                        
                    response = requests.post(url, json=msg_data, timeout=10)
                    
                    if response.status_code == 200:
                        if i == 0:  # Only print success once
                            print("✅ Telegram message sent successfully!")
                    else:
                        print(f"⚠️ Telegram API returned status {response.status_code}")
                        print(f"Error: {response.text}")
                        
                except Exception as e:
                    print(f"❌ Failed to send Telegram message: {e}")
                    return False
        else:
            try:
                response = requests.post(url, json=data, timeout=10)
                
                if response.status_code == 200:
                    print("✅ Telegram message sent successfully!")
                    return True
                else:
                    print(f"⚠️ Telegram API returned status {response.status_code}")
                    print(f"Error: {response.text}")
                    return False
                    
            except Exception as e:
                print(f"❌ Failed to send Telegram message: {e}")
                return False
                
        return True
        
    def _split_message(self, text, max_length):
        """Split long message into chunks"""
        lines = text.split('\n')
        messages = []
        current_message = []
        current_length = 0
        
        for line in lines:
            line_length = len(line) + 1  # +1 for newline
            
            if current_length + line_length > max_length and current_message:
                messages.append('\n'.join(current_message))
                current_message = [line]
                current_length = line_length
            else:
                current_message.append(line)
                current_length += line_length
                
        if current_message:
            messages.append('\n'.join(current_message))
            
        return messages
        
    def test_connection(self):
        """Test if bot can send messages - FIXED VERSION"""
        test_message = "🤖 Claude_ML Bot Test\n✅ Connection successful!"
        return self.send_message(test_message, parse_mode=None)  # Don't use Markdown for test

def format_predictions_for_telegram(prediction_data):
    """Format predictions for Telegram delivery"""
    if not prediction_data or not prediction_data.get('predictions'):
        return "❌ No predictions available today"
        
    predictions = prediction_data['predictions']
    games = prediction_data.get('games', [])
    
    # Header
    message = [
        "🏟️ TODAY'S MLB HOME RUN PREDICTIONS",
        f"📅 {datetime.now().strftime('%A, %B %d, %Y')}",
        ""
    ]
    
    # Games info
    if games:
        message.extend([
            f"📋 {len(games)} SCHEDULED GAMES",
            ""
        ])
        for game in games[:5]:  # Show first 5 games
            message.append(f"🏟️ {game.get('away_abbr', 'UNK')} @ {game.get('home_abbr', 'UNK')}")
        
        if len(games) > 5:
            message.append(f"   ... and {len(games) - 5} more games")
        message.append("")
    
    # Top picks
    message.extend([
        "🎯 TOP 10 PICKS",
        ""
    ])
    
    for i, pred in enumerate(predictions[:10], 1):
        prob = pred['hr_probability']
        name = pred['player_name']
        team = pred.get('team', 'UNK')
        recent_hrs = pred.get('recent_hrs', 0)
        recent_games = pred.get('recent_games', 0)
        
        # Confidence tier
        if prob >= 0.10:
            tier = "💎"
        elif prob >= 0.07:
            tier = "⭐"
        else:
            tier = "💰"
            
        # WITH PITCHER
        pitcher = pred.get('pitcher_name', 'TBD')    
        message.append(f"{i:2d}. {tier} {name} ({prob:4.1%})")
        message.append(f"    {team} | {recent_hrs} HRs in {recent_games} games")
        if i % 3 == 0:  # Add spacing every 3 picks
            message.append("")
    
    # Tier breakdown
    premium = [p for p in predictions if p['hr_probability'] >= 0.10]
    standard = [p for p in predictions if 0.07 <= p['hr_probability'] < 0.10]
    value = [p for p in predictions if p['hr_probability'] < 0.07]
    
    message.extend([
        "",
        "📊 SUMMARY",
        f"💎 Premium: {len(premium)} players (10%+ probability)",
        f"⭐ Standard: {len(standard)} players (7-10% probability)",
        f"💰 Value: {len(value)} players (<7% probability)",
        ""
    ])
    
    # Stats
    if predictions:
        avg_prob = sum(p['hr_probability'] for p in predictions) / len(predictions)
        max_prob = max(p['hr_probability'] for p in predictions)
        
        message.extend([
            f"📈 Highest: {max_prob:.1%} | Average: {avg_prob:.1%}",
            f"📊 Total predictions: {len(predictions)}",
            ""
        ])
    
    # Footer
    message.extend([
        "🎊 Good luck with today's games!",
        "",
        "⚠️ These are AI predictions for entertainment.",
        "Past performance doesn't guarantee future results."
    ])
    
    return '\n'.join(message)

def send_daily_predictions_to_telegram(bot_token, chat_id):
    """Load and send today's predictions via Telegram"""
    print("📱 Sending daily predictions to Telegram...")
    
    # Load today's predictions
    today_str = datetime.now().strftime('%Y%m%d')
    prediction_files = [
        f"data/predictions/calibrated_predictions_{today_str}.json",
        f"data/predictions/fixed_predictions_{today_str}.json",
        f"data/predictions/daily_predictions_{today_str}.json"
    ]
    
    prediction_data = None
    for file_path in prediction_files:
        if Path(file_path).exists():
            try:
                with open(file_path, 'r') as f:
                    prediction_data = json.load(f)
                print(f"✅ Loaded predictions from {file_path}")
                break
            except Exception as e:
                print(f"⚠️ Could not load {file_path}: {e}")
                continue
    
    if not prediction_data:
        print("❌ No prediction files found for today")
        return False
    
    # Format for Telegram
    telegram_message = format_predictions_for_telegram(prediction_data)
    
    # Send via Telegram
    bot = TelegramBot(bot_token, chat_id)
    return bot.send_message(telegram_message, parse_mode=None)  # Don't use Markdown to avoid formatting issues

def main():
    """Test Telegram integration"""
    print("🤖 Testing FIXED Telegram Integration")
    print("=" * 40)
    
    # Load environment variables
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not bot_token or not chat_id:
        print("❌ Please set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env file")
        return
    
    # Test connection
    bot = TelegramBot(bot_token, chat_id)
    
    if bot.test_connection():
        print("✅ FIXED Telegram bot is working!")
        
        # Try to send today's predictions
        if send_daily_predictions_to_telegram(bot_token, chat_id):
            print("🎊 Daily predictions sent successfully!")
        else:
            print("⚠️ Could not send daily predictions (no prediction files found)")
    else:
        print("❌ FIXED Telegram bot test failed")

if __name__ == "__main__":
    main()
