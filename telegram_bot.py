"""
Telegram Bot for Claude_ML
Delivers daily predictions and handles user interactions
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import sqlite3
import pandas as pd

# Telegram bot libraries
from telegram import Update, Bot, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

from prediction_engine import PredictionEngine
from backtesting import Backtester

logger = logging.getLogger('TelegramBot')


class ClaudeMLBot:
    """Telegram bot for delivering ML predictions"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.bot_token = config['telegram']['bot_token']
        self.chat_id = config['telegram']['chat_id']
        self.db_path = config.get('database', {}).get('path', 'data/mlb_predictions.db')
        
        # Initialize prediction engine
        self.prediction_engine = PredictionEngine(config)
        self.backtester = Backtester(config)
        
        # Bot application
        self.application = None
        
    async def initialize_bot(self):
        """Initialize the Telegram bot"""
        self.application = Application.builder().token(self.bot_token).build()
        
        # Add command handlers
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("predictions", self.predictions_command))
        self.application.add_handler(CommandHandler("today", self.today_command))
        self.application.add_handler(CommandHandler("yesterday", self.yesterday_command))
        self.application.add_handler(CommandHandler("stats", self.stats_command))
        self.application.add_handler(CommandHandler("help", self.help_command))
        self.application.add_handler(CommandHandler("backtest", self.backtest_command))
        
        # Add callback query handler for interactive buttons
        self.application.add_handler(CallbackQueryHandler(self.button_callback))
        
        logger.info("Telegram bot initialized")
        
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        welcome_message = """
🏟️ **Welcome to Claude_ML!** ⚾

I'm your AI-powered home run prediction bot. I analyze player stats, matchups, weather, and park factors to predict which players are most likely to hit home runs today.

**Available Commands:**
/predictions - Get today's predictions
/today - Alias for /predictions
/yesterday - Check yesterday's results
/stats - View recent performance stats
/backtest - Run performance analysis
/help - Show this help message

Let's hit some dingers! 🚀
        """
        
        keyboard = [
            [InlineKeyboardButton("📊 Today's Predictions", callback_data='predictions')],
            [InlineKeyboardButton("📈 Performance Stats", callback_data='stats')],
            [InlineKeyboardButton("❓ Help", callback_data='help')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            welcome_message, 
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )
        
    async def predictions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /predictions command"""
        await self.send_daily_predictions(update.effective_chat.id)
        
    async def today_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /today command"""
        await self.send_daily_predictions(update.effective_chat.id)
        
    async def yesterday_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /yesterday command"""
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        await self.send_prediction_results(update.effective_chat.id, yesterday)
        
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command"""
        await self.send_performance_stats(update.effective_chat.id)
        
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        help_message = """
🤖 **Claude_ML Help** 📚

**Commands:**
• `/predictions` or `/today` - Get today's home run predictions
• `/yesterday` - Check how yesterday's predictions performed
• `/stats` - View recent accuracy and performance metrics
• `/backtest` - Run backtesting analysis

**Prediction Tiers:**
• 💎 **Premium** - Highest confidence picks (top 3)
• ⭐ **Standard** - Good confidence picks (next 4)
• 💰 **Value** - Lower confidence but potential value (next 3)

**How It Works:**
I analyze 50+ features including:
- Recent performance (3-30 day windows)
- Player vs pitcher matchups
- Park factors and weather conditions
- Statcast quality metrics
- Hot/cold streaks and trends

**Disclaimer:**
These are AI predictions for entertainment. Past performance doesn't guarantee future results. Please gamble responsibly.

For technical details, visit: github.com/your-repo
        """
        
        await update.message.reply_text(help_message, parse_mode='Markdown')
        
    async def backtest_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /backtest command"""
        await update.message.reply_text("🔄 Running backtest analysis... This may take a moment.")
        
        try:
            # Run quick 7-day backtest
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            # This would be run in background in production
            results = self.backtester.run_historical_backtest(start_date, end_date)
            
            # Format results
            message = self._format_backtest_results(results, start_date, end_date)
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            await update.message.reply_text("❌ Backtest analysis failed. Please try again later.")
            
    async def button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks"""
        query = update.callback_query
        await query.answer()
        
        if query.data == 'predictions':
            await self.send_daily_predictions(query.message.chat.id)
        elif query.data == 'stats':
            await self.send_performance_stats(query.message.chat.id)
        elif query.data == 'help':
            await self.help_command(update, context)
            
    async def send_daily_predictions(self, chat_id: int):
        """Send today's predictions"""
        try:
            today = datetime.now().strftime('%Y-%m-%d')
            
            # Generate predictions
            predictions = self.prediction_engine.generate_daily_predictions(today)
            
            if predictions.empty:
                message = f"🤔 No predictions available for {today}.\n\nThis could be because:\n• No games scheduled\n• All players filtered out\n• Insufficient data"
                await self.application.bot.send_message(chat_id=chat_id, text=message)
                return
                
            # Format predictions
            formatted_message = self.prediction_engine.format_predictions_for_output(predictions)
            
            # Add interactive buttons
            keyboard = [
                [InlineKeyboardButton("📊 Performance Stats", callback_data='stats')],
                [InlineKeyboardButton("🔄 Refresh Predictions", callback_data='predictions')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Split message if too long (Telegram 4096 char limit)
            if len(formatted_message) > 4000:
                # Send in chunks
                chunks = self._split_message(formatted_message, 4000)
                for i, chunk in enumerate(chunks):
                    if i == len(chunks) - 1:  # Last chunk gets buttons
                        await self.application.bot.send_message(
                            chat_id=chat_id, 
                            text=chunk, 
                            reply_markup=reply_markup,
                            parse_mode='Markdown'
                        )
                    else:
                        await self.application.bot.send_message(
                            chat_id=chat_id, 
                            text=chunk,
                            parse_mode='Markdown'
                        )
            else:
                await self.application.bot.send_message(
                    chat_id=chat_id, 
                    text=formatted_message, 
                    reply_markup=reply_markup,
                    parse_mode='Markdown'
                )
                
        except Exception as e:
            logger.error(f"Failed to send predictions: {e}")
            error_message = "❌ Failed to generate predictions. Please try again later."
            await self.application.bot.send_message(chat_id=chat_id, text=error_message)
            
    async def send_prediction_results(self, chat_id: int, date: str):
        """Send results for a specific date"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get predictions and results
            query = '''
                SELECT 
                    player_name,
                    hr_probability,
                    confidence_tier,
                    matchup,
                    actual_result
                FROM daily_predictions
                WHERE game_date = ?
                ORDER BY hr_probability DESC
            '''
            
            df = pd.read_sql_query(query, conn, params=(date,))
            conn.close()
            
            if df.empty:
                message = f"📅 No predictions found for {date}"
                await self.application.bot.send_message(chat_id=chat_id, text=message)
                return
                
            # Format results
            message = self._format_prediction_results(df, date)
            await self.application.bot.send_message(
                chat_id=chat_id, 
                text=message,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"Failed to get prediction results: {e}")
            error_message = "❌ Failed to retrieve prediction results."
            await self.application.bot.send_message(chat_id=chat_id, text=error_message)
            
    async def send_performance_stats(self, chat_id: int):
        """Send recent performance statistics"""
        try:
            # Get recent accuracy
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            accuracy_metrics = self.prediction_engine.get_prediction_accuracy(start_date, end_date)
            
            message = self._format_performance_stats(accuracy_metrics, start_date, end_date)
            await self.application.bot.send_message(
                chat_id=chat_id, 
                text=message,
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"Failed to get performance stats: {e}")
            error_message = "❌ Failed to retrieve performance statistics."
            await self.application.bot.send_message(chat_id=chat_id, text=error_message)
            
    def _format_prediction_results(self, df: pd.DataFrame, date: str) -> str:
        """Format prediction results for display"""
        
        total_predictions = len(df)
        actual_hrs = df['actual_result'].sum() if 'actual_result' in df.columns else 0
        accuracy = actual_hrs / total_predictions if total_predictions > 0 else 0
        
        message = [
            f"📊 **Results for {date}** ⚾",
            f"🎯 Accuracy: {accuracy:.1%} ({actual_hrs}/{total_predictions})",
            ""
        ]
        
        # Group by tier
        tiers = {
            'premium': '💎 PREMIUM',
            'standard': '⭐ STANDARD', 
            'value': '💰 VALUE'
        }
        
        for tier_key, tier_name in tiers.items():
            tier_df = df[df['confidence_tier'] == tier_key]
            
            if not tier_df.empty:
                tier_hits = tier_df['actual_result'].sum() if 'actual_result' in tier_df.columns else 0
                tier_accuracy = tier_hits / len(tier_df) if len(tier_df) > 0 else 0
                
                message.append(f"{tier_name} ({tier_accuracy:.1%})")
                
                for _, row in tier_df.iterrows():
                    result_emoji = "✅" if row.get('actual_result', 0) == 1 else "❌"
                    prob = row['hr_probability']
                    player = row['player_name']
                    
                    message.append(f"{result_emoji} {player} ({prob:.1%})")
                    
                message.append("")
                
        return "\n".join(message)
        
    def _format_performance_stats(self, metrics: Dict, start_date: str, end_date: str) -> str:
        """Format performance statistics"""
        
        if not metrics:
            return f"📊 No performance data available for {start_date} to {end_date}"
            
        message = [
            f"📈 **Performance Stats** 📊",
            f"📅 Period: {start_date} to {end_date}",
            ""
        ]
        
        # Overall stats
        overall = metrics.get('overall', {})
        if overall:
            message.extend([
                "🎯 **Overall Performance**",
                f"Total Predictions: {overall.get('total_predictions', 0)}",
                f"Correct: {overall.get('correct_predictions', 0)}",
                f"Accuracy: {overall.get('accuracy', 0):.1%}",
                f"Avg Probability: {overall.get('avg_probability', 0):.1%}",
                ""
            ])
            
        # Tier breakdown
        tiers = {
            'premium': '💎 Premium',
            'standard': '⭐ Standard',
            'value': '💰 Value'
        }
        
        message.append("📊 **By Confidence Tier**")
        for tier_key, tier_name in tiers.items():
            tier_data = metrics.get(tier_key, {})
            if tier_data:
                accuracy = tier_data.get('accuracy', 0)
                total = tier_data.get('total_predictions', 0)
                correct = tier_data.get('correct_predictions', 0)
                
                message.append(f"{tier_name}: {accuracy:.1%} ({correct}/{total})")
                
        message.extend([
            "",
            "💡 **Tips:**",
            "• Premium picks have highest confidence",
            "• Check weather and lineup changes",
            "• Past performance doesn't guarantee future results"
        ])
        
        return "\n".join(message)
        
    def _format_backtest_results(self, results: Dict, start_date: str, end_date: str) -> str:
        """Format backtest results"""
        
        overall = results.get('overall_metrics', {})
        
        if not overall:
            return "❌ Backtest analysis incomplete"
            
        message = [
            f"🔬 **Backtest Analysis** 📈",
            f"📅 Period: {start_date} to {end_date}",
            "",
            f"🎯 **Results:**",
            f"Total Predictions: {overall.get('total_predictions', 0)}",
            f"Home Runs Hit: {overall.get('total_home_runs', 0)}",
            f"Overall Accuracy: {overall.get('overall_accuracy', 0):.1%}",
            f"Avg Daily Predictions: {overall.get('avg_daily_predictions', 0):.1f}",
            ""
        ]
        
        # ROI analysis
        roi = overall.get('roi_analysis', {})
        if roi:
            roi_pct = roi.get('roi_percentage', 0)
            roi_emoji = "📈" if roi_pct > 0 else "📉"
            
            message.extend([
                f"💰 **Hypothetical ROI:**",
                f"{roi_emoji} ROI: {roi_pct:.1f}%",
                f"Net Profit: ${roi.get('net_profit', 0):.2f}",
                ""
            ])
            
        message.extend([
            "⚠️ **Note:** Backtesting uses historical data.",
            "Future performance may differ significantly."
        ])
        
        return "\n".join(message)
        
    def _split_message(self, message: str, max_length: int) -> List[str]:
        """Split long message into chunks"""
        
        lines = message.split('\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for line in lines:
            line_length = len(line) + 1  # +1 for newline
            
            if current_length + line_length > max_length and current_chunk:
                # Start new chunk
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_length = line_length
            else:
                current_chunk.append(line)
                current_length += line_length
                
        # Add final chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
            
        return chunks
        
    async def send_scheduled_predictions(self):
        """Send scheduled daily predictions"""
        logger.info("Sending scheduled predictions")
        
        try:
            await self.send_daily_predictions(self.chat_id)
            
            # Update yesterday's results if available
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            self.prediction_engine.update_prediction_results(yesterday)
            
        except Exception as e:
            logger.error(f"Failed to send scheduled predictions: {e}")
            
            # Send error notification
            error_message = f"❌ Failed to generate daily predictions: {str(e)}"
            await self.application.bot.send_message(chat_id=self.chat_id, text=error_message)
            
    async def run_bot(self):
        """Run the bot"""
        await self.initialize_bot()
        
        # Start the bot
        await self.application.initialize()
        await self.application.start()
        
        logger.info("Claude_ML Telegram bot is running")
        
        # Keep running
        await self.application.updater.start_polling()
        
        # Wait until the bot is stopped
        await self.application.updater.idle()


async def send_test_message(config: Dict):
    """Send a test message to verify bot setup"""
    bot = Bot(token=config['telegram']['bot_token'])
    chat_id = config['telegram']['chat_id']
    
    test_message = """
🤖 **Claude_ML Bot Test** ⚾

Bot is properly configured and ready to send predictions!

Use /start to begin or /predictions to get today's picks.
    """
    
    try:
        await bot.send_message(chat_id=chat_id, text=test_message, parse_mode='Markdown')
        print("✅ Test message sent successfully!")
    except Exception as e:
        print(f"❌ Failed to send test message: {e}")


def main():
    """Run the Telegram bot"""
    import yaml
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
        
    # Create and run bot
    bot = ClaudeMLBot(config)
    
    try:
        asyncio.run(bot.run_bot())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Bot crashed: {e}")


if __name__ == "__main__":
    # Uncomment to test bot setup
    # import yaml
    # with open('config.yaml', 'r') as f:
    #     config = yaml.safe_load(f)
    # asyncio.run(send_test_message(config))
    
    main()