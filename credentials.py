"""
Credentials Management for Claude_ML
Handles API keys and sensitive configuration securely
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional
from dotenv import load_dotenv

logger = logging.getLogger('Credentials')


class CredentialsManager:
    """Secure management of API keys and sensitive data"""
    
    def __init__(self, credentials_file: str = '.env'):
        self.credentials_file = credentials_file
        self.credentials = {}
        
        # Load environment variables
        load_dotenv(credentials_file)
        
        # Load credentials
        self._load_credentials()
        
    def _load_credentials(self):
        """Load credentials from environment variables"""
        
        # Weather API
        self.credentials['weather_api_key'] = os.getenv('WEATHER_API_KEY', 'your_openweather_api_key_here')
        
        # Telegram Bot
        self.credentials['telegram_bot_token'] = os.getenv('TELEGRAM_BOT_TOKEN', 'your_telegram_bot_token_here')
        self.credentials['telegram_chat_id'] = os.getenv('TELEGRAM_CHAT_ID', 'your_telegram_chat_id_here')
        
        # Optional: Database credentials (if using external DB)
        self.credentials['db_host'] = os.getenv('DB_HOST', 'localhost')
        self.credentials['db_port'] = os.getenv('DB_PORT', '5432')
        self.credentials['db_name'] = os.getenv('DB_NAME', 'claude_ml')
        self.credentials['db_user'] = os.getenv('DB_USER', 'claude_ml_user')
        self.credentials['db_password'] = os.getenv('DB_PASSWORD', 'your_db_password_here')
        
        # Optional: External APIs
        self.credentials['statcast_api_key'] = os.getenv('STATCAST_API_KEY', '')
        self.credentials['fangraphs_api_key'] = os.getenv('FANGRAPHS_API_KEY', '')
        
        # Rate limiting settings
        self.credentials['api_rate_limit'] = float(os.getenv('API_RATE_LIMIT', '0.5'))
        
        logger.info("Credentials loaded from environment")
        
    def get_credential(self, key: str) -> Optional[str]:
        """Get a specific credential"""
        return self.credentials.get(key)
        
    def update_config_with_credentials(self, config: Dict) -> Dict:
        """Update config dictionary with credentials"""
        updated_config = config.copy()
        
        # Update weather API key
        if 'weather' in updated_config and 'api_key' in updated_config['weather']:
            updated_config['weather']['api_key'] = self.get_credential('weather_api_key')
            
        # Update Telegram credentials
        if 'telegram' in updated_config:
            updated_config['telegram']['bot_token'] = self.get_credential('telegram_bot_token')
            updated_config['telegram']['chat_id'] = self.get_credential('telegram_chat_id')
            
        # Update rate limiting
        if 'data_sources' in updated_config:
            if 'mlb_stats_api' in updated_config['data_sources']:
                updated_config['data_sources']['mlb_stats_api']['rate_limit_delay'] = self.get_credential('api_rate_limit')
                
        return updated_config
        
    def validate_credentials(self) -> Dict[str, bool]:
        """Validate that required credentials are present"""
        validation_results = {}
        
        # Check required credentials
        required_creds = {
            'weather_api_key': 'Weather API functionality',
            'telegram_bot_token': 'Telegram bot notifications',
            'telegram_chat_id': 'Telegram message delivery'
        }
        
        for cred_key, description in required_creds.items():
            cred_value = self.get_credential(cred_key)
            
            # Custom validation per credential type
            if cred_key == 'telegram_chat_id':
                # Chat ID should be a number (can be 8-15 digits)
                is_valid = cred_value and cred_value.isdigit() and len(cred_value) >= 8
            else:
                # For other credentials, check they exist and aren't placeholders
                is_valid = cred_value and not cred_value.startswith('your_') and len(cred_value) > 10
                
            validation_results[cred_key] = is_valid
            
            if not is_valid:
                logger.warning(f"Invalid or missing credential: {cred_key} ({description})")
            else:
                logger.info(f"Valid credential found: {cred_key}")
                
        return validation_results
        
    def create_sample_env_file(self):
        """Create a sample .env file with placeholder values"""
        sample_content = """# Claude_ML Environment Variables
# Copy this file to .env and fill in your actual values

# OpenWeather API Key (required for weather data)
# Get from: https://openweathermap.org/api
WEATHER_API_KEY=your_openweather_api_key_here

# Telegram Bot Configuration (required for notifications)
# Create bot: https://t.me/botfather
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here

# Optional: External Database (leave as default for SQLite)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=claude_ml
DB_USER=claude_ml_user
DB_PASSWORD=your_db_password_here

# Optional: Additional API Keys
STATCAST_API_KEY=
FANGRAPHS_API_KEY=

# API Rate Limiting (seconds between requests)
API_RATE_LIMIT=0.5

# Optional: Email notifications (if implementing)
EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=587
EMAIL_USER=your_email@gmail.com
EMAIL_PASSWORD=your_email_app_password
"""
        
        sample_file = Path('.env.sample')
        with open(sample_file, 'w') as f:
            f.write(sample_content)
            
        logger.info(f"Sample environment file created: {sample_file}")
        print(f"✅ Sample .env file created at {sample_file}")
        print("📝 Copy this to .env and fill in your actual credentials")


def setup_credentials():
    """Interactive setup for credentials"""
    print("🔐 Claude_ML Credentials Setup")
    print("=" * 40)
    
    env_file = Path('.env')
    
    if env_file.exists():
        overwrite = input(f"\n.env file already exists. Overwrite? (y/n): ").lower().strip()
        if overwrite != 'y':
            print("❌ Setup cancelled")
            return
            
    credentials = {}
    
    # Weather API
    print("\n🌤️  Weather API Setup")
    print("Get your free API key from: https://openweathermap.org/api")
    weather_key = input("Enter your OpenWeather API key: ").strip()
    credentials['WEATHER_API_KEY'] = weather_key
    
    # Telegram Bot
    print("\n🤖 Telegram Bot Setup")
    print("1. Message @BotFather on Telegram")
    print("2. Create a new bot with /newbot")
    print("3. Get your bot token")
    bot_token = input("Enter your Telegram bot token: ").strip()
    credentials['TELEGRAM_BOT_TOKEN'] = bot_token
    
    print("\n4. Get your chat ID:")
    print("   - Start a chat with your bot")
    print("   - Send any message")
    print("   - Visit: https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates")
    print("   - Look for 'chat':{'id': YOUR_CHAT_ID}")
    chat_id = input("Enter your Telegram chat ID: ").strip()
    credentials['TELEGRAM_CHAT_ID'] = chat_id
    
    # Optional settings
    print("\n⚙️  Optional Settings")
    rate_limit = input("API rate limit in seconds (default: 0.5): ").strip()
    credentials['API_RATE_LIMIT'] = rate_limit or '0.5'
    
    # Write to .env file
    with open('.env', 'w') as f:
        f.write("# Claude_ML Environment Variables\n")
        f.write("# Generated by credentials setup\n\n")
        
        for key, value in credentials.items():
            f.write(f"{key}={value}\n")
            
    print("\n✅ Credentials saved to .env file")
    
    # Validate
    manager = CredentialsManager()
    validation = manager.validate_credentials()
    
    if all(validation.values()):
        print("🎉 All credentials validated successfully!")
    else:
        print("⚠️  Some credentials may need attention:")
        for cred, is_valid in validation.items():
            status = "✅" if is_valid else "❌"
            print(f"   {status} {cred}")


def main():
    """Command line interface for credentials management"""
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "setup":
            setup_credentials()
        elif command == "validate":
            manager = CredentialsManager()
            validation = manager.validate_credentials()
            
            print("\n🔐 Credential Validation Results:")
            for cred, is_valid in validation.items():
                status = "✅ Valid" if is_valid else "❌ Invalid/Missing"
                print(f"   {cred}: {status}")
                
            if all(validation.values()):
                print("\n🎉 All credentials are valid!")
            else:
                print("\n⚠️  Some credentials need attention. Run 'python credentials.py setup' to fix.")
                
        elif command == "sample":
            manager = CredentialsManager()
            manager.create_sample_env_file()
        else:
            print("Available commands: setup, validate, sample")
    else:
        print("Claude_ML Credentials Manager")
        print("Usage: python credentials.py [setup|validate|sample]")


if __name__ == "__main__":
    main()