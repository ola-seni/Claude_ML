#!/usr/bin/env python3
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
            print("\n🎉 Daily predictions complete!")
        else:
            print("\n❌ Daily workflow had issues")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
