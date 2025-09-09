#!/usr/bin/env python3
"""
Main entry point for LingBot Healthcare Assistant.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from deployment.gradio_interface import LingBotInterface


def main():
    """Main function to run LingBot."""
    print("🤖 Starting LingBot Healthcare Assistant...")
    
    try:
        # Create and launch interface
        lingbot = LingBotInterface()
        lingbot.launch(share=False, server_port=7860)
        
    except KeyboardInterrupt:
        print("\n👋 LingBot stopped by user")
    except Exception as e:
        print(f"❌ Error starting LingBot: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
