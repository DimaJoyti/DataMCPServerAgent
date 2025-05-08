"""
Main entry point for DataMCPServerAgent.
This file provides a simple interface to launch different versions of the agent.
"""

import argparse
import asyncio
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.advanced_enhanced_main import chat_with_advanced_enhanced_agent
from src.core.advanced_main import chat_with_advanced_agent
from src.core.enhanced_main import chat_with_enhanced_agent
from src.core.main import chat_with_agent
from src.core.multi_agent_main import chat_with_multi_agent_learning_system
from src.utils.env_config import load_dotenv

# Load environment variables
load_dotenv()


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description="DataMCPServerAgent - Advanced Agent Architectures"
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["basic", "advanced", "enhanced", "advanced_enhanced", "multi_agent"],
        default="basic",
        help="Agent mode to run (basic, advanced, enhanced, advanced_enhanced, or multi_agent)",
    )

    args = parser.parse_args()

    print(f"Starting DataMCPServerAgent in {args.mode} mode...")

    if args.mode == "basic":
        asyncio.run(chat_with_agent())
    elif args.mode == "advanced":
        asyncio.run(chat_with_advanced_agent())
    elif args.mode == "enhanced":
        asyncio.run(chat_with_enhanced_agent())
    elif args.mode == "advanced_enhanced":
        asyncio.run(chat_with_advanced_enhanced_agent())
    elif args.mode == "multi_agent":
        asyncio.run(chat_with_multi_agent_learning_system())


if __name__ == "__main__":
    main()
