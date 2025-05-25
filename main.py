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
from src.core.distributed_memory_main import chat_with_distributed_memory_agent
from src.core.enhanced_main import chat_with_enhanced_agent
from src.core.error_recovery_main import chat_with_error_recovery_agent
from src.core.knowledge_graph_main import chat_with_knowledge_graph_agent
from src.core.main import chat_with_agent
from src.core.multi_agent_main import chat_with_multi_agent_learning_system
from src.core.reinforcement_learning_main import chat_with_rl_agent
from src.core.seo_main import chat_with_seo_agent
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
        choices=[
            "basic",
            "advanced",
            "enhanced",
            "advanced_enhanced",
            "multi_agent",
            "reinforcement_learning",
            "distributed_memory",
            "knowledge_graph",
            "error_recovery",
            "research_reports",
            "seo",
            "api",
        ],
        default="basic",
        help="Agent mode to run",
    )

    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("API_HOST", "0.0.0.0"),
        help="Host to bind the API server to (only used with --mode=api)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("API_PORT", "8000")),
        help="Port to bind the API server to (only used with --mode=api)",
    )

    parser.add_argument(
        "--reload",
        action="store_true",
        default=os.getenv("API_RELOAD", "false").lower() == "true",
        help="Enable auto-reload on code changes (only used with --mode=api)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        default=os.getenv("API_DEBUG", "false").lower() == "true",
        help="Enable debug mode (only used with --mode=api)",
    )

    args = parser.parse_args()

    print(f"Starting DataMCPServerAgent in {args.mode} mode...")

    if args.mode == "api":
        # Set environment variables for API
        os.environ["API_HOST"] = args.host
        os.environ["API_PORT"] = str(args.port)
        os.environ["API_RELOAD"] = str(args.reload).lower()
        os.environ["API_DEBUG"] = str(args.debug).lower()

        # Import and start the API server
        from src.api.main import start_api

        start_api()
    elif args.mode == "basic":
        asyncio.run(chat_with_agent())
    elif args.mode == "advanced":
        asyncio.run(chat_with_advanced_agent())
    elif args.mode == "enhanced":
        asyncio.run(chat_with_enhanced_agent())
    elif args.mode == "advanced_enhanced":
        asyncio.run(chat_with_advanced_enhanced_agent())
    elif args.mode == "multi_agent":
        asyncio.run(chat_with_multi_agent_learning_system())
    elif args.mode == "reinforcement_learning":
        asyncio.run(chat_with_rl_agent())
    elif args.mode == "distributed_memory":
        asyncio.run(chat_with_distributed_memory_agent())
    elif args.mode == "knowledge_graph":
        asyncio.run(chat_with_knowledge_graph_agent())
    elif args.mode == "error_recovery":
        asyncio.run(chat_with_error_recovery_agent())
    elif args.mode == "research_reports":
        # Import here to avoid circular imports
        from research_reports_runner import run_research_reports_agent

        asyncio.run(run_research_reports_agent())
    elif args.mode == "seo":
        asyncio.run(chat_with_seo_agent())


if __name__ == "__main__":
    main()
