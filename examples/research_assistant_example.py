#!/usr/bin/env python3
"""
Research Assistant Example

This example demonstrates how to use the Research Assistant to gather information
on various topics.
"""

import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agents.research_assistant import run_research_assistant


def run_example():
    """Run the research assistant example."""
    print("Running research assistant example...")

    # Run the research assistant
    run_research_assistant()

if __name__ == "__main__":
    run_example()
