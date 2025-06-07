"""
Example script demonstrating how to use the SEO Agent with advanced features.
"""

import asyncio
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.seo_main import chat_with_seo_agent

async def run_example():
    """Run the SEO agent example with advanced features."""
    print("Running SEO agent example with advanced features...")

    # Configure the agent
    config = {
        "verbose": True
    }

    print("\nSEO Agent Features:")
    print("1. Basic SEO Analysis: Analyze a webpage for SEO factors")
    print("2. Keyword Research: Research keywords related to a topic")
    print("3. Content Optimization: Optimize content for SEO")
    print("4. Metadata Generation: Generate optimized metadata for SEO")
    print("5. Backlink Analysis: Analyze backlinks to a website")
    print("6. Competitor Analysis: Analyze competitors and compare websites")
    print("7. Rank Tracking: Track keyword rankings over time")
    print("8. Bulk Analysis: Analyze multiple pages or entire websites")
    print("9. ML Content Optimization: Use machine learning to optimize content")
    print("10. ML Ranking Prediction: Predict search rankings with machine learning")
    print("11. Scheduled Reporting: Schedule regular SEO reports")
    print("12. Visualization: Generate charts and graphs for SEO metrics")
    print("\nExample commands:")
    print("- 'Analyze example.com for SEO issues'")
    print("- 'Research keywords for artificial intelligence'")
    print("- 'Optimize this content for SEO: [your content here]'")
    print("- 'Generate metadata for my blog post about machine learning'")
    print("- 'Analyze backlinks for example.com'")
    print("- 'Compare my site example.com with competitor competitor.com'")
    print("- 'Track rankings for example.com for these keywords: AI, machine learning, deep learning'")
    print("- 'Analyze all pages on example.com'")
    print("- 'Use machine learning to optimize this content: [your content here] for these keywords: AI, machine learning'")
    print("- 'Predict search ranking for example.com for the keyword artificial intelligence'")
    print("- 'Schedule a weekly comprehensive SEO report for example.com and send it to me@example.com'")
    print("- 'Generate a visualization of keyword rankings for example.com'")
    print("\nType 'exit' to quit the agent.")

    await chat_with_seo_agent(config=config)

if __name__ == "__main__":
    asyncio.run(run_example())
