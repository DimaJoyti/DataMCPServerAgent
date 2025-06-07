"""
Example script demonstrating how to use the enhanced Bright Data MCP tools
for social media content analysis.
"""

import asyncio
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from bright_data_tools import BrightDataToolkit
from error_handlers import format_error_for_user, with_retry
from result_processors import format_social_media_data

load_dotenv()

# Set up the MCP server parameters
server_params = StdioServerParameters(
    command="npx",
    env={
        "API_TOKEN": os.getenv("API_TOKEN"),
        "BROWSER_AUTH": os.getenv("BROWSER_AUTH"),
        "WEB_UNLOCKER_ZONE": os.getenv("WEB_UNLOCKER_ZONE"),
    },
    args=["@brightdata/mcp"],
)

# Example social media URLs to analyze
SOCIAL_MEDIA_URLS = [
    "https://www.instagram.com/apple/",  # Instagram profile
    "https://twitter.com/Apple",         # Twitter/X profile
]

async def analyze_social_media():
    """Analyze social media content using the enhanced Bright Data MCP tools."""
    print("Starting social media analysis example...")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Create the Bright Data toolkit
            bright_data_toolkit = BrightDataToolkit(session)

            # Get the custom tools
            custom_tools = await bright_data_toolkit.create_custom_tools()

            # Find the social media analyzer tool
            social_media_tool = None
            for tool in custom_tools:
                if tool.name == "social_media_analyzer":
                    social_media_tool = tool
                    break

            if not social_media_tool:
                print("Social media analyzer tool not found. Falling back to standard tools...")

                # Load standard MCP tools
                standard_tools = await load_mcp_tools(session)

                # Find appropriate social media tools
                instagram_tool = None
                twitter_tool = None

                for tool in standard_tools:
                    if tool.name == "web_data_instagram_profiles_Bright_Data":
                        instagram_tool = tool
                    elif tool.name == "web_data_x_posts_Bright_Data":
                        twitter_tool = tool

                # Analyze each URL with the appropriate tool
                for url in SOCIAL_MEDIA_URLS:
                    try:
                        print(f"Analyzing social media content for {url}...")

                        if "instagram.com" in url and instagram_tool:
                            data = await with_retry(instagram_tool.invoke, {"url": url})
                            formatted_data = format_social_media_data(data, "detailed")
                            print("\nInstagram Analysis Results:\n")
                            print(formatted_data)
                        elif ("twitter.com" in url or "x.com" in url) and twitter_tool:
                            data = await with_retry(twitter_tool.invoke, {"url": url})
                            formatted_data = format_social_media_data(data, "detailed")
                            print("\nTwitter/X Analysis Results:\n")
                            print(formatted_data)
                        else:
                            print(f"No appropriate tool found for {url}")
                    except Exception as e:
                        error_message = format_error_for_user(e)
                        print(f"Error analyzing social media content:\n{error_message}")
            else:
                # Use the enhanced social media analyzer tool
                for url in SOCIAL_MEDIA_URLS:
                    try:
                        print(f"Analyzing social media content for {url}...")

                        # Try different analysis types
                        for analysis_type in ["basic", "detailed", "engagement"]:
                            result = await with_retry(
                                social_media_tool.invoke,
                                {"url": url, "analysis_type": analysis_type}
                            )
                            print(f"\n{analysis_type.title()} Analysis Results:\n")
                            print(result)
                            print("\n" + "-" * 50 + "\n")
                    except Exception as e:
                        error_message = format_error_for_user(e)
                        print(f"Error analyzing social media content:\n{error_message}")

if __name__ == "__main__":
    asyncio.run(analyze_social_media())
