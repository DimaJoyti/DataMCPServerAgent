"""
Example script demonstrating how to use the enhanced Bright Data MCP tools
for product comparison across e-commerce sites.
"""

import asyncio
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bright_data_tools import BrightDataToolkit
from dotenv import load_dotenv
from error_handlers import format_error_for_user, with_retry
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from result_processors import format_product_comparison

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

# Example product URLs to compare
PRODUCT_URLS = [
    "https://www.amazon.com/Apple-iPhone-13-128GB-Blue/dp/B09G9F5J7F",
    "https://www.amazon.com/Apple-iPhone-14-128GB-Blue/dp/B0BN93JNXT",
]

async def compare_products():
    """Compare products using the enhanced Bright Data MCP tools."""
    print("Starting product comparison example...")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Create the Bright Data toolkit
            bright_data_toolkit = BrightDataToolkit(session)

            # Get the custom tools
            custom_tools = await bright_data_toolkit.create_custom_tools()

            # Find the product comparison tool
            product_comparison_tool = None
            for tool in custom_tools:
                if tool.name == "product_comparison":
                    product_comparison_tool = tool
                    break

            if not product_comparison_tool:
                print("Product comparison tool not found. Falling back to standard tools...")

                # Load standard MCP tools
                standard_tools = await load_mcp_tools(session)

                # Find the Amazon product tool
                amazon_product_tool = None
                for tool in standard_tools:
                    if tool.name == "web_data_amazon_product_Bright_Data":
                        amazon_product_tool = tool
                        break

                if not amazon_product_tool:
                    print("Amazon product tool not found. Exiting...")
                    return

                # Manually compare products
                products = []
                for url in PRODUCT_URLS:
                    try:
                        print(f"Fetching product data for {url}...")
                        product_data = await with_retry(amazon_product_tool.invoke, {"url": url})
                        products.append(product_data)
                    except Exception as e:
                        print(f"Error fetching product data: {e}")
                        products.append({"error": str(e), "url": url})

                # Format the comparison
                comparison = format_product_comparison(products)
                print("\nProduct Comparison Results:\n")
                print(comparison)
            else:
                # Use the enhanced product comparison tool
                try:
                    print(f"Comparing products: {', '.join(PRODUCT_URLS)}")
                    comparison = await with_retry(product_comparison_tool.invoke, {"urls": PRODUCT_URLS})
                    print("\nProduct Comparison Results:\n")
                    print(comparison)
                except Exception as e:
                    error_message = format_error_for_user(e)
                    print(f"Error comparing products:\n{error_message}")

if __name__ == "__main__":
    asyncio.run(compare_products())
