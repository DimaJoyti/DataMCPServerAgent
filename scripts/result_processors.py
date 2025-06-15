"""
Utility functions for processing and formatting results from Bright Data MCP tools.
These functions help clean, structure, and enhance the raw data returned by MCP tools.
"""

import re
from typing import Any, Dict, List, Union


def clean_html_content(content: str) -> str:
    """Clean HTML content by removing scripts, styles, and excessive whitespace.
    
    Args:
        content: Raw HTML content
        
    Returns:
        Cleaned content
    """
    # Remove script tags and their contents
    content = re.sub(r'<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>', '', content)

    # Remove style tags and their contents
    content = re.sub(r'<style\b[^<]*(?:(?!<\/style>)<[^<]*)*<\/style>', '', content)

    # Remove HTML comments
    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)

    # Replace multiple newlines with a single newline
    content = re.sub(r'\n\s*\n', '\n\n', content)

    return content.strip()


def extract_main_content(markdown_content: str) -> str:
    """Extract the main content from a markdown document by removing headers, footers, etc.
    
    Args:
        markdown_content: Markdown content to process
        
    Returns:
        Main content section
    """
    lines = markdown_content.split('\n')

    # Skip initial navigation/header (usually first 10-15% of content)
    start_idx = min(int(len(lines) * 0.15), 20)

    # Skip footer (usually last 10% of content)
    end_idx = max(int(len(lines) * 0.9), len(lines) - 15)

    # Extract the main content
    main_content = '\n'.join(lines[start_idx:end_idx])

    return main_content


def extract_tables(markdown_content: str) -> List[str]:
    """Extract markdown tables from content.
    
    Args:
        markdown_content: Markdown content to process
        
    Returns:
        List of extracted tables
    """
    tables = re.findall(r"\|.*\|[\s\S]*?\n\|[-:| ]+\|[\s\S]*?(?=\n\n|\Z)", markdown_content)
    return tables


def extract_links(markdown_content: str) -> List[Dict[str, str]]:
    """Extract links from markdown content.
    
    Args:
        markdown_content: Markdown content to process
        
    Returns:
        List of dictionaries with text and url keys
    """
    links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", markdown_content)
    return [{"text": text, "url": url} for text, url in links]


def extract_headings(markdown_content: str) -> List[Dict[str, Union[str, int]]]:
    """Extract headings from markdown content with their levels.
    
    Args:
        markdown_content: Markdown content to process
        
    Returns:
        List of dictionaries with text and level keys
    """
    headings = re.findall(r"^(#{1,6})\s+(.+)$", markdown_content, re.MULTILINE)
    return [{"level": len(level), "text": text.strip()} for level, text in headings]


def format_search_results(results: Dict[str, Any]) -> str:
    """Format search results into a readable markdown structure.
    
    Args:
        results: Search results from Brave search
        
    Returns:
        Formatted markdown
    """
    if not results or "results" not in results:
        return "No search results found."

    formatted_results = "## Search Results\n\n"

    for i, result in enumerate(results.get("results", []), 1):
        title = result.get("title", "No Title")
        url = result.get("url", "")
        description = result.get("description", "No description available.")

        formatted_results += f"### {i}. {title}\n"
        formatted_results += f"**URL**: {url}\n\n"
        formatted_results += f"{description}\n\n"
        formatted_results += "---\n\n"

    return formatted_results


def format_product_data(product_data: Dict[str, Any]) -> str:
    """Format product data into a readable markdown structure.
    
    Args:
        product_data: Product data from Amazon or other e-commerce site
        
    Returns:
        Formatted markdown
    """
    if not product_data or isinstance(product_data, str):
        return "No product data available or invalid format."

    output = "## Product Information\n\n"

    # Extract key product information
    title = product_data.get("title", "Unknown Product")
    price = product_data.get("price", "Price not available")
    rating = product_data.get("rating", "No rating")
    reviews_count = product_data.get("reviews_count", "No reviews")
    availability = product_data.get("availability", "Unknown availability")

    output += f"### {title}\n\n"
    output += f"**Price**: {price}\n"
    output += f"**Rating**: {rating} ({reviews_count} reviews)\n"
    output += f"**Availability**: {availability}\n\n"

    # Add features if available
    features = product_data.get("features", [])
    if features:
        output += "### Features\n\n"
        for feature in features:
            output += f"- {feature}\n"
        output += "\n"

    # Add description if available
    description = product_data.get("description", "")
    if description:
        output += "### Description\n\n"
        output += f"{description}\n\n"

    return output


def format_product_comparison(products: List[Dict[str, Any]]) -> str:
    """Format multiple products' data into a comparison table.
    
    Args:
        products: List of product data to compare
        
    Returns:
        Formatted product comparison
    """
    if not products:
        return "No products to compare."

    output = "## Product Comparison\n\n"

    # Create comparison table header
    output += "| Product | Price | Rating | Availability |\n"
    output += "|---------|-------|--------|-------------|\n"

    # Add each product to the table
    for product in products:
        if "error" in product:
            output += "| Error retrieving product | - | - | - |\n"
            continue

        title = product.get("title", "Unknown Product")
        price = product.get("price", "N/A")
        rating = product.get("rating", "N/A")
        availability = product.get("availability", "Unknown")

        output += f"| {title} | {price} | {rating} | {availability} |\n"

    output += "\n### Detailed Comparison\n\n"

    # Add detailed comparison for each product
    for i, product in enumerate(products, 1):
        if "error" in product:
            output += f"### Product {i}: Error retrieving data\n\n"
            continue

        title = product.get("title", f"Product {i}")
        output += f"### {title}\n\n"

        # Add features comparison
        features = product.get("features", [])
        if features:
            output += "**Features**:\n"
            for feature in features[:5]:  # Limit to top 5 features
                output += f"- {feature}\n"
            output += "\n"

    return output


def format_social_media_data(data: Dict[str, Any], analysis_type: str = "basic") -> str:
    """Format social media data based on the requested analysis type.
    
    Args:
        data: The social media data to format
        analysis_type: Type of analysis to perform
        
    Returns:
        Formatted social media analysis
    """
    if not data or isinstance(data, str):
        return "No social media data available or invalid format."

    if analysis_type == "basic":
        return format_basic_social_media(data)
    elif analysis_type == "detailed":
        return format_detailed_social_media(data)
    elif analysis_type == "engagement":
        return format_engagement_analysis(data)
    else:
        return format_basic_social_media(data)


def format_basic_social_media(data: Dict[str, Any]) -> str:
    """Format basic social media information.
    
    Args:
        data: Social media data
        
    Returns:
        Basic formatted information
    """
    output = "## Social Media Content\n\n"

    # Handle different data structures based on platform
    if "username" in data:
        output += f"**Username**: {data.get('username', 'Unknown')}\n"
    if "name" in data:
        output += f"**Name**: {data.get('name', 'Unknown')}\n"
    if "followers" in data:
        output += f"**Followers**: {data.get('followers', 'Unknown')}\n"
    if "following" in data:
        output += f"**Following**: {data.get('following', 'Unknown')}\n"
    if "posts_count" in data:
        output += f"**Posts**: {data.get('posts_count', 'Unknown')}\n"
    if "text" in data:
        output += f"\n**Content**: {data.get('text', '')}\n"
    if "caption" in data:
        output += f"\n**Caption**: {data.get('caption', '')}\n"

    # Add engagement metrics if available
    if "likes" in data or "comments" in data or "shares" in data:
        output += "\n**Engagement**:\n"
        if "likes" in data:
            output += f"- Likes: {data.get('likes', 0)}\n"
        if "comments" in data:
            output += f"- Comments: {data.get('comments', 0)}\n"
        if "shares" in data:
            output += f"- Shares: {data.get('shares', 0)}\n"

    return output


def format_detailed_social_media(data: Dict[str, Any]) -> str:
    """Format detailed social media information.
    
    Args:
        data: Social media data
        
    Returns:
        Detailed formatted information
    """
    # Start with basic formatting
    output = format_basic_social_media(data)

    # Add more detailed information
    if "bio" in data:
        output += f"\n### Bio\n{data.get('bio', 'No bio available.')}\n"

    if "website" in data:
        output += f"\n**Website**: {data.get('website', 'None')}\n"

    # Add hashtags if available
    hashtags = []
    if "text" in data:
        hashtags = re.findall(r"#(\w+)", data.get("text", ""))
    elif "caption" in data:
        hashtags = re.findall(r"#(\w+)", data.get("caption", ""))

    if hashtags:
        output += "\n### Hashtags\n"
        for tag in hashtags:
            output += f"- #{tag}\n"

    return output


def format_engagement_analysis(data: Dict[str, Any]) -> str:
    """Format engagement analysis for social media content.
    
    Args:
        data: Social media data
        
    Returns:
        Engagement analysis
    """
    output = "## Social Media Engagement Analysis\n\n"

    # Basic content info
    if "username" in data:
        output += f"**Account**: {data.get('username', 'Unknown')}\n"
    if "text" in data:
        output += f"**Content**: {data.get('text', '')[:100]}...\n\n"
    elif "caption" in data:
        output += f"**Content**: {data.get('caption', '')[:100]}...\n\n"

    # Engagement metrics
    output += "### Engagement Metrics\n\n"

    likes = data.get("likes", 0)
    comments = data.get("comments", 0)
    shares = data.get("shares", 0)

    output += f"- **Likes**: {likes}\n"
    output += f"- **Comments**: {comments}\n"
    output += f"- **Shares**: {shares}\n"

    # Calculate engagement rate if followers are available
    followers = data.get("followers", 0)
    if followers and followers > 0:
        engagement = (likes + comments + shares) / followers * 100
        output += f"\n**Engagement Rate**: {engagement:.2f}%\n"

        # Add engagement assessment
        if engagement > 5:
            output += "\n**Assessment**: High engagement rate (>5%)\n"
        elif engagement > 2:
            output += "\n**Assessment**: Average engagement rate (2-5%)\n"
        else:
            output += "\n**Assessment**: Low engagement rate (<2%)\n"

    return output
