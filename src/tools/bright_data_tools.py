"""
Custom Bright Data MCP tools and utilities for enhanced web scraping and data collection.
This module extends the basic MCP tools with specialized functions for common scraping tasks.
"""

import json
import re
from typing import Any, Dict, List, Optional, Union

from langchain_core.tools import BaseTool
from mcp import ClientSession


class BrightDataToolkit:
    """A toolkit for specialized Bright Data MCP operations."""

    def __init__(self, session: ClientSession):
        """Initialize the toolkit with an MCP client session.
        
        Args:
            session: An initialized MCP ClientSession
        """
        self.session = session
        
    async def create_custom_tools(self) -> List[BaseTool]:
        """Create and return custom Bright Data MCP tools.
        
        Returns:
            A list of custom BaseTool instances
        """
        # Get all available tools from the session
        available_tools = {}
        for plugin in await self.session.list_plugins():
            for tool in plugin.tools:
                available_tools[tool.name] = tool
        
        # Create custom tools that combine or enhance the base tools
        custom_tools = []
        
        # Add specialized search tool
        if "brave_web_search_Brave" in available_tools:
            custom_tools.append(
                self._create_enhanced_search_tool(available_tools["brave_web_search_Brave"])
            )
        
        # Add specialized scraping tool
        if "scrape_as_markdown_Bright_Data" in available_tools:
            custom_tools.append(
                self._create_enhanced_scraping_tool(
                    available_tools["scrape_as_markdown_Bright_Data"]
                )
            )
            
        # Add specialized product data tool
        if "web_data_amazon_product_Bright_Data" in available_tools:
            custom_tools.append(
                self._create_product_comparison_tool(
                    available_tools["web_data_amazon_product_Bright_Data"]
                )
            )
            
        # Add specialized social media tool
        social_media_tools = [
            "web_data_instagram_profiles_Bright_Data",
            "web_data_facebook_posts_Bright_Data",
            "web_data_x_posts_Bright_Data"
        ]
        
        available_social_tools = [t for t in social_media_tools if t in available_tools]
        if available_social_tools:
            custom_tools.append(
                self._create_social_media_analyzer(
                    {name: available_tools[name] for name in available_social_tools}
                )
            )
            
        return custom_tools
    
    def _create_enhanced_search_tool(self, base_tool: BaseTool) -> BaseTool:
        """Create an enhanced search tool that provides better formatting and filtering.
        
        Args:
            base_tool: The base search tool to enhance
            
        Returns:
            An enhanced search tool
        """
        async def _run(query: str, count: int = 10) -> str:
            """Run the enhanced search with better result formatting."""
            results = await base_tool.invoke({"query": query, "count": count})
            
            # Process and format the results
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
        
        return BaseTool(
            name="enhanced_web_search",
            description="An enhanced web search tool that provides better formatted results from Brave Search",
            func=_run,
            args_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "count": {"type": "integer", "description": "Number of results (1-20)", "default": 10}
                },
                "required": ["query"]
            }
        )
    
    def _create_enhanced_scraping_tool(self, base_tool: BaseTool) -> BaseTool:
        """Create an enhanced scraping tool with better content extraction.
        
        Args:
            base_tool: The base scraping tool to enhance
            
        Returns:
            An enhanced scraping tool
        """
        async def _run(url: str, extract_type: str = "all") -> str:
            """Run the enhanced scraper with content type filtering."""
            result = await base_tool.invoke({"url": url})
            
            if extract_type == "all":
                return result
            
            # Extract specific content based on type
            if extract_type == "main_content":
                # Try to extract the main content by removing headers, footers, sidebars
                content_lines = result.split("\n")
                # Skip initial navigation/header (usually first 10-15% of content)
                start_idx = min(int(len(content_lines) * 0.15), 20)
                # Skip footer (usually last 10% of content)
                end_idx = max(int(len(content_lines) * 0.9), len(content_lines) - 15)
                
                return "\n".join(content_lines[start_idx:end_idx])
            
            if extract_type == "tables":
                # Extract markdown tables
                tables = re.findall(r"\|.*\|[\s\S]*?\n\|[-:| ]+\|[\s\S]*?(?=\n\n|\Z)", result)
                if tables:
                    return "\n\n".join(tables)
                return "No tables found in the content."
                
            if extract_type == "links":
                # Extract all links
                links = re.findall(r"\[([^\]]+)\]\(([^)]+)\)", result)
                if links:
                    return "\n".join([f"- [{text}]({url})" for text, url in links])
                return "No links found in the content."
            
            return result
        
        return BaseTool(
            name="enhanced_web_scraper",
            description="An enhanced web scraper that can extract and filter specific types of content from websites",
            func=_run,
            args_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to scrape"},
                    "extract_type": {
                        "type": "string", 
                        "description": "Type of content to extract: 'all', 'main_content', 'tables', or 'links'",
                        "default": "all",
                        "enum": ["all", "main_content", "tables", "links"]
                    }
                },
                "required": ["url"]
            }
        )
    
    def _create_product_comparison_tool(self, base_tool: BaseTool) -> BaseTool:
        """Create a product comparison tool that can analyze multiple product pages.
        
        Args:
            base_tool: The base product data tool to enhance
            
        Returns:
            A product comparison tool
        """
        async def _run(urls: List[str]) -> str:
            """Run the product comparison on multiple URLs."""
            if not urls:
                return "No URLs provided for comparison."
                
            if len(urls) == 1:
                # Single product analysis
                result = await base_tool.invoke({"url": urls[0]})
                return self._format_product_data(result)
            
            # Multiple product comparison
            products = []
            for url in urls:
                try:
                    product_data = await base_tool.invoke({"url": url})
                    products.append(product_data)
                except Exception as e:
                    products.append({"error": str(e), "url": url})
            
            return self._format_product_comparison(products)
        
        return BaseTool(
            name="product_comparison",
            description="Compare product details across multiple product pages",
            func=_run,
            args_schema={
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of product URLs to compare"
                    }
                },
                "required": ["urls"]
            }
        )
    
    def _format_product_data(self, product_data: Dict[str, Any]) -> str:
        """Format a single product's data into a readable markdown format.
        
        Args:
            product_data: The product data to format
            
        Returns:
            Formatted product information
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
    
    def _format_product_comparison(self, products: List[Dict[str, Any]]) -> str:
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
                output += f"| Error retrieving product | - | - | - |\n"
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
    
    def _create_social_media_analyzer(self, tools: Dict[str, BaseTool]) -> BaseTool:
        """Create a social media analysis tool that works across platforms.
        
        Args:
            tools: Dictionary of social media tools by name
            
        Returns:
            A social media analyzer tool
        """
        async def _run(url: str, analysis_type: str = "basic") -> str:
            """Run social media analysis on the provided URL."""
            # Determine which platform tool to use based on the URL
            if "instagram.com" in url:
                if "/p/" in url and "web_data_instagram_posts_Bright_Data" in tools:
                    tool = tools["web_data_instagram_posts_Bright_Data"]
                elif "web_data_instagram_profiles_Bright_Data" in tools:
                    tool = tools["web_data_instagram_profiles_Bright_Data"]
                else:
                    return "No appropriate Instagram tool available."
            elif "facebook.com" in url and "web_data_facebook_posts_Bright_Data" in tools:
                tool = tools["web_data_facebook_posts_Bright_Data"]
            elif ("twitter.com" in url or "x.com" in url) and "web_data_x_posts_Bright_Data" in tools:
                tool = tools["web_data_x_posts_Bright_Data"]
            else:
                return "Unsupported social media platform or URL format."
                
            # Get the data from the appropriate tool
            try:
                result = await tool.invoke({"url": url})
                return self._format_social_media_data(result, analysis_type)
            except Exception as e:
                return f"Error analyzing social media content: {str(e)}"
        
        return BaseTool(
            name="social_media_analyzer",
            description="Analyze social media content across platforms (Instagram, Facebook, Twitter/X)",
            func=_run,
            args_schema={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL of the social media post or profile"},
                    "analysis_type": {
                        "type": "string",
                        "description": "Type of analysis to perform",
                        "enum": ["basic", "detailed", "engagement"],
                        "default": "basic"
                    }
                },
                "required": ["url"]
            }
        )
    
    def _format_social_media_data(self, data: Dict[str, Any], analysis_type: str) -> str:
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
            return self._format_basic_social_media(data)
        elif analysis_type == "detailed":
            return self._format_detailed_social_media(data)
        elif analysis_type == "engagement":
            return self._format_engagement_analysis(data)
        else:
            return self._format_basic_social_media(data)
    
    def _format_basic_social_media(self, data: Dict[str, Any]) -> str:
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
    
    def _format_detailed_social_media(self, data: Dict[str, Any]) -> str:
        """Format detailed social media information.
        
        Args:
            data: Social media data
            
        Returns:
            Detailed formatted information
        """
        # Start with basic formatting
        output = self._format_basic_social_media(data)
        
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
    
    def _format_engagement_analysis(self, data: Dict[str, Any]) -> str:
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

    async def create_osint_tools(self) -> List[BaseTool]:
        """Create OSINT-specific tools using Bright Data capabilities"""
        osint_tools = []

        # Social media intelligence tool
        osint_tools.append(self._create_social_media_intel_tool())

        # Domain intelligence tool
        osint_tools.append(self._create_domain_intel_tool())

        # Dark web monitoring tool
        osint_tools.append(self._create_dark_web_monitor_tool())

        # Threat intelligence tool
        osint_tools.append(self._create_threat_intel_tool())

        # Company intelligence tool
        osint_tools.append(self._create_company_intel_tool())

        # Email intelligence tool
        osint_tools.append(self._create_email_intel_tool())

        return osint_tools

    def _create_social_media_intel_tool(self) -> BaseTool:
        """Create social media intelligence gathering tool"""
        async def _run(target_name: str, platforms: str = "all") -> str:
            """Gather social media intelligence about a target"""
            try:
                results = await self._gather_social_media_intelligence(target_name, platforms)
                return self._format_social_media_intel_results(results)
            except Exception as e:
                return f"Social media intelligence gathering failed: {str(e)}"

        return BaseTool(
            name="social_media_intelligence",
            description="Gather intelligence from social media platforms about a target",
            func=_run,
            args_schema={
                "type": "object",
                "properties": {
                    "target_name": {"type": "string", "description": "Target name or company to investigate"},
                    "platforms": {
                        "type": "string",
                        "description": "Social media platforms to search",
                        "enum": ["all", "linkedin", "twitter", "facebook", "instagram"],
                        "default": "all"
                    }
                },
                "required": ["target_name"]
            }
        )

    def _create_domain_intel_tool(self) -> BaseTool:
        """Create domain intelligence tool"""
        async def _run(domain: str, intel_type: str = "comprehensive") -> str:
            """Gather comprehensive domain intelligence"""
            try:
                results = await self._gather_domain_intelligence(domain, intel_type)
                return self._format_domain_intel_results(results)
            except Exception as e:
                return f"Domain intelligence gathering failed: {str(e)}"

        return BaseTool(
            name="domain_intelligence",
            description="Gather comprehensive intelligence about a domain",
            func=_run,
            args_schema={
                "type": "object",
                "properties": {
                    "domain": {"type": "string", "description": "Domain to investigate"},
                    "intel_type": {
                        "type": "string",
                        "description": "Type of intelligence to gather",
                        "enum": ["comprehensive", "subdomains", "certificates", "history"],
                        "default": "comprehensive"
                    }
                },
                "required": ["domain"]
            }
        )

    def _create_dark_web_monitor_tool(self) -> BaseTool:
        """Create dark web monitoring tool"""
        async def _run(keywords: str, search_type: str = "mentions") -> str:
            """Monitor dark web for specific keywords or indicators"""
            try:
                results = await self._monitor_dark_web(keywords, search_type)
                return self._format_dark_web_results(results)
            except Exception as e:
                return f"Dark web monitoring failed: {str(e)}"

        return BaseTool(
            name="dark_web_monitor",
            description="Monitor dark web for mentions of keywords, domains, or indicators",
            func=_run,
            args_schema={
                "type": "object",
                "properties": {
                    "keywords": {"type": "string", "description": "Keywords, domains, or indicators to monitor"},
                    "search_type": {
                        "type": "string",
                        "description": "Type of search to perform",
                        "enum": ["mentions", "credentials", "data_breaches", "threats"],
                        "default": "mentions"
                    }
                },
                "required": ["keywords"]
            }
        )

    def _create_threat_intel_tool(self) -> BaseTool:
        """Create threat intelligence tool"""
        async def _run(indicator: str, indicator_type: str = "auto") -> str:
            """Gather threat intelligence about an indicator"""
            try:
                results = await self._gather_threat_intelligence(indicator, indicator_type)
                return self._format_threat_intel_results(results)
            except Exception as e:
                return f"Threat intelligence gathering failed: {str(e)}"

        return BaseTool(
            name="threat_intelligence",
            description="Gather threat intelligence about IPs, domains, or hashes",
            func=_run,
            args_schema={
                "type": "object",
                "properties": {
                    "indicator": {"type": "string", "description": "Indicator to investigate (IP, domain, hash)"},
                    "indicator_type": {
                        "type": "string",
                        "description": "Type of indicator",
                        "enum": ["auto", "ip", "domain", "hash", "url"],
                        "default": "auto"
                    }
                },
                "required": ["indicator"]
            }
        )

    def _create_company_intel_tool(self) -> BaseTool:
        """Create company intelligence tool"""
        async def _run(company_name: str, intel_type: str = "comprehensive") -> str:
            """Gather comprehensive company intelligence"""
            try:
                results = await self._gather_company_intelligence(company_name, intel_type)
                return self._format_company_intel_results(results)
            except Exception as e:
                return f"Company intelligence gathering failed: {str(e)}"

        return BaseTool(
            name="company_intelligence",
            description="Gather comprehensive intelligence about a company",
            func=_run,
            args_schema={
                "type": "object",
                "properties": {
                    "company_name": {"type": "string", "description": "Company name to investigate"},
                    "intel_type": {
                        "type": "string",
                        "description": "Type of intelligence to gather",
                        "enum": ["comprehensive", "employees", "technologies", "news", "financials"],
                        "default": "comprehensive"
                    }
                },
                "required": ["company_name"]
            }
        )

    def _create_email_intel_tool(self) -> BaseTool:
        """Create email intelligence tool"""
        async def _run(email: str, intel_type: str = "comprehensive") -> str:
            """Gather intelligence about an email address"""
            try:
                results = await self._gather_email_intelligence(email, intel_type)
                return self._format_email_intel_results(results)
            except Exception as e:
                return f"Email intelligence gathering failed: {str(e)}"

        return BaseTool(
            name="email_intelligence",
            description="Gather intelligence about an email address",
            func=_run,
            args_schema={
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "Email address to investigate"},
                    "intel_type": {
                        "type": "string",
                        "description": "Type of intelligence to gather",
                        "enum": ["comprehensive", "breaches", "social_media", "professional"],
                        "default": "comprehensive"
                    }
                },
                "required": ["email"]
            }
        )

    # OSINT Intelligence Gathering Methods

    async def _gather_social_media_intelligence(self, target_name: str, platforms: str) -> Dict[str, Any]:
        """Gather social media intelligence using Bright Data"""
        from datetime import datetime

        intelligence = {
            "target": target_name,
            "platforms": {},
            "timestamp": datetime.now().isoformat(),
            "summary": {}
        }

        # Use Bright Data's social media scraping capabilities
        try:
            if platforms == "all" or "linkedin" in platforms:
                # LinkedIn intelligence
                linkedin_data = await self._scrape_linkedin_intelligence(target_name)
                intelligence["platforms"]["linkedin"] = linkedin_data

            if platforms == "all" or "twitter" in platforms:
                # Twitter/X intelligence
                twitter_data = await self._scrape_twitter_intelligence(target_name)
                intelligence["platforms"]["twitter"] = twitter_data

            if platforms == "all" or "facebook" in platforms:
                # Facebook intelligence
                facebook_data = await self._scrape_facebook_intelligence(target_name)
                intelligence["platforms"]["facebook"] = facebook_data

            if platforms == "all" or "instagram" in platforms:
                # Instagram intelligence
                instagram_data = await self._scrape_instagram_intelligence(target_name)
                intelligence["platforms"]["instagram"] = instagram_data

            # Generate summary
            intelligence["summary"] = self._generate_social_media_summary(intelligence["platforms"])

        except Exception as e:
            intelligence["error"] = str(e)

        return intelligence

    async def _gather_domain_intelligence(self, domain: str, intel_type: str) -> Dict[str, Any]:
        """Gather domain intelligence using Bright Data"""
        from datetime import datetime

        intelligence = {
            "domain": domain,
            "intelligence_type": intel_type,
            "data": {},
            "timestamp": datetime.now().isoformat()
        }

        try:
            if intel_type == "comprehensive" or intel_type == "subdomains":
                # Subdomain discovery using various sources
                subdomains = await self._discover_subdomains_via_bright_data(domain)
                intelligence["data"]["subdomains"] = subdomains

            if intel_type == "comprehensive" or intel_type == "certificates":
                # Certificate transparency logs
                certificates = await self._gather_certificate_intelligence(domain)
                intelligence["data"]["certificates"] = certificates

            if intel_type == "comprehensive" or intel_type == "history":
                # Domain history and changes
                history = await self._gather_domain_history(domain)
                intelligence["data"]["history"] = history

            if intel_type == "comprehensive":
                # Additional comprehensive data
                whois_data = await self._gather_whois_intelligence(domain)
                intelligence["data"]["whois"] = whois_data

                dns_data = await self._gather_dns_intelligence(domain)
                intelligence["data"]["dns"] = dns_data

        except Exception as e:
            intelligence["error"] = str(e)

        return intelligence

    async def _monitor_dark_web(self, keywords: str, search_type: str) -> Dict[str, Any]:
        """Monitor dark web using Bright Data's capabilities"""
        from datetime import datetime

        monitoring_result = {
            "keywords": keywords,
            "search_type": search_type,
            "findings": [],
            "timestamp": datetime.now().isoformat(),
            "risk_level": "low"
        }

        try:
            # Use Bright Data to access dark web sources (with proper authorization)
            # This would require special Bright Data configurations for dark web access

            if search_type == "mentions":
                findings = await self._search_dark_web_mentions(keywords)
                monitoring_result["findings"] = findings

            elif search_type == "credentials":
                findings = await self._search_credential_dumps(keywords)
                monitoring_result["findings"] = findings
                monitoring_result["risk_level"] = "high" if findings else "low"

            elif search_type == "data_breaches":
                findings = await self._search_data_breaches(keywords)
                monitoring_result["findings"] = findings
                monitoring_result["risk_level"] = "critical" if findings else "low"

            elif search_type == "threats":
                findings = await self._search_threat_mentions(keywords)
                monitoring_result["findings"] = findings
                monitoring_result["risk_level"] = "high" if findings else "medium"

        except Exception as e:
            monitoring_result["error"] = str(e)

        return monitoring_result

    async def _gather_threat_intelligence(self, indicator: str, indicator_type: str) -> Dict[str, Any]:
        """Gather threat intelligence using Bright Data"""
        from datetime import datetime

        intelligence = {
            "indicator": indicator,
            "indicator_type": indicator_type,
            "threat_data": {},
            "risk_score": 0,
            "timestamp": datetime.now().isoformat()
        }

        try:
            # Auto-detect indicator type if needed
            if indicator_type == "auto":
                indicator_type = self._detect_indicator_type(indicator)
                intelligence["indicator_type"] = indicator_type

            # Gather threat intelligence from various sources
            if indicator_type == "ip":
                threat_data = await self._gather_ip_threat_intelligence(indicator)
            elif indicator_type == "domain":
                threat_data = await self._gather_domain_threat_intelligence(indicator)
            elif indicator_type == "hash":
                threat_data = await self._gather_hash_threat_intelligence(indicator)
            elif indicator_type == "url":
                threat_data = await self._gather_url_threat_intelligence(indicator)
            else:
                threat_data = {"error": f"Unsupported indicator type: {indicator_type}"}

            intelligence["threat_data"] = threat_data
            intelligence["risk_score"] = self._calculate_risk_score(threat_data)

        except Exception as e:
            intelligence["error"] = str(e)

        return intelligence

    async def _gather_company_intelligence(self, company_name: str, intel_type: str) -> Dict[str, Any]:
        """Gather company intelligence using Bright Data"""
        from datetime import datetime

        intelligence = {
            "company_name": company_name,
            "intelligence_type": intel_type,
            "data": {},
            "timestamp": datetime.now().isoformat()
        }

        try:
            if intel_type == "comprehensive" or intel_type == "employees":
                # Employee intelligence from LinkedIn and other sources
                employees = await self._gather_employee_intelligence(company_name)
                intelligence["data"]["employees"] = employees

            if intel_type == "comprehensive" or intel_type == "technologies":
                # Technology stack intelligence
                technologies = await self._gather_technology_intelligence(company_name)
                intelligence["data"]["technologies"] = technologies

            if intel_type == "comprehensive" or intel_type == "news":
                # News and media mentions
                news = await self._gather_news_intelligence(company_name)
                intelligence["data"]["news"] = news

            if intel_type == "comprehensive" or intel_type == "financials":
                # Financial intelligence
                financials = await self._gather_financial_intelligence(company_name)
                intelligence["data"]["financials"] = financials

            if intel_type == "comprehensive":
                # Additional comprehensive data
                domains = await self._discover_company_domains(company_name)
                intelligence["data"]["domains"] = domains

                social_media = await self._gather_company_social_media(company_name)
                intelligence["data"]["social_media"] = social_media

        except Exception as e:
            intelligence["error"] = str(e)

        return intelligence