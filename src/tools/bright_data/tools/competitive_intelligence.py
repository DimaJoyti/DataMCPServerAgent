"""
Competitive Intelligence Tools for Bright Data MCP Integration

This module provides advanced competitive intelligence capabilities including:
- Competitor price monitoring
- Product availability tracking
- Feature comparison analysis
- Market positioning insights
- Brand mention monitoring
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.tools import BaseTool

from ..core.enhanced_client import EnhancedBrightDataClient


@dataclass
class CompetitorProduct:
    """Competitor product information"""

    name: str
    price: float
    availability: str
    features: List[str]
    rating: Optional[float]
    reviews_count: Optional[int]
    url: str
    last_updated: datetime


@dataclass
class PriceHistory:
    """Price history tracking"""

    product_url: str
    prices: List[Dict[str, Any]]  # [{"price": float, "timestamp": datetime}]

    def add_price(self, price: float, timestamp: Optional[datetime] = None) -> None:
        """Add price point to history"""
        if timestamp is None:
            timestamp = datetime.now()

        self.prices.append({"price": price, "timestamp": timestamp})

        # Keep only last 100 price points
        if len(self.prices) > 100:
            self.prices = self.prices[-100:]

    def get_price_trend(self) -> str:
        """Get price trend analysis"""
        if len(self.prices) < 2:
            return "insufficient_data"

        recent_prices = self.prices[-10:]  # Last 10 price points
        if len(recent_prices) < 2:
            return "insufficient_data"

        first_price = recent_prices[0]["price"]
        last_price = recent_prices[-1]["price"]

        change_percent = ((last_price - first_price) / first_price) * 100

        if change_percent > 5:
            return "increasing"
        elif change_percent < -5:
            return "decreasing"
        else:
            return "stable"


class CompetitiveIntelligenceTools:
    """Competitive intelligence tools using Bright Data"""

    def __init__(self, client: EnhancedBrightDataClient):
        self.client = client
        self.price_histories: Dict[str, PriceHistory] = {}
        self.competitor_products: Dict[str, List[CompetitorProduct]] = {}

    def create_tools(self) -> List[BaseTool]:
        """Create competitive intelligence tools"""
        return [
            self._create_price_monitoring_tool(),
            self._create_competitor_analysis_tool(),
            self._create_feature_comparison_tool(),
            self._create_market_positioning_tool(),
            self._create_availability_tracker_tool(),
        ]

    def _create_price_monitoring_tool(self) -> BaseTool:
        """Create price monitoring tool"""

        async def _run(product_urls: List[str], competitor_name: str = "unknown") -> str:
            """Monitor competitor prices across multiple products"""
            try:
                results = await self._monitor_competitor_prices(product_urls, competitor_name)
                return self._format_price_monitoring_results(results)
            except Exception as e:
                return f"Price monitoring failed: {str(e)}"

        return BaseTool(
            name="competitive_price_monitoring",
            description="Monitor competitor prices across multiple products and track changes",
            func=_run,
            args_schema={
                "type": "object",
                "properties": {
                    "product_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of competitor product URLs to monitor",
                    },
                    "competitor_name": {
                        "type": "string",
                        "description": "Name of the competitor",
                        "default": "unknown",
                    },
                },
                "required": ["product_urls"],
            },
        )

    def _create_competitor_analysis_tool(self) -> BaseTool:
        """Create comprehensive competitor analysis tool"""

        async def _run(competitor_domain: str, analysis_type: str = "comprehensive") -> str:
            """Analyze competitor's overall strategy and positioning"""
            try:
                results = await self._analyze_competitor(competitor_domain, analysis_type)
                return self._format_competitor_analysis_results(results)
            except Exception as e:
                return f"Competitor analysis failed: {str(e)}"

        return BaseTool(
            name="competitor_analysis",
            description="Comprehensive analysis of competitor strategy and market positioning",
            func=_run,
            args_schema={
                "type": "object",
                "properties": {
                    "competitor_domain": {
                        "type": "string",
                        "description": "Competitor's domain or website URL",
                    },
                    "analysis_type": {
                        "type": "string",
                        "description": "Type of analysis to perform",
                        "enum": ["comprehensive", "pricing", "products", "marketing"],
                        "default": "comprehensive",
                    },
                },
                "required": ["competitor_domain"],
            },
        )

    def _create_feature_comparison_tool(self) -> BaseTool:
        """Create feature comparison tool"""

        async def _run(product_urls: List[str], comparison_criteria: List[str] = None) -> str:
            """Compare features across competitor products"""
            try:
                results = await self._compare_product_features(product_urls, comparison_criteria)
                return self._format_feature_comparison_results(results)
            except Exception as e:
                return f"Feature comparison failed: {str(e)}"

        return BaseTool(
            name="feature_comparison",
            description="Compare features and specifications across competitor products",
            func=_run,
            args_schema={
                "type": "object",
                "properties": {
                    "product_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of product URLs to compare",
                    },
                    "comparison_criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Specific features or criteria to compare",
                        "default": None,
                    },
                },
                "required": ["product_urls"],
            },
        )

    def _create_market_positioning_tool(self) -> BaseTool:
        """Create market positioning analysis tool"""

        async def _run(
            industry: str, competitors: List[str], analysis_depth: str = "standard"
        ) -> str:
            """Analyze market positioning of competitors in an industry"""
            try:
                results = await self._analyze_market_positioning(
                    industry, competitors, analysis_depth
                )
                return self._format_market_positioning_results(results)
            except Exception as e:
                return f"Market positioning analysis failed: {str(e)}"

        return BaseTool(
            name="market_positioning_analysis",
            description="Analyze competitive positioning and market dynamics",
            func=_run,
            args_schema={
                "type": "object",
                "properties": {
                    "industry": {
                        "type": "string",
                        "description": "Industry or market segment to analyze",
                    },
                    "competitors": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of competitor names or domains",
                    },
                    "analysis_depth": {
                        "type": "string",
                        "description": "Depth of analysis",
                        "enum": ["standard", "detailed", "comprehensive"],
                        "default": "standard",
                    },
                },
                "required": ["industry", "competitors"],
            },
        )

    def _create_availability_tracker_tool(self) -> BaseTool:
        """Create product availability tracking tool"""

        async def _run(product_urls: List[str], check_frequency: str = "daily") -> str:
            """Track product availability across competitor sites"""
            try:
                results = await self._track_product_availability(product_urls, check_frequency)
                return self._format_availability_results(results)
            except Exception as e:
                return f"Availability tracking failed: {str(e)}"

        return BaseTool(
            name="product_availability_tracker",
            description="Track product availability and stock status across competitor sites",
            func=_run,
            args_schema={
                "type": "object",
                "properties": {
                    "product_urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of product URLs to track",
                    },
                    "check_frequency": {
                        "type": "string",
                        "description": "How often to check availability",
                        "enum": ["hourly", "daily", "weekly"],
                        "default": "daily",
                    },
                },
                "required": ["product_urls"],
            },
        )

    # Implementation methods

    async def _monitor_competitor_prices(
        self, product_urls: List[str], competitor_name: str
    ) -> Dict[str, Any]:
        """Monitor competitor prices"""
        results = {
            "competitor": competitor_name,
            "products": [],
            "price_changes": [],
            "summary": {},
            "timestamp": datetime.now().isoformat(),
        }

        for url in product_urls:
            try:
                # Get current product data
                product_data = await self.client.get_product_data(url)

                if product_data and "price" in product_data:
                    current_price = float(product_data["price"].replace("$", "").replace(",", ""))

                    # Update price history
                    if url not in self.price_histories:
                        self.price_histories[url] = PriceHistory(url, [])

                    self.price_histories[url].add_price(current_price)

                    # Analyze price trend
                    trend = self.price_histories[url].get_price_trend()

                    product_info = {
                        "url": url,
                        "name": product_data.get("title", "Unknown Product"),
                        "current_price": current_price,
                        "price_trend": trend,
                        "availability": product_data.get("availability", "Unknown"),
                        "rating": product_data.get("rating"),
                        "reviews_count": product_data.get("reviews_count"),
                    }

                    results["products"].append(product_info)

                    # Check for significant price changes
                    if len(self.price_histories[url].prices) > 1:
                        previous_price = self.price_histories[url].prices[-2]["price"]
                        change_percent = ((current_price - previous_price) / previous_price) * 100

                        if abs(change_percent) > 5:  # 5% change threshold
                            results["price_changes"].append(
                                {
                                    "product": product_data.get("title", "Unknown Product"),
                                    "url": url,
                                    "previous_price": previous_price,
                                    "current_price": current_price,
                                    "change_percent": change_percent,
                                    "change_type": "increase" if change_percent > 0 else "decrease",
                                }
                            )

            except Exception as e:
                results["products"].append({"url": url, "error": str(e), "status": "failed"})

        # Generate summary
        successful_products = [p for p in results["products"] if "error" not in p]
        if successful_products:
            prices = [p["current_price"] for p in successful_products]
            results["summary"] = {
                "total_products": len(product_urls),
                "successful_checks": len(successful_products),
                "failed_checks": len(product_urls) - len(successful_products),
                "average_price": sum(prices) / len(prices),
                "min_price": min(prices),
                "max_price": max(prices),
                "significant_changes": len(results["price_changes"]),
            }

        return results

    def _format_price_monitoring_results(self, results: Dict[str, Any]) -> str:
        """Format price monitoring results"""
        output = f"## Price Monitoring Report - {results['competitor']}\n\n"

        if "summary" in results:
            summary = results["summary"]
            output += "### Summary\n"
            output += f"- **Total Products**: {summary['total_products']}\n"
            output += f"- **Successful Checks**: {summary['successful_checks']}\n"
            output += f"- **Average Price**: ${summary['average_price']:.2f}\n"
            output += (
                f"- **Price Range**: ${summary['min_price']:.2f} - ${summary['max_price']:.2f}\n"
            )
            output += f"- **Significant Changes**: {summary['significant_changes']}\n\n"

        # Price changes
        if results["price_changes"]:
            output += "### Significant Price Changes\n\n"
            for change in results["price_changes"]:
                direction = "📈" if change["change_type"] == "increase" else "📉"
                output += f"{direction} **{change['product']}**\n"
                output += f"- Previous: ${change['previous_price']:.2f}\n"
                output += f"- Current: ${change['current_price']:.2f}\n"
                output += f"- Change: {change['change_percent']:.1f}%\n\n"

        # Product details
        output += "### Product Details\n\n"
        for product in results["products"]:
            if "error" not in product:
                trend_emoji = {"increasing": "📈", "decreasing": "📉", "stable": "➡️"}.get(
                    product["price_trend"], "❓"
                )
                output += f"**{product['name']}**\n"
                output += f"- Price: ${product['current_price']:.2f} {trend_emoji}\n"
                output += f"- Availability: {product['availability']}\n"
                if product.get("rating"):
                    output += f"- Rating: {product['rating']} ({product.get('reviews_count', 0)} reviews)\n"
                output += f"- URL: {product['url']}\n\n"
            else:
                output += f"**Error**: {product['url']} - {product['error']}\n\n"

        return output
