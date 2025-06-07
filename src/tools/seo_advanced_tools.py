"""
Advanced SEO tools for the SEO Agent.

This module provides advanced SEO tools for competitor analysis, rank tracking,
and other advanced SEO tasks.
"""

import os
import json
import re
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup

from langchain.tools import Tool
from src.tools.seo_api_clients import SEMrushClient, MozClient
from src.tools.seo_tools import SEOAnalyzerTool

class CompetitorAnalysisTool:
    """Tool for analyzing competitors for SEO."""

    def __init__(self):
        """Initialize the competitor analysis tool."""
        self.semrush_client = SEMrushClient()
        self.moz_client = MozClient()
        self.seo_analyzer = SEOAnalyzerTool()

    def identify_competitors(self, domain: str, limit: int = 5) -> Dict[str, Any]:
        """
        Identify competitors for a domain based on keyword overlap.

        Args:
            domain: The domain to identify competitors for
            limit: Maximum number of competitors to return

        Returns:
            Dictionary with competitor analysis results
        """
        print(f"Identifying competitors for domain: {domain} (limit: {limit})...")

        try:
            # In a real implementation, this would call the SEMrush API to get competitors
            # For this example, we'll return mock data

            # Mock competitor data
            mock_competitors = [
                {
                    "domain": f"competitor1.com",
                    "overlap_score": 85,
                    "common_keywords": 250,
                    "domain_authority": 75,
                    "estimated_traffic": 45000
                },
                {
                    "domain": f"competitor2.com",
                    "overlap_score": 72,
                    "common_keywords": 180,
                    "domain_authority": 68,
                    "estimated_traffic": 38000
                },
                {
                    "domain": f"competitor3.com",
                    "overlap_score": 65,
                    "common_keywords": 150,
                    "domain_authority": 72,
                    "estimated_traffic": 42000
                },
                {
                    "domain": f"competitor4.com",
                    "overlap_score": 58,
                    "common_keywords": 120,
                    "domain_authority": 65,
                    "estimated_traffic": 35000
                },
                {
                    "domain": f"competitor5.com",
                    "overlap_score": 52,
                    "common_keywords": 100,
                    "domain_authority": 70,
                    "estimated_traffic": 40000
                },
                {
                    "domain": f"competitor6.com",
                    "overlap_score": 45,
                    "common_keywords": 90,
                    "domain_authority": 62,
                    "estimated_traffic": 32000
                },
                {
                    "domain": f"competitor7.com",
                    "overlap_score": 40,
                    "common_keywords": 80,
                    "domain_authority": 58,
                    "estimated_traffic": 28000
                },
                {
                    "domain": f"competitor8.com",
                    "overlap_score": 35,
                    "common_keywords": 70,
                    "domain_authority": 55,
                    "estimated_traffic": 25000
                }
            ]

            # Sort by overlap score and limit results
            sorted_competitors = sorted(mock_competitors, key=lambda x: x["overlap_score"], reverse=True)[:limit]

            # Prepare result
            result = {
                "domain": domain,
                "competitors": sorted_competitors,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            return result

        except Exception as e:
            return {
                "error": str(e),
                "domain": domain
            }

    def compare_with_competitor(self, domain: str, competitor_domain: str) -> Dict[str, Any]:
        """
        Compare a domain with a specific competitor.

        Args:
            domain: The main domain to analyze
            competitor_domain: The competitor domain to compare with

        Returns:
            Dictionary with comparison results
        """
        print(f"Comparing {domain} with competitor {competitor_domain}...")

        try:
            # Analyze both domains
            main_domain_analysis = self.seo_analyzer.analyze(f"https://{domain}", "detailed")
            competitor_analysis = self.seo_analyzer.analyze(f"https://{competitor_domain}", "detailed")

            # Get keyword data for both domains
            main_keywords = self.semrush_client.keyword_research(domain, "us", 20)
            competitor_keywords = self.semrush_client.keyword_research(competitor_domain, "us", 20)

            # Identify common keywords
            main_keyword_list = [kw["keyword"] for kw in main_keywords.get("keywords", [])]
            competitor_keyword_list = [kw["keyword"] for kw in competitor_keywords.get("keywords", [])]
            common_keywords = list(set(main_keyword_list) & set(competitor_keyword_list))

            # Compare metrics
            comparison = {
                "content_metrics": {
                    "word_count": {
                        "main": main_domain_analysis.get("word_count", 0),
                        "competitor": competitor_analysis.get("word_count", 0),
                        "difference": main_domain_analysis.get("word_count", 0) - competitor_analysis.get("word_count", 0)
                    },
                    "internal_links": {
                        "main": main_domain_analysis.get("internal_links", 0),
                        "competitor": competitor_analysis.get("internal_links", 0),
                        "difference": main_domain_analysis.get("internal_links", 0) - competitor_analysis.get("internal_links", 0)
                    },
                    "external_links": {
                        "main": main_domain_analysis.get("external_links", 0),
                        "competitor": competitor_analysis.get("external_links", 0),
                        "difference": main_domain_analysis.get("external_links", 0) - competitor_analysis.get("external_links", 0)
                    },
                    "image_count": {
                        "main": main_domain_analysis.get("image_count", 0),
                        "competitor": competitor_analysis.get("image_count", 0),
                        "difference": main_domain_analysis.get("image_count", 0) - competitor_analysis.get("image_count", 0)
                    }
                },
                "seo_metrics": {
                    "seo_score": {
                        "main": main_domain_analysis.get("seo_score", 0),
                        "competitor": competitor_analysis.get("seo_score", 0),
                        "difference": main_domain_analysis.get("seo_score", 0) - competitor_analysis.get("seo_score", 0)
                    },
                    "title_length": {
                        "main": main_domain_analysis.get("title_length", 0),
                        "competitor": competitor_analysis.get("title_length", 0),
                        "difference": main_domain_analysis.get("title_length", 0) - competitor_analysis.get("title_length", 0)
                    },
                    "meta_description_length": {
                        "main": main_domain_analysis.get("meta_description_length", 0),
                        "competitor": competitor_analysis.get("meta_description_length", 0),
                        "difference": main_domain_analysis.get("meta_description_length", 0) - competitor_analysis.get("meta_description_length", 0)
                    }
                },
                "keyword_metrics": {
                    "total_keywords": {
                        "main": len(main_keywords.get("keywords", [])),
                        "competitor": len(competitor_keywords.get("keywords", [])),
                        "difference": len(main_keywords.get("keywords", [])) - len(competitor_keywords.get("keywords", []))
                    },
                    "common_keywords": len(common_keywords),
                    "common_keyword_list": common_keywords[:10]  # Limit to 10 common keywords
                }
            }

            # Generate recommendations based on comparison
            recommendations = []

            # Content recommendations
            if comparison["content_metrics"]["word_count"]["difference"] < 0:
                recommendations.append(f"Increase content length to match or exceed competitor ({abs(comparison['content_metrics']['word_count']['difference'])} words difference)")

            if comparison["content_metrics"]["internal_links"]["difference"] < 0:
                recommendations.append(f"Add more internal links to improve site structure ({abs(comparison['content_metrics']['internal_links']['difference'])} links difference)")

            # SEO recommendations
            if comparison["seo_metrics"]["seo_score"]["difference"] < 0:
                recommendations.append(f"Improve overall SEO score to match or exceed competitor ({abs(comparison['seo_metrics']['seo_score']['difference'])} points difference)")

            # Keyword recommendations
            if comparison["keyword_metrics"]["total_keywords"]["difference"] < 0:
                recommendations.append(f"Target more keywords to expand your keyword portfolio ({abs(comparison['keyword_metrics']['total_keywords']['difference'])} keywords difference)")

            # Prepare result
            result = {
                "main_domain": domain,
                "competitor_domain": competitor_domain,
                "comparison": comparison,
                "recommendations": recommendations,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            return result

        except Exception as e:
            return {
                "error": str(e),
                "main_domain": domain,
                "competitor_domain": competitor_domain
            }

    def run(self, domain: str, competitor_domain: str = None, limit: int = 5) -> str:
        """
        Run the competitor analysis tool and return formatted results.

        Args:
            domain: The domain to analyze
            competitor_domain: Optional specific competitor to compare with
            limit: Maximum number of competitors to identify if no specific competitor is provided

        Returns:
            Formatted string with analysis results
        """
        if competitor_domain:
            # Compare with specific competitor
            result = self.compare_with_competitor(domain, competitor_domain)

            if "error" in result:
                return f"Error comparing {domain} with {competitor_domain}: {result['error']}"

            # Format the comparison results
            output = f"# Competitor Comparison: {result['main_domain']} vs {result['competitor_domain']}\n\n"

            output += "## Content Metrics\n"
            output += "| Metric | Your Site | Competitor | Difference |\n"
            output += "|--------|-----------|------------|------------|\n"

            for metric, values in result["comparison"]["content_metrics"].items():
                diff = values["difference"]
                diff_str = f"+{diff}" if diff > 0 else str(diff)
                output += f"| {metric.replace('_', ' ').title()} | {values['main']} | {values['competitor']} | {diff_str} |\n"

            output += "\n## SEO Metrics\n"
            output += "| Metric | Your Site | Competitor | Difference |\n"
            output += "|--------|-----------|------------|------------|\n"

            for metric, values in result["comparison"]["seo_metrics"].items():
                diff = values["difference"]
                diff_str = f"+{diff}" if diff > 0 else str(diff)
                output += f"| {metric.replace('_', ' ').title()} | {values['main']} | {values['competitor']} | {diff_str} |\n"

            output += "\n## Keyword Metrics\n"
            output += "| Metric | Your Site | Competitor | Difference |\n"
            output += "|--------|-----------|------------|------------|\n"

            total_kw = result["comparison"]["keyword_metrics"]["total_keywords"]
            diff = total_kw["difference"]
            diff_str = f"+{diff}" if diff > 0 else str(diff)
            output += f"| Total Keywords | {total_kw['main']} | {total_kw['competitor']} | {diff_str} |\n"
            output += f"| Common Keywords | {result['comparison']['keyword_metrics']['common_keywords']} | - | - |\n"

            output += "\n### Common Keywords\n"
            for keyword in result["comparison"]["keyword_metrics"]["common_keyword_list"]:
                output += f"- {keyword}\n"

            output += "\n## Recommendations\n"
            for recommendation in result["recommendations"]:
                output += f"- {recommendation}\n"

        else:
            # Identify competitors
            result = self.identify_competitors(domain, limit)

            if "error" in result:
                return f"Error identifying competitors for {domain}: {result['error']}"

            # Format the competitor identification results
            output = f"# Competitor Analysis for {result['domain']}\n\n"

            output += "## Top Competitors\n"
            output += "| Competitor | Overlap Score | Common Keywords | Domain Authority | Est. Traffic |\n"
            output += "|------------|---------------|-----------------|------------------|-------------|\n"

            for competitor in result["competitors"]:
                output += f"| {competitor['domain']} | {competitor['overlap_score']}% | {competitor['common_keywords']} | {competitor['domain_authority']} | {competitor['estimated_traffic']:,} |\n"

            output += f"\n*Analysis conducted on {result['timestamp']}*\n\n"

            output += "## Next Steps\n"
            output += "- Run a detailed comparison with a specific competitor using the 'competitor_domain' parameter\n"
            output += "- Analyze the content strategy of your top competitors\n"
            output += "- Identify keyword gaps between your site and competitors\n"
            output += "- Evaluate backlink profiles of competitors for link building opportunities\n"

        return output

class RankTrackingTool:
    """Tool for tracking keyword rankings over time."""

    def __init__(self):
        """Initialize the rank tracking tool."""
        self.semrush_client = SEMrushClient()
        self.rankings_db = {}  # In a real implementation, this would be a database

    def track_rankings(self, domain: str, keywords: List[str] = None, limit: int = 10) -> Dict[str, Any]:
        """
        Track keyword rankings for a domain.

        Args:
            domain: The domain to track rankings for
            keywords: Optional list of specific keywords to track
            limit: Maximum number of keywords to track if no specific keywords are provided

        Returns:
            Dictionary with ranking results
        """
        print(f"Tracking rankings for domain: {domain}...")

        try:
            # If no specific keywords are provided, get top keywords for the domain
            if not keywords:
                keyword_data = self.semrush_client.keyword_research(domain, "us", limit)
                keywords = [kw["keyword"] for kw in keyword_data.get("keywords", [])]

            # In a real implementation, this would call a rank tracking API
            # For this example, we'll return mock data

            # Generate mock ranking data
            current_date = datetime.now().strftime("%Y-%m-%d")
            previous_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

            rankings = []
            for keyword in keywords:
                # Generate a stable but random-looking ranking based on the keyword
                keyword_hash = sum(ord(c) for c in keyword) % 100
                current_rank = max(1, min(100, keyword_hash))
                previous_rank = max(1, min(100, current_rank + (hash(keyword) % 20 - 10)))

                rankings.append({
                    "keyword": keyword,
                    "current_rank": current_rank,
                    "previous_rank": previous_rank,
                    "change": previous_rank - current_rank,
                    "search_volume": 500 + (keyword_hash * 100),
                    "url": f"https://{domain}/{keyword.replace(' ', '-').lower()}"
                })

            # Sort by current rank
            sorted_rankings = sorted(rankings, key=lambda x: x["current_rank"])

            # Store in the "database" for historical tracking
            if domain not in self.rankings_db:
                self.rankings_db[domain] = {}

            self.rankings_db[domain][current_date] = sorted_rankings

            # Prepare result
            result = {
                "domain": domain,
                "date": current_date,
                "previous_date": previous_date,
                "rankings": sorted_rankings,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            return result

        except Exception as e:
            return {
                "error": str(e),
                "domain": domain
            }

    def run(self, domain: str, keywords: str = None, limit: int = 10) -> str:
        """
        Run the rank tracking tool and return formatted results.

        Args:
            domain: The domain to track rankings for
            keywords: Optional comma-separated list of specific keywords to track
            limit: Maximum number of keywords to track if no specific keywords are provided

        Returns:
            Formatted string with ranking results
        """
        # Parse keywords if provided
        keyword_list = None
        if keywords:
            keyword_list = [k.strip() for k in keywords.split(',')]

        result = self.track_rankings(domain, keyword_list, limit)

        if "error" in result:
            return f"Error tracking rankings for {domain}: {result['error']}"

        # Format the results as a readable string
        output = f"# Keyword Rankings for {result['domain']}\n\n"

        output += f"## Rankings as of {result['date']} (compared to {result['previous_date']})\n"
        output += "| Keyword | Current Rank | Previous Rank | Change | Search Volume | URL |\n"
        output += "|---------|--------------|---------------|--------|---------------|-----|\n"

        for ranking in result["rankings"]:
            change = ranking["change"]
            change_str = f"↑ +{change}" if change > 0 else f"↓ {change}" if change < 0 else "→ 0"
            output += f"| {ranking['keyword']} | {ranking['current_rank']} | {ranking['previous_rank']} | {change_str} | {ranking['search_volume']:,} | {ranking['url']} |\n"

        output += f"\n*Rankings tracked on {result['timestamp']}*\n\n"

        output += "## Summary\n"
        improved = sum(1 for r in result["rankings"] if r["change"] > 0)
        declined = sum(1 for r in result["rankings"] if r["change"] < 0)
        unchanged = sum(1 for r in result["rankings"] if r["change"] == 0)

        output += f"- Improved Rankings: {improved}\n"
        output += f"- Declined Rankings: {declined}\n"
        output += f"- Unchanged Rankings: {unchanged}\n"

        output += "\n## Recommendations\n"
        output += "- Focus on improving content for keywords with declining rankings\n"
        output += "- Analyze top-ranking pages for keywords with good rankings to identify success factors\n"
        output += "- Consider creating new content for high-volume keywords not currently in the top 10\n"

        return output

# Create tool instances
competitor_analysis = CompetitorAnalysisTool()
rank_tracking = RankTrackingTool()

# Create LangChain tools
competitor_analysis_tool = Tool(
    name="competitor_analysis",
    func=competitor_analysis.run,
    description="Analyze competitors for SEO. Identifies top competitors and compares your site with them.",
)

rank_tracking_tool = Tool(
    name="rank_tracking",
    func=rank_tracking.run,
    description="Track keyword rankings over time. Monitors position changes and provides recommendations.",
)
