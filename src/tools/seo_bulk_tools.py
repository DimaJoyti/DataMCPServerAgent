"""
Bulk SEO analysis tools for the SEO Agent.

This module provides tools for analyzing multiple pages or websites at once,
generating comprehensive reports, and performing site-wide analysis.
"""

import concurrent.futures
import json
from datetime import datetime
from typing import Any, Dict, List
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain.tools import Tool

from src.tools.seo_tools import SEOAnalyzerTool


class BulkAnalysisManager:
    """Manager for bulk SEO analysis tasks."""

    def __init__(self):
        """Initialize the bulk analysis manager."""
        self.seo_analyzer = SEOAnalyzerTool()
        self.max_workers = 5  # Maximum number of concurrent workers

    def discover_pages(self, domain: str, max_pages: int = 50) -> List[str]:
        """
        Discover pages on a domain through crawling.

        Args:
            domain: The domain to crawl
            max_pages: Maximum number of pages to discover

        Returns:
            List of discovered URLs
        """
        print(f"Discovering pages on {domain} (max: {max_pages})...")

        try:
            # Start with the homepage
            start_url = f"https://{domain}"
            discovered_urls = set([start_url])
            urls_to_crawl = [start_url]
            crawled_urls = set()

            while urls_to_crawl and len(discovered_urls) < max_pages:
                # Get the next URL to crawl
                current_url = urls_to_crawl.pop(0)

                # Skip if already crawled
                if current_url in crawled_urls:
                    continue

                try:
                    # Fetch the page
                    response = requests.get(
                        current_url,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                        },
                        timeout=10,
                    )
                    response.raise_for_status()

                    # Parse the HTML
                    soup = BeautifulSoup(response.text, "html.parser")

                    # Find all links
                    links = soup.find_all("a", href=True)

                    # Process each link
                    for link in links:
                        href = link["href"]

                        # Skip empty links, anchors, javascript, etc.
                        if not href or href.startswith(("#", "javascript:", "mailto:", "tel:")):
                            continue

                        # Resolve relative URLs
                        if not href.startswith(("http://", "https://")):
                            if href.startswith("/"):
                                # Absolute path
                                parsed_url = urlparse(current_url)
                                href = f"{parsed_url.scheme}://{parsed_url.netloc}{href}"
                            else:
                                # Relative path
                                href = f"{current_url.rstrip('/')}/{href}"

                        # Skip external links
                        parsed_href = urlparse(href)
                        if parsed_href.netloc and parsed_href.netloc != urlparse(start_url).netloc:
                            continue

                        # Add to discovered URLs if not already discovered
                        if href not in discovered_urls and len(discovered_urls) < max_pages:
                            discovered_urls.add(href)
                            urls_to_crawl.append(href)

                    # Mark as crawled
                    crawled_urls.add(current_url)

                except Exception as e:
                    print(f"Error crawling {current_url}: {str(e)}")
                    # Mark as crawled even if there was an error
                    crawled_urls.add(current_url)

            return list(discovered_urls)

        except Exception as e:
            print(f"Error discovering pages: {str(e)}")
            return [f"https://{domain}"]  # Return just the homepage if there's an error

    def analyze_multiple_urls(self, urls: List[str], depth: str = "basic") -> Dict[str, Any]:
        """
        Analyze multiple URLs for SEO factors.

        Args:
            urls: List of URLs to analyze
            depth: Analysis depth ('basic', 'detailed', or 'comprehensive')

        Returns:
            Dictionary with analysis results for all URLs
        """
        print(f"Analyzing {len(urls)} URLs (depth: {depth})...")

        results = []
        errors = []

        # Use ThreadPoolExecutor for parallel processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all analysis tasks
            future_to_url = {
                executor.submit(self.seo_analyzer.analyze, url, depth): url for url in urls
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    result = future.result()
                    if "error" in result:
                        errors.append({"url": url, "error": result["error"]})
                    else:
                        results.append(result)
                except Exception as e:
                    errors.append({"url": url, "error": str(e)})

        # Calculate aggregate statistics
        avg_score = sum(r.get("seo_score", 0) for r in results) / len(results) if results else 0
        avg_word_count = (
            sum(r.get("word_count", 0) for r in results) / len(results) if results else 0
        )

        # Count common issues
        issues = {
            "missing_meta_description": sum(1 for r in results if not r.get("meta_description")),
            "title_too_short": sum(1 for r in results if r.get("title_length", 0) < 30),
            "title_too_long": sum(1 for r in results if r.get("title_length", 0) > 60),
            "missing_h1": sum(1 for r in results if r.get("h1_count", 0) == 0),
            "multiple_h1": sum(1 for r in results if r.get("h1_count", 0) > 1),
            "low_word_count": sum(1 for r in results if r.get("word_count", 0) < 300),
            "missing_alt_text": sum(
                1
                for r in results
                if r.get("image_count", 0) > 0
                and r.get("images_with_alt", 0) < r.get("image_count", 0)
            ),
        }

        # Prepare result
        result = {
            "total_urls": len(urls),
            "successful_analyses": len(results),
            "failed_analyses": len(errors),
            "average_seo_score": round(avg_score, 2),
            "average_word_count": round(avg_word_count, 2),
            "common_issues": issues,
            "page_results": results,
            "errors": errors,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        return result

    def analyze_site(
        self, domain: str, max_pages: int = 50, depth: str = "basic"
    ) -> Dict[str, Any]:
        """
        Analyze an entire website by discovering and analyzing pages.

        Args:
            domain: The domain to analyze
            max_pages: Maximum number of pages to analyze
            depth: Analysis depth ('basic', 'detailed', or 'comprehensive')

        Returns:
            Dictionary with site-wide analysis results
        """
        print(f"Analyzing site {domain} (max pages: {max_pages}, depth: {depth})...")

        # Discover pages
        urls = self.discover_pages(domain, max_pages)

        # Analyze discovered pages
        result = self.analyze_multiple_urls(urls, depth)

        # Add site-specific information
        result["domain"] = domain
        result["analyzed_pages"] = len(urls)

        return result

    def export_results(self, results: Dict[str, Any], format: str = "markdown") -> str:
        """
        Export analysis results in the specified format.

        Args:
            results: Analysis results to export
            format: Export format ('markdown', 'csv', or 'json')

        Returns:
            Exported results as a string
        """
        if format == "json":
            return json.dumps(results, indent=2)

        elif format == "csv":
            # Create CSV content
            csv_content = []

            # Add header row
            header = [
                "URL",
                "SEO Score",
                "Title Length",
                "Meta Description Length",
                "Word Count",
                "H1 Count",
                "H2 Count",
                "Issues",
            ]
            csv_content.append(header)

            # Add data rows
            for page in results.get("page_results", []):
                issues = []
                if not page.get("meta_description"):
                    issues.append("Missing meta description")
                if page.get("title_length", 0) < 30:
                    issues.append("Title too short")
                if page.get("title_length", 0) > 60:
                    issues.append("Title too long")
                if page.get("h1_count", 0) == 0:
                    issues.append("Missing H1")
                if page.get("h1_count", 0) > 1:
                    issues.append("Multiple H1s")
                if page.get("word_count", 0) < 300:
                    issues.append("Low word count")

                row = [
                    page.get("url", ""),
                    page.get("seo_score", 0),
                    page.get("title_length", 0),
                    page.get("meta_description_length", 0),
                    page.get("word_count", 0),
                    page.get("h1_count", 0),
                    page.get("h2_count", 0),
                    "; ".join(issues),
                ]
                csv_content.append(row)

            # Convert to CSV string
            output = []
            for row in csv_content:
                output.append(",".join([str(cell) for cell in row]))

            return "\n".join(output)

        else:  # Default to markdown
            # Create markdown content
            output = "# Bulk SEO Analysis Report\n\n"

            output += "## Overview\n"
            output += f"- Domain: {results.get('domain', 'Multiple URLs')}\n"
            output += f"- Total URLs: {results.get('total_urls', 0)}\n"
            output += f"- Successful Analyses: {results.get('successful_analyses', 0)}\n"
            output += f"- Failed Analyses: {results.get('failed_analyses', 0)}\n"
            output += f"- Average SEO Score: {results.get('average_seo_score', 0)}/100\n"
            output += f"- Average Word Count: {results.get('average_word_count', 0)} words\n\n"

            output += "## Common Issues\n"
            issues = results.get("common_issues", {})
            output += (
                f"- Missing Meta Descriptions: {issues.get('missing_meta_description', 0)} pages\n"
            )
            output += f"- Titles Too Short: {issues.get('title_too_short', 0)} pages\n"
            output += f"- Titles Too Long: {issues.get('title_too_long', 0)} pages\n"
            output += f"- Missing H1 Tags: {issues.get('missing_h1', 0)} pages\n"
            output += f"- Multiple H1 Tags: {issues.get('multiple_h1', 0)} pages\n"
            output += f"- Low Word Count: {issues.get('low_word_count', 0)} pages\n"
            output += f"- Missing Alt Text: {issues.get('missing_alt_text', 0)} pages\n\n"

            output += "## Page Analysis\n"
            output += (
                "| URL | SEO Score | Title Length | Meta Desc Length | Word Count | H1 | H2 |\n"
            )
            output += "|-----|-----------|--------------|------------------|------------|----|----|"

            for page in results.get("page_results", [])[:20]:  # Limit to 20 pages in the table
                output += f"\n| {page.get('url', '')} | {page.get('seo_score', 0)} | {page.get('title_length', 0)} | {page.get('meta_description_length', 0)} | {page.get('word_count', 0)} | {page.get('h1_count', 0)} | {page.get('h2_count', 0)} |"

            if len(results.get("page_results", [])) > 20:
                output += f"\n\n*Table truncated. {len(results.get('page_results', [])) - 20} more pages analyzed but not shown.*\n"

            output += f"\n\n*Analysis conducted on {results.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}*\n\n"

            output += "## Recommendations\n"
            output += "- Fix missing meta descriptions to improve click-through rates\n"
            output += "- Ensure all pages have a single H1 tag\n"
            output += "- Optimize title lengths to be between 30-60 characters\n"
            output += "- Increase content length on pages with low word counts\n"
            output += "- Add alt text to all images for better accessibility and SEO\n"

            return output

    def run(
        self,
        domain: str = None,
        urls: str = None,
        max_pages: int = 50,
        depth: str = "basic",
        format: str = "markdown",
    ) -> str:
        """
        Run the bulk analysis tool and return formatted results.

        Args:
            domain: The domain to analyze (for site-wide analysis)
            urls: Comma-separated list of URLs to analyze (for multi-page analysis)
            max_pages: Maximum number of pages to analyze for site-wide analysis
            depth: Analysis depth ('basic', 'detailed', or 'comprehensive')
            format: Export format ('markdown', 'csv', or 'json')

        Returns:
            Formatted string with analysis results
        """
        try:
            if domain:
                # Site-wide analysis
                results = self.analyze_site(domain, max_pages, depth)
            elif urls:
                # Multi-page analysis
                url_list = [url.strip() for url in urls.split(",")]
                results = self.analyze_multiple_urls(url_list, depth)
            else:
                return "Error: Either 'domain' or 'urls' parameter must be provided."

            # Export results in the specified format
            return self.export_results(results, format)

        except Exception as e:
            return f"Error performing bulk analysis: {str(e)}"


# Create tool instance
bulk_analysis = BulkAnalysisManager()

# Create LangChain tool
bulk_analysis_tool = Tool(
    name="bulk_analysis",
    func=bulk_analysis.run,
    description="Analyze multiple pages or entire websites for SEO. Provides comprehensive reports and site-wide recommendations.",
)
