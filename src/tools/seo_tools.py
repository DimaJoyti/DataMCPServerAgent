"""
SEO tools for the SEO Agent.

This module provides tools for SEO analysis, keyword research, content optimization,
metadata generation, and backlink analysis.
"""

import json
import re
from typing import Any, Dict, List

import requests
from bs4 import BeautifulSoup
from langchain.tools import Tool


class SEOAnalyzerTool:
    """Tool for analyzing a webpage for SEO factors."""

    def analyze(self, url: str, depth: str = "basic") -> Dict[str, Any]:
        """
        Analyze a webpage for SEO factors.

        Args:
            url: The URL of the webpage to analyze
            depth: Analysis depth ('basic', 'detailed', or 'comprehensive')

        Returns:
            Dictionary with SEO analysis results
        """
        print(f"Analyzing {url} for SEO factors (depth: {depth})...")

        try:
            # Fetch the webpage
            response = requests.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                },
            )
            response.raise_for_status()

            # Parse the HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Basic analysis
            title = soup.title.string if soup.title else None
            meta_description = soup.find("meta", attrs={"name": "description"})
            meta_description = meta_description["content"] if meta_description else None

            h1_tags = soup.find_all("h1")
            h2_tags = soup.find_all("h2")
            h3_tags = soup.find_all("h3")

            images = soup.find_all("img")
            images_with_alt = [img for img in images if img.get("alt")]

            links = soup.find_all("a")
            internal_links = [
                link
                for link in links
                if link.get("href") and not link["href"].startswith(("http", "https", "//"))
            ]
            external_links = [
                link
                for link in links
                if link.get("href") and link["href"].startswith(("http", "https", "//"))
            ]

            # Calculate word count
            text = soup.get_text()
            words = re.findall(r"\w+", text)
            word_count = len(words)

            # Basic SEO score calculation
            score = 0
            max_score = 0

            # Title check
            max_score += 10
            if title:
                title_length = len(title)
                if 30 <= title_length <= 60:
                    score += 10
                elif 20 <= title_length < 30 or 60 < title_length <= 70:
                    score += 5

            # Meta description check
            max_score += 10
            if meta_description:
                desc_length = len(meta_description)
                if 120 <= desc_length <= 160:
                    score += 10
                elif 80 <= desc_length < 120 or 160 < desc_length <= 200:
                    score += 5

            # H1 check
            max_score += 10
            if len(h1_tags) == 1:
                score += 10
            elif len(h1_tags) > 1:
                score += 5

            # Image alt text check
            max_score += 10
            if images:
                alt_ratio = len(images_with_alt) / len(images)
                score += int(alt_ratio * 10)

            # Word count check
            max_score += 10
            if word_count >= 300:
                score += 10
            elif word_count >= 200:
                score += 5

            # Calculate percentage score
            percentage_score = int((score / max_score) * 100) if max_score > 0 else 0

            # Prepare recommendations
            recommendations = []

            if not title or len(title) < 30 or len(title) > 60:
                recommendations.append("Optimize title tag length (30-60 characters)")

            if not meta_description or len(meta_description) < 120 or len(meta_description) > 160:
                recommendations.append("Optimize meta description length (120-160 characters)")

            if not h1_tags:
                recommendations.append("Add an H1 tag")
            elif len(h1_tags) > 1:
                recommendations.append("Use only one H1 tag per page")

            if images and len(images_with_alt) < len(images):
                recommendations.append("Add alt text to all images")

            if word_count < 300:
                recommendations.append("Increase content length (aim for at least 300 words)")

            # Prepare the result
            result = {
                "url": url,
                "title": title,
                "title_length": len(title) if title else 0,
                "meta_description": meta_description,
                "meta_description_length": len(meta_description) if meta_description else 0,
                "h1_count": len(h1_tags),
                "h2_count": len(h2_tags),
                "h3_count": len(h3_tags),
                "image_count": len(images),
                "images_with_alt": len(images_with_alt),
                "internal_links": len(internal_links),
                "external_links": len(external_links),
                "word_count": word_count,
                "seo_score": percentage_score,
                "recommendations": recommendations,
            }

            # Add detailed analysis if requested
            if depth in ["detailed", "comprehensive"]:
                # Add keyword density analysis
                keyword_density = self._analyze_keyword_density(text)
                result["keyword_density"] = keyword_density

                # Add heading structure analysis
                heading_structure = {
                    "h1": [h.get_text() for h in h1_tags],
                    "h2": [h.get_text() for h in h2_tags],
                    "h3": [h.get_text() for h in h3_tags],
                }
                result["heading_structure"] = heading_structure

                # Add more detailed recommendations
                if keyword_density:
                    top_keywords = sorted(
                        keyword_density.items(), key=lambda x: x[1], reverse=True
                    )[:5]
                    if top_keywords and top_keywords[0][1] > 5:
                        recommendations.append(
                            f"Keyword '{top_keywords[0][0]}' may be overused ({top_keywords[0][1]}%)"
                        )

            # Add comprehensive analysis if requested
            if depth == "comprehensive":
                # Add page speed analysis (mock data for now)
                result["page_speed"] = {
                    "mobile_score": 75,
                    "desktop_score": 85,
                    "load_time": "2.5s",
                }

                # Add mobile-friendliness check (mock data for now)
                result["mobile_friendly"] = True

                # Add structured data analysis
                structured_data = self._extract_structured_data(soup)
                result["structured_data"] = structured_data

            return result

        except Exception as e:
            return {"error": str(e), "url": url}

    def _analyze_keyword_density(self, text: str) -> Dict[str, float]:
        """
        Analyze keyword density in the text.

        Args:
            text: The text to analyze

        Returns:
            Dictionary with keywords and their density percentages
        """
        # Remove common stop words
        stop_words = {
            "a",
            "an",
            "the",
            "and",
            "or",
            "but",
            "is",
            "are",
            "was",
            "were",
            "in",
            "on",
            "at",
            "to",
            "for",
            "with",
            "by",
            "about",
            "as",
            "of",
        }

        # Extract words
        words = re.findall(r"\b\w+\b", text.lower())

        # Filter out stop words and short words
        filtered_words = [word for word in words if word not in stop_words and len(word) > 3]

        # Count occurrences
        word_count = {}
        for word in filtered_words:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1

        # Calculate density
        total_words = len(filtered_words)
        density = {}

        if total_words > 0:
            for word, count in word_count.items():
                density[word] = round((count / total_words) * 100, 2)

        # Return top keywords by density
        sorted_density = dict(sorted(density.items(), key=lambda x: x[1], reverse=True)[:10])
        return sorted_density

    def _extract_structured_data(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """
        Extract structured data from the webpage.

        Args:
            soup: BeautifulSoup object of the webpage

        Returns:
            List of structured data items
        """
        structured_data = []

        # Look for JSON-LD
        ld_scripts = soup.find_all("script", type="application/ld+json")
        for script in ld_scripts:
            try:
                data = json.loads(script.string)
                structured_data.append({"type": "JSON-LD", "data": data})
            except:
                pass

        # Look for microdata
        itemscope_elements = soup.find_all(itemscope=True)
        for element in itemscope_elements:
            item_type = element.get("itemtype", "")
            structured_data.append({"type": "Microdata", "itemType": item_type})

        return structured_data

    def run(self, url: str, depth: str = "basic") -> str:
        """
        Run the SEO analyzer and return formatted results.

        Args:
            url: The URL to analyze
            depth: Analysis depth ('basic', 'detailed', or 'comprehensive')

        Returns:
            Formatted string with analysis results
        """
        result = self.analyze(url, depth)

        if "error" in result:
            return f"Error analyzing {url}: {result['error']}"

        # Format the results as a readable string
        output = f"# SEO Analysis for {result['url']}\n\n"

        output += "## Overview\n"
        output += f"- SEO Score: {result['seo_score']}/100\n"
        output += f"- Title: {result['title']} ({result['title_length']} characters)\n"
        output += f"- Meta Description: {result['meta_description']} ({result['meta_description_length']} characters)\n"
        output += f"- Word Count: {result['word_count']} words\n\n"

        output += "## Content Structure\n"
        output += f"- H1 Tags: {result['h1_count']}\n"
        output += f"- H2 Tags: {result['h2_count']}\n"
        output += f"- H3 Tags: {result['h3_count']}\n"
        output += (
            f"- Images: {result['image_count']} (with alt text: {result['images_with_alt']})\n"
        )
        output += f"- Internal Links: {result['internal_links']}\n"
        output += f"- External Links: {result['external_links']}\n\n"

        if "keyword_density" in result:
            output += "## Keyword Density\n"
            for keyword, density in result["keyword_density"].items():
                output += f"- {keyword}: {density}%\n"
            output += "\n"

        output += "## Recommendations\n"
        for recommendation in result["recommendations"]:
            output += f"- {recommendation}\n"

        return output


# Create tool instances
seo_analyzer = SEOAnalyzerTool()

# Create LangChain tools
seo_analyzer_tool = Tool(
    name="seo_analyzer",
    func=seo_analyzer.run,
    description="Analyze a webpage for SEO factors. Provides SEO score, content analysis, and recommendations for improvement.",
)


class KeywordResearchTool:
    """Tool for researching keywords related to a topic."""

    def __init__(self):
        """Initialize the keyword research tool."""
        from src.tools.seo_api_clients import SEMrushClient

        self.api_client = SEMrushClient()

    def research(self, topic: str, limit: int = 10, database: str = "us") -> Dict[str, Any]:
        """
        Research keywords related to a topic.

        Args:
            topic: The topic to research keywords for
            limit: Maximum number of keywords to return
            database: SEMrush database to use (e.g., "us", "uk", "ca")

        Returns:
            Dictionary with keyword research results
        """
        print(f"Researching keywords for topic: {topic} (limit: {limit}, database: {database})...")

        try:
            # Use the SEMrush API client to get keyword data
            result = self.api_client.keyword_research(topic, database, limit)

            # Check if we have an error but still got mock data
            if "error" in result:
                print(f"Warning: {result['error']}")

            return result

        except Exception as e:
            return {"error": str(e), "topic": topic}

    def run(self, topic: str, limit: int = 10) -> str:
        """
        Run the keyword research tool and return formatted results.

        Args:
            topic: The topic to research keywords for
            limit: Maximum number of keywords to return

        Returns:
            Formatted string with research results
        """
        result = self.research(topic, limit)

        if "error" in result:
            return f"Error researching keywords for {topic}: {result['error']}"

        # Format the results as a readable string
        output = f"# Keyword Research for '{result['topic']}'\n\n"

        output += "## Top Keywords by Opportunity\n"
        output += "| Keyword | Monthly Volume | Difficulty | CPC | Opportunity |\n"
        output += "|---------|----------------|------------|-----|-------------|\n"

        for keyword in result["keywords"]:
            output += f"| {keyword['keyword']} | {keyword['volume']} | {keyword['difficulty']}/100 | ${keyword['cpc']} | {keyword['opportunity']} |\n"

        output += f"\n*Research conducted on {result['timestamp']}*\n\n"

        output += "## Recommendations\n"
        output += "- Focus on keywords with high opportunity scores for the best ROI\n"
        output += "- Consider long-tail variations of top keywords for easier ranking\n"
        output += "- Target a mix of high-volume and low-difficulty keywords\n"

        return output


class ContentOptimizerTool:
    """Tool for optimizing content for SEO."""

    def optimize(self, content: str, target_keywords: List[str]) -> Dict[str, Any]:
        """
        Optimize content for SEO.

        Args:
            content: The content to optimize
            target_keywords: List of target keywords

        Returns:
            Dictionary with optimization results
        """
        print(f"Optimizing content for keywords: {', '.join(target_keywords)}...")

        try:
            # Calculate word count
            words = re.findall(r"\w+", content)
            word_count = len(words)

            # Calculate keyword density
            keyword_density = {}
            for keyword in target_keywords:
                # Count occurrences (case insensitive)
                count = len(re.findall(re.escape(keyword.lower()), content.lower()))
                if word_count > 0:
                    density = (count / word_count) * 100
                    keyword_density[keyword] = round(density, 2)

            # Check readability (Flesch Reading Ease)
            sentences = re.split(r"[.!?]+", content)
            sentence_count = len([s for s in sentences if s.strip()])

            syllable_count = 0
            for word in words:
                syllable_count += self._count_syllables(word)

            if sentence_count > 0 and word_count > 0:
                flesch_score = (
                    206.835
                    - 1.015 * (word_count / sentence_count)
                    - 84.6 * (syllable_count / word_count)
                )
                flesch_score = min(100, max(0, round(flesch_score, 2)))
            else:
                flesch_score = 0

            # Analyze heading structure
            headings = {
                "h1": re.findall(r"# (.*?)(?:\n|$)", content),
                "h2": re.findall(r"## (.*?)(?:\n|$)", content),
                "h3": re.findall(r"### (.*?)(?:\n|$)", content),
            }

            # Check for keyword in headings
            keywords_in_headings = {}
            for keyword in target_keywords:
                keyword_lower = keyword.lower()
                h1_matches = sum(1 for h in headings["h1"] if keyword_lower in h.lower())
                h2_matches = sum(1 for h in headings["h2"] if keyword_lower in h.lower())
                h3_matches = sum(1 for h in headings["h3"] if keyword_lower in h.lower())

                keywords_in_headings[keyword] = {
                    "h1": h1_matches,
                    "h2": h2_matches,
                    "h3": h3_matches,
                    "total": h1_matches + h2_matches + h3_matches,
                }

            # Generate recommendations
            recommendations = []

            # Word count recommendations
            if word_count < 300:
                recommendations.append(
                    "Increase content length to at least 300 words for better SEO"
                )

            # Keyword density recommendations
            for keyword, density in keyword_density.items():
                if density < 0.5:
                    recommendations.append(
                        f"Increase density of keyword '{keyword}' (currently {density}%)"
                    )
                elif density > 3:
                    recommendations.append(
                        f"Reduce density of keyword '{keyword}' (currently {density}%, aim for 1-2%)"
                    )

            # Heading recommendations
            if not headings["h1"]:
                recommendations.append("Add an H1 heading (# Heading) to your content")

            for keyword in target_keywords:
                if keywords_in_headings[keyword]["total"] == 0:
                    recommendations.append(f"Include keyword '{keyword}' in at least one heading")

            # Readability recommendations
            if flesch_score < 60:
                recommendations.append(
                    f"Improve readability (current score: {flesch_score}/100, aim for 60+)"
                )

            # Prepare result
            result = {
                "word_count": word_count,
                "keyword_density": keyword_density,
                "readability_score": flesch_score,
                "headings": headings,
                "keywords_in_headings": keywords_in_headings,
                "recommendations": recommendations,
            }

            return result

        except Exception as e:
            return {"error": str(e)}

    def _count_syllables(self, word: str) -> int:
        """
        Count syllables in a word.

        Args:
            word: The word to count syllables for

        Returns:
            Number of syllables
        """
        word = word.lower()

        # Remove non-alphabetic characters
        word = re.sub(r"[^a-z]", "", word)

        if not word:
            return 0

        # Count vowel groups
        count = len(re.findall(r"[aeiouy]+", word))

        # Adjust for silent e at the end
        if word.endswith("e"):
            count -= 1

        # Ensure at least one syllable
        return max(1, count)

    def run(self, content: str, target_keywords: str) -> str:
        """
        Run the content optimizer and return formatted results.

        Args:
            content: The content to optimize
            target_keywords: Comma-separated list of target keywords

        Returns:
            Formatted string with optimization results
        """
        # Parse keywords
        keywords = [k.strip() for k in target_keywords.split(",")]

        result = self.optimize(content, keywords)

        if "error" in result:
            return f"Error optimizing content: {result['error']}"

        # Format the results as a readable string
        output = "# Content Optimization Analysis\n\n"

        output += "## Overview\n"
        output += f"- Word Count: {result['word_count']} words\n"
        output += f"- Readability Score: {result['readability_score']}/100\n\n"

        output += "## Keyword Density\n"
        for keyword, density in result["keyword_density"].items():
            status = "✅" if 0.5 <= density <= 3 else "⚠️"
            output += f"- {keyword}: {density}% {status}\n"
        output += "\n"

        output += "## Heading Structure\n"
        output += f"- H1 Headings: {len(result['headings']['h1'])}\n"
        output += f"- H2 Headings: {len(result['headings']['h2'])}\n"
        output += f"- H3 Headings: {len(result['headings']['h3'])}\n\n"

        output += "## Keywords in Headings\n"
        for keyword, counts in result["keywords_in_headings"].items():
            output += f"- '{keyword}': {counts['total']} occurrences (H1: {counts['h1']}, H2: {counts['h2']}, H3: {counts['h3']})\n"
        output += "\n"

        output += "## Recommendations\n"
        for recommendation in result["recommendations"]:
            output += f"- {recommendation}\n"

        return output


# Create additional tool instances
keyword_research = KeywordResearchTool()
content_optimizer = ContentOptimizerTool()

# Create additional LangChain tools
keyword_research_tool = Tool(
    name="keyword_research",
    func=keyword_research.run,
    description="Research keywords related to a topic. Provides search volume, difficulty, and opportunity scores.",
)

content_optimizer_tool = Tool(
    name="content_optimizer",
    func=content_optimizer.run,
    description="Optimize content for SEO. Analyzes keyword density, readability, and heading structure.",
)


class MetadataGeneratorTool:
    """Tool for generating optimized metadata for SEO."""

    def generate(
        self, title: str, content: str, keywords: List[str], url: str = None
    ) -> Dict[str, Any]:
        """
        Generate optimized metadata for SEO.

        Args:
            title: The page title
            content: The page content
            keywords: List of target keywords
            url: Optional URL of the page

        Returns:
            Dictionary with generated metadata
        """
        print(f"Generating metadata for title: {title}...")

        try:
            # Extract first paragraph as a base for description
            paragraphs = re.split(r"\n\s*\n", content)
            first_paragraph = paragraphs[0] if paragraphs else ""

            # Clean up the paragraph (remove markdown, etc.)
            clean_paragraph = re.sub(r"[#*_`]", "", first_paragraph)

            # Generate meta description
            description = clean_paragraph[:160]
            if len(clean_paragraph) > 160:
                # Try to cut at a sentence boundary
                last_period = description.rfind(".")
                if last_period > 100:  # Only truncate if we have a decent length
                    description = description[: last_period + 1]
                else:
                    # Cut at a word boundary
                    description = description[: description.rfind(" ")] + "..."

            # Optimize title
            optimized_title = title
            primary_keyword = keywords[0] if keywords else None

            if primary_keyword and primary_keyword.lower() not in title.lower():
                # Try to include the primary keyword in the title
                if len(title) + len(primary_keyword) + 3 <= 60:
                    optimized_title = f"{title} - {primary_keyword}"
                else:
                    # Title is already too long, try to replace some words
                    optimized_title = title

            # Truncate title if too long
            if len(optimized_title) > 60:
                optimized_title = optimized_title[:57] + "..."

            # Generate JSON-LD structured data
            structured_data = {
                "@context": "https://schema.org",
                "@type": "WebPage",
                "name": optimized_title,
                "description": description,
            }

            if url:
                structured_data["url"] = url

            # Generate Open Graph metadata
            og_metadata = {
                "og:title": optimized_title,
                "og:description": description,
                "og:type": "website",
            }

            if url:
                og_metadata["og:url"] = url

            # Generate Twitter Card metadata
            twitter_metadata = {
                "twitter:card": "summary",
                "twitter:title": optimized_title,
                "twitter:description": description,
            }

            # Prepare result
            result = {
                "meta_title": optimized_title,
                "meta_description": description,
                "structured_data": structured_data,
                "open_graph": og_metadata,
                "twitter_card": twitter_metadata,
            }

            return result

        except Exception as e:
            return {"error": str(e)}

    def run(self, title: str, content: str, keywords: str, url: str = None) -> str:
        """
        Run the metadata generator and return formatted results.

        Args:
            title: The page title
            content: The page content
            keywords: Comma-separated list of target keywords
            url: Optional URL of the page

        Returns:
            Formatted string with generated metadata
        """
        # Parse keywords
        keyword_list = [k.strip() for k in keywords.split(",")]

        result = self.generate(title, content, keyword_list, url)

        if "error" in result:
            return f"Error generating metadata: {result['error']}"

        # Format the results as a readable string
        output = "# SEO Metadata Generator Results\n\n"

        output += "## Meta Tags\n"
        output += "```html\n"
        output += f'<title>{result["meta_title"]}</title>\n'
        output += f'<meta name="description" content="{result["meta_description"]}">\n'

        # Add keyword meta tag (though it's not as important for modern SEO)
        output += f'<meta name="keywords" content="{keywords}">\n'
        output += "```\n\n"

        output += "## Open Graph Metadata\n"
        output += "```html\n"
        for key, value in result["open_graph"].items():
            output += f'<meta property="{key}" content="{value}">\n'
        output += "```\n\n"

        output += "## Twitter Card Metadata\n"
        output += "```html\n"
        for key, value in result["twitter_card"].items():
            output += f'<meta name="{key}" content="{value}">\n'
        output += "```\n\n"

        output += "## JSON-LD Structured Data\n"
        output += "```html\n"
        output += '<script type="application/ld+json">\n'
        output += json.dumps(result["structured_data"], indent=2)
        output += "\n</script>\n"
        output += "```\n\n"

        output += "## Recommendations\n"
        output += "- Add these metadata tags to the `<head>` section of your HTML\n"
        output += (
            "- Ensure your meta title is under 60 characters (current: "
            + str(len(result["meta_title"]))
            + ")\n"
        )
        output += (
            "- Ensure your meta description is under 160 characters (current: "
            + str(len(result["meta_description"]))
            + ")\n"
        )

        return output


class BacklinkAnalyzerTool:
    """Tool for analyzing backlinks to a website."""

    def __init__(self):
        """Initialize the backlink analyzer tool."""
        from src.tools.seo_api_clients import MozClient

        self.api_client = MozClient()

    def analyze(self, domain: str, limit: int = 10) -> Dict[str, Any]:
        """
        Analyze backlinks to a domain.

        Args:
            domain: The domain to analyze backlinks for
            limit: Maximum number of backlinks to return

        Returns:
            Dictionary with backlink analysis results
        """
        print(f"Analyzing backlinks for domain: {domain} (limit: {limit})...")

        try:
            # Use the Moz API client to get backlink data
            result = self.api_client.analyze_backlinks(domain, limit)

            # Check if we have an error but still got mock data
            if "error" in result:
                print(f"Warning: {result['error']}")

            return result

        except Exception as e:
            return {"error": str(e), "domain": domain}

    def run(self, domain: str, limit: int = 10) -> str:
        """
        Run the backlink analyzer and return formatted results.

        Args:
            domain: The domain to analyze backlinks for
            limit: Maximum number of backlinks to return

        Returns:
            Formatted string with analysis results
        """
        result = self.analyze(domain, limit)

        if "error" in result:
            return f"Error analyzing backlinks for {domain}: {result['error']}"

        # Format the results as a readable string
        output = f"# Backlink Analysis for {result['domain']}\n\n"

        output += "## Overview\n"
        output += f"- Total Backlinks: {result['total_backlinks']}\n"
        output += f"- Analysis Date: {result['timestamp']}\n\n"

        output += "## Top Backlinks by Domain Authority\n"
        output += "| Source Domain | Target URL | Anchor Text | Domain Authority |\n"
        output += "|---------------|------------|-------------|------------------|\n"

        for backlink in result["top_backlinks"]:
            output += f"| {backlink['source']} | {backlink['target_url']} | {backlink['anchor_text']} | {backlink['domain_authority']} |\n"

        output += "\n## Domain Authority Distribution\n"
        for range_name, count in result["domain_authority_distribution"].items():
            output += f"- {range_name}: {count} backlinks\n"

        output += "\n## Anchor Text Distribution\n"
        for type_name, count in result["anchor_text_distribution"].items():
            output += f"- {type_name.replace('_', ' ').title()}: {count} backlinks\n"

        output += "\n## Recommendations\n"
        output += "- Focus on acquiring backlinks from high-authority domains (DA 60+)\n"
        output += "- Diversify your anchor text to include branded, partial match, and exact match keywords\n"
        output += "- Target relevant websites in your industry for the most effective backlinks\n"

        return output


# Create additional tool instances
metadata_generator = MetadataGeneratorTool()
backlink_analyzer = BacklinkAnalyzerTool()

# Create additional LangChain tools
metadata_generator_tool = Tool(
    name="metadata_generator",
    func=metadata_generator.run,
    description="Generate optimized metadata for SEO. Creates meta tags, structured data, and social media metadata.",
)

backlink_analyzer_tool = Tool(
    name="backlink_analyzer",
    func=backlink_analyzer.run,
    description="Analyze backlinks to a website. Provides information on backlink quality, anchor text, and domain authority.",
)
