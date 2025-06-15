"""
Machine learning tools for SEO optimization.

This module provides tools that use machine learning models for content optimization,
ranking prediction, and other SEO-related tasks.
"""

import os
import re
from typing import Any, Dict, List

from langchain.tools import Tool

from src.tools.seo_ml_models import ContentOptimizationModel, RankingPredictionModel

# Directory for storing trained models
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "seo"
)
os.makedirs(MODELS_DIR, exist_ok=True)

# Default model paths
DEFAULT_CONTENT_MODEL_PATH = os.path.join(MODELS_DIR, "content_optimization_model.pkl")
DEFAULT_RANKING_MODEL_PATH = os.path.join(MODELS_DIR, "ranking_prediction_model.pkl")


class MLContentOptimizerTool:
    """Tool for optimizing content using machine learning."""

    def __init__(self):
        """Initialize the ML content optimizer tool."""
        # Load the content optimization model if it exists
        if os.path.exists(DEFAULT_CONTENT_MODEL_PATH):
            self.model = ContentOptimizationModel(DEFAULT_CONTENT_MODEL_PATH)
        else:
            self.model = ContentOptimizationModel()
            # Train a simple model with mock data if no model exists
            self._train_with_mock_data()

    def _train_with_mock_data(self):
        """Train the model with mock data."""
        print("Training content optimization model with mock data...")

        # Mock training data
        contents = [
            "# SEO Best Practices\n\nSearch engine optimization is important for websites. It helps improve visibility in search results.\n\n## Keywords\n\nChoosing the right keywords is essential for SEO success.",
            "# Keyword Research Guide\n\nKeyword research is the foundation of SEO. Finding the right keywords can make or break your SEO strategy.\n\n## Tools for Keyword Research\n\nThere are many tools available for keyword research, including SEMrush, Ahrefs, and Google Keyword Planner.\n\n## Long-tail Keywords\n\nLong-tail keywords are more specific and less competitive.",
            "# Content Optimization\n\nOptimizing your content for search engines is crucial. This includes using keywords naturally, structuring content with headings, and providing value to readers.\n\n## Headings\n\nUse headings to structure your content. This helps both readers and search engines understand your content better.\n\n## Images\n\nInclude relevant images with alt text. This improves user experience and provides additional ranking opportunities.",
            "# Technical SEO Guide\n\nTechnical SEO focuses on improving the technical aspects of a website to increase its rankings in search engines.\n\n## Page Speed\n\nPage speed is a ranking factor. Faster websites provide better user experience and rank higher in search results.\n\n## Mobile-Friendly\n\nEnsure your website is mobile-friendly. Google uses mobile-first indexing, meaning it primarily uses the mobile version of a site for ranking.",
            "# Link Building Strategies\n\nLink building is an important part of SEO. Quality backlinks from reputable websites can significantly improve your rankings.\n\n## Guest Posting\n\nGuest posting on relevant websites can help build backlinks and establish authority.\n\n## Broken Link Building\n\nFind broken links on other websites and suggest your content as a replacement.",
        ]

        target_keywords_list = [
            ["seo", "search engine optimization", "visibility"],
            ["keyword research", "seo", "long-tail keywords"],
            ["content optimization", "headings", "images"],
            ["technical seo", "page speed", "mobile-friendly"],
            ["link building", "backlinks", "guest posting"],
        ]

        scores = [65, 78, 82, 75, 70]

        # Train the model
        self.model.train(contents, target_keywords_list, scores)

        # Save the model
        self.model.save_model(DEFAULT_CONTENT_MODEL_PATH)

    def optimize(self, content: str, target_keywords: List[str]) -> Dict[str, Any]:
        """
        Optimize content for SEO using machine learning.

        Args:
            content: The content to optimize
            target_keywords: List of target keywords

        Returns:
            Dictionary with optimization results
        """
        print(f"Optimizing content for keywords: {', '.join(target_keywords)}...")

        try:
            # Predict SEO score
            score = self.model.predict(content, target_keywords)

            # Get improvement suggestions
            suggestions = self.model.get_improvement_suggestions(content, target_keywords)

            # Preprocess content to get features
            features = self.model.preprocess_content(content)

            # Calculate keyword density
            keyword_density = {}
            for keyword in target_keywords:
                count = len(re.findall(re.escape(keyword.lower()), content.lower()))
                if features["word_count"] > 0:
                    density = (count / features["word_count"]) * 100
                    keyword_density[keyword] = round(density, 2)

            # Check for keywords in headings
            keywords_in_headings = {}
            headings = []
            headings.extend(re.findall(r"# (.*?)(?:\n|$)", content))
            headings.extend(re.findall(r"## (.*?)(?:\n|$)", content))
            headings.extend(re.findall(r"### (.*?)(?:\n|$)", content))

            for keyword in target_keywords:
                keywords_in_headings[keyword] = False
                for heading in headings:
                    if keyword.lower() in heading.lower():
                        keywords_in_headings[keyword] = True
                        break

            # Prepare result
            result = {
                "seo_score": round(score, 2),
                "word_count": features["word_count"],
                "readability_score": round(features["readability_score"], 2),
                "keyword_density": keyword_density,
                "keywords_in_headings": keywords_in_headings,
                "headings": {
                    "h1": features["h1_count"],
                    "h2": features["h2_count"],
                    "h3": features["h3_count"],
                },
                "links": {
                    "internal": features["internal_links"],
                    "external": features["external_links"],
                },
                "images": features["images"],
                "suggestions": suggestions,
            }

            return result

        except Exception as e:
            return {"error": str(e)}

    def run(self, content: str, target_keywords: str) -> str:
        """
        Run the ML content optimizer and return formatted results.

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
        output = "# ML Content Optimization Analysis\n\n"

        output += "## Overview\n"
        output += f"- SEO Score: {result['seo_score']}/100\n"
        output += f"- Word Count: {result['word_count']} words\n"
        output += f"- Readability Score: {result['readability_score']}/100\n\n"

        output += "## Keyword Analysis\n"
        output += "### Keyword Density\n"
        for keyword, density in result["keyword_density"].items():
            status = "✅" if 0.5 <= density <= 3 else "⚠️"
            output += f"- {keyword}: {density}% {status}\n"
        output += "\n"

        output += "### Keywords in Headings\n"
        for keyword, in_heading in result["keywords_in_headings"].items():
            status = "✅" if in_heading else "⚠️"
            output += f"- {keyword}: {status}\n"
        output += "\n"

        output += "## Content Structure\n"
        output += f"- H1 Headings: {result['headings']['h1']}\n"
        output += f"- H2 Headings: {result['headings']['h2']}\n"
        output += f"- H3 Headings: {result['headings']['h3']}\n"
        output += f"- Internal Links: {result['links']['internal']}\n"
        output += f"- External Links: {result['links']['external']}\n"
        output += f"- Images: {result['images']}\n\n"

        output += "## ML-Based Recommendations\n"

        # Group suggestions by importance
        high_importance = [s for s in result["suggestions"] if s["importance"] == "high"]
        medium_importance = [s for s in result["suggestions"] if s["importance"] == "medium"]
        low_importance = [s for s in result["suggestions"] if s["importance"] == "low"]

        if high_importance:
            output += "### High Priority\n"
            for suggestion in high_importance:
                output += f"- {suggestion['suggestion']}\n  - *{suggestion['reason']}*\n"
            output += "\n"

        if medium_importance:
            output += "### Medium Priority\n"
            for suggestion in medium_importance:
                output += f"- {suggestion['suggestion']}\n  - *{suggestion['reason']}*\n"
            output += "\n"

        if low_importance:
            output += "### Low Priority\n"
            for suggestion in low_importance:
                output += f"- {suggestion['suggestion']}\n  - *{suggestion['reason']}*\n"
            output += "\n"

        return output


class MLRankingPredictionTool:
    """Tool for predicting search rankings using machine learning."""

    def __init__(self):
        """Initialize the ML ranking prediction tool."""
        # Load the ranking prediction model if it exists
        if os.path.exists(DEFAULT_RANKING_MODEL_PATH):
            self.model = RankingPredictionModel(DEFAULT_RANKING_MODEL_PATH)
        else:
            self.model = RankingPredictionModel()
            # Train a simple model with mock data if no model exists
            self._train_with_mock_data()

    def _train_with_mock_data(self):
        """Train the model with mock data."""
        print("Training ranking prediction model with mock data...")

        # Mock training data
        urls = [
            "https://example.com/seo-guide",
            "https://example.com/keyword-research",
            "https://example.com/content-optimization",
            "https://example.com/technical-seo",
            "https://example.com/link-building",
        ]

        keywords = [
            "seo guide",
            "keyword research",
            "content optimization",
            "technical seo",
            "link building",
        ]

        content_features_list = [
            {
                "word_count": 1500,
                "keyword_density": 1.8,
                "readability_score": 75,
                "headings_count": 8,
                "images_count": 5,
                "internal_links": 12,
                "external_links": 8,
                "keyword_in_title": 1,
                "keyword_in_headings": 1,
                "keyword_in_first_paragraph": 1,
            },
            {
                "word_count": 2000,
                "keyword_density": 1.5,
                "readability_score": 80,
                "headings_count": 10,
                "images_count": 7,
                "internal_links": 15,
                "external_links": 10,
                "keyword_in_title": 1,
                "keyword_in_headings": 1,
                "keyword_in_first_paragraph": 1,
            },
            {
                "word_count": 1200,
                "keyword_density": 2.0,
                "readability_score": 70,
                "headings_count": 6,
                "images_count": 4,
                "internal_links": 8,
                "external_links": 6,
                "keyword_in_title": 1,
                "keyword_in_headings": 1,
                "keyword_in_first_paragraph": 0,
            },
            {
                "word_count": 1800,
                "keyword_density": 1.2,
                "readability_score": 85,
                "headings_count": 9,
                "images_count": 6,
                "internal_links": 14,
                "external_links": 9,
                "keyword_in_title": 1,
                "keyword_in_headings": 1,
                "keyword_in_first_paragraph": 1,
            },
            {
                "word_count": 1000,
                "keyword_density": 2.2,
                "readability_score": 65,
                "headings_count": 5,
                "images_count": 3,
                "internal_links": 6,
                "external_links": 4,
                "keyword_in_title": 1,
                "keyword_in_headings": 0,
                "keyword_in_first_paragraph": 1,
            },
        ]

        backlink_features_list = [
            {
                "backlink_count": 500,
                "referring_domains": 120,
                "domain_authority": 45,
                "page_authority": 38,
                "dofollow_ratio": 0.7,
            },
            {
                "backlink_count": 800,
                "referring_domains": 200,
                "domain_authority": 55,
                "page_authority": 48,
                "dofollow_ratio": 0.8,
            },
            {
                "backlink_count": 300,
                "referring_domains": 80,
                "domain_authority": 40,
                "page_authority": 35,
                "dofollow_ratio": 0.6,
            },
            {
                "backlink_count": 600,
                "referring_domains": 150,
                "domain_authority": 50,
                "page_authority": 42,
                "dofollow_ratio": 0.75,
            },
            {
                "backlink_count": 200,
                "referring_domains": 50,
                "domain_authority": 35,
                "page_authority": 30,
                "dofollow_ratio": 0.5,
            },
        ]

        technical_features_list = [
            {
                "page_speed_mobile": 75,
                "page_speed_desktop": 85,
                "is_https": 1,
                "is_mobile_friendly": 1,
                "has_structured_data": 1,
            },
            {
                "page_speed_mobile": 80,
                "page_speed_desktop": 90,
                "is_https": 1,
                "is_mobile_friendly": 1,
                "has_structured_data": 1,
            },
            {
                "page_speed_mobile": 65,
                "page_speed_desktop": 80,
                "is_https": 1,
                "is_mobile_friendly": 1,
                "has_structured_data": 0,
            },
            {
                "page_speed_mobile": 70,
                "page_speed_desktop": 85,
                "is_https": 1,
                "is_mobile_friendly": 1,
                "has_structured_data": 1,
            },
            {
                "page_speed_mobile": 60,
                "page_speed_desktop": 75,
                "is_https": 1,
                "is_mobile_friendly": 0,
                "has_structured_data": 0,
            },
        ]

        rankings = [5, 2, 8, 4, 12]

        # Train the model
        self.model.train(
            urls,
            keywords,
            content_features_list,
            backlink_features_list,
            technical_features_list,
            rankings,
        )

        # Save the model
        self.model.save_model(DEFAULT_RANKING_MODEL_PATH)

    def predict_ranking(
        self,
        url: str,
        keyword: str,
        content_features: Dict[str, Any],
        backlink_features: Dict[str, Any],
        technical_features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Predict search ranking for a URL-keyword pair.

        Args:
            url: The URL to predict ranking for
            keyword: The target keyword
            content_features: Content-related features
            backlink_features: Backlink-related features
            technical_features: Technical SEO features

        Returns:
            Dictionary with prediction results
        """
        print(f"Predicting ranking for {url} and keyword '{keyword}'...")

        try:
            # Predict ranking category
            category, probabilities = self.model.predict(
                url, keyword, content_features, backlink_features, technical_features
            )

            # Get ranking factors
            ranking_factors = self.model.get_ranking_factors()

            # Convert category to ranking range
            ranking_ranges = {0: "1-3", 1: "4-10", 2: "11-20", 3: "21+"}

            # Prepare result
            result = {
                "url": url,
                "keyword": keyword,
                "predicted_ranking_range": ranking_ranges[category],
                "ranking_probabilities": probabilities,
                "ranking_factors": ranking_factors,
                "content_features": content_features,
                "backlink_features": backlink_features,
                "technical_features": technical_features,
            }

            return result

        except Exception as e:
            return {"error": str(e), "url": url, "keyword": keyword}

    def run(self, url: str, keyword: str) -> str:
        """
        Run the ML ranking prediction tool and return formatted results.

        Args:
            url: The URL to predict ranking for
            keyword: The target keyword

        Returns:
            Formatted string with prediction results
        """
        # In a real implementation, we would fetch these features from APIs
        # For now, we'll use mock data
        content_features = {
            "word_count": 1500,
            "keyword_density": 1.8,
            "readability_score": 75,
            "headings_count": 8,
            "images_count": 5,
            "internal_links": 12,
            "external_links": 8,
            "keyword_in_title": 1,
            "keyword_in_headings": 1,
            "keyword_in_first_paragraph": 1,
        }

        backlink_features = {
            "backlink_count": 500,
            "referring_domains": 120,
            "domain_authority": 45,
            "page_authority": 38,
            "dofollow_ratio": 0.7,
        }

        technical_features = {
            "page_speed_mobile": 75,
            "page_speed_desktop": 85,
            "is_https": 1,
            "is_mobile_friendly": 1,
            "has_structured_data": 1,
        }

        result = self.predict_ranking(
            url, keyword, content_features, backlink_features, technical_features
        )

        if "error" in result:
            return f"Error predicting ranking: {result['error']}"

        # Format the results as a readable string
        output = f"# ML Ranking Prediction for '{result['keyword']}'\n\n"

        output += f"## Prediction for {result['url']}\n"
        output += f"- Predicted Ranking Range: {result['predicted_ranking_range']}\n\n"

        output += "## Ranking Probabilities\n"
        output += f"- Top Positions (1-3): {result['ranking_probabilities']['top_positions']:.2%}\n"
        output += f"- First Page (4-10): {result['ranking_probabilities']['first_page']:.2%}\n"
        output += f"- Second Page (11-20): {result['ranking_probabilities']['second_page']:.2%}\n"
        output += f"- Beyond Second Page (21+): {result['ranking_probabilities']['beyond_second_page']:.2%}\n\n"

        output += "## Top Ranking Factors\n"
        top_factors = list(result["ranking_factors"].items())[:5]
        for factor, importance in top_factors:
            output += f"- {factor.replace('_', ' ').title()}: {importance:.4f}\n"
        output += "\n"

        output += "## Recommendations\n"

        # Generate recommendations based on ranking factors and features
        recommendations = []

        # Content recommendations
        if content_features["word_count"] < 1500:
            recommendations.append("Increase content length to at least 1500 words")

        if content_features["keyword_density"] < 1.0:
            recommendations.append(
                f"Increase keyword density for '{keyword}' (currently {content_features['keyword_density']}%)"
            )
        elif content_features["keyword_density"] > 2.5:
            recommendations.append(
                f"Reduce keyword density for '{keyword}' (currently {content_features['keyword_density']}%)"
            )

        if content_features["keyword_in_headings"] == 0:
            recommendations.append(f"Include keyword '{keyword}' in at least one heading")

        if content_features["keyword_in_first_paragraph"] == 0:
            recommendations.append(f"Include keyword '{keyword}' in the first paragraph")

        # Backlink recommendations
        if backlink_features["referring_domains"] < 100:
            recommendations.append("Build more backlinks from different domains")

        # Technical recommendations
        if technical_features["page_speed_mobile"] < 70:
            recommendations.append("Improve mobile page speed")

        if technical_features["is_mobile_friendly"] == 0:
            recommendations.append("Make the page mobile-friendly")

        if technical_features["has_structured_data"] == 0:
            recommendations.append("Add structured data (Schema.org) to the page")

        for recommendation in recommendations:
            output += f"- {recommendation}\n"

        return output


# Create tool instances
ml_content_optimizer = MLContentOptimizerTool()
ml_ranking_prediction = MLRankingPredictionTool()

# Create LangChain tools
ml_content_optimizer_tool = Tool(
    name="ml_content_optimizer",
    func=ml_content_optimizer.run,
    description="Optimize content for SEO using machine learning. Provides ML-based recommendations for improving content.",
)

ml_ranking_prediction_tool = Tool(
    name="ml_ranking_prediction",
    func=ml_ranking_prediction.run,
    description="Predict search rankings using machine learning. Estimates the potential ranking position for a URL and keyword.",
)
