"""
Machine learning models for SEO optimization.

This module provides machine learning models for content optimization,
ranking prediction, and other SEO-related tasks.
"""

import os
import pickle
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Directory for storing trained models
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "seo"
)
os.makedirs(MODELS_DIR, exist_ok=True)


class ContentOptimizationModel:
    """Machine learning model for content optimization."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the content optimization model.

        Args:
            model_path: Optional path to a pre-trained model
        """
        self.vectorizer = TfidfVectorizer(
            max_features=5000, stop_words="english", ngram_range=(1, 2)
        )

        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

        self.scaler = StandardScaler()
        self.is_trained = False

        # Load pre-trained model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def preprocess_content(self, content: str) -> Dict[str, Any]:
        """
        Preprocess content for feature extraction.

        Args:
            content: The content to preprocess

        Returns:
            Dictionary with extracted features
        """
        # Clean the content
        clean_content = re.sub(r"[^\w\s]", "", content.lower())

        # Extract basic features
        word_count = len(content.split())
        sentence_count = len(re.split(r"[.!?]+", content))
        avg_word_length = sum(len(word) for word in content.split()) / max(1, word_count)
        avg_sentence_length = word_count / max(1, sentence_count)

        # Extract heading features
        h1_count = len(re.findall(r"# (.*?)(?:\n|$)", content))
        h2_count = len(re.findall(r"## (.*?)(?:\n|$)", content))
        h3_count = len(re.findall(r"### (.*?)(?:\n|$)", content))
        total_headings = h1_count + h2_count + h3_count

        # Extract link features
        internal_links = len(re.findall(r"\[.*?\]\((?!http).*?\)", content))
        external_links = len(re.findall(r"\[.*?\]\(http.*?\)", content))
        total_links = internal_links + external_links

        # Extract image features
        images = len(re.findall(r"!\[.*?\]\(.*?\)", content))

        # Calculate readability (Flesch Reading Ease)
        words = re.findall(r"\b\w+\b", content.lower())
        syllable_count = sum(self._count_syllables(word) for word in words)
        if sentence_count > 0 and word_count > 0:
            flesch_score = (
                206.835
                - 1.015 * (word_count / sentence_count)
                - 84.6 * (syllable_count / word_count)
            )
            flesch_score = min(100, max(0, flesch_score))
        else:
            flesch_score = 0

        # Return all features
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            "h1_count": h1_count,
            "h2_count": h2_count,
            "h3_count": h3_count,
            "total_headings": total_headings,
            "internal_links": internal_links,
            "external_links": external_links,
            "total_links": total_links,
            "images": images,
            "readability_score": flesch_score,
            "clean_content": clean_content,
        }

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

    def extract_features(self, content: str, target_keywords: List[str]) -> np.ndarray:
        """
        Extract features from content for model input.

        Args:
            content: The content to extract features from
            target_keywords: List of target keywords

        Returns:
            Feature vector
        """
        # Preprocess content
        features = self.preprocess_content(content)
        clean_content = features.pop("clean_content")

        # Calculate keyword density
        keyword_density = {}
        for keyword in target_keywords:
            # Count occurrences (case insensitive)
            count = len(re.findall(re.escape(keyword.lower()), content.lower()))
            if features["word_count"] > 0:
                density = (count / features["word_count"]) * 100
                keyword_density[keyword] = round(density, 2)

        # Calculate average keyword density
        avg_keyword_density = sum(keyword_density.values()) / max(1, len(keyword_density))
        features["avg_keyword_density"] = avg_keyword_density

        # Check for keywords in headings
        keywords_in_headings = 0
        headings = []
        headings.extend(re.findall(r"# (.*?)(?:\n|$)", content))
        headings.extend(re.findall(r"## (.*?)(?:\n|$)", content))
        headings.extend(re.findall(r"### (.*?)(?:\n|$)", content))

        for keyword in target_keywords:
            for heading in headings:
                if keyword.lower() in heading.lower():
                    keywords_in_headings += 1
                    break

        features["keywords_in_headings"] = keywords_in_headings

        # Convert features to numpy array
        feature_vector = np.array(
            [
                features["word_count"],
                features["sentence_count"],
                features["avg_word_length"],
                features["avg_sentence_length"],
                features["h1_count"],
                features["h2_count"],
                features["h3_count"],
                features["total_headings"],
                features["internal_links"],
                features["external_links"],
                features["total_links"],
                features["images"],
                features["readability_score"],
                features["avg_keyword_density"],
                features["keywords_in_headings"],
            ]
        ).reshape(1, -1)

        # Add TF-IDF features if the model is trained
        if self.is_trained:
            tfidf_features = self.vectorizer.transform([clean_content])
            return np.hstack([feature_vector, tfidf_features.toarray()])

        return feature_vector

    def train(
        self, contents: List[str], target_keywords_list: List[List[str]], scores: List[float]
    ) -> float:
        """
        Train the model on a dataset of content, keywords, and SEO scores.

        Args:
            contents: List of content samples
            target_keywords_list: List of target keywords for each content sample
            scores: List of SEO scores for each content sample

        Returns:
            Model accuracy (R^2 score)
        """
        # Extract features for each content sample
        X = []
        for content, target_keywords in zip(contents, target_keywords_list):
            features = self.preprocess_content(content)
            clean_content = features.pop("clean_content")

            # Calculate keyword density
            keyword_density = {}
            for keyword in target_keywords:
                count = len(re.findall(re.escape(keyword.lower()), content.lower()))
                if features["word_count"] > 0:
                    density = (count / features["word_count"]) * 100
                    keyword_density[keyword] = round(density, 2)

            # Calculate average keyword density
            avg_keyword_density = sum(keyword_density.values()) / max(1, len(keyword_density))
            features["avg_keyword_density"] = avg_keyword_density

            # Check for keywords in headings
            keywords_in_headings = 0
            headings = []
            headings.extend(re.findall(r"# (.*?)(?:\n|$)", content))
            headings.extend(re.findall(r"## (.*?)(?:\n|$)", content))
            headings.extend(re.findall(r"### (.*?)(?:\n|$)", content))

            for keyword in target_keywords:
                for heading in headings:
                    if keyword.lower() in heading.lower():
                        keywords_in_headings += 1
                        break

            features["keywords_in_headings"] = keywords_in_headings

            # Add to feature list
            X.append(
                [
                    features["word_count"],
                    features["sentence_count"],
                    features["avg_word_length"],
                    features["avg_sentence_length"],
                    features["h1_count"],
                    features["h2_count"],
                    features["h3_count"],
                    features["total_headings"],
                    features["internal_links"],
                    features["external_links"],
                    features["total_links"],
                    features["images"],
                    features["readability_score"],
                    features["avg_keyword_density"],
                    features["keywords_in_headings"],
                ]
            )

        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(scores)

        # Fit TF-IDF vectorizer
        clean_contents = [self.preprocess_content(content)["clean_content"] for content in contents]
        tfidf_features = self.vectorizer.fit_transform(clean_contents)

        # Combine features
        X_combined = np.hstack([X, tfidf_features.toarray()])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = self.model.score(X_test_scaled, y_test)

        print(f"Model trained with MSE: {mse:.4f}, R^2: {r2:.4f}")

        self.is_trained = True
        return r2

    def predict(self, content: str, target_keywords: List[str]) -> float:
        """
        Predict the SEO score for content.

        Args:
            content: The content to predict the score for
            target_keywords: List of target keywords

        Returns:
            Predicted SEO score (0-100)
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet")

        # Extract features
        features = self.extract_features(content, target_keywords)

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Predict score
        score = self.model.predict(features_scaled)[0]

        # Ensure score is in range 0-100
        return min(100, max(0, score))

    def get_improvement_suggestions(
        self, content: str, target_keywords: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Get suggestions for improving content SEO.

        Args:
            content: The content to analyze
            target_keywords: List of target keywords

        Returns:
            List of improvement suggestions
        """
        # Preprocess content
        features = self.preprocess_content(content)

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

        # Generate suggestions
        suggestions = []

        # Word count suggestions
        if features["word_count"] < 300:
            suggestions.append(
                {
                    "type": "word_count",
                    "importance": "high",
                    "suggestion": f"Increase content length to at least 300 words (currently {features['word_count']} words)",
                    "reason": "Longer content tends to rank better in search results",
                }
            )
        elif features["word_count"] < 600:
            suggestions.append(
                {
                    "type": "word_count",
                    "importance": "medium",
                    "suggestion": f"Consider increasing content length to 600+ words (currently {features['word_count']} words)",
                    "reason": "More comprehensive content often ranks better for competitive keywords",
                }
            )

        # Heading suggestions
        if features["h1_count"] == 0:
            suggestions.append(
                {
                    "type": "headings",
                    "importance": "high",
                    "suggestion": "Add an H1 heading (# Heading) to your content",
                    "reason": "H1 headings are important for SEO and content structure",
                }
            )
        elif features["h1_count"] > 1:
            suggestions.append(
                {
                    "type": "headings",
                    "importance": "medium",
                    "suggestion": f"Reduce the number of H1 headings to 1 (currently {features['h1_count']})",
                    "reason": "Multiple H1 headings can confuse search engines about the main topic",
                }
            )

        if features["total_headings"] < 3 and features["word_count"] > 300:
            suggestions.append(
                {
                    "type": "headings",
                    "importance": "medium",
                    "suggestion": "Add more headings to structure your content",
                    "reason": "Well-structured content with clear headings improves readability and SEO",
                }
            )

        # Keyword suggestions
        for keyword, density in keyword_density.items():
            if density < 0.5:
                suggestions.append(
                    {
                        "type": "keyword_density",
                        "importance": "high",
                        "suggestion": f"Increase usage of keyword '{keyword}' (currently {density}%)",
                        "reason": "Keywords should appear naturally throughout the content",
                    }
                )
            elif density > 3:
                suggestions.append(
                    {
                        "type": "keyword_density",
                        "importance": "high",
                        "suggestion": f"Reduce usage of keyword '{keyword}' (currently {density}%)",
                        "reason": "Keyword stuffing can negatively impact SEO",
                    }
                )

        for keyword, in_heading in keywords_in_headings.items():
            if not in_heading:
                suggestions.append(
                    {
                        "type": "keywords_in_headings",
                        "importance": "medium",
                        "suggestion": f"Include keyword '{keyword}' in at least one heading",
                        "reason": "Keywords in headings help search engines understand your content",
                    }
                )

        # Readability suggestions
        if features["readability_score"] < 60:
            suggestions.append(
                {
                    "type": "readability",
                    "importance": "medium",
                    "suggestion": f"Improve readability (current score: {features['readability_score']:.1f}/100)",
                    "reason": "More readable content keeps users engaged and reduces bounce rate",
                }
            )

        # Link suggestions
        if features["total_links"] == 0 and features["word_count"] > 300:
            suggestions.append(
                {
                    "type": "links",
                    "importance": "medium",
                    "suggestion": "Add internal and/or external links to your content",
                    "reason": "Links help search engines understand the context and relevance of your content",
                }
            )

        # Image suggestions
        if features["images"] == 0 and features["word_count"] > 300:
            suggestions.append(
                {
                    "type": "images",
                    "importance": "low",
                    "suggestion": "Add images to your content",
                    "reason": "Images make content more engaging and can rank in image search",
                }
            )

        return suggestions

    def save_model(self, model_path: Optional[str] = None) -> str:
        """
        Save the trained model to disk.

        Args:
            model_path: Optional path to save the model to

        Returns:
            Path where the model was saved
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        if model_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(MODELS_DIR, f"content_optimization_model_{timestamp}.pkl")

        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save model components
        with open(model_path, "wb") as f:
            pickle.dump(
                {"vectorizer": self.vectorizer, "model": self.model, "scaler": self.scaler}, f
            )

        print(f"Model saved to {model_path}")
        return model_path

    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from disk.

        Args:
            model_path: Path to the saved model
        """
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        self.vectorizer = model_data["vectorizer"]
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.is_trained = True

        print(f"Model loaded from {model_path}")


class RankingPredictionModel:
    """Machine learning model for predicting search rankings."""

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the ranking prediction model.

        Args:
            model_path: Optional path to a pre-trained model
        """
        self.model = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
        )

        self.scaler = StandardScaler()
        self.is_trained = False

        # Load pre-trained model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def extract_features(
        self,
        url: str,
        keyword: str,
        content_features: Dict[str, Any],
        backlink_features: Dict[str, Any],
        technical_features: Dict[str, Any],
    ) -> np.ndarray:
        """
        Extract features for ranking prediction.

        Args:
            url: The URL to predict ranking for
            keyword: The target keyword
            content_features: Content-related features
            backlink_features: Backlink-related features
            technical_features: Technical SEO features

        Returns:
            Feature vector
        """
        # Combine all features
        features = []

        # Content features
        features.extend(
            [
                content_features.get("word_count", 0),
                content_features.get("keyword_density", 0),
                content_features.get("readability_score", 0),
                content_features.get("headings_count", 0),
                content_features.get("images_count", 0),
                content_features.get("internal_links", 0),
                content_features.get("external_links", 0),
                content_features.get("keyword_in_title", 0),
                content_features.get("keyword_in_headings", 0),
                content_features.get("keyword_in_first_paragraph", 0),
            ]
        )

        # Backlink features
        features.extend(
            [
                backlink_features.get("backlink_count", 0),
                backlink_features.get("referring_domains", 0),
                backlink_features.get("domain_authority", 0),
                backlink_features.get("page_authority", 0),
                backlink_features.get("dofollow_ratio", 0),
            ]
        )

        # Technical features
        features.extend(
            [
                technical_features.get("page_speed_mobile", 0),
                technical_features.get("page_speed_desktop", 0),
                technical_features.get("is_https", 0),
                technical_features.get("is_mobile_friendly", 0),
                technical_features.get("has_structured_data", 0),
            ]
        )

        return np.array(features).reshape(1, -1)

    def train(
        self,
        urls: List[str],
        keywords: List[str],
        content_features_list: List[Dict[str, Any]],
        backlink_features_list: List[Dict[str, Any]],
        technical_features_list: List[Dict[str, Any]],
        rankings: List[int],
    ) -> float:
        """
        Train the model on a dataset of URLs, keywords, features, and rankings.

        Args:
            urls: List of URLs
            keywords: List of target keywords for each URL
            content_features_list: List of content features for each URL
            backlink_features_list: List of backlink features for each URL
            technical_features_list: List of technical features for each URL
            rankings: List of rankings for each URL-keyword pair

        Returns:
            Model accuracy
        """
        # Extract features for each URL-keyword pair
        X = []
        for i in range(len(urls)):
            features = []

            # Content features
            features.extend(
                [
                    content_features_list[i].get("word_count", 0),
                    content_features_list[i].get("keyword_density", 0),
                    content_features_list[i].get("readability_score", 0),
                    content_features_list[i].get("headings_count", 0),
                    content_features_list[i].get("images_count", 0),
                    content_features_list[i].get("internal_links", 0),
                    content_features_list[i].get("external_links", 0),
                    content_features_list[i].get("keyword_in_title", 0),
                    content_features_list[i].get("keyword_in_headings", 0),
                    content_features_list[i].get("keyword_in_first_paragraph", 0),
                ]
            )

            # Backlink features
            features.extend(
                [
                    backlink_features_list[i].get("backlink_count", 0),
                    backlink_features_list[i].get("referring_domains", 0),
                    backlink_features_list[i].get("domain_authority", 0),
                    backlink_features_list[i].get("page_authority", 0),
                    backlink_features_list[i].get("dofollow_ratio", 0),
                ]
            )

            # Technical features
            features.extend(
                [
                    technical_features_list[i].get("page_speed_mobile", 0),
                    technical_features_list[i].get("page_speed_desktop", 0),
                    technical_features_list[i].get("is_https", 0),
                    technical_features_list[i].get("is_mobile_friendly", 0),
                    technical_features_list[i].get("has_structured_data", 0),
                ]
            )

            X.append(features)

        # Convert to numpy arrays
        X = np.array(X)

        # Convert rankings to position categories
        # 1-3: Top positions, 4-10: First page, 11-20: Second page, >20: Beyond second page
        y = np.array([0 if r <= 3 else 1 if r <= 10 else 2 if r <= 20 else 3 for r in rankings])

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        self.model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Model trained with accuracy: {accuracy:.4f}")

        self.is_trained = True
        return accuracy

    def predict(
        self,
        url: str,
        keyword: str,
        content_features: Dict[str, Any],
        backlink_features: Dict[str, Any],
        technical_features: Dict[str, Any],
    ) -> Tuple[int, Dict[str, float]]:
        """
        Predict the ranking category for a URL-keyword pair.

        Args:
            url: The URL to predict ranking for
            keyword: The target keyword
            content_features: Content-related features
            backlink_features: Backlink-related features
            technical_features: Technical SEO features

        Returns:
            Tuple of (ranking category, probability distribution)
            Ranking categories: 0 = Top positions (1-3), 1 = First page (4-10),
                               2 = Second page (11-20), 3 = Beyond second page (>20)
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet")

        # Extract features
        features = self.extract_features(
            url, keyword, content_features, backlink_features, technical_features
        )

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Predict ranking category
        category = self.model.predict(features_scaled)[0]

        # Get probability distribution
        probabilities = self.model.predict_proba(features_scaled)[0]
        probability_dict = {
            "top_positions": probabilities[0],
            "first_page": probabilities[1],
            "second_page": probabilities[2],
            "beyond_second_page": probabilities[3],
        }

        return category, probability_dict

    def get_ranking_factors(self) -> Dict[str, float]:
        """
        Get the importance of different ranking factors.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet")

        # Feature names
        feature_names = [
            # Content features
            "word_count",
            "keyword_density",
            "readability_score",
            "headings_count",
            "images_count",
            "internal_links",
            "external_links",
            "keyword_in_title",
            "keyword_in_headings",
            "keyword_in_first_paragraph",
            # Backlink features
            "backlink_count",
            "referring_domains",
            "domain_authority",
            "page_authority",
            "dofollow_ratio",
            # Technical features
            "page_speed_mobile",
            "page_speed_desktop",
            "is_https",
            "is_mobile_friendly",
            "has_structured_data",
        ]

        # Get feature importances
        importances = self.model.feature_importances_

        # Create dictionary mapping feature names to importances
        importance_dict = dict(zip(feature_names, importances))

        # Sort by importance (descending)
        importance_dict = {
            k: v for k, v in sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)
        }

        return importance_dict

    def save_model(self, model_path: Optional[str] = None) -> str:
        """
        Save the trained model to disk.

        Args:
            model_path: Optional path to save the model to

        Returns:
            Path where the model was saved
        """
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        if model_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = os.path.join(MODELS_DIR, f"ranking_prediction_model_{timestamp}.pkl")

        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # Save model components
        with open(model_path, "wb") as f:
            pickle.dump({"model": self.model, "scaler": self.scaler}, f)

        print(f"Model saved to {model_path}")
        return model_path

    def load_model(self, model_path: str) -> None:
        """
        Load a trained model from disk.

        Args:
            model_path: Path to the saved model
        """
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.is_trained = True

        print(f"Model loaded from {model_path}")
