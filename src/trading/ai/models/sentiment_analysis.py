"""
Sentiment analysis for trading applications.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class SentimentAnalyzer:
    """
    Sentiment analysis for news and social media data.

    Features:
    - Text preprocessing
    - Sentiment scoring
    - Financial keyword detection
    - Real-time sentiment tracking
    - Sentiment aggregation
    """

    def __init__(self, name: str = "SentimentAnalyzer"):
        self.name = name
        self.logger = logging.getLogger(f"SentimentAnalyzer.{name}")

        # Sentiment lexicons
        self.positive_words = {
            "bullish",
            "buy",
            "strong",
            "growth",
            "profit",
            "gain",
            "rise",
            "up",
            "positive",
            "good",
            "excellent",
            "outperform",
            "beat",
            "exceed",
            "upgrade",
            "rally",
            "surge",
            "boom",
            "optimistic",
            "confident",
        }

        self.negative_words = {
            "bearish",
            "sell",
            "weak",
            "decline",
            "loss",
            "fall",
            "down",
            "negative",
            "bad",
            "poor",
            "underperform",
            "miss",
            "disappoint",
            "downgrade",
            "crash",
            "plunge",
            "recession",
            "pessimistic",
            "concern",
        }

        self.financial_keywords = {
            "earnings",
            "revenue",
            "profit",
            "eps",
            "guidance",
            "forecast",
            "dividend",
            "buyback",
            "merger",
            "acquisition",
            "ipo",
            "split",
            "fed",
            "interest",
            "rate",
            "inflation",
            "gdp",
            "unemployment",
        }

        # Sentiment history
        self.sentiment_history: Dict[str, List[Dict]] = {}

        # Performance tracking
        self.analysis_count = 0

    def analyze_text(self, text: str, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze
            symbol: Optional symbol for context

        Returns:
            Sentiment analysis results
        """
        try:
            # Preprocess text
            processed_text = self._preprocess_text(text)
            words = processed_text.split()

            # Calculate sentiment scores
            positive_score = self._calculate_positive_score(words)
            negative_score = self._calculate_negative_score(words)
            financial_relevance = self._calculate_financial_relevance(words)

            # Overall sentiment
            net_sentiment = positive_score - negative_score
            sentiment_label = self._get_sentiment_label(net_sentiment)

            # Confidence based on word count and financial relevance
            confidence = min(1.0, (len(words) / 50) * financial_relevance)

            result = {
                "timestamp": datetime.utcnow(),
                "text": text[:200] + "..." if len(text) > 200 else text,
                "symbol": symbol,
                "sentiment_score": net_sentiment,
                "sentiment_label": sentiment_label,
                "positive_score": positive_score,
                "negative_score": negative_score,
                "financial_relevance": financial_relevance,
                "confidence": confidence,
                "word_count": len(words),
                "keywords_found": self._extract_keywords(words),
            }

            # Store in history
            if symbol:
                if symbol not in self.sentiment_history:
                    self.sentiment_history[symbol] = []
                self.sentiment_history[symbol].append(result)

                # Keep only recent history
                if len(self.sentiment_history[symbol]) > 1000:
                    self.sentiment_history[symbol] = self.sentiment_history[symbol][-1000:]

            self.analysis_count += 1
            return result

        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {str(e)}")
            return {
                "timestamp": datetime.utcnow(),
                "text": text,
                "symbol": symbol,
                "sentiment_score": 0.0,
                "sentiment_label": "neutral",
                "confidence": 0.0,
                "error": str(e),
            }

    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis."""
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        # Remove special characters but keep spaces
        text = re.sub(r"[^a-zA-Z\s]", " ", text)

        # Remove extra whitespace
        text = " ".join(text.split())

        return text

    def _calculate_positive_score(self, words: List[str]) -> float:
        """Calculate positive sentiment score."""
        positive_count = sum(1 for word in words if word in self.positive_words)
        return positive_count / max(1, len(words))

    def _calculate_negative_score(self, words: List[str]) -> float:
        """Calculate negative sentiment score."""
        negative_count = sum(1 for word in words if word in self.negative_words)
        return negative_count / max(1, len(words))

    def _calculate_financial_relevance(self, words: List[str]) -> float:
        """Calculate financial relevance score."""
        financial_count = sum(1 for word in words if word in self.financial_keywords)
        return min(1.0, financial_count / max(1, len(words)) * 10)

    def _get_sentiment_label(self, sentiment_score: float) -> str:
        """Convert sentiment score to label."""
        if sentiment_score > 0.02:
            return "positive"
        elif sentiment_score < -0.02:
            return "negative"
        else:
            return "neutral"

    def _extract_keywords(self, words: List[str]) -> List[str]:
        """Extract relevant keywords from text."""
        keywords = []

        # Financial keywords
        keywords.extend([word for word in words if word in self.financial_keywords])

        # Sentiment keywords
        keywords.extend(
            [word for word in words if word in self.positive_words or word in self.negative_words]
        )

        return list(set(keywords))

    def analyze_batch(self, texts: List[str], symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Analyze sentiment for multiple texts."""
        return [self.analyze_text(text, symbol) for text in texts]

    def get_aggregated_sentiment(
        self, symbol: str, time_window_hours: int = 24
    ) -> Optional[Dict[str, Any]]:
        """Get aggregated sentiment for a symbol over time window."""
        try:
            if symbol not in self.sentiment_history:
                return None

            # Filter by time window
            cutoff_time = datetime.utcnow() - pd.Timedelta(hours=time_window_hours)
            recent_sentiments = [
                s for s in self.sentiment_history[symbol] if s["timestamp"] >= cutoff_time
            ]

            if not recent_sentiments:
                return None

            # Calculate aggregated metrics
            sentiment_scores = [s["sentiment_score"] for s in recent_sentiments]
            confidences = [s["confidence"] for s in recent_sentiments]

            # Weighted average by confidence
            if sum(confidences) > 0:
                weighted_sentiment = sum(
                    score * conf for score, conf in zip(sentiment_scores, confidences)
                ) / sum(confidences)
            else:
                weighted_sentiment = np.mean(sentiment_scores)

            # Count by sentiment labels
            labels = [s["sentiment_label"] for s in recent_sentiments]
            label_counts = {
                "positive": labels.count("positive"),
                "negative": labels.count("negative"),
                "neutral": labels.count("neutral"),
            }

            return {
                "symbol": symbol,
                "time_window_hours": time_window_hours,
                "timestamp": datetime.utcnow(),
                "total_articles": len(recent_sentiments),
                "weighted_sentiment": weighted_sentiment,
                "average_sentiment": np.mean(sentiment_scores),
                "sentiment_std": np.std(sentiment_scores),
                "average_confidence": np.mean(confidences),
                "label_distribution": label_counts,
                "dominant_sentiment": max(label_counts, key=label_counts.get),
            }

        except Exception as e:
            self.logger.error(f"Error calculating aggregated sentiment: {str(e)}")
            return None

    def get_sentiment_trend(self, symbol: str, periods: int = 24) -> Optional[List[Dict[str, Any]]]:
        """Get sentiment trend over time periods."""
        try:
            if symbol not in self.sentiment_history:
                return None

            # Group sentiments by hour
            hourly_sentiments = {}

            for sentiment in self.sentiment_history[symbol]:
                hour_key = sentiment["timestamp"].replace(minute=0, second=0, microsecond=0)

                if hour_key not in hourly_sentiments:
                    hourly_sentiments[hour_key] = []

                hourly_sentiments[hour_key].append(sentiment)

            # Calculate hourly averages
            trend_data = []

            for hour, sentiments in sorted(hourly_sentiments.items()):
                if sentiments:
                    avg_sentiment = np.mean([s["sentiment_score"] for s in sentiments])
                    avg_confidence = np.mean([s["confidence"] for s in sentiments])

                    trend_data.append(
                        {
                            "timestamp": hour,
                            "sentiment_score": avg_sentiment,
                            "confidence": avg_confidence,
                            "article_count": len(sentiments),
                        }
                    )

            # Return most recent periods
            return trend_data[-periods:] if len(trend_data) >= periods else trend_data

        except Exception as e:
            self.logger.error(f"Error calculating sentiment trend: {str(e)}")
            return None

    def get_sentiment_signal(
        self, symbol: str, threshold: float = 0.05
    ) -> Optional[Dict[str, Any]]:
        """Generate trading signal based on sentiment."""
        try:
            aggregated = self.get_aggregated_sentiment(symbol, time_window_hours=4)

            if not aggregated:
                return None

            sentiment_score = aggregated["weighted_sentiment"]
            confidence = aggregated["average_confidence"]

            # Generate signal
            if sentiment_score > threshold and confidence > 0.3:
                signal = "BUY"
                strength = min(1.0, sentiment_score * confidence * 2)
            elif sentiment_score < -threshold and confidence > 0.3:
                signal = "SELL"
                strength = min(1.0, abs(sentiment_score) * confidence * 2)
            else:
                signal = "HOLD"
                strength = 0.0

            return {
                "symbol": symbol,
                "timestamp": datetime.utcnow(),
                "signal": signal,
                "strength": strength,
                "sentiment_score": sentiment_score,
                "confidence": confidence,
                "reasoning": f"Sentiment: {sentiment_score:.3f}, Confidence: {confidence:.3f}",
            }

        except Exception as e:
            self.logger.error(f"Error generating sentiment signal: {str(e)}")
            return None

    def get_analyzer_stats(self) -> Dict[str, Any]:
        """Get sentiment analyzer statistics."""
        total_sentiments = sum(len(history) for history in self.sentiment_history.values())

        return {
            "analysis_count": self.analysis_count,
            "symbols_tracked": len(self.sentiment_history),
            "total_sentiments_stored": total_sentiments,
            "positive_words_count": len(self.positive_words),
            "negative_words_count": len(self.negative_words),
            "financial_keywords_count": len(self.financial_keywords),
        }
