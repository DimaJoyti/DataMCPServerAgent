"""
Trend Analyzer

Advanced trend analysis and predictive insights for monitoring data.
"""

import json
import logging
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import statistics

logger = logging.getLogger(__name__)


@dataclass
class TrendData:
    """Trend analysis data"""
    metric_name: str
    direction: str  # "improving", "declining", "stable"
    slope: float
    confidence: float
    prediction_7d: float
    prediction_30d: float
    volatility: float
    anomalies: List[Dict[str, Any]]


@dataclass
class InsightRecommendation:
    """Data-driven recommendation"""
    priority: str  # "high", "medium", "low"
    category: str
    title: str
    description: str
    impact: str
    effort: str
    data_points: List[str]


class TrendAnalyzer:
    """Analyzes trends and generates data-driven insights"""
    
    def __init__(self, data_directory: str):
        self.data_directory = Path(data_directory)
        self.min_data_points = 5
        self.anomaly_threshold = 2.0  # Standard deviations
    
    async def analyze_trends(self, metrics_history: List) -> Dict[str, Any]:
        """Analyze trends from metrics history"""
        try:
            if len(metrics_history) < self.min_data_points:
                return {"error": "Insufficient data for trend analysis"}
            
            trends = {}
            insights = []
            recommendations = []
            
            # Extract time series data for each metric
            metric_series = self._extract_metric_series(metrics_history)
            
            # Analyze each metric
            for metric_name, values in metric_series.items():
                if len(values) >= self.min_data_points:
                    trend_data = self._analyze_metric_trend(metric_name, values)
                    trends[metric_name] = trend_data
                    
                    # Generate insights
                    metric_insights = self._generate_metric_insights(metric_name, trend_data, values)
                    insights.extend(metric_insights)
            
            # Generate system-wide recommendations
            system_recommendations = self._generate_system_recommendations(trends, metrics_history)
            recommendations.extend(system_recommendations)
            
            # Detect patterns and correlations
            patterns = self._detect_patterns(metric_series)
            
            # Generate predictive alerts
            predictive_alerts = self._generate_predictive_alerts(trends)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "analysis_period": {
                    "start": metrics_history[0].timestamp.isoformat(),
                    "end": metrics_history[-1].timestamp.isoformat(),
                    "data_points": len(metrics_history)
                },
                "trends": {k: self._trend_to_dict(v) for k, v in trends.items()},
                "insights": insights,
                "recommendations": [self._recommendation_to_dict(r) for r in recommendations],
                "patterns": patterns,
                "predictive_alerts": predictive_alerts,
                "summary": self._generate_trend_summary(trends)
            }
            
        except Exception as e:
            logger.error(f"❌ Trend analysis error: {e}")
            return {"error": str(e)}
    
    def _extract_metric_series(self, metrics_history: List) -> Dict[str, List[Tuple[datetime, float]]]:
        """Extract time series data for each metric"""
        metric_series = {}
        
        for snapshot in metrics_history:
            for metric_name, metric_snapshot in snapshot.metrics.items():
                if metric_name not in metric_series:
                    metric_series[metric_name] = []
                
                metric_series[metric_name].append((
                    metric_snapshot.timestamp,
                    metric_snapshot.value
                ))
        
        # Sort by timestamp
        for metric_name in metric_series:
            metric_series[metric_name].sort(key=lambda x: x[0])
        
        return metric_series
    
    def _analyze_metric_trend(self, metric_name: str, values: List[Tuple[datetime, float]]) -> TrendData:
        """Analyze trend for a specific metric"""
        try:
            # Extract values and timestamps
            timestamps = [v[0] for v in values]
            metric_values = [v[1] for v in values]
            
            # Convert timestamps to numeric values (hours since first measurement)
            base_time = timestamps[0]
            time_hours = [(t - base_time).total_seconds() / 3600 for t in timestamps]
            
            # Calculate linear regression
            slope, intercept, confidence = self._linear_regression(time_hours, metric_values)
            
            # Determine trend direction
            direction = self._determine_trend_direction(slope, metric_name)
            
            # Calculate volatility
            volatility = self._calculate_volatility(metric_values)
            
            # Detect anomalies
            anomalies = self._detect_anomalies(timestamps, metric_values)
            
            # Make predictions
            current_time_hours = time_hours[-1]
            prediction_7d = slope * (current_time_hours + 168) + intercept  # 168 hours = 7 days
            prediction_30d = slope * (current_time_hours + 720) + intercept  # 720 hours = 30 days
            
            # Ensure predictions are within reasonable bounds
            prediction_7d = max(0, min(100, prediction_7d))
            prediction_30d = max(0, min(100, prediction_30d))
            
            return TrendData(
                metric_name=metric_name,
                direction=direction,
                slope=slope,
                confidence=confidence,
                prediction_7d=prediction_7d,
                prediction_30d=prediction_30d,
                volatility=volatility,
                anomalies=anomalies
            )
            
        except Exception as e:
            logger.error(f"❌ Trend analysis error for {metric_name}: {e}")
            return TrendData(
                metric_name=metric_name,
                direction="unknown",
                slope=0.0,
                confidence=0.0,
                prediction_7d=0.0,
                prediction_30d=0.0,
                volatility=0.0,
                anomalies=[]
            )
    
    def _linear_regression(self, x: List[float], y: List[float]) -> Tuple[float, float, float]:
        """Calculate linear regression"""
        try:
            n = len(x)
            if n < 2:
                return 0.0, 0.0, 0.0
            
            # Calculate means
            x_mean = sum(x) / n
            y_mean = sum(y) / n
            
            # Calculate slope and intercept
            numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
            
            if denominator == 0:
                return 0.0, y_mean, 0.0
            
            slope = numerator / denominator
            intercept = y_mean - slope * x_mean
            
            # Calculate R-squared (confidence)
            y_pred = [slope * x[i] + intercept for i in range(n)]
            ss_res = sum((y[i] - y_pred[i]) ** 2 for i in range(n))
            ss_tot = sum((y[i] - y_mean) ** 2 for i in range(n))
            
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            confidence = max(0, min(1, r_squared))
            
            return slope, intercept, confidence
            
        except Exception as e:
            logger.error(f"❌ Linear regression error: {e}")
            return 0.0, 0.0, 0.0
    
    def _determine_trend_direction(self, slope: float, metric_name: str) -> str:
        """Determine trend direction based on slope and metric type"""
        threshold = 0.01  # Minimum slope to consider significant
        
        if abs(slope) < threshold:
            return "stable"
        
        # For security risk, higher values are worse
        if metric_name == "security_risk":
            return "declining" if slope > 0 else "improving"
        else:
            # For other metrics, higher values are better
            return "improving" if slope > 0 else "declining"
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility (standard deviation)"""
        try:
            if len(values) < 2:
                return 0.0
            
            return statistics.stdev(values)
            
        except Exception:
            return 0.0
    
    def _detect_anomalies(self, timestamps: List[datetime], values: List[float]) -> List[Dict[str, Any]]:
        """Detect anomalies in the data"""
        try:
            if len(values) < 3:
                return []
            
            mean_val = statistics.mean(values)
            std_val = statistics.stdev(values)
            
            anomalies = []
            for i, (timestamp, value) in enumerate(zip(timestamps, values)):
                z_score = abs(value - mean_val) / std_val if std_val > 0 else 0
                
                if z_score > self.anomaly_threshold:
                    anomalies.append({
                        "timestamp": timestamp.isoformat(),
                        "value": value,
                        "z_score": z_score,
                        "type": "outlier"
                    })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"❌ Anomaly detection error: {e}")
            return []
    
    def _generate_metric_insights(self, metric_name: str, trend_data: TrendData, 
                                values: List[Tuple[datetime, float]]) -> List[str]:
        """Generate insights for a specific metric"""
        insights = []
        
        try:
            # Trend insights
            if trend_data.direction == "improving":
                insights.append(f"📈 {metric_name.replace('_', ' ').title()} is improving with {trend_data.confidence:.1%} confidence")
            elif trend_data.direction == "declining":
                insights.append(f"📉 {metric_name.replace('_', ' ').title()} is declining with {trend_data.confidence:.1%} confidence")
            else:
                insights.append(f"📊 {metric_name.replace('_', ' ').title()} remains stable")
            
            # Volatility insights
            if trend_data.volatility > 10:
                insights.append(f"⚡ {metric_name.replace('_', ' ').title()} shows high volatility ({trend_data.volatility:.1f})")
            
            # Anomaly insights
            if trend_data.anomalies:
                insights.append(f"🔍 {len(trend_data.anomalies)} anomalies detected in {metric_name.replace('_', ' ').title()}")
            
            # Prediction insights
            current_value = values[-1][1]
            if abs(trend_data.prediction_7d - current_value) > 5:
                direction = "increase" if trend_data.prediction_7d > current_value else "decrease"
                insights.append(f"🔮 {metric_name.replace('_', ' ').title()} predicted to {direction} to {trend_data.prediction_7d:.1f} in 7 days")
            
        except Exception as e:
            logger.error(f"❌ Insight generation error for {metric_name}: {e}")
        
        return insights
    
    def _generate_system_recommendations(self, trends: Dict[str, TrendData], 
                                       metrics_history: List) -> List[InsightRecommendation]:
        """Generate system-wide recommendations"""
        recommendations = []
        
        try:
            # Security recommendations
            if "security_risk" in trends:
                security_trend = trends["security_risk"]
                if security_trend.direction == "declining":  # Getting worse
                    recommendations.append(InsightRecommendation(
                        priority="high",
                        category="security",
                        title="Security Risk Increasing",
                        description="Security risk metrics show a declining trend. Immediate action required.",
                        impact="High - potential security vulnerabilities",
                        effort="Medium - security review and fixes",
                        data_points=[f"Trend slope: {security_trend.slope:.3f}", f"Confidence: {security_trend.confidence:.1%}"]
                    ))
            
            # Code quality recommendations
            if "code_quality" in trends:
                quality_trend = trends["code_quality"]
                if quality_trend.direction == "declining":
                    recommendations.append(InsightRecommendation(
                        priority="medium",
                        category="quality",
                        title="Code Quality Declining",
                        description="Code quality metrics show a declining trend. Consider implementing stricter quality gates.",
                        impact="Medium - technical debt accumulation",
                        effort="Low - automated quality checks",
                        data_points=[f"Trend slope: {quality_trend.slope:.3f}", f"Prediction 7d: {quality_trend.prediction_7d:.1f}"]
                    ))
            
            # Test health recommendations
            if "test_health" in trends:
                test_trend = trends["test_health"]
                if test_trend.direction == "declining":
                    recommendations.append(InsightRecommendation(
                        priority="medium",
                        category="testing",
                        title="Test Health Declining",
                        description="Test health metrics show a declining trend. Focus on improving test coverage and performance.",
                        impact="Medium - reduced confidence in releases",
                        effort="Medium - test improvements",
                        data_points=[f"Volatility: {test_trend.volatility:.1f}", f"Anomalies: {len(test_trend.anomalies)}"]
                    ))
            
            # CI/CD recommendations
            if "cicd_health" in trends:
                cicd_trend = trends["cicd_health"]
                if cicd_trend.volatility > 15:
                    recommendations.append(InsightRecommendation(
                        priority="medium",
                        category="cicd",
                        title="CI/CD Performance Unstable",
                        description="CI/CD metrics show high volatility. Consider pipeline optimization.",
                        impact="Medium - unpredictable build times",
                        effort="Medium - pipeline optimization",
                        data_points=[f"Volatility: {cicd_trend.volatility:.1f}"]
                    ))
            
            # Cross-metric recommendations
            declining_metrics = [name for name, trend in trends.items() 
                                if trend.direction == "declining" and trend.confidence > 0.5]
            
            if len(declining_metrics) >= 2:
                recommendations.append(InsightRecommendation(
                    priority="high",
                    category="system",
                    title="Multiple Metrics Declining",
                    description=f"Multiple metrics ({', '.join(declining_metrics)}) show declining trends. System-wide review recommended.",
                    impact="High - overall system health",
                    effort="High - comprehensive review",
                    data_points=[f"Declining metrics: {len(declining_metrics)}"]
                ))
            
        except Exception as e:
            logger.error(f"❌ Recommendation generation error: {e}")
        
        return recommendations
    
    def _detect_patterns(self, metric_series: Dict[str, List[Tuple[datetime, float]]]) -> Dict[str, Any]:
        """Detect patterns and correlations between metrics"""
        patterns = {}
        
        try:
            # Correlation analysis
            correlations = {}
            metric_names = list(metric_series.keys())
            
            for i, metric1 in enumerate(metric_names):
                for metric2 in metric_names[i+1:]:
                    correlation = self._calculate_correlation(metric_series[metric1], metric_series[metric2])
                    if abs(correlation) > 0.5:  # Significant correlation
                        correlations[f"{metric1}_vs_{metric2}"] = correlation
            
            patterns["correlations"] = correlations
            
            # Cyclical patterns (simplified)
            cyclical_metrics = []
            for metric_name, values in metric_series.items():
                if self._detect_cyclical_pattern(values):
                    cyclical_metrics.append(metric_name)
            
            patterns["cyclical_metrics"] = cyclical_metrics
            
        except Exception as e:
            logger.error(f"❌ Pattern detection error: {e}")
        
        return patterns
    
    def _calculate_correlation(self, series1: List[Tuple[datetime, float]], 
                             series2: List[Tuple[datetime, float]]) -> float:
        """Calculate correlation between two time series"""
        try:
            # Align time series by timestamp
            values1, values2 = [], []
            
            # Create dictionaries for faster lookup
            dict1 = {t: v for t, v in series1}
            dict2 = {t: v for t, v in series2}
            
            # Find common timestamps
            common_times = set(dict1.keys()) & set(dict2.keys())
            
            if len(common_times) < 3:
                return 0.0
            
            for t in sorted(common_times):
                values1.append(dict1[t])
                values2.append(dict2[t])
            
            # Calculate Pearson correlation
            if len(values1) < 2:
                return 0.0
            
            mean1 = sum(values1) / len(values1)
            mean2 = sum(values2) / len(values2)
            
            numerator = sum((v1 - mean1) * (v2 - mean2) for v1, v2 in zip(values1, values2))
            denominator1 = sum((v1 - mean1) ** 2 for v1 in values1)
            denominator2 = sum((v2 - mean2) ** 2 for v2 in values2)
            
            if denominator1 == 0 or denominator2 == 0:
                return 0.0
            
            correlation = numerator / (denominator1 * denominator2) ** 0.5
            return correlation
            
        except Exception as e:
            logger.error(f"❌ Correlation calculation error: {e}")
            return 0.0
    
    def _detect_cyclical_pattern(self, values: List[Tuple[datetime, float]]) -> bool:
        """Detect if a metric shows cyclical patterns"""
        try:
            if len(values) < 10:
                return False
            
            # Simple cyclical detection based on variance in different time periods
            # This is a simplified approach - more sophisticated methods could be used
            metric_values = [v[1] for v in values]
            
            # Check if there's significant variation
            if statistics.stdev(metric_values) < 1:
                return False
            
            # Look for repeating patterns (simplified)
            # In a real implementation, you might use FFT or other signal processing techniques
            return False  # Placeholder
            
        except Exception:
            return False
    
    def _generate_predictive_alerts(self, trends: Dict[str, TrendData]) -> List[Dict[str, Any]]:
        """Generate predictive alerts based on trends"""
        alerts = []
        
        try:
            for metric_name, trend in trends.items():
                # Check if metric is predicted to cross critical thresholds
                if metric_name == "security_risk":
                    if trend.prediction_7d > 80 and trend.confidence > 0.6:
                        alerts.append({
                            "type": "predictive",
                            "severity": "warning",
                            "metric": metric_name,
                            "message": f"Security risk predicted to reach critical levels ({trend.prediction_7d:.1f}) within 7 days",
                            "confidence": trend.confidence
                        })
                
                elif metric_name in ["code_quality", "test_health", "documentation_health"]:
                    if trend.prediction_7d < 60 and trend.confidence > 0.6:
                        alerts.append({
                            "type": "predictive",
                            "severity": "warning",
                            "metric": metric_name,
                            "message": f"{metric_name.replace('_', ' ').title()} predicted to drop below 60 ({trend.prediction_7d:.1f}) within 7 days",
                            "confidence": trend.confidence
                        })
                
                # High volatility alerts
                if trend.volatility > 20:
                    alerts.append({
                        "type": "volatility",
                        "severity": "info",
                        "metric": metric_name,
                        "message": f"{metric_name.replace('_', ' ').title()} showing high volatility ({trend.volatility:.1f})",
                        "confidence": 1.0
                    })
            
        except Exception as e:
            logger.error(f"❌ Predictive alert generation error: {e}")
        
        return alerts
    
    def _generate_trend_summary(self, trends: Dict[str, TrendData]) -> Dict[str, Any]:
        """Generate overall trend summary"""
        try:
            improving_count = len([t for t in trends.values() if t.direction == "improving"])
            declining_count = len([t for t in trends.values() if t.direction == "declining"])
            stable_count = len([t for t in trends.values() if t.direction == "stable"])
            
            avg_confidence = sum(t.confidence for t in trends.values()) / len(trends) if trends else 0
            high_volatility_count = len([t for t in trends.values() if t.volatility > 15])
            
            return {
                "total_metrics": len(trends),
                "improving_metrics": improving_count,
                "declining_metrics": declining_count,
                "stable_metrics": stable_count,
                "average_confidence": avg_confidence,
                "high_volatility_metrics": high_volatility_count,
                "overall_trend": "improving" if improving_count > declining_count else "declining" if declining_count > improving_count else "stable"
            }
            
        except Exception as e:
            logger.error(f"❌ Trend summary error: {e}")
            return {}
    
    def _trend_to_dict(self, trend: TrendData) -> Dict[str, Any]:
        """Convert TrendData to dictionary"""
        return {
            "metric_name": trend.metric_name,
            "direction": trend.direction,
            "slope": trend.slope,
            "confidence": trend.confidence,
            "prediction_7d": trend.prediction_7d,
            "prediction_30d": trend.prediction_30d,
            "volatility": trend.volatility,
            "anomalies": trend.anomalies
        }
    
    def _recommendation_to_dict(self, rec: InsightRecommendation) -> Dict[str, Any]:
        """Convert InsightRecommendation to dictionary"""
        return {
            "priority": rec.priority,
            "category": rec.category,
            "title": rec.title,
            "description": rec.description,
            "impact": rec.impact,
            "effort": rec.effort,
            "data_points": rec.data_points
        }
