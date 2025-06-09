"""
Learning Service - Machine learning and continuous improvement for brand agents.
Analyzes conversation patterns and optimizes agent responses over time.
"""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

from app.core.logging import LoggerMixin, get_logger
from app.domain.models.analytics import ConversationAnalytics, MetricType, MetricValue
from app.domain.models.base import DomainService
from app.domain.models.brand_agent import BrandAgent, BrandPersonality
from app.domain.models.conversation import ConversationMessage, IntentType, SentimentType

logger = get_logger(__name__)


class LearningInsight:
    """A learning insight derived from conversation analysis."""
    
    def __init__(
        self,
        insight_type: str,
        title: str,
        description: str,
        confidence: float,
        impact_score: float,
        recommendations: List[str],
        data_points: int,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.id = str(uuid4())
        self.insight_type = insight_type
        self.title = title
        self.description = description
        self.confidence = confidence
        self.impact_score = impact_score
        self.recommendations = recommendations
        self.data_points = data_points
        self.metadata = metadata or {}
        self.created_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "insight_type": self.insight_type,
            "title": self.title,
            "description": self.description,
            "confidence": self.confidence,
            "impact_score": self.impact_score,
            "recommendations": self.recommendations,
            "data_points": self.data_points,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }


class ResponsePattern:
    """A pattern identified in agent responses."""
    
    def __init__(
        self,
        pattern_type: str,
        trigger_conditions: Dict[str, Any],
        response_template: str,
        success_rate: float,
        usage_count: int,
        avg_satisfaction: float,
    ):
        self.id = str(uuid4())
        self.pattern_type = pattern_type
        self.trigger_conditions = trigger_conditions
        self.response_template = response_template
        self.success_rate = success_rate
        self.usage_count = usage_count
        self.avg_satisfaction = avg_satisfaction
        self.created_at = datetime.now(timezone.utc)
        self.last_used = datetime.now(timezone.utc)
    
    def matches_conditions(self, context: Dict[str, Any]) -> bool:
        """Check if context matches trigger conditions."""
        for key, expected_value in self.trigger_conditions.items():
            if key not in context:
                return False
            
            actual_value = context[key]
            
            # Handle different comparison types
            if isinstance(expected_value, dict):
                if "min" in expected_value and actual_value < expected_value["min"]:
                    return False
                if "max" in expected_value and actual_value > expected_value["max"]:
                    return False
                if "equals" in expected_value and actual_value != expected_value["equals"]:
                    return False
                if "contains" in expected_value and expected_value["contains"] not in str(actual_value):
                    return False
            else:
                if actual_value != expected_value:
                    return False
        
        return True


class LearningService(DomainService, LoggerMixin):
    """Service for machine learning and continuous improvement of brand agents."""
    
    def __init__(self):
        super().__init__()
        self._response_patterns: Dict[str, List[ResponsePattern]] = {}
        self._learning_insights: List[LearningInsight] = []
        self._personality_adaptations: Dict[str, Dict[str, Any]] = {}
    
    async def analyze_conversation_patterns(
        self, 
        agent_id: str, 
        conversations: List[ConversationAnalytics],
        time_window_days: int = 30
    ) -> List[LearningInsight]:
        """Analyze conversation patterns to generate learning insights."""
        insights = []
        
        if not conversations:
            return insights
        
        # Analyze response time patterns
        response_time_insight = await self._analyze_response_time_patterns(agent_id, conversations)
        if response_time_insight:
            insights.append(response_time_insight)
        
        # Analyze satisfaction patterns
        satisfaction_insight = await self._analyze_satisfaction_patterns(agent_id, conversations)
        if satisfaction_insight:
            insights.append(satisfaction_insight)
        
        # Analyze escalation patterns
        escalation_insight = await self._analyze_escalation_patterns(agent_id, conversations)
        if escalation_insight:
            insights.append(escalation_insight)
        
        # Analyze topic effectiveness
        topic_insight = await self._analyze_topic_effectiveness(agent_id, conversations)
        if topic_insight:
            insights.append(topic_insight)
        
        # Analyze knowledge usage patterns
        knowledge_insight = await self._analyze_knowledge_usage(agent_id, conversations)
        if knowledge_insight:
            insights.append(knowledge_insight)
        
        # Store insights
        self._learning_insights.extend(insights)
        
        self.logger.info(f"Generated {len(insights)} learning insights for agent {agent_id}")
        return insights
    
    async def optimize_response_strategy(
        self, 
        agent: BrandAgent, 
        conversation_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Optimize response strategy based on learned patterns."""
        optimizations = {
            "suggested_tone": None,
            "suggested_length": None,
            "recommended_knowledge": [],
            "confidence_boost": 0.0,
            "pattern_match": None,
        }
        
        # Find matching response patterns
        agent_patterns = self._response_patterns.get(agent.id, [])
        best_pattern = None
        best_score = 0.0
        
        for pattern in agent_patterns:
            if pattern.matches_conditions(conversation_context):
                # Score pattern based on success rate and usage
                score = pattern.success_rate * 0.7 + min(pattern.usage_count / 100, 1.0) * 0.3
                if score > best_score:
                    best_score = score
                    best_pattern = pattern
        
        if best_pattern:
            optimizations["pattern_match"] = {
                "id": best_pattern.id,
                "type": best_pattern.pattern_type,
                "success_rate": best_pattern.success_rate,
                "template": best_pattern.response_template,
            }
            optimizations["confidence_boost"] = best_pattern.success_rate - 0.5
        
        # Analyze context for optimizations
        user_sentiment = conversation_context.get("user_sentiment")
        if user_sentiment == "frustrated":
            optimizations["suggested_tone"] = "empathetic"
            optimizations["suggested_length"] = "concise"
        elif user_sentiment == "confused":
            optimizations["suggested_tone"] = "explanatory"
            optimizations["suggested_length"] = "detailed"
        
        # Recommend knowledge based on intent
        user_intent = conversation_context.get("user_intent")
        if user_intent:
            optimizations["recommended_knowledge"] = await self._get_effective_knowledge_for_intent(
                agent.id, user_intent
            )
        
        return optimizations
    
    async def adapt_personality(
        self, 
        agent: BrandAgent, 
        performance_feedback: Dict[str, float]
    ) -> Optional[BrandPersonality]:
        """Adapt agent personality based on performance feedback."""
        current_personality = agent.personality
        adaptations = self._personality_adaptations.get(agent.id, {})
        
        # Analyze performance metrics
        satisfaction = performance_feedback.get("avg_satisfaction", 0.0)
        resolution_rate = performance_feedback.get("resolution_rate", 0.0)
        escalation_rate = performance_feedback.get("escalation_rate", 0.0)
        
        suggested_changes = {}
        
        # Satisfaction-based adaptations
        if satisfaction < 3.0:  # Low satisfaction
            if current_personality.tone != "empathetic":
                suggested_changes["tone"] = "empathetic"
            if current_personality.formality_level != "formal":
                suggested_changes["formality_level"] = "formal"
        elif satisfaction > 4.5:  # High satisfaction
            # Keep current successful approach
            pass
        
        # Resolution rate adaptations
        if resolution_rate < 0.7:  # Low resolution rate
            if current_personality.response_length != "detailed":
                suggested_changes["response_length"] = "detailed"
            if "helpful" not in current_personality.traits:
                new_traits = current_personality.traits + ["helpful"]
                suggested_changes["traits"] = new_traits
        
        # Escalation rate adaptations
        if escalation_rate > 0.3:  # High escalation rate
            if current_personality.tone != "calm":
                suggested_changes["tone"] = "calm"
            if "patient" not in current_personality.traits:
                new_traits = current_personality.traits + ["patient"]
                suggested_changes["traits"] = new_traits
        
        if suggested_changes:
            # Create adapted personality
            adapted_personality = BrandPersonality(
                traits=suggested_changes.get("traits", current_personality.traits),
                tone=suggested_changes.get("tone", current_personality.tone),
                communication_style=current_personality.communication_style,
                response_length=suggested_changes.get("response_length", current_personality.response_length),
                formality_level=suggested_changes.get("formality_level", current_personality.formality_level),
                emoji_usage=current_personality.emoji_usage,
                custom_phrases=current_personality.custom_phrases,
            )
            
            # Store adaptation
            adaptations[datetime.now().isoformat()] = {
                "changes": suggested_changes,
                "reason": "performance_optimization",
                "metrics": performance_feedback,
            }
            self._personality_adaptations[agent.id] = adaptations
            
            self.logger.info(f"Adapted personality for agent {agent.id}: {suggested_changes}")
            return adapted_personality
        
        return None
    
    async def learn_from_feedback(
        self, 
        agent_id: str, 
        conversation_id: str, 
        user_feedback: Dict[str, Any]
    ) -> None:
        """Learn from user feedback to improve future responses."""
        feedback_type = user_feedback.get("type", "satisfaction")
        rating = user_feedback.get("rating")
        comments = user_feedback.get("comments", "")
        
        # Extract learning signals
        if feedback_type == "satisfaction" and rating:
            if rating >= 4:
                # Positive feedback - reinforce current approach
                await self._reinforce_successful_pattern(agent_id, conversation_id)
            elif rating <= 2:
                # Negative feedback - identify improvement areas
                await self._identify_improvement_areas(agent_id, conversation_id, comments)
        
        # Analyze feedback comments for specific insights
        if comments:
            insights = await self._analyze_feedback_comments(agent_id, comments)
            self._learning_insights.extend(insights)
        
        self.logger.info(f"Processed feedback for agent {agent_id}, conversation {conversation_id}")
    
    async def get_learning_recommendations(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get learning-based recommendations for agent improvement."""
        recommendations = []
        
        # Get recent insights for this agent
        agent_insights = [
            insight for insight in self._learning_insights
            if insight.metadata.get("agent_id") == agent_id
        ]
        
        # Sort by impact score and confidence
        agent_insights.sort(key=lambda x: x.impact_score * x.confidence, reverse=True)
        
        for insight in agent_insights[:5]:  # Top 5 insights
            recommendations.append({
                "type": insight.insight_type,
                "title": insight.title,
                "description": insight.description,
                "confidence": insight.confidence,
                "impact": insight.impact_score,
                "actions": insight.recommendations,
                "priority": "high" if insight.impact_score > 0.8 else "medium" if insight.impact_score > 0.5 else "low",
            })
        
        return recommendations
    
    async def _analyze_response_time_patterns(
        self, 
        agent_id: str, 
        conversations: List[ConversationAnalytics]
    ) -> Optional[LearningInsight]:
        """Analyze response time patterns."""
        response_times = [c.avg_response_time_ms for c in conversations if c.avg_response_time_ms > 0]
        
        if len(response_times) < 10:  # Need sufficient data
            return None
        
        avg_response_time = sum(response_times) / len(response_times)
        
        # Correlate with satisfaction
        satisfaction_by_speed = {"fast": [], "medium": [], "slow": []}
        
        for conv in conversations:
            if conv.avg_response_time_ms > 0 and conv.user_satisfaction:
                if conv.avg_response_time_ms < 2000:
                    satisfaction_by_speed["fast"].append(conv.user_satisfaction)
                elif conv.avg_response_time_ms < 5000:
                    satisfaction_by_speed["medium"].append(conv.user_satisfaction)
                else:
                    satisfaction_by_speed["slow"].append(conv.user_satisfaction)
        
        # Calculate average satisfaction for each speed category
        avg_satisfaction = {}
        for speed, ratings in satisfaction_by_speed.items():
            if ratings:
                avg_satisfaction[speed] = sum(ratings) / len(ratings)
        
        if len(avg_satisfaction) >= 2:
            # Generate insight
            best_speed = max(avg_satisfaction.keys(), key=lambda k: avg_satisfaction[k])
            worst_speed = min(avg_satisfaction.keys(), key=lambda k: avg_satisfaction[k])
            
            satisfaction_diff = avg_satisfaction[best_speed] - avg_satisfaction[worst_speed]
            
            if satisfaction_diff > 0.5:  # Significant difference
                return LearningInsight(
                    insight_type="response_time_optimization",
                    title="Response Time Impact on Satisfaction",
                    description=f"Users are {satisfaction_diff:.1f} points more satisfied with {best_speed} responses",
                    confidence=min(len(response_times) / 50, 1.0),
                    impact_score=satisfaction_diff / 5.0,  # Normalize to 0-1
                    recommendations=[
                        f"Optimize for {best_speed} response times",
                        f"Current average: {avg_response_time:.0f}ms",
                        "Consider response caching for common queries",
                    ],
                    data_points=len(response_times),
                    metadata={"agent_id": agent_id, "avg_response_time": avg_response_time},
                )
        
        return None
    
    async def _analyze_satisfaction_patterns(
        self, 
        agent_id: str, 
        conversations: List[ConversationAnalytics]
    ) -> Optional[LearningInsight]:
        """Analyze satisfaction patterns."""
        satisfaction_data = [
            (c.user_satisfaction, c.topics_discussed, c.sentiment_scores)
            for c in conversations 
            if c.user_satisfaction is not None
        ]
        
        if len(satisfaction_data) < 20:
            return None
        
        # Analyze topic correlation with satisfaction
        topic_satisfaction = {}
        for satisfaction, topics, sentiments in satisfaction_data:
            for topic in topics:
                if topic not in topic_satisfaction:
                    topic_satisfaction[topic] = []
                topic_satisfaction[topic].append(satisfaction)
        
        # Find topics with significant impact
        topic_impact = {}
        for topic, ratings in topic_satisfaction.items():
            if len(ratings) >= 5:  # Minimum data points
                avg_rating = sum(ratings) / len(ratings)
                topic_impact[topic] = avg_rating
        
        if topic_impact:
            best_topic = max(topic_impact.keys(), key=lambda k: topic_impact[k])
            worst_topic = min(topic_impact.keys(), key=lambda k: topic_impact[k])
            
            impact_diff = topic_impact[best_topic] - topic_impact[worst_topic]
            
            if impact_diff > 1.0:  # Significant difference
                return LearningInsight(
                    insight_type="topic_satisfaction_correlation",
                    title="Topic Impact on User Satisfaction",
                    description=f"'{best_topic}' topics lead to {impact_diff:.1f} higher satisfaction than '{worst_topic}'",
                    confidence=min(len(satisfaction_data) / 100, 1.0),
                    impact_score=impact_diff / 5.0,
                    recommendations=[
                        f"Emphasize discussions about '{best_topic}'",
                        f"Improve handling of '{worst_topic}' topics",
                        "Train on successful conversation patterns",
                    ],
                    data_points=len(satisfaction_data),
                    metadata={"agent_id": agent_id, "topic_impact": topic_impact},
                )
        
        return None
    
    async def _analyze_escalation_patterns(
        self, 
        agent_id: str, 
        conversations: List[ConversationAnalytics]
    ) -> Optional[LearningInsight]:
        """Analyze escalation patterns."""
        escalated_conversations = [c for c in conversations if c.escalated]
        total_conversations = len(conversations)
        
        if total_conversations < 20 or len(escalated_conversations) < 3:
            return None
        
        escalation_rate = len(escalated_conversations) / total_conversations
        
        # Analyze common patterns in escalated conversations
        escalation_triggers = {}
        
        for conv in escalated_conversations:
            # Analyze topics
            for topic in conv.topics_discussed:
                escalation_triggers[f"topic:{topic}"] = escalation_triggers.get(f"topic:{topic}", 0) + 1
            
            # Analyze sentiment patterns
            if conv.sentiment_scores:
                avg_sentiment = sum(conv.sentiment_scores) / len(conv.sentiment_scores)
                if avg_sentiment < 0.3:
                    escalation_triggers["negative_sentiment"] = escalation_triggers.get("negative_sentiment", 0) + 1
            
            # Analyze conversation length
            if conv.message_count > 10:
                escalation_triggers["long_conversation"] = escalation_triggers.get("long_conversation", 0) + 1
        
        # Find most common triggers
        if escalation_triggers:
            top_trigger = max(escalation_triggers.keys(), key=lambda k: escalation_triggers[k])
            trigger_frequency = escalation_triggers[top_trigger] / len(escalated_conversations)
            
            if trigger_frequency > 0.5:  # Appears in >50% of escalations
                return LearningInsight(
                    insight_type="escalation_pattern",
                    title="Common Escalation Trigger Identified",
                    description=f"'{top_trigger}' appears in {trigger_frequency:.1%} of escalated conversations",
                    confidence=min(len(escalated_conversations) / 20, 1.0),
                    impact_score=escalation_rate,
                    recommendations=[
                        f"Develop specific strategies for handling '{top_trigger}'",
                        "Create escalation prevention protocols",
                        "Train on de-escalation techniques",
                    ],
                    data_points=len(escalated_conversations),
                    metadata={"agent_id": agent_id, "escalation_triggers": escalation_triggers},
                )
        
        return None
    
    async def _analyze_topic_effectiveness(
        self, 
        agent_id: str, 
        conversations: List[ConversationAnalytics]
    ) -> Optional[LearningInsight]:
        """Analyze topic handling effectiveness."""
        # Implementation would analyze which topics lead to better outcomes
        return None
    
    async def _analyze_knowledge_usage(
        self, 
        agent_id: str, 
        conversations: List[ConversationAnalytics]
    ) -> Optional[LearningInsight]:
        """Analyze knowledge usage patterns."""
        # Implementation would analyze knowledge item effectiveness
        return None
    
    async def _get_effective_knowledge_for_intent(self, agent_id: str, intent: str) -> List[str]:
        """Get most effective knowledge items for a given intent."""
        # Mock implementation
        return ["knowledge_item_1", "knowledge_item_2"]
    
    async def _reinforce_successful_pattern(self, agent_id: str, conversation_id: str) -> None:
        """Reinforce successful conversation patterns."""
        # Implementation would identify and strengthen successful patterns
        pass
    
    async def _identify_improvement_areas(
        self, 
        agent_id: str, 
        conversation_id: str, 
        feedback: str
    ) -> None:
        """Identify areas for improvement based on negative feedback."""
        # Implementation would analyze negative feedback for improvement opportunities
        pass
    
    async def _analyze_feedback_comments(self, agent_id: str, comments: str) -> List[LearningInsight]:
        """Analyze feedback comments for insights."""
        # Implementation would use NLP to extract insights from feedback text
        return []
