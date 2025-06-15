"""
Knowledge Integration Service - Integrates brand knowledge with RAG system.
Provides intelligent knowledge retrieval for conversation context.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from app.core.logging import LoggerMixin, get_logger
from app.domain.models.base import DomainService
from app.domain.models.brand_agent import BrandKnowledge, KnowledgeType
from app.domain.models.conversation import ConversationMessage, IntentType

logger = get_logger(__name__)


class KnowledgeSearchResult:
    """Result from knowledge search."""

    def __init__(
        self,
        knowledge_id: str,
        title: str,
        content: str,
        knowledge_type: KnowledgeType,
        relevance_score: float,
        source_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.knowledge_id = knowledge_id
        self.title = title
        self.content = content
        self.knowledge_type = knowledge_type
        self.relevance_score = relevance_score
        self.source_url = source_url
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.knowledge_id,
            "title": self.title,
            "content": self.content,
            "type": self.knowledge_type,
            "relevance_score": self.relevance_score,
            "source_url": self.source_url,
            "metadata": self.metadata,
        }


class KnowledgeIntegrationService(DomainService, LoggerMixin):
    """Service for integrating brand knowledge with conversations."""

    def __init__(self):
        super().__init__()
        self._knowledge_cache: Dict[str, BrandKnowledge] = {}
        self._search_cache: Dict[str, List[KnowledgeSearchResult]] = {}

    async def search_relevant_knowledge(
        self,
        query: str,
        brand_id: str,
        knowledge_types: Optional[List[KnowledgeType]] = None,
        limit: int = 5,
        min_relevance: float = 0.3,
        intent: Optional[IntentType] = None,
    ) -> List[KnowledgeSearchResult]:
        """Search for relevant knowledge items."""
        self.logger.info(f"Searching knowledge for query: {query[:50]}...")

        # Check cache first
        cache_key = f"{brand_id}:{query}:{','.join(knowledge_types or [])}:{limit}"
        if cache_key in self._search_cache:
            self.logger.debug("Returning cached knowledge search results")
            return self._search_cache[cache_key]

        # Get all knowledge items for the brand
        knowledge_items = await self._get_brand_knowledge(brand_id, knowledge_types)

        # Score and rank knowledge items
        scored_items = []
        for item in knowledge_items:
            relevance_score = await self._calculate_relevance_score(query, item, intent)

            if relevance_score >= min_relevance:
                result = KnowledgeSearchResult(
                    knowledge_id=item.id,
                    title=item.title,
                    content=item.content,
                    knowledge_type=item.knowledge_type,
                    relevance_score=relevance_score,
                    source_url=item.source_url,
                    metadata={
                        "priority": item.priority,
                        "tags": item.tags,
                        "last_updated": item.last_updated.isoformat(),
                    },
                )
                scored_items.append(result)

        # Sort by relevance score and priority
        scored_items.sort(
            key=lambda x: (x.relevance_score, x.metadata.get("priority", 1)), reverse=True
        )

        # Limit results
        results = scored_items[:limit]

        # Cache results
        self._search_cache[cache_key] = results

        self.logger.info(f"Found {len(results)} relevant knowledge items")
        return results

    async def get_contextual_knowledge(
        self,
        message: ConversationMessage,
        brand_id: str,
        conversation_context: Dict[str, Any],
    ) -> List[KnowledgeSearchResult]:
        """Get contextual knowledge based on message and conversation."""
        # Extract search terms from message
        search_terms = self._extract_search_terms(message)

        # Determine knowledge types based on intent
        knowledge_types = self._get_relevant_knowledge_types(
            message.analysis.intent if message.analysis else None
        )

        # Search for relevant knowledge
        results = await self.search_relevant_knowledge(
            query=" ".join(search_terms),
            brand_id=brand_id,
            knowledge_types=knowledge_types,
            intent=message.analysis.intent if message.analysis else None,
        )

        # Re-rank based on conversation context
        results = self._rerank_by_context(results, conversation_context)

        return results

    async def _get_brand_knowledge(
        self, brand_id: str, knowledge_types: Optional[List[KnowledgeType]] = None
    ) -> List[BrandKnowledge]:
        """Get all knowledge items for a brand."""
        knowledge_repo = self.get_repository("brand_knowledge")

        # Build filter criteria
        filters = {"metadata.brand_id": brand_id, "is_active": True}
        if knowledge_types:
            filters["knowledge_type__in"] = knowledge_types

        # Get knowledge items
        knowledge_items = await knowledge_repo.list(**filters)

        # Cache items
        for item in knowledge_items:
            self._knowledge_cache[item.id] = item

        return knowledge_items

    async def _calculate_relevance_score(
        self, query: str, knowledge_item: BrandKnowledge, intent: Optional[IntentType] = None
    ) -> float:
        """Calculate relevance score for a knowledge item."""
        score = 0.0
        query_lower = query.lower()

        # Title match (highest weight)
        title_lower = knowledge_item.title.lower()
        if query_lower in title_lower:
            score += 0.4
        elif any(word in title_lower for word in query_lower.split()):
            score += 0.2

        # Content match
        content_lower = knowledge_item.content.lower()
        query_words = query_lower.split()
        content_words = content_lower.split()

        # Calculate word overlap
        common_words = set(query_words) & set(content_words)
        if query_words:
            word_overlap = len(common_words) / len(query_words)
            score += word_overlap * 0.3

        # Tag match
        for tag in knowledge_item.tags:
            if tag.lower() in query_lower:
                score += 0.1

        # Intent-based scoring
        if intent:
            intent_boost = self._get_intent_knowledge_boost(intent, knowledge_item.knowledge_type)
            score += intent_boost

        # Priority boost
        priority_boost = (knowledge_item.priority - 1) * 0.05  # 0-0.45 boost
        score += priority_boost

        # Recency boost (newer content gets slight boost)
        days_old = (datetime.now() - knowledge_item.last_updated).days
        if days_old < 30:
            score += 0.05
        elif days_old < 90:
            score += 0.02

        return min(1.0, score)

    def _get_intent_knowledge_boost(
        self, intent: IntentType, knowledge_type: KnowledgeType
    ) -> float:
        """Get knowledge type boost based on user intent."""
        intent_knowledge_mapping = {
            IntentType.PRODUCT_INFO: {
                KnowledgeType.PRODUCT_INFO: 0.2,
                KnowledgeType.COMPANY_INFO: 0.1,
            },
            IntentType.SUPPORT: {
                KnowledgeType.FAQ: 0.2,
                KnowledgeType.PROCEDURES: 0.15,
                KnowledgeType.POLICIES: 0.1,
            },
            IntentType.COMPLAINT: {
                KnowledgeType.POLICIES: 0.2,
                KnowledgeType.PROCEDURES: 0.15,
                KnowledgeType.FAQ: 0.1,
            },
            IntentType.SALES_INQUIRY: {
                KnowledgeType.PRODUCT_INFO: 0.2,
                KnowledgeType.COMPETITOR_INFO: 0.1,
            },
            IntentType.PRICING: {
                KnowledgeType.PRODUCT_INFO: 0.15,
                KnowledgeType.POLICIES: 0.1,
            },
        }

        return intent_knowledge_mapping.get(intent, {}).get(knowledge_type, 0.0)

    def _extract_search_terms(self, message: ConversationMessage) -> List[str]:
        """Extract search terms from message."""
        content = message.content.lower()

        # Remove common stop words
        stop_words = {
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "am",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "having",
            "do",
            "does",
            "did",
            "doing",
            "a",
            "an",
            "the",
            "and",
            "but",
            "if",
            "or",
            "because",
            "as",
            "until",
            "while",
            "of",
            "at",
            "by",
            "for",
            "with",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "up",
            "down",
            "in",
            "out",
            "on",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "can",
            "could",
            "should",
            "would",
            "will",
            "shall",
        }

        # Extract words
        words = [word.strip(".,!?;:") for word in content.split()]

        # Filter out stop words and short words
        search_terms = [word for word in words if len(word) > 2 and word not in stop_words]

        # Add keywords from analysis if available
        if message.analysis and message.analysis.keywords:
            search_terms.extend(message.analysis.keywords)

        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in search_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)

        return unique_terms[:10]  # Limit to top 10 terms

    def _get_relevant_knowledge_types(
        self, intent: Optional[IntentType]
    ) -> Optional[List[KnowledgeType]]:
        """Get relevant knowledge types based on intent."""
        if not intent:
            return None

        intent_type_mapping = {
            IntentType.PRODUCT_INFO: [KnowledgeType.PRODUCT_INFO, KnowledgeType.COMPANY_INFO],
            IntentType.SUPPORT: [
                KnowledgeType.FAQ,
                KnowledgeType.PROCEDURES,
                KnowledgeType.POLICIES,
            ],
            IntentType.COMPLAINT: [KnowledgeType.POLICIES, KnowledgeType.PROCEDURES],
            IntentType.SALES_INQUIRY: [KnowledgeType.PRODUCT_INFO, KnowledgeType.COMPETITOR_INFO],
            IntentType.PRICING: [KnowledgeType.PRODUCT_INFO, KnowledgeType.POLICIES],
            IntentType.TECHNICAL_ISSUE: [KnowledgeType.FAQ, KnowledgeType.PROCEDURES],
        }

        return intent_type_mapping.get(intent)

    def _rerank_by_context(
        self, results: List[KnowledgeSearchResult], context: Dict[str, Any]
    ) -> List[KnowledgeSearchResult]:
        """Re-rank results based on conversation context."""
        # Get conversation topics
        topics = context.get("topics_discussed", [])

        # Boost results that match conversation topics
        for result in results:
            topic_boost = 0.0
            for topic in topics:
                if topic.lower() in result.title.lower() or topic.lower() in result.content.lower():
                    topic_boost += 0.1

            result.relevance_score = min(1.0, result.relevance_score + topic_boost)

        # Re-sort by updated scores
        results.sort(key=lambda x: x.relevance_score, reverse=True)

        return results

    async def update_knowledge_usage(
        self,
        knowledge_ids: List[str],
        conversation_id: str,
        effectiveness_score: Optional[float] = None,
    ) -> None:
        """Update knowledge usage statistics."""
        for knowledge_id in knowledge_ids:
            if knowledge_id in self._knowledge_cache:
                knowledge_item = self._knowledge_cache[knowledge_id]

                # Update usage metadata
                if "usage_stats" not in knowledge_item.metadata:
                    knowledge_item.metadata["usage_stats"] = {
                        "total_uses": 0,
                        "conversations": [],
                        "effectiveness_scores": [],
                    }

                stats = knowledge_item.metadata["usage_stats"]
                stats["total_uses"] += 1
                stats["conversations"].append(conversation_id)

                if effectiveness_score is not None:
                    stats["effectiveness_scores"].append(effectiveness_score)

                # Keep only recent data
                if len(stats["conversations"]) > 100:
                    stats["conversations"] = stats["conversations"][-100:]
                if len(stats["effectiveness_scores"]) > 100:
                    stats["effectiveness_scores"] = stats["effectiveness_scores"][-100:]

                # Save updated knowledge item
                knowledge_repo = self.get_repository("brand_knowledge")
                await knowledge_repo.save(knowledge_item)

    async def get_knowledge_analytics(self, brand_id: str) -> Dict[str, Any]:
        """Get analytics for brand knowledge usage."""
        knowledge_items = await self._get_brand_knowledge(brand_id)

        total_items = len(knowledge_items)
        used_items = 0
        total_uses = 0
        avg_effectiveness = 0.0

        type_usage = {}
        top_items = []

        for item in knowledge_items:
            usage_stats = item.metadata.get("usage_stats", {})
            uses = usage_stats.get("total_uses", 0)

            if uses > 0:
                used_items += 1
                total_uses += uses

                # Calculate average effectiveness
                effectiveness_scores = usage_stats.get("effectiveness_scores", [])
                if effectiveness_scores:
                    item_effectiveness = sum(effectiveness_scores) / len(effectiveness_scores)
                    avg_effectiveness += item_effectiveness

                # Track usage by type
                knowledge_type = item.knowledge_type
                if knowledge_type not in type_usage:
                    type_usage[knowledge_type] = 0
                type_usage[knowledge_type] += uses

                # Track top items
                top_items.append(
                    {
                        "id": item.id,
                        "title": item.title,
                        "type": knowledge_type,
                        "uses": uses,
                        "effectiveness": item_effectiveness if effectiveness_scores else None,
                    }
                )

        # Calculate averages
        if used_items > 0:
            avg_effectiveness /= used_items

        # Sort top items
        top_items.sort(key=lambda x: x["uses"], reverse=True)

        return {
            "total_items": total_items,
            "used_items": used_items,
            "usage_rate": used_items / total_items if total_items > 0 else 0,
            "total_uses": total_uses,
            "avg_effectiveness": avg_effectiveness,
            "usage_by_type": type_usage,
            "top_items": top_items[:10],
        }

    async def suggest_knowledge_gaps(
        self, brand_id: str, recent_conversations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Suggest knowledge gaps based on conversation analysis."""
        gaps = []

        # Analyze failed searches and unresolved queries
        for conversation in recent_conversations:
            messages = conversation.get("messages", [])
            for message in messages:
                if message.get("sender_type") == "user":
                    # Check if this query had low knowledge relevance
                    query = message.get("content", "")
                    results = await self.search_relevant_knowledge(query, brand_id, limit=3)

                    if not results or max(r.relevance_score for r in results) < 0.5:
                        # Potential knowledge gap
                        gaps.append(
                            {
                                "query": query,
                                "conversation_id": conversation.get("id"),
                                "timestamp": message.get("timestamp"),
                                "suggested_type": self._suggest_knowledge_type(query),
                            }
                        )

        # Group similar gaps
        grouped_gaps = self._group_similar_gaps(gaps)

        return grouped_gaps[:10]  # Return top 10 gaps

    def _suggest_knowledge_type(self, query: str) -> KnowledgeType:
        """Suggest knowledge type for a query."""
        query_lower = query.lower()

        if any(word in query_lower for word in ["how", "what", "why", "when", "where"]):
            return KnowledgeType.FAQ
        elif any(word in query_lower for word in ["product", "feature", "specification"]):
            return KnowledgeType.PRODUCT_INFO
        elif any(word in query_lower for word in ["policy", "rule", "guideline"]):
            return KnowledgeType.POLICIES
        elif any(word in query_lower for word in ["procedure", "process", "step"]):
            return KnowledgeType.PROCEDURES
        else:
            return KnowledgeType.COMPANY_INFO

    def _group_similar_gaps(self, gaps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Group similar knowledge gaps."""
        # Simple grouping by common keywords
        grouped = {}

        for gap in gaps:
            query = gap["query"].lower()
            words = set(query.split())

            # Find existing group with similar keywords
            best_group = None
            best_overlap = 0

            for group_key, group_gaps in grouped.items():
                group_words = set(group_key.split())
                overlap = len(words & group_words)
                if overlap > best_overlap and overlap >= 2:
                    best_overlap = overlap
                    best_group = group_key

            if best_group:
                grouped[best_group].append(gap)
            else:
                # Create new group
                key_words = [word for word in words if len(word) > 3][:3]
                group_key = " ".join(sorted(key_words))
                grouped[group_key] = [gap]

        # Convert to list format
        result = []
        for group_key, group_gaps in grouped.items():
            result.append(
                {
                    "topic": group_key,
                    "frequency": len(group_gaps),
                    "examples": [gap["query"] for gap in group_gaps[:3]],
                    "suggested_type": group_gaps[0]["suggested_type"],
                }
            )

        # Sort by frequency
        result.sort(key=lambda x: x["frequency"], reverse=True)

        return result
