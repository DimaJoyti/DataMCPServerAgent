"""
Reranking for RAG Architecture.

This module provides result reranking capabilities:
- Multiple reranking strategies
- Relevance scoring
- Context-aware ranking
- Performance optimization
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from pydantic import BaseModel, Field

from app.core.logging import get_logger, LoggerMixin


class RerankingStrategy(str, Enum):
    """Reranking strategies."""
    SCORE_BASED = "score_based"
    SEMANTIC = "semantic"
    DIVERSITY = "diversity"
    HYBRID = "hybrid"


@dataclass
class ScoredResult:
    """Result with relevance score."""
    
    content: str
    original_score: float
    reranked_score: float
    metadata: Dict[str, Any]
    rank_position: int = 0


class RerankingMetrics(BaseModel):
    """Metrics for reranking performance."""
    
    total_results: int = Field(..., description="Total number of results")
    reranked_results: int = Field(..., description="Number of reranked results")
    avg_score_improvement: float = Field(default=0.0, description="Average score improvement")
    processing_time_ms: float = Field(default=0.0, description="Processing time in milliseconds")


class ReRanker(LoggerMixin):
    """Result reranker for improving relevance."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize reranker."""
        self.config = config or {}
        self.logger = get_logger(self.__class__.__name__)
        
        # Configuration
        self.default_strategy = RerankingStrategy(
            self.config.get("default_strategy", RerankingStrategy.SCORE_BASED)
        )
        self.diversity_threshold = self.config.get("diversity_threshold", 0.8)
        self.semantic_weight = self.config.get("semantic_weight", 0.7)
        
        self.logger.info("ReRanker initialized")
    
    async def rerank(self, results: List[Dict[str, Any]], query: str,
                    strategy: Optional[RerankingStrategy] = None) -> List[ScoredResult]:
        """Rerank search results."""
        
        if not results:
            return []
        
        strategy = strategy or self.default_strategy
        
        # Convert to scored results
        scored_results = [
            ScoredResult(
                content=result.get("content", ""),
                original_score=result.get("score", 0.0),
                reranked_score=result.get("score", 0.0),
                metadata=result.get("metadata", {}),
                rank_position=i
            )
            for i, result in enumerate(results)
        ]
        
        # Apply reranking strategy
        if strategy == RerankingStrategy.SCORE_BASED:
            reranked = await self._score_based_reranking(scored_results, query)
        elif strategy == RerankingStrategy.SEMANTIC:
            reranked = await self._semantic_reranking(scored_results, query)
        elif strategy == RerankingStrategy.DIVERSITY:
            reranked = await self._diversity_reranking(scored_results, query)
        else:  # HYBRID
            reranked = await self._hybrid_reranking(scored_results, query)
        
        # Update rank positions
        for i, result in enumerate(reranked):
            result.rank_position = i
        
        return reranked
    
    async def _score_based_reranking(self, results: List[ScoredResult], query: str) -> List[ScoredResult]:
        """Simple score-based reranking."""
        # Sort by original score (already done, but ensure consistency)
        return sorted(results, key=lambda x: x.original_score, reverse=True)
    
    async def _semantic_reranking(self, results: List[ScoredResult], query: str) -> List[ScoredResult]:
        """Semantic similarity-based reranking."""
        # Placeholder implementation
        # In production, use semantic similarity models
        
        query_words = set(query.lower().split())
        
        for result in results:
            content_words = set(result.content.lower().split())
            
            # Simple word overlap similarity
            overlap = len(query_words.intersection(content_words))
            total_words = len(query_words.union(content_words))
            
            semantic_score = overlap / max(total_words, 1) if total_words > 0 else 0
            
            # Combine with original score
            result.reranked_score = (
                self.semantic_weight * semantic_score + 
                (1 - self.semantic_weight) * result.original_score
            )
        
        return sorted(results, key=lambda x: x.reranked_score, reverse=True)
    
    async def _diversity_reranking(self, results: List[ScoredResult], query: str) -> List[ScoredResult]:
        """Diversity-based reranking to avoid redundant results."""
        if not results:
            return results
        
        reranked = []
        remaining = results.copy()
        
        # Start with highest scoring result
        remaining.sort(key=lambda x: x.original_score, reverse=True)
        reranked.append(remaining.pop(0))
        
        # Add diverse results
        while remaining and len(reranked) < len(results):
            best_candidate = None
            best_diversity_score = -1
            
            for candidate in remaining:
                # Calculate diversity score (how different from already selected)
                diversity_score = self._calculate_diversity(candidate, reranked)
                
                # Combine diversity with relevance
                combined_score = (
                    0.6 * candidate.original_score + 
                    0.4 * diversity_score
                )
                
                if combined_score > best_diversity_score:
                    best_diversity_score = combined_score
                    best_candidate = candidate
            
            if best_candidate:
                best_candidate.reranked_score = best_diversity_score
                reranked.append(best_candidate)
                remaining.remove(best_candidate)
        
        return reranked
    
    def _calculate_diversity(self, candidate: ScoredResult, selected: List[ScoredResult]) -> float:
        """Calculate diversity score for a candidate."""
        if not selected:
            return 1.0
        
        candidate_words = set(candidate.content.lower().split())
        
        min_similarity = 1.0
        for selected_result in selected:
            selected_words = set(selected_result.content.lower().split())
            
            # Calculate similarity
            overlap = len(candidate_words.intersection(selected_words))
            total_words = len(candidate_words.union(selected_words))
            
            similarity = overlap / max(total_words, 1) if total_words > 0 else 0
            min_similarity = min(min_similarity, similarity)
        
        # Diversity is inverse of similarity
        return 1.0 - min_similarity
    
    async def _hybrid_reranking(self, results: List[ScoredResult], query: str) -> List[ScoredResult]:
        """Hybrid reranking combining multiple strategies."""
        # Apply semantic reranking first
        semantic_results = await self._semantic_reranking(results, query)
        
        # Then apply diversity
        final_results = await self._diversity_reranking(semantic_results, query)
        
        return final_results
    
    def calculate_metrics(self, original_results: List[Dict[str, Any]], 
                         reranked_results: List[ScoredResult]) -> RerankingMetrics:
        """Calculate reranking performance metrics."""
        
        if not original_results or not reranked_results:
            return RerankingMetrics(
                total_results=len(original_results),
                reranked_results=len(reranked_results)
            )
        
        # Calculate average score improvement
        original_scores = [r.get("score", 0.0) for r in original_results]
        reranked_scores = [r.reranked_score for r in reranked_results]
        
        avg_original = sum(original_scores) / len(original_scores)
        avg_reranked = sum(reranked_scores) / len(reranked_scores)
        
        score_improvement = avg_reranked - avg_original
        
        return RerankingMetrics(
            total_results=len(original_results),
            reranked_results=len(reranked_results),
            avg_score_improvement=score_improvement
        )
