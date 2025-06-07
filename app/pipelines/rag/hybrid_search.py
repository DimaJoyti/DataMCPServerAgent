"""
Hybrid Search Engine for RAG Architecture.

This module implements a sophisticated hybrid search system that combines:
- Vector similarity search for semantic understanding
- Keyword search for exact matches and specific terms
- Semantic search with contextual understanding
- Advanced result fusion and ranking algorithms
"""

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from app.core.logging import get_logger

class SearchType(str, Enum):
    """Types of search strategies."""
    VECTOR = "vector"
    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"

class FusionStrategy(str, Enum):
    """Result fusion strategies."""
    RRF = "reciprocal_rank_fusion"  # Reciprocal Rank Fusion
    WEIGHTED = "weighted_average"
    BORDA = "borda_count"
    CONDORCET = "condorcet"

@dataclass
class SearchFilters:
    """Filters for search queries."""
    content_types: Optional[List[str]] = None
    date_range: Optional[Tuple[str, str]] = None
    metadata_filters: Optional[Dict[str, Any]] = None
    min_score: Optional[float] = None
    max_results: Optional[int] = None

class SearchQuery(BaseModel):
    """Search query with multiple search strategies."""

    # Query content
    text: str = Field(..., description="Main search text")
    vector: Optional[List[float]] = Field(None, description="Query vector embedding")

    # Search configuration
    search_types: List[SearchType] = Field(default=[SearchType.HYBRID], description="Search strategies to use")
    top_k: int = Field(default=10, description="Number of results to return")

    # Weights for fusion
    vector_weight: float = Field(default=0.5, description="Weight for vector search")
    keyword_weight: float = Field(default=0.3, description="Weight for keyword search")
    semantic_weight: float = Field(default=0.2, description="Weight for semantic search")

    # Filters and options
    filters: Optional[SearchFilters] = Field(None, description="Search filters")
    fusion_strategy: FusionStrategy = Field(default=FusionStrategy.RRF, description="Result fusion strategy")

    # Advanced options
    enable_expansion: bool = Field(default=True, description="Enable query expansion")
    enable_reranking: bool = Field(default=True, description="Enable result reranking")
    context: Optional[str] = Field(None, description="Additional context for search")

class SearchResult(BaseModel):
    """Individual search result."""

    # Content identification
    document_id: str = Field(..., description="Document identifier")
    chunk_id: Optional[str] = Field(None, description="Chunk identifier")

    # Content
    content: str = Field(..., description="Result content")
    title: Optional[str] = Field(None, description="Content title")

    # Scoring
    score: float = Field(..., description="Relevance score")
    vector_score: Optional[float] = Field(None, description="Vector similarity score")
    keyword_score: Optional[float] = Field(None, description="Keyword match score")
    semantic_score: Optional[float] = Field(None, description="Semantic relevance score")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    highlights: List[str] = Field(default_factory=list, description="Highlighted text snippets")

    # Source information
    source: Optional[str] = Field(None, description="Source document/URL")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Update timestamp")

class RankedResults(BaseModel):
    """Ranked search results with metadata."""

    # Results
    results: List[SearchResult] = Field(..., description="Ranked search results")
    total_found: int = Field(..., description="Total number of results found")

    # Query information
    query: SearchQuery = Field(..., description="Original search query")
    search_time_ms: float = Field(..., description="Search time in milliseconds")

    # Fusion information
    fusion_strategy: FusionStrategy = Field(..., description="Fusion strategy used")
    search_types_used: List[SearchType] = Field(..., description="Search types that were executed")

    # Quality metrics
    avg_score: float = Field(..., description="Average relevance score")
    score_distribution: Dict[str, float] = Field(default_factory=dict, description="Score distribution stats")

class VectorSearchEngine:
    """Vector similarity search engine."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    async def search(self, query_vector: List[float], top_k: int = 10,
                    filters: Optional[SearchFilters] = None) -> List[SearchResult]:
        """Perform vector similarity search."""
        # Placeholder implementation
        # In production, integrate with actual vector stores like:
        # - Chroma, Pinecone, Weaviate, Qdrant, etc.

        results = []
        for i in range(min(top_k, 5)):  # Simulate results
            score = 0.9 - (i * 0.1)  # Decreasing scores
            result = SearchResult(
                document_id=f"doc_{i}",
                chunk_id=f"chunk_{i}",
                content=f"Vector search result {i} content...",
                title=f"Document {i}",
                score=score,
                vector_score=score,
                metadata={"search_type": "vector"}
            )
            results.append(result)

        self.logger.debug(f"Vector search returned {len(results)} results")
        return results

class KeywordSearchEngine:
    """Keyword-based search engine."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    async def search(self, query_text: str, top_k: int = 10,
                    filters: Optional[SearchFilters] = None) -> List[SearchResult]:
        """Perform keyword search."""
        # Placeholder implementation
        # In production, integrate with search engines like:
        # - Elasticsearch, Solr, Whoosh, etc.

        results = []
        keywords = query_text.lower().split()

        for i in range(min(top_k, 5)):  # Simulate results
            # Simulate keyword matching score
            score = 0.8 - (i * 0.15)

            result = SearchResult(
                document_id=f"kw_doc_{i}",
                chunk_id=f"kw_chunk_{i}",
                content=f"Keyword search result {i} containing: {', '.join(keywords[:2])}...",
                title=f"Keyword Document {i}",
                score=score,
                keyword_score=score,
                metadata={"search_type": "keyword", "matched_keywords": keywords[:2]},
                highlights=[f"<mark>{kw}</mark>" for kw in keywords[:2]]
            )
            results.append(result)

        self.logger.debug(f"Keyword search returned {len(results)} results")
        return results

class SemanticSearchEngine:
    """Semantic search with contextual understanding."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    async def search(self, query_text: str, context: Optional[str] = None,
                    top_k: int = 10, filters: Optional[SearchFilters] = None) -> List[SearchResult]:
        """Perform semantic search."""
        # Placeholder implementation
        # In production, use advanced semantic models like:
        # - Sentence transformers, BERT variants, etc.

        results = []

        for i in range(min(top_k, 5)):  # Simulate results
            # Simulate semantic relevance score
            score = 0.85 - (i * 0.12)

            result = SearchResult(
                document_id=f"sem_doc_{i}",
                chunk_id=f"sem_chunk_{i}",
                content=f"Semantic search result {i} with contextual understanding...",
                title=f"Semantic Document {i}",
                score=score,
                semantic_score=score,
                metadata={
                    "search_type": "semantic",
                    "context_used": bool(context),
                    "semantic_concepts": ["concept1", "concept2"]
                }
            )
            results.append(result)

        self.logger.debug(f"Semantic search returned {len(results)} results")
        return results

class ResultFusion:
    """Handles fusion of results from multiple search strategies."""

    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)

    async def fuse_results(self, result_sets: Dict[SearchType, List[SearchResult]],
                          query: SearchQuery) -> List[SearchResult]:
        """Fuse results from multiple search strategies."""

        if query.fusion_strategy == FusionStrategy.RRF:
            return await self._reciprocal_rank_fusion(result_sets, query)
        elif query.fusion_strategy == FusionStrategy.WEIGHTED:
            return await self._weighted_fusion(result_sets, query)
        elif query.fusion_strategy == FusionStrategy.BORDA:
            return await self._borda_count_fusion(result_sets, query)
        else:
            # Default to RRF
            return await self._reciprocal_rank_fusion(result_sets, query)

    async def _reciprocal_rank_fusion(self, result_sets: Dict[SearchType, List[SearchResult]],
                                     query: SearchQuery) -> List[SearchResult]:
        """Reciprocal Rank Fusion algorithm."""
        k = 60  # RRF parameter
        doc_scores = {}

        # Weight mapping
        weights = {
            SearchType.VECTOR: query.vector_weight,
            SearchType.KEYWORD: query.keyword_weight,
            SearchType.SEMANTIC: query.semantic_weight
        }

        for search_type, results in result_sets.items():
            weight = weights.get(search_type, 1.0)

            for rank, result in enumerate(results, 1):
                doc_key = f"{result.document_id}_{result.chunk_id or ''}"

                if doc_key not in doc_scores:
                    doc_scores[doc_key] = {
                        "result": result,
                        "score": 0.0,
                        "search_types": []
                    }

                # RRF score calculation
                rrf_score = weight / (k + rank)
                doc_scores[doc_key]["score"] += rrf_score
                doc_scores[doc_key]["search_types"].append(search_type)

        # Sort by fused score and create final results
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1]["score"], reverse=True)

        fused_results = []
        for doc_key, doc_data in sorted_docs[:query.top_k]:
            result = doc_data["result"]
            result.score = doc_data["score"]
            result.metadata["fusion_score"] = doc_data["score"]
            result.metadata["search_types_matched"] = doc_data["search_types"]
            fused_results.append(result)

        self.logger.debug(f"RRF fusion produced {len(fused_results)} results")
        return fused_results

    async def _weighted_fusion(self, result_sets: Dict[SearchType, List[SearchResult]],
                              query: SearchQuery) -> List[SearchResult]:
        """Weighted average fusion."""
        doc_scores = {}

        weights = {
            SearchType.VECTOR: query.vector_weight,
            SearchType.KEYWORD: query.keyword_weight,
            SearchType.SEMANTIC: query.semantic_weight
        }

        for search_type, results in result_sets.items():
            weight = weights.get(search_type, 1.0)

            for result in results:
                doc_key = f"{result.document_id}_{result.chunk_id or ''}"

                if doc_key not in doc_scores:
                    doc_scores[doc_key] = {
                        "result": result,
                        "weighted_scores": [],
                        "weights": []
                    }

                doc_scores[doc_key]["weighted_scores"].append(result.score * weight)
                doc_scores[doc_key]["weights"].append(weight)

        # Calculate weighted averages
        for doc_key, doc_data in doc_scores.items():
            total_weighted_score = sum(doc_data["weighted_scores"])
            total_weight = sum(doc_data["weights"])
            doc_data["final_score"] = total_weighted_score / total_weight if total_weight > 0 else 0

        # Sort and return
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1]["final_score"], reverse=True)

        fused_results = []
        for doc_key, doc_data in sorted_docs[:query.top_k]:
            result = doc_data["result"]
            result.score = doc_data["final_score"]
            result.metadata["fusion_score"] = doc_data["final_score"]
            fused_results.append(result)

        return fused_results

    async def _borda_count_fusion(self, result_sets: Dict[SearchType, List[SearchResult]],
                                 query: SearchQuery) -> List[SearchResult]:
        """Borda count fusion method."""
        doc_scores = {}

        for search_type, results in result_sets.items():
            max_rank = len(results)

            for rank, result in enumerate(results):
                doc_key = f"{result.document_id}_{result.chunk_id or ''}"

                if doc_key not in doc_scores:
                    doc_scores[doc_key] = {
                        "result": result,
                        "borda_score": 0
                    }

                # Borda count: higher rank = higher score
                borda_points = max_rank - rank
                doc_scores[doc_key]["borda_score"] += borda_points

        # Sort by Borda score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1]["borda_score"], reverse=True)

        fused_results = []
        for doc_key, doc_data in sorted_docs[:query.top_k]:
            result = doc_data["result"]
            result.score = doc_data["borda_score"]
            result.metadata["borda_score"] = doc_data["borda_score"]
            fused_results.append(result)

        return fused_results

class HybridSearchEngine:
    """Main hybrid search engine coordinating all search strategies."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the hybrid search engine."""
        self.config = config or {}
        self.logger = get_logger(self.__class__.__name__)

        # Initialize search engines
        self.vector_engine = VectorSearchEngine()
        self.keyword_engine = KeywordSearchEngine()
        self.semantic_engine = SemanticSearchEngine()
        self.result_fusion = ResultFusion()

        self.logger.info("HybridSearchEngine initialized")

    async def search(self, query: SearchQuery) -> RankedResults:
        """Perform hybrid search using multiple strategies."""
        start_time = asyncio.get_event_loop().time()

        self.logger.info(f"Starting hybrid search with strategies: {query.search_types}")

        # Execute searches in parallel
        search_tasks = {}

        if SearchType.VECTOR in query.search_types and query.vector:
            search_tasks[SearchType.VECTOR] = self.vector_engine.search(
                query.vector, query.top_k, query.filters
            )

        if SearchType.KEYWORD in query.search_types:
            search_tasks[SearchType.KEYWORD] = self.keyword_engine.search(
                query.text, query.top_k, query.filters
            )

        if SearchType.SEMANTIC in query.search_types:
            search_tasks[SearchType.SEMANTIC] = self.semantic_engine.search(
                query.text, query.context, query.top_k, query.filters
            )

        # Wait for all searches to complete
        search_results = {}
        if search_tasks:
            completed_searches = await asyncio.gather(
                *search_tasks.values(),
                return_exceptions=True
            )

            for (search_type, _), result in zip(search_tasks.items(), completed_searches):
                if isinstance(result, Exception):
                    self.logger.error(f"Search failed for {search_type}: {result}")
                    search_results[search_type] = []
                else:
                    search_results[search_type] = result

        # Fuse results if multiple search types
        if len(search_results) > 1:
            final_results = await self.result_fusion.fuse_results(search_results, query)
        elif len(search_results) == 1:
            final_results = list(search_results.values())[0]
        else:
            final_results = []

        # Calculate metrics
        search_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        avg_score = np.mean([r.score for r in final_results]) if final_results else 0.0

        # Create ranked results
        ranked_results = RankedResults(
            results=final_results,
            total_found=len(final_results),
            query=query,
            search_time_ms=search_time_ms,
            fusion_strategy=query.fusion_strategy,
            search_types_used=list(search_results.keys()),
            avg_score=avg_score,
            score_distribution={
                "min": min([r.score for r in final_results]) if final_results else 0.0,
                "max": max([r.score for r in final_results]) if final_results else 0.0,
                "avg": avg_score
            }
        )

        self.logger.info(f"Hybrid search completed in {search_time_ms:.2f}ms, "
                        f"returned {len(final_results)} results")

        return ranked_results
