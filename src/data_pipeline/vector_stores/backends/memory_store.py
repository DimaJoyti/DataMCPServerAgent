"""
In-memory vector store implementation for testing and development.
"""

import math
import time
from typing import Any, Dict, List, Optional

from .base_store import BaseVectorStore, VectorStoreStats
from ..schemas.base_schema import VectorRecord
from ..schemas.search_models import SearchQuery, SearchResults, SearchResult, SearchType


class MemoryVectorStore(BaseVectorStore):
    """In-memory vector store implementation."""
    
    def __init__(self, config):
        """Initialize memory store."""
        super().__init__(config)
        self.vectors = {}  # id -> VectorRecord
        self.created_at = None
    
    async def initialize(self) -> None:
        """Initialize memory store."""
        from datetime import datetime
        self.vectors = {}
        self.created_at = datetime.now()
        self._is_initialized = True
        self.logger.info(f"Initialized memory vector store: {self.config.collection_name}")
    
    async def close(self) -> None:
        """Close memory store."""
        self.vectors.clear()
        self._is_initialized = False
    
    async def create_collection(self, schema: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new collection (clear existing data)."""
        try:
            self.vectors.clear()
            return True
        except Exception as e:
            self.logger.error(f"Failed to create collection: {e}")
            return False
    
    async def delete_collection(self) -> bool:
        """Delete the collection."""
        try:
            self.vectors.clear()
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete collection: {e}")
            return False
    
    async def collection_exists(self) -> bool:
        """Check if collection exists (always true for memory store)."""
        return self._is_initialized
    
    async def insert_vectors(self, records: List[VectorRecord]) -> List[str]:
        """Insert vector records into memory."""
        if not records:
            return []
        
        self._validate_records(records)
        
        try:
            inserted_ids = []
            
            for record in records:
                if record.id in self.vectors:
                    self.logger.warning(f"Vector {record.id} already exists, skipping")
                    continue
                
                self.vectors[record.id] = record
                inserted_ids.append(record.id)
            
            self.logger.debug(f"Inserted {len(inserted_ids)} vectors into memory store")
            return inserted_ids
            
        except Exception as e:
            self.logger.error(f"Failed to insert vectors: {e}")
            raise
    
    async def update_vectors(self, records: List[VectorRecord]) -> List[str]:
        """Update existing vector records."""
        if not records:
            return []
        
        self._validate_records(records)
        
        try:
            updated_ids = []
            
            for record in records:
                if record.id not in self.vectors:
                    self.logger.warning(f"Vector {record.id} does not exist, inserting instead")
                
                # Update timestamp
                from datetime import datetime
                record.updated_at = datetime.now()
                
                self.vectors[record.id] = record
                updated_ids.append(record.id)
            
            self.logger.debug(f"Updated {len(updated_ids)} vectors in memory store")
            return updated_ids
            
        except Exception as e:
            self.logger.error(f"Failed to update vectors: {e}")
            raise
    
    async def delete_vectors(self, ids: List[str]) -> int:
        """Delete vectors by IDs."""
        if not ids:
            return 0
        
        try:
            deleted_count = 0
            
            for vector_id in ids:
                if vector_id in self.vectors:
                    del self.vectors[vector_id]
                    deleted_count += 1
            
            self.logger.debug(f"Deleted {deleted_count} vectors from memory store")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to delete vectors: {e}")
            return 0
    
    async def get_vector(self, id: str) -> Optional[VectorRecord]:
        """Get a vector by ID."""
        return self.vectors.get(id)
    
    async def search_vectors(self, query: SearchQuery) -> SearchResults:
        """Search for similar vectors."""
        start_time = time.time()
        
        try:
            if query.search_type == SearchType.VECTOR and query.query_vector:
                results = await self._vector_search(query)
            elif query.search_type == SearchType.KEYWORD and query.query_text:
                results = await self._keyword_search(query)
            elif query.search_type == SearchType.HYBRID:
                results = await self._hybrid_search(query)
            else:
                raise ValueError(f"Unsupported search type: {query.search_type}")
            
            search_time = time.time() - start_time
            
            return SearchResults(
                results=results,
                query=query,
                total_results=len(results),
                search_time=search_time,
                offset=query.offset,
                limit=query.limit,
                has_more=len(results) == query.limit
            )
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            raise
    
    async def _vector_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform vector similarity search."""
        # Filter vectors based on filters
        candidate_vectors = self._apply_filters(query.filters)
        
        if not candidate_vectors:
            return []
        
        # Calculate similarities
        similarities = []
        
        for vector_id, record in candidate_vectors.items():
            similarity = self._calculate_similarity(
                query.query_vector,
                record.vector,
                self.config.distance_metric.value
            )
            
            # Apply similarity threshold if specified
            if query.similarity_threshold and similarity < query.similarity_threshold:
                continue
            
            similarities.append((vector_id, record, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[2], reverse=True)
        
        # Apply offset and limit
        start_idx = query.offset
        end_idx = start_idx + query.limit
        selected_similarities = similarities[start_idx:end_idx]
        
        # Convert to SearchResult objects
        search_results = []
        
        for rank, (vector_id, record, similarity) in enumerate(selected_similarities, 1):
            search_result = SearchResult(
                id=vector_id,
                score=similarity,
                text=record.text,
                metadata=record.metadata,
                rank=rank,
                distance=1.0 - similarity  # Convert similarity to distance
            )
            
            if query.include_vectors:
                search_result.vector = record.vector
            
            search_results.append(search_result)
        
        return search_results
    
    async def _keyword_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform keyword search."""
        # Filter vectors based on filters
        candidate_vectors = self._apply_filters(query.filters)
        
        if not candidate_vectors:
            return []
        
        # Calculate keyword relevance scores
        scores = []
        query_terms = query.query_text.lower().split()
        
        for vector_id, record in candidate_vectors.items():
            score = self._calculate_keyword_score(query_terms, record.text)
            
            if score > 0:  # Only include results with some relevance
                scores.append((vector_id, record, score))
        
        # Sort by score (descending)
        scores.sort(key=lambda x: x[2], reverse=True)
        
        # Apply offset and limit
        start_idx = query.offset
        end_idx = start_idx + query.limit
        selected_scores = scores[start_idx:end_idx]
        
        # Convert to SearchResult objects
        search_results = []
        
        for rank, (vector_id, record, score) in enumerate(selected_scores, 1):
            search_result = SearchResult(
                id=vector_id,
                score=score,
                text=record.text,
                metadata=record.metadata,
                rank=rank
            )
            
            if query.include_vectors:
                search_result.vector = record.vector
            
            search_results.append(search_result)
        
        return search_results
    
    async def _hybrid_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform hybrid search combining vector and keyword search."""
        vector_results = []
        keyword_results = []
        
        # Perform vector search if vector query is provided
        if query.query_vector:
            vector_query = SearchQuery(
                query_vector=query.query_vector,
                search_type=SearchType.VECTOR,
                limit=query.limit * 2,  # Get more results for fusion
                filters=query.filters
            )
            vector_results = await self._vector_search(vector_query)
        
        # Perform keyword search if text query is provided
        if query.query_text:
            keyword_query = SearchQuery(
                query_text=query.query_text,
                search_type=SearchType.KEYWORD,
                limit=query.limit * 2,  # Get more results for fusion
                filters=query.filters
            )
            keyword_results = await self._keyword_search(keyword_query)
        
        # Combine and rerank results
        return self._combine_search_results(
            vector_results,
            keyword_results,
            query.vector_weight,
            query.keyword_weight,
            query.limit
        )
    
    def _apply_filters(self, filters) -> Dict[str, VectorRecord]:
        """Apply filters to vectors."""
        if not filters or not filters.filters:
            return self.vectors.copy()
        
        filtered_vectors = {}
        
        for vector_id, record in self.vectors.items():
            if self._record_matches_filters(record, filters):
                filtered_vectors[vector_id] = record
        
        return filtered_vectors
    
    def _record_matches_filters(self, record: VectorRecord, filters) -> bool:
        """Check if record matches filters."""
        if not filters or not filters.filters:
            return True
        
        results = []
        
        for filter_obj in filters.filters:
            field = filter_obj.field
            operator = filter_obj.operator.value
            value = filter_obj.value
            
            # Get field value from record
            field_value = self._get_field_value(record, field)
            
            # Apply filter
            match = self._apply_filter_operator(field_value, operator, value)
            results.append(match)
        
        # Combine results based on operator
        if filters.operator.upper() == "AND":
            return all(results)
        else:  # OR
            return any(results)
    
    def _get_field_value(self, record: VectorRecord, field: str):
        """Get field value from record."""
        # Check direct attributes first
        if hasattr(record, field):
            return getattr(record, field)
        
        # Check metadata
        if field in record.metadata:
            return record.metadata[field]
        
        return None
    
    def _apply_filter_operator(self, field_value, operator: str, filter_value) -> bool:
        """Apply filter operator."""
        if field_value is None:
            return operator in ["not_exists", "ne"]
        
        if operator == "eq":
            return field_value == filter_value
        elif operator == "ne":
            return field_value != filter_value
        elif operator == "gt":
            return field_value > filter_value
        elif operator == "gte":
            return field_value >= filter_value
        elif operator == "lt":
            return field_value < filter_value
        elif operator == "lte":
            return field_value <= filter_value
        elif operator == "in":
            return field_value in filter_value
        elif operator == "not_in":
            return field_value not in filter_value
        elif operator == "contains":
            return str(filter_value).lower() in str(field_value).lower()
        elif operator == "not_contains":
            return str(filter_value).lower() not in str(field_value).lower()
        elif operator == "starts_with":
            return str(field_value).lower().startswith(str(filter_value).lower())
        elif operator == "ends_with":
            return str(field_value).lower().endswith(str(filter_value).lower())
        elif operator == "exists":
            return True  # Field exists if we got here
        elif operator == "not_exists":
            return False  # Field exists if we got here
        
        return False
    
    def _calculate_similarity(self, vector1: List[float], vector2: List[float], metric: str) -> float:
        """Calculate similarity between two vectors."""
        if metric == "cosine":
            return self._cosine_similarity(vector1, vector2)
        elif metric == "euclidean":
            distance = self._euclidean_distance(vector1, vector2)
            return 1.0 / (1.0 + distance)  # Convert distance to similarity
        elif metric == "dot_product":
            return self._dot_product(vector1, vector2)
        else:
            return self._cosine_similarity(vector1, vector2)  # Default
    
    def _cosine_similarity(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate cosine similarity."""
        dot_product = sum(a * b for a, b in zip(vector1, vector2))
        norm1 = math.sqrt(sum(a * a for a in vector1))
        norm2 = math.sqrt(sum(b * b for b in vector2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _euclidean_distance(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate Euclidean distance."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vector1, vector2)))
    
    def _dot_product(self, vector1: List[float], vector2: List[float]) -> float:
        """Calculate dot product."""
        return sum(a * b for a, b in zip(vector1, vector2))
    
    def _calculate_keyword_score(self, query_terms: List[str], text: str) -> float:
        """Calculate keyword relevance score."""
        text_lower = text.lower()
        text_terms = text_lower.split()
        
        if not query_terms or not text_terms:
            return 0.0
        
        # Simple TF scoring
        matches = 0
        for term in query_terms:
            if term in text_lower:
                # Count term frequency
                tf = text_terms.count(term)
                matches += tf
        
        # Normalize by query length
        return matches / len(query_terms)
    
    def _combine_search_results(
        self,
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult],
        vector_weight: float,
        keyword_weight: float,
        limit: int
    ) -> List[SearchResult]:
        """Combine vector and keyword search results."""
        # Create a map of all unique results
        result_map = {}
        
        # Add vector results
        for result in vector_results:
            result.score = result.score * vector_weight
            result_map[result.id] = result
        
        # Add/combine keyword results
        for result in keyword_results:
            if result.id in result_map:
                # Combine scores
                result_map[result.id].score += result.score * keyword_weight
            else:
                result.score = result.score * keyword_weight
                result_map[result.id] = result
        
        # Sort by combined score and return top results
        combined_results = list(result_map.values())
        combined_results.sort(key=lambda x: x.score, reverse=True)
        
        # Update ranks
        for i, result in enumerate(combined_results[:limit]):
            result.rank = i + 1
        
        return combined_results[:limit]
    
    async def get_stats(self) -> VectorStoreStats:
        """Get memory store statistics."""
        try:
            return VectorStoreStats(
                total_vectors=len(self.vectors),
                index_type="Memory",
                is_trained=True,
                created_at=self.created_at
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return VectorStoreStats(total_vectors=0)
