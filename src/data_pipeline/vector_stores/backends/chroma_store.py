"""
ChromaDB vector store implementation.
"""

import time
from typing import Any, Dict, List, Optional

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

from .base_store import BaseVectorStore, VectorStoreStats
from ..schemas.base_schema import VectorRecord
from ..schemas.search_models import SearchQuery, SearchResults, SearchResult, SearchType


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB vector store implementation."""
    
    def __init__(self, config):
        """Initialize ChromaDB store."""
        if not HAS_CHROMA:
            raise ImportError(
                "ChromaDB not available. Install with: pip install chromadb"
            )
        
        super().__init__(config)
        self.client = None
        self.collection = None
    
    async def initialize(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            # Create client
            if self.config.persist_directory:
                # Persistent client
                self.client = chromadb.PersistentClient(
                    path=self.config.persist_directory,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
            else:
                # In-memory client
                self.client = chromadb.Client(
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
            
            # Get or create collection
            distance_mapping = {
                "cosine": "cosine",
                "euclidean": "l2", 
                "dot_product": "ip"
            }
            
            distance_function = distance_mapping.get(
                self.config.distance_metric.value,
                "cosine"
            )
            
            try:
                self.collection = self.client.get_collection(
                    name=self.config.collection_name
                )
                self.logger.info(f"Retrieved existing collection: {self.config.collection_name}")
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self.config.collection_name,
                    metadata={
                        "hnsw:space": distance_function,
                        "description": "Document embeddings collection"
                    }
                )
                self.logger.info(f"Created new collection: {self.config.collection_name}")
            
            self._is_initialized = True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    async def close(self) -> None:
        """Close ChromaDB connection."""
        # ChromaDB doesn't require explicit closing
        self.client = None
        self.collection = None
        self._is_initialized = False
    
    async def create_collection(self, schema: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new collection."""
        try:
            if await self.collection_exists():
                self.logger.warning(f"Collection {self.config.collection_name} already exists")
                return True
            
            distance_mapping = {
                "cosine": "cosine",
                "euclidean": "l2",
                "dot_product": "ip"
            }
            
            distance_function = distance_mapping.get(
                self.config.distance_metric.value,
                "cosine"
            )
            
            metadata = {
                "hnsw:space": distance_function,
                "description": "Document embeddings collection"
            }
            
            if schema:
                metadata.update(schema)
            
            self.collection = self.client.create_collection(
                name=self.config.collection_name,
                metadata=metadata
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create collection: {e}")
            return False
    
    async def delete_collection(self) -> bool:
        """Delete the collection."""
        try:
            self.client.delete_collection(name=self.config.collection_name)
            self.collection = None
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete collection: {e}")
            return False
    
    async def collection_exists(self) -> bool:
        """Check if collection exists."""
        try:
            collections = self.client.list_collections()
            return any(col.name == self.config.collection_name for col in collections)
        except Exception:
            return False
    
    async def insert_vectors(self, records: List[VectorRecord]) -> List[str]:
        """Insert vector records into ChromaDB."""
        if not records:
            return []
        
        self._validate_records(records)
        
        try:
            # Prepare data for ChromaDB
            ids = [record.id for record in records]
            embeddings = [record.vector for record in records]
            documents = [record.text for record in records]
            metadatas = []
            
            for record in records:
                metadata = record.metadata.copy()
                metadata.update({
                    "created_at": record.created_at.isoformat(),
                    "source": record.source or "",
                    "source_type": record.source_type or ""
                })
                if record.updated_at:
                    metadata["updated_at"] = record.updated_at.isoformat()
                metadatas.append(metadata)
            
            # Insert into ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            self.logger.debug(f"Inserted {len(records)} vectors into ChromaDB")
            return ids
            
        except Exception as e:
            self.logger.error(f"Failed to insert vectors: {e}")
            raise
    
    async def update_vectors(self, records: List[VectorRecord]) -> List[str]:
        """Update existing vector records."""
        if not records:
            return []
        
        self._validate_records(records)
        
        try:
            # Prepare data for ChromaDB
            ids = [record.id for record in records]
            embeddings = [record.vector for record in records]
            documents = [record.text for record in records]
            metadatas = []
            
            for record in records:
                metadata = record.metadata.copy()
                metadata.update({
                    "created_at": record.created_at.isoformat(),
                    "source": record.source or "",
                    "source_type": record.source_type or "",
                    "updated_at": record.updated_at.isoformat() if record.updated_at else ""
                })
                metadatas.append(metadata)
            
            # Update in ChromaDB
            self.collection.update(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            self.logger.debug(f"Updated {len(records)} vectors in ChromaDB")
            return ids
            
        except Exception as e:
            self.logger.error(f"Failed to update vectors: {e}")
            raise
    
    async def delete_vectors(self, ids: List[str]) -> int:
        """Delete vectors by IDs."""
        if not ids:
            return 0
        
        try:
            self.collection.delete(ids=ids)
            self.logger.debug(f"Deleted {len(ids)} vectors from ChromaDB")
            return len(ids)
        except Exception as e:
            self.logger.error(f"Failed to delete vectors: {e}")
            return 0
    
    async def get_vector(self, id: str) -> Optional[VectorRecord]:
        """Get a vector by ID."""
        try:
            result = self.collection.get(
                ids=[id],
                include=["embeddings", "documents", "metadatas"]
            )
            
            if not result["ids"]:
                return None
            
            # Convert back to VectorRecord
            from datetime import datetime
            
            metadata = result["metadatas"][0]
            created_at = datetime.fromisoformat(metadata.pop("created_at"))
            updated_at = None
            if metadata.get("updated_at"):
                updated_at = datetime.fromisoformat(metadata.pop("updated_at"))
            
            source = metadata.pop("source", None)
            source_type = metadata.pop("source_type", None)
            
            return VectorRecord(
                id=result["ids"][0],
                vector=result["embeddings"][0],
                text=result["documents"][0],
                metadata=metadata,
                created_at=created_at,
                updated_at=updated_at,
                source=source if source else None,
                source_type=source_type if source_type else None
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get vector {id}: {e}")
            return None
    
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
        # Build where clause for filters
        where_clause = self._build_where_clause(query.filters)
        
        # Perform search
        results = self.collection.query(
            query_embeddings=[query.query_vector],
            n_results=query.limit,
            where=where_clause,
            include=["embeddings", "documents", "metadatas", "distances"]
        )
        
        # Convert to SearchResult objects
        search_results = []
        
        if results["ids"] and results["ids"][0]:
            for i, (id_, distance, document, metadata) in enumerate(zip(
                results["ids"][0],
                results["distances"][0],
                results["documents"][0],
                results["metadatas"][0]
            )):
                # Convert distance to similarity score
                score = 1.0 - distance if self.config.distance_metric.value == "cosine" else 1.0 / (1.0 + distance)
                
                # Apply similarity threshold if specified
                if query.similarity_threshold and score < query.similarity_threshold:
                    continue
                
                search_result = SearchResult(
                    id=id_,
                    score=score,
                    text=document,
                    metadata=metadata,
                    rank=i + 1,
                    distance=distance
                )
                
                if query.include_vectors and results.get("embeddings"):
                    search_result.vector = results["embeddings"][0][i]
                
                search_results.append(search_result)
        
        return search_results
    
    async def _keyword_search(self, query: SearchQuery) -> List[SearchResult]:
        """Perform keyword search (using metadata filtering)."""
        # ChromaDB doesn't have built-in full-text search
        # We'll use a simple contains filter on the document text
        where_clause = {
            "$and": [
                {"$contains": query.query_text}
            ]
        }
        
        if query.filters:
            additional_filters = self._build_where_clause(query.filters)
            if additional_filters:
                where_clause["$and"].append(additional_filters)
        
        # Get all matching documents
        results = self.collection.get(
            where_document=where_clause,
            limit=query.limit,
            offset=query.offset,
            include=["documents", "metadatas"]
        )
        
        # Convert to SearchResult objects with simple scoring
        search_results = []
        
        for i, (id_, document, metadata) in enumerate(zip(
            results["ids"],
            results["documents"], 
            results["metadatas"]
        )):
            # Simple keyword scoring based on term frequency
            score = self._calculate_keyword_score(query.query_text, document)
            
            search_result = SearchResult(
                id=id_,
                score=score,
                text=document,
                metadata=metadata,
                rank=i + 1
            )
            
            search_results.append(search_result)
        
        # Sort by score
        search_results.sort(key=lambda x: x.score, reverse=True)
        
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
    
    def _build_where_clause(self, filters) -> Optional[Dict[str, Any]]:
        """Build ChromaDB where clause from filters."""
        if not filters or not filters.filters:
            return None
        
        conditions = []
        
        for filter_obj in filters.filters:
            field = filter_obj.field
            operator = filter_obj.operator.value
            value = filter_obj.value
            
            if operator == "eq":
                conditions.append({field: {"$eq": value}})
            elif operator == "ne":
                conditions.append({field: {"$ne": value}})
            elif operator == "gt":
                conditions.append({field: {"$gt": value}})
            elif operator == "gte":
                conditions.append({field: {"$gte": value}})
            elif operator == "lt":
                conditions.append({field: {"$lt": value}})
            elif operator == "lte":
                conditions.append({field: {"$lte": value}})
            elif operator == "in":
                conditions.append({field: {"$in": value}})
            elif operator == "not_in":
                conditions.append({field: {"$nin": value}})
            elif operator == "contains":
                conditions.append({field: {"$contains": value}})
        
        if not conditions:
            return None
        
        if len(conditions) == 1:
            return conditions[0]
        
        # Combine with AND or OR
        operator = "$and" if filters.operator.upper() == "AND" else "$or"
        return {operator: conditions}
    
    def _calculate_keyword_score(self, query_text: str, document: str) -> float:
        """Calculate simple keyword relevance score."""
        query_terms = query_text.lower().split()
        doc_terms = document.lower().split()
        
        if not query_terms or not doc_terms:
            return 0.0
        
        # Simple term frequency scoring
        matches = sum(1 for term in query_terms if term in doc_terms)
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
        """Get ChromaDB statistics."""
        try:
            # Get collection count
            count_result = self.collection.count()
            
            return VectorStoreStats(
                total_vectors=count_result,
                index_type="HNSW",
                is_trained=True
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return VectorStoreStats(total_vectors=0)
