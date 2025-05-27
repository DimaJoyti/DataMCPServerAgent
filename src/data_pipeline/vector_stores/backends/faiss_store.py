"""
FAISS vector store implementation.
"""

import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import faiss
    import numpy as np
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

from .base_store import BaseVectorStore, VectorStoreStats
from ..schemas.base_schema import VectorRecord
from ..schemas.search_models import SearchQuery, SearchResults, SearchResult, SearchType


class FAISSVectorStore(BaseVectorStore):
    """FAISS vector store implementation."""
    
    def __init__(self, config):
        """Initialize FAISS store."""
        if not HAS_FAISS:
            raise ImportError(
                "FAISS not available. Install with: pip install faiss-cpu or faiss-gpu"
            )
        
        super().__init__(config)
        self.index = None
        self.id_map = {}  # Maps internal FAISS IDs to external IDs
        self.metadata_store = {}  # Stores metadata for each vector
        self.text_store = {}  # Stores text for each vector
        self.next_internal_id = 0
        
        # File paths for persistence
        if self.config.persist_directory:
            self.persist_path = Path(self.config.persist_directory)
            self.persist_path.mkdir(parents=True, exist_ok=True)
            self.index_file = self.persist_path / f"{self.config.collection_name}.index"
            self.metadata_file = self.persist_path / f"{self.config.collection_name}_metadata.pkl"
        else:
            self.persist_path = None
            self.index_file = None
            self.metadata_file = None
    
    async def initialize(self) -> None:
        """Initialize FAISS index."""
        try:
            # Try to load existing index
            if self.index_file and self.index_file.exists():
                await self._load_index()
                self.logger.info(f"Loaded existing FAISS index: {self.config.collection_name}")
            else:
                # Create new index
                await self._create_index()
                self.logger.info(f"Created new FAISS index: {self.config.collection_name}")
            
            self._is_initialized = True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize FAISS: {e}")
            raise
    
    async def close(self) -> None:
        """Close FAISS store and save if persistent."""
        if self.persist_path and self.index:
            await self._save_index()
        
        self.index = None
        self.id_map.clear()
        self.metadata_store.clear()
        self.text_store.clear()
        self._is_initialized = False
    
    async def _create_index(self) -> None:
        """Create a new FAISS index."""
        dimension = self.config.embedding_dimension
        
        # Choose index type based on configuration
        if self.config.index_type.value == "flat":
            # Flat index for exact search
            if self.config.distance_metric.value == "cosine":
                self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine
            elif self.config.distance_metric.value == "euclidean":
                self.index = faiss.IndexFlatL2(dimension)  # L2 for euclidean
            else:
                self.index = faiss.IndexFlatIP(dimension)  # Default to IP
        
        elif self.config.index_type.value == "hnsw":
            # HNSW index for approximate search
            M = self.config.index_params.get("M", 16)  # Number of connections
            ef_construction = self.config.index_params.get("ef_construction", 200)
            
            if self.config.distance_metric.value == "cosine":
                self.index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
            elif self.config.distance_metric.value == "euclidean":
                self.index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_L2)
            else:
                self.index = faiss.IndexHNSWFlat(dimension, M, faiss.METRIC_INNER_PRODUCT)
            
            self.index.hnsw.efConstruction = ef_construction
        
        elif self.config.index_type.value == "ivf":
            # IVF index for large datasets
            nlist = self.config.index_params.get("nlist", 100)  # Number of clusters
            
            if self.config.distance_metric.value == "cosine":
                quantizer = faiss.IndexFlatIP(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            elif self.config.distance_metric.value == "euclidean":
                quantizer = faiss.IndexFlatL2(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_L2)
            else:
                quantizer = faiss.IndexFlatIP(dimension)
                self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        
        else:
            # Default to flat index
            self.index = faiss.IndexFlatIP(dimension)
        
        # Reset stores
        self.id_map.clear()
        self.metadata_store.clear()
        self.text_store.clear()
        self.next_internal_id = 0
    
    async def _load_index(self) -> None:
        """Load existing FAISS index from disk."""
        # Load FAISS index
        self.index = faiss.read_index(str(self.index_file))
        
        # Load metadata
        if self.metadata_file.exists():
            with open(self.metadata_file, 'rb') as f:
                data = pickle.load(f)
                self.id_map = data.get('id_map', {})
                self.metadata_store = data.get('metadata_store', {})
                self.text_store = data.get('text_store', {})
                self.next_internal_id = data.get('next_internal_id', 0)
    
    async def _save_index(self) -> None:
        """Save FAISS index to disk."""
        if not self.persist_path:
            return
        
        # Save FAISS index
        faiss.write_index(self.index, str(self.index_file))
        
        # Save metadata
        data = {
            'id_map': self.id_map,
            'metadata_store': self.metadata_store,
            'text_store': self.text_store,
            'next_internal_id': self.next_internal_id
        }
        
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(data, f)
    
    async def create_collection(self, schema: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new collection (recreate index)."""
        try:
            await self._create_index()
            if self.persist_path:
                await self._save_index()
            return True
        except Exception as e:
            self.logger.error(f"Failed to create collection: {e}")
            return False
    
    async def delete_collection(self) -> bool:
        """Delete the collection (remove files)."""
        try:
            if self.index_file and self.index_file.exists():
                self.index_file.unlink()
            if self.metadata_file and self.metadata_file.exists():
                self.metadata_file.unlink()
            
            self.index = None
            self.id_map.clear()
            self.metadata_store.clear()
            self.text_store.clear()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to delete collection: {e}")
            return False
    
    async def collection_exists(self) -> bool:
        """Check if collection exists."""
        if self.index_file:
            return self.index_file.exists()
        return self.index is not None
    
    async def insert_vectors(self, records: List[VectorRecord]) -> List[str]:
        """Insert vector records into FAISS."""
        if not records:
            return []
        
        self._validate_records(records)
        
        try:
            # Prepare vectors for FAISS
            vectors = np.array([record.vector for record in records], dtype=np.float32)
            
            # Normalize vectors for cosine similarity
            if self.config.distance_metric.value == "cosine":
                faiss.normalize_L2(vectors)
            
            # Check if IVF index needs training
            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                if vectors.shape[0] >= self.index.nlist:
                    self.index.train(vectors)
                    self.logger.info("Trained IVF index")
                else:
                    self.logger.warning(f"Not enough vectors to train IVF index. Need at least {self.index.nlist}")
            
            # Add vectors to index
            internal_ids = list(range(self.next_internal_id, self.next_internal_id + len(records)))
            self.index.add_with_ids(vectors, np.array(internal_ids, dtype=np.int64))
            
            # Update mappings and stores
            inserted_ids = []
            for i, record in enumerate(records):
                internal_id = internal_ids[i]
                external_id = record.id
                
                self.id_map[internal_id] = external_id
                self.metadata_store[external_id] = {
                    'metadata': record.metadata,
                    'created_at': record.created_at.isoformat(),
                    'updated_at': record.updated_at.isoformat() if record.updated_at else None,
                    'source': record.source,
                    'source_type': record.source_type
                }
                self.text_store[external_id] = record.text
                inserted_ids.append(external_id)
            
            self.next_internal_id += len(records)
            
            # Save if persistent
            if self.persist_path:
                await self._save_index()
            
            self.logger.debug(f"Inserted {len(records)} vectors into FAISS")
            return inserted_ids
            
        except Exception as e:
            self.logger.error(f"Failed to insert vectors: {e}")
            raise
    
    async def update_vectors(self, records: List[VectorRecord]) -> List[str]:
        """Update existing vector records."""
        # FAISS doesn't support direct updates, so we need to remove and re-add
        if not records:
            return []
        
        try:
            # Find existing records to remove
            ids_to_remove = [record.id for record in records if record.id in self.text_store]
            
            # Remove existing vectors
            if ids_to_remove:
                await self.delete_vectors(ids_to_remove)
            
            # Insert updated vectors
            return await self.insert_vectors(records)
            
        except Exception as e:
            self.logger.error(f"Failed to update vectors: {e}")
            raise
    
    async def delete_vectors(self, ids: List[str]) -> int:
        """Delete vectors by IDs."""
        if not ids:
            return 0
        
        try:
            # FAISS doesn't support direct deletion, so we need to rebuild the index
            # For now, we'll just remove from our metadata stores
            deleted_count = 0
            
            for external_id in ids:
                if external_id in self.text_store:
                    del self.text_store[external_id]
                    del self.metadata_store[external_id]
                    deleted_count += 1
                    
                    # Remove from id_map
                    internal_id_to_remove = None
                    for internal_id, mapped_external_id in self.id_map.items():
                        if mapped_external_id == external_id:
                            internal_id_to_remove = internal_id
                            break
                    
                    if internal_id_to_remove is not None:
                        del self.id_map[internal_id_to_remove]
            
            # For a complete implementation, we would need to rebuild the index
            # without the deleted vectors. For now, we'll mark this as a limitation.
            
            if self.persist_path:
                await self._save_index()
            
            self.logger.debug(f"Marked {deleted_count} vectors as deleted in FAISS")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Failed to delete vectors: {e}")
            return 0
    
    async def get_vector(self, id: str) -> Optional[VectorRecord]:
        """Get a vector by ID."""
        try:
            if id not in self.text_store:
                return None
            
            # Get metadata and text
            metadata_info = self.metadata_store[id]
            text = self.text_store[id]
            
            # Find the vector in the index (this is expensive for FAISS)
            # For a production implementation, we might want to cache vectors
            
            from datetime import datetime
            
            created_at = datetime.fromisoformat(metadata_info['created_at'])
            updated_at = None
            if metadata_info['updated_at']:
                updated_at = datetime.fromisoformat(metadata_info['updated_at'])
            
            return VectorRecord(
                id=id,
                vector=[],  # We don't store the actual vector for retrieval
                text=text,
                metadata=metadata_info['metadata'],
                created_at=created_at,
                updated_at=updated_at,
                source=metadata_info['source'],
                source_type=metadata_info['source_type']
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
            else:
                raise ValueError(f"FAISS only supports vector search. Got: {query.search_type}")
            
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
        # Prepare query vector
        query_vector = np.array([query.query_vector], dtype=np.float32)
        
        # Normalize for cosine similarity
        if self.config.distance_metric.value == "cosine":
            faiss.normalize_L2(query_vector)
        
        # Set search parameters for HNSW
        if hasattr(self.index, 'hnsw'):
            ef_search = self.config.index_params.get("ef_search", 50)
            self.index.hnsw.efSearch = ef_search
        
        # Perform search
        k = min(query.limit + query.offset, self.index.ntotal)
        if k == 0:
            return []
        
        distances, internal_ids = self.index.search(query_vector, k)
        
        # Convert results
        search_results = []
        
        for i, (distance, internal_id) in enumerate(zip(distances[0], internal_ids[0])):
            if internal_id == -1:  # FAISS returns -1 for empty slots
                continue
            
            # Skip offset results
            if i < query.offset:
                continue
            
            # Check if we have enough results
            if len(search_results) >= query.limit:
                break
            
            # Get external ID
            external_id = self.id_map.get(internal_id)
            if not external_id or external_id not in self.text_store:
                continue
            
            # Convert distance to similarity score
            if self.config.distance_metric.value == "cosine":
                score = float(distance)  # FAISS returns similarity for IP
            else:
                score = 1.0 / (1.0 + float(distance))  # Convert distance to similarity
            
            # Apply similarity threshold if specified
            if query.similarity_threshold and score < query.similarity_threshold:
                continue
            
            # Get metadata and text
            metadata_info = self.metadata_store[external_id]
            text = self.text_store[external_id]
            
            search_result = SearchResult(
                id=external_id,
                score=score,
                text=text,
                metadata=metadata_info['metadata'],
                rank=len(search_results) + 1,
                distance=float(distance)
            )
            
            if query.include_vectors:
                search_result.vector = query.query_vector  # We don't store original vectors
            
            search_results.append(search_result)
        
        return search_results
    
    async def get_stats(self) -> VectorStoreStats:
        """Get FAISS statistics."""
        try:
            total_vectors = self.index.ntotal if self.index else 0
            
            # Estimate index size
            index_size = None
            if self.index_file and self.index_file.exists():
                index_size = self.index_file.stat().st_size
            
            index_type = "Unknown"
            is_trained = True
            
            if hasattr(self.index, 'hnsw'):
                index_type = "HNSW"
            elif hasattr(self.index, 'nlist'):
                index_type = "IVF"
                is_trained = self.index.is_trained
            else:
                index_type = "Flat"
            
            return VectorStoreStats(
                total_vectors=total_vectors,
                index_size=index_size,
                index_type=index_type,
                is_trained=is_trained
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get stats: {e}")
            return VectorStoreStats(total_vectors=0)
