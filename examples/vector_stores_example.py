"""
Example demonstrating vector store usage with different backends.
"""

import asyncio
import logging
from pathlib import Path
from typing import List

import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add src to path for imports
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data_pipeline.vector_stores.schemas import (
    DistanceMetric,
    DocumentVectorSchema,
    SearchFilters,
    SearchQuery,
    SearchType,
    VectorStoreConfig,
    VectorStoreType,
)
from src.data_pipeline.vector_stores.vector_store_manager import (
    VectorStoreFactory,
    VectorStoreManager,
)


class VectorStoreDemo:
    """Vector store demonstration."""

    def __init__(self):
        """Initialize demo."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.manager = VectorStoreManager()

    def create_sample_data(self) -> List[dict]:
        """Create sample vector data."""
        # Sample documents about AI/ML topics
        sample_docs = [
            {
                "id": "doc1_chunk1",
                "text": "Machine learning is a subset of artificial intelligence that enables computers to learn without explicit programming.",
                "vector": np.random.rand(384).tolist(),
                "metadata": {
                    "document_id": "ai_overview",
                    "document_title": "AI Overview",
                    "document_type": "article",
                    "chunk_index": 0,
                    "topic": "machine_learning",
                    "word_count": 16
                }
            },
            {
                "id": "doc1_chunk2",
                "text": "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
                "vector": np.random.rand(384).tolist(),
                "metadata": {
                    "document_id": "ai_overview",
                    "document_title": "AI Overview",
                    "document_type": "article",
                    "chunk_index": 1,
                    "topic": "deep_learning",
                    "word_count": 15
                }
            },
            {
                "id": "doc2_chunk1",
                "text": "Natural language processing enables computers to understand, interpret, and generate human language.",
                "vector": np.random.rand(384).tolist(),
                "metadata": {
                    "document_id": "nlp_guide",
                    "document_title": "NLP Guide",
                    "document_type": "tutorial",
                    "chunk_index": 0,
                    "topic": "nlp",
                    "word_count": 13
                }
            },
            {
                "id": "doc2_chunk2",
                "text": "Computer vision allows machines to interpret and make decisions based on visual data from images and videos.",
                "vector": np.random.rand(384).tolist(),
                "metadata": {
                    "document_id": "nlp_guide",
                    "document_title": "NLP Guide",
                    "document_type": "tutorial",
                    "chunk_index": 1,
                    "topic": "computer_vision",
                    "word_count": 16
                }
            },
            {
                "id": "doc3_chunk1",
                "text": "Reinforcement learning is a type of machine learning where agents learn through interaction with an environment.",
                "vector": np.random.rand(384).tolist(),
                "metadata": {
                    "document_id": "rl_basics",
                    "document_title": "Reinforcement Learning Basics",
                    "document_type": "research",
                    "chunk_index": 0,
                    "topic": "reinforcement_learning",
                    "word_count": 16
                }
            }
        ]

        return sample_docs

    def create_vector_records(self, sample_data: List[dict], schema: DocumentVectorSchema):
        """Convert sample data to vector records."""
        from datetime import datetime

        from src.data_pipeline.vector_stores.schemas.base_schema import VectorRecord

        records = []

        for data in sample_data:
            record = VectorRecord(
                id=data["id"],
                vector=data["vector"],
                text=data["text"],
                metadata=data["metadata"],
                created_at=datetime.now(),
                source="demo",
                source_type="example"
            )
            records.append(record)

        return records

    async def demo_memory_store(self):
        """Demonstrate memory vector store."""
        print("\n" + "="*60)
        print("MEMORY VECTOR STORE DEMO")
        print("="*60)

        # Create configuration
        config = VectorStoreConfig(
            store_type=VectorStoreType.MEMORY,
            collection_name="demo_memory",
            embedding_dimension=384,
            distance_metric=DistanceMetric.COSINE
        )

        # Create schema
        schema = DocumentVectorSchema(config)

        # Create and initialize store
        store = await self.manager.create_store("memory_demo", config)

        # Create sample data
        sample_data = self.create_sample_data()
        records = self.create_vector_records(sample_data, schema)

        print(f"1. Created {len(records)} sample records")

        # Insert vectors
        inserted_ids = await store.insert_vectors(records)
        print(f"2. Inserted {len(inserted_ids)} vectors")

        # Get statistics
        stats = await store.get_stats()
        print(f"3. Store stats: {stats.total_vectors} vectors")

        # Perform vector search
        query_vector = np.random.rand(384).tolist()
        search_query = SearchQuery(
            query_vector=query_vector,
            search_type=SearchType.VECTOR,
            limit=3
        )

        results = await store.search_vectors(search_query)
        print(f"4. Vector search returned {len(results.results)} results")

        for i, result in enumerate(results.results):
            print(f"   Result {i+1}: {result.id} (score: {result.score:.3f})")
            print(f"   Text: {result.text[:80]}...")

        # Perform keyword search
        keyword_query = SearchQuery(
            query_text="machine learning",
            search_type=SearchType.KEYWORD,
            limit=3
        )

        keyword_results = await store.search_vectors(keyword_query)
        print(f"5. Keyword search returned {len(keyword_results.results)} results")

        for i, result in enumerate(keyword_results.results):
            print(f"   Result {i+1}: {result.id} (score: {result.score:.3f})")

        # Perform filtered search
        filters = SearchFilters()
        filters.add_text_filter("topic", "deep_learning", exact=True)

        filtered_query = SearchQuery(
            query_vector=query_vector,
            search_type=SearchType.VECTOR,
            limit=5,
            filters=filters
        )

        filtered_results = await store.search_vectors(filtered_query)
        print(f"6. Filtered search returned {len(filtered_results.results)} results")

        # Test update and delete
        if records:
            # Update a record
            record_to_update = records[0]
            record_to_update.text = "Updated: " + record_to_update.text
            updated_ids = await store.update_vectors([record_to_update])
            print(f"7. Updated {len(updated_ids)} records")

            # Delete a record
            deleted_count = await store.delete_vectors([records[-1].id])
            print(f"8. Deleted {deleted_count} records")

        # Final stats
        final_stats = await store.get_stats()
        print(f"9. Final stats: {final_stats.total_vectors} vectors")

    async def demo_chroma_store(self):
        """Demonstrate ChromaDB vector store."""
        print("\n" + "="*60)
        print("CHROMADB VECTOR STORE DEMO")
        print("="*60)

        try:
            # Create configuration
            config = VectorStoreConfig(
                store_type=VectorStoreType.CHROMA,
                collection_name="demo_chroma",
                embedding_dimension=384,
                distance_metric=DistanceMetric.COSINE,
                persist_directory="data/chroma_demo"
            )

            # Create schema
            schema = DocumentVectorSchema(config)

            # Create and initialize store
            store = await self.manager.create_store("chroma_demo", config)

            # Create sample data
            sample_data = self.create_sample_data()
            records = self.create_vector_records(sample_data, schema)

            print(f"1. Created {len(records)} sample records")

            # Insert vectors
            inserted_ids = await store.insert_vectors(records)
            print(f"2. Inserted {len(inserted_ids)} vectors into ChromaDB")

            # Get statistics
            stats = await store.get_stats()
            print(f"3. ChromaDB stats: {stats.total_vectors} vectors")

            # Perform search
            query_vector = np.random.rand(384).tolist()
            search_query = SearchQuery(
                query_vector=query_vector,
                search_type=SearchType.VECTOR,
                limit=3
            )

            results = await store.search_vectors(search_query)
            print(f"4. ChromaDB search returned {len(results.results)} results")

            for i, result in enumerate(results.results):
                print(f"   Result {i+1}: {result.id} (score: {result.score:.3f})")

            # Test hybrid search
            hybrid_query = SearchQuery(
                query_vector=query_vector,
                query_text="neural networks",
                search_type=SearchType.HYBRID,
                limit=3,
                vector_weight=0.7,
                keyword_weight=0.3
            )

            hybrid_results = await store.search_vectors(hybrid_query)
            print(f"5. Hybrid search returned {len(hybrid_results.results)} results")

        except ImportError:
            print("ChromaDB not available - skipping demo")
        except Exception as e:
            print(f"ChromaDB demo failed: {e}")

    async def demo_faiss_store(self):
        """Demonstrate FAISS vector store."""
        print("\n" + "="*60)
        print("FAISS VECTOR STORE DEMO")
        print("="*60)

        try:
            # Create configuration
            config = VectorStoreConfig(
                store_type=VectorStoreType.FAISS,
                collection_name="demo_faiss",
                embedding_dimension=384,
                distance_metric=DistanceMetric.COSINE,
                persist_directory="data/faiss_demo",
                index_type="flat"  # Use flat index for demo
            )

            # Create schema
            schema = DocumentVectorSchema(config)

            # Create and initialize store
            store = await self.manager.create_store("faiss_demo", config)

            # Create sample data
            sample_data = self.create_sample_data()
            records = self.create_vector_records(sample_data, schema)

            print(f"1. Created {len(records)} sample records")

            # Insert vectors
            inserted_ids = await store.insert_vectors(records)
            print(f"2. Inserted {len(inserted_ids)} vectors into FAISS")

            # Get statistics
            stats = await store.get_stats()
            print(f"3. FAISS stats: {stats.total_vectors} vectors, index type: {stats.index_type}")

            # Perform search
            query_vector = np.random.rand(384).tolist()
            search_query = SearchQuery(
                query_vector=query_vector,
                search_type=SearchType.VECTOR,
                limit=3
            )

            results = await store.search_vectors(search_query)
            print(f"4. FAISS search returned {len(results.results)} results")

            for i, result in enumerate(results.results):
                print(f"   Result {i+1}: {result.id} (score: {result.score:.3f})")

        except ImportError:
            print("FAISS not available - skipping demo")
        except Exception as e:
            print(f"FAISS demo failed: {e}")

    async def demo_store_manager(self):
        """Demonstrate vector store manager."""
        print("\n" + "="*60)
        print("VECTOR STORE MANAGER DEMO")
        print("="*60)

        # Show available store types
        factory = VectorStoreFactory()
        available_stores = factory.get_available_stores()
        print(f"1. Available store types: {[store.value for store in available_stores]}")

        # List current stores
        current_stores = self.manager.list_stores()
        print(f"2. Current stores: {current_stores}")

        # Health check all stores
        health_results = await self.manager.health_check_all()
        print("3. Health check results:")
        for store_name, is_healthy in health_results.items():
            status = "✓ Healthy" if is_healthy else "✗ Unhealthy"
            print(f"   {store_name}: {status}")

        # Get stats for all stores
        all_stats = await self.manager.get_stats_all()
        print("4. Store statistics:")
        for store_name, stats in all_stats.items():
            if "error" in stats:
                print(f"   {store_name}: Error - {stats['error']}")
            else:
                print(f"   {store_name}: {stats.get('total_vectors', 0)} vectors")

        print(f"5. Total managed stores: {len(self.manager)}")

    async def run_all_demos(self):
        """Run all vector store demonstrations."""
        print("VECTOR STORES DEMONSTRATION")
        print("=" * 80)

        try:
            # Run individual store demos
            await self.demo_memory_store()
            await self.demo_chroma_store()
            await self.demo_faiss_store()

            # Run manager demo
            await self.demo_store_manager()

            print("\n" + "="*80)
            print("ALL DEMOS COMPLETED SUCCESSFULLY!")
            print("="*80)

        except Exception as e:
            print(f"\nError during demos: {e}")
            import traceback
            traceback.print_exc()

        finally:
            # Clean up
            await self.manager.close_all_stores()
            print("\nCleaned up all stores")

async def main():
    """Main demonstration function."""
    demo = VectorStoreDemo()
    await demo.run_all_demos()

if __name__ == "__main__":
    asyncio.run(main())
