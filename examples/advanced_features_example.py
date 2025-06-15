"""
Example demonstrating advanced features: vector stores, web interface, new formats, and async processing.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add src to path for imports
import sys

sys.path.append(str(Path(__file__).parent.parent))

from src.data_pipeline.async_processing import AsyncDocumentProcessor, TaskManager, TaskPriority
from src.data_pipeline.document_processing import DocumentProcessor
from src.data_pipeline.vector_stores.schemas import (
    DistanceMetric,
    VectorStoreConfig,
    VectorStoreType,
)
from src.data_pipeline.vector_stores.vector_store_manager import VectorStoreManager
from src.data_pipeline.vectorization import (
    BatchVectorProcessor,
    EmbeddingConfig,
    HuggingFaceEmbedder,
)


class AdvancedFeaturesDemo:
    """Demonstration of advanced pipeline features."""

    def __init__(self):
        """Initialize demo."""
        self.logger = logging.getLogger(self.__class__.__name__)

    async def demo_new_document_formats(self):
        """Demonstrate new document format support."""
        print("\n" + "="*60)
        print("NEW DOCUMENT FORMATS DEMO")
        print("="*60)

        # Create processor
        processor = DocumentProcessor()

        # Demo data directory
        demo_dir = Path("demo_documents")
        demo_dir.mkdir(exist_ok=True)

        # Create sample files for demonstration
        sample_files = self._create_sample_files(demo_dir)

        print(f"1. Created {len(sample_files)} sample files")

        # Process each file type
        for file_path in sample_files:
            try:
                print(f"\n2. Processing {file_path.name} ({file_path.suffix})")

                result = processor.process_file(file_path)

                print(f"   - Document ID: {result.document_id}")
                print(f"   - Text length: {len(result.get_text())} characters")
                print(f"   - Chunks: {len(result.chunks)}")
                print(f"   - Processing time: {result.processing_time:.2f}s")
                print(f"   - Document type: {result.get_metadata().document_type}")

                # Show first 100 characters of text
                text_preview = result.get_text()[:100] + "..." if len(result.get_text()) > 100 else result.get_text()
                print(f"   - Preview: {text_preview}")

            except Exception as e:
                print(f"   - Error processing {file_path.name}: {e}")

        # Cleanup
        self._cleanup_demo_files(demo_dir)

    async def demo_vector_stores(self):
        """Demonstrate vector store functionality."""
        print("\n" + "="*60)
        print("VECTOR STORES DEMO")
        print("="*60)

        # Create vector store manager
        manager = VectorStoreManager()

        # Create different types of vector stores
        stores_config = [
            ("memory_store", VectorStoreType.MEMORY),
            ("chroma_store", VectorStoreType.CHROMA),
            ("faiss_store", VectorStoreType.FAISS)
        ]

        created_stores = []

        for store_name, store_type in stores_config:
            try:
                config = VectorStoreConfig(
                    store_type=store_type,
                    collection_name=store_name,
                    embedding_dimension=384,
                    distance_metric=DistanceMetric.COSINE,
                    persist_directory=f"data/{store_name}" if store_type != VectorStoreType.MEMORY else None
                )

                store = await manager.create_store(store_name, config)
                created_stores.append((store_name, store))
                print(f"1. Created {store_type.value} vector store: {store_name}")

            except ImportError as e:
                print(f"1. Skipped {store_type.value} (missing dependencies): {e}")
            except Exception as e:
                print(f"1. Failed to create {store_type.value}: {e}")

        if not created_stores:
            print("No vector stores available for demo")
            return

        # Create sample vectors
        embedder = HuggingFaceEmbedder(
            EmbeddingConfig(
                model_name="all-MiniLM-L6-v2",
                embedding_dimension=384
            )
        )

        sample_texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks with multiple layers.",
            "Natural language processing enables computers to understand human language.",
            "Computer vision allows machines to interpret visual data.",
            "Reinforcement learning involves agents learning through interaction."
        ]

        print(f"\n2. Generating embeddings for {len(sample_texts)} texts")

        # Generate embeddings
        batch_processor = BatchVectorProcessor(embedder)
        embedding_result = batch_processor.process_texts(sample_texts)

        print(f"   - Generated {len(embedding_result.results)} embeddings")
        print(f"   - Processing time: {embedding_result.total_time:.2f}s")

        # Test each vector store
        for store_name, store in created_stores:
            try:
                print(f"\n3. Testing {store_name}")

                # Create vector records
                from datetime import datetime

                from src.data_pipeline.vector_stores.schemas.base_schema import VectorRecord

                records = []
                for i, (text, embedding_result) in enumerate(zip(sample_texts, embedding_result.results)):
                    if embedding_result:
                        record = VectorRecord(
                            id=f"doc_{i}",
                            vector=embedding_result.embedding,
                            text=text,
                            metadata={"index": i, "topic": "ai"},
                            created_at=datetime.now(),
                            source="demo"
                        )
                        records.append(record)

                # Insert vectors
                inserted_ids = await store.insert_vectors(records)
                print(f"   - Inserted {len(inserted_ids)} vectors")

                # Search vectors
                from src.data_pipeline.vector_stores.schemas.search_models import (
                    SearchQuery,
                    SearchType,
                )

                query_embedding = embedding_result.results[0].embedding
                search_query = SearchQuery(
                    query_vector=query_embedding,
                    search_type=SearchType.VECTOR,
                    limit=3
                )

                search_results = await store.search_vectors(search_query)
                print(f"   - Search returned {len(search_results.results)} results")
                print(f"   - Search time: {search_results.search_time:.3f}s")

                # Get statistics
                stats = await store.get_stats()
                print(f"   - Total vectors: {stats.total_vectors}")

            except Exception as e:
                print(f"   - Error testing {store_name}: {e}")

        # Cleanup
        await manager.close_all_stores()

    async def demo_async_processing(self):
        """Demonstrate asynchronous processing."""
        print("\n" + "="*60)
        print("ASYNC PROCESSING DEMO")
        print("="*60)

        # Create async document processor
        async_processor = AsyncDocumentProcessor(max_workers=2)

        # Create sample files
        demo_dir = Path("demo_async")
        demo_dir.mkdir(exist_ok=True)
        sample_files = self._create_sample_files(demo_dir)

        print(f"1. Created {len(sample_files)} files for async processing")

        # Process files asynchronously
        start_time = time.time()

        async def progress_callback(completed, total, progress):
            print(f"   Progress: {completed}/{total} ({progress:.1f}%)")

        results = await async_processor.process_files_async(
            sample_files,
            progress_callback=progress_callback
        )

        async_time = time.time() - start_time

        print("\n2. Async processing completed:")
        print(f"   - Processed {len(results)} files")
        print(f"   - Total time: {async_time:.2f}s")
        print(f"   - Average time per file: {async_time/len(sample_files):.2f}s")

        # Compare with synchronous processing
        sync_processor = DocumentProcessor()
        start_time = time.time()

        sync_results = []
        for file_path in sample_files:
            try:
                result = sync_processor.process_file(file_path)
                sync_results.append(result)
            except Exception as e:
                print(f"   Error processing {file_path}: {e}")

        sync_time = time.time() - start_time

        print("\n3. Sync processing comparison:")
        print(f"   - Processed {len(sync_results)} files")
        print(f"   - Total time: {sync_time:.2f}s")
        print(f"   - Average time per file: {sync_time/len(sample_files):.2f}s")
        print(f"   - Speedup: {sync_time/async_time:.1f}x")

        # Cleanup
        await async_processor.close()
        self._cleanup_demo_files(demo_dir)

    async def demo_task_queue(self):
        """Demonstrate task queue system."""
        print("\n" + "="*60)
        print("TASK QUEUE DEMO")
        print("="*60)

        # Create task manager
        task_manager = TaskManager(max_workers=3)
        await task_manager.start()

        print("1. Started task manager with 3 workers")

        # Define sample tasks
        async def sample_task(name: str, duration: float) -> str:
            """Sample async task."""
            await asyncio.sleep(duration)
            return f"Task {name} completed after {duration}s"

        def sample_sync_task(name: str, duration: float) -> str:
            """Sample sync task."""
            time.sleep(duration)
            return f"Sync task {name} completed after {duration}s"

        # Submit various tasks
        task_ids = []

        # High priority tasks
        for i in range(3):
            task_id = await task_manager.submit_task(
                sample_task,
                f"high_{i}",
                0.5,
                name=f"High Priority Task {i}",
                priority=TaskPriority.HIGH
            )
            task_ids.append(task_id)

        # Normal priority tasks
        for i in range(5):
            task_id = await task_manager.submit_task(
                sample_sync_task,
                f"normal_{i}",
                1.0,
                name=f"Normal Priority Task {i}",
                priority=TaskPriority.NORMAL
            )
            task_ids.append(task_id)

        # Low priority tasks
        for i in range(2):
            task_id = await task_manager.submit_task(
                sample_task,
                f"low_{i}",
                0.3,
                name=f"Low Priority Task {i}",
                priority=TaskPriority.LOW
            )
            task_ids.append(task_id)

        print(f"2. Submitted {len(task_ids)} tasks")

        # Monitor task progress
        completed_tasks = 0
        while completed_tasks < len(task_ids):
            await asyncio.sleep(0.5)

            stats = task_manager.get_stats()
            completed_tasks = stats['tasks_processed'] + stats['tasks_failed']

            print(f"   Progress: {completed_tasks}/{len(task_ids)} tasks completed")
            print(f"   Queue size: {stats['queue_size']}")
            print(f"   Running: {stats['running_tasks']}")

        # Show final results
        print("\n3. All tasks completed:")
        stats = task_manager.get_stats()
        print(f"   - Processed: {stats['tasks_processed']}")
        print(f"   - Failed: {stats['tasks_failed']}")
        print(f"   - Success rate: {stats['success_rate']:.1f}%")
        print(f"   - Average execution time: {stats['average_execution_time']:.2f}s")

        # Stop task manager
        await task_manager.stop()
        print("4. Task manager stopped")

    def _create_sample_files(self, demo_dir: Path) -> List[Path]:
        """Create sample files for testing."""
        files = []

        # Text file
        text_file = demo_dir / "sample.txt"
        text_file.write_text("This is a sample text file for testing document processing.")
        files.append(text_file)

        # Markdown file
        md_file = demo_dir / "sample.md"
        md_file.write_text("""# Sample Markdown

This is a **sample** markdown file with:
- Lists
- *Emphasis*
- `Code`

## Section 2
More content here.
""")
        files.append(md_file)

        # CSV file
        csv_file = demo_dir / "sample.csv"
        csv_file.write_text("""Name,Age,City
John,25,New York
Jane,30,London
Bob,35,Paris
""")
        files.append(csv_file)

        # HTML file
        html_file = demo_dir / "sample.html"
        html_file.write_text("""<!DOCTYPE html>
<html>
<head><title>Sample HTML</title></head>
<body>
<h1>Sample HTML Document</h1>
<p>This is a sample HTML file for testing.</p>
<ul>
<li>Item 1</li>
<li>Item 2</li>
</ul>
</body>
</html>
""")
        files.append(html_file)

        return files

    def _cleanup_demo_files(self, demo_dir: Path):
        """Clean up demo files."""
        try:
            import shutil
            if demo_dir.exists():
                shutil.rmtree(demo_dir)
        except Exception as e:
            self.logger.warning(f"Failed to cleanup demo files: {e}")

    async def run_all_demos(self):
        """Run all demonstrations."""
        print("ADVANCED FEATURES DEMONSTRATION")
        print("=" * 80)

        try:
            await self.demo_new_document_formats()
            await self.demo_vector_stores()
            await self.demo_async_processing()
            await self.demo_task_queue()

            print("\n" + "="*80)
            print("ALL ADVANCED DEMOS COMPLETED SUCCESSFULLY!")
            print("="*80)

        except Exception as e:
            print(f"\nError during demos: {e}")
            import traceback
            traceback.print_exc()

async def main():
    """Main demonstration function."""
    demo = AdvancedFeaturesDemo()
    await demo.run_all_demos()

if __name__ == "__main__":
    asyncio.run(main())
