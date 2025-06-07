"""
Complete pipeline example demonstrating document processing, vectorization, and vector storage.
"""

import asyncio
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.data_pipeline.document_processing import (
    DocumentProcessor,
    DocumentProcessingConfig,
    ParsingConfig,
    ChunkingConfig
)
from src.data_pipeline.vectorization import (
    EmbeddingConfig,
    HuggingFaceEmbedder,
    BatchVectorProcessor,
    BatchProcessingConfig,
    VectorCache,
    CacheConfig
)
from src.data_pipeline.vector_stores.schemas import (
    DocumentVectorSchema,
    VectorStoreConfig,
    VectorStoreType,
    DistanceMetric
)

class CompletePipelineDemo:
    """Complete pipeline demonstration."""

    def __init__(self):
        """Initialize pipeline components."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize document processor
        self.document_processor = self._create_document_processor()

        # Initialize embedder
        self.embedder = self._create_embedder()

        # Initialize batch processor
        self.batch_processor = self._create_batch_processor()

        # Initialize vector store schema
        self.vector_schema = self._create_vector_schema()

    def _create_document_processor(self) -> DocumentProcessor:
        """Create document processor with optimized configuration."""
        parsing_config = ParsingConfig(
            extract_metadata=True,
            extract_tables=False,  # Disable for performance
            extract_images=False,  # Disable for performance
            normalize_whitespace=True,
            preserve_formatting=False
        )

        chunking_config = ChunkingConfig(
            chunk_size=512,  # Optimal for most embedding models
            chunk_overlap=50,
            strategy="text",
            preserve_sentences=True,
            preserve_paragraphs=True
        )

        processing_config = DocumentProcessingConfig(
            parsing_config=parsing_config,
            chunking_config=chunking_config,
            enable_chunking=True,
            enable_metadata_enrichment=True
        )

        return DocumentProcessor(processing_config)

    def _create_embedder(self) -> HuggingFaceEmbedder:
        """Create HuggingFace embedder."""
        embedding_config = EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",  # Fast and efficient model
            model_provider="huggingface",
            embedding_dimension=384,
            max_input_length=512,
            normalize_embeddings=True,
            batch_size=32
        )

        return HuggingFaceEmbedder(
            config=embedding_config,
            use_sentence_transformers=True
        )

    def _create_batch_processor(self) -> BatchVectorProcessor:
        """Create batch vector processor with caching."""
        cache_config = CacheConfig(
            backend="file",
            cache_dir="cache/embeddings",
            ttl=86400 * 7,  # 1 week
            max_size=50000
        )

        batch_config = BatchProcessingConfig(
            batch_size=32,
            max_workers=2,
            enable_caching=True,
            cache_config=cache_config,
            show_progress=True,
            continue_on_error=True
        )

        return BatchVectorProcessor(self.embedder, batch_config)

    def _create_vector_schema(self) -> DocumentVectorSchema:
        """Create vector store schema."""
        store_config = VectorStoreConfig(
            store_type=VectorStoreType.CHROMA,
            collection_name="document_chunks",
            embedding_dimension=384,
            distance_metric=DistanceMetric.COSINE
        )

        return DocumentVectorSchema(store_config)

    def create_sample_documents(self) -> Path:
        """Create sample documents for testing."""
        sample_dir = Path("data/pipeline_demo")
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Create multiple sample documents
        documents = {
            "ai_overview.md": """
# Artificial Intelligence Overview

## Introduction
Artificial Intelligence (AI) represents one of the most significant technological
advancements of our time. It encompasses machine learning, deep learning, natural
language processing, and computer vision.

## Machine Learning
Machine learning is a subset of AI that enables systems to learn and improve from
experience without explicit programming. Key approaches include:

### Supervised Learning
- Uses labeled training data
- Common algorithms: Linear Regression, Decision Trees, Random Forest
- Applications: Classification, Regression

### Unsupervised Learning
- Finds patterns in unlabeled data
- Techniques: Clustering, Dimensionality Reduction
- Applications: Customer Segmentation, Anomaly Detection

### Reinforcement Learning
- Learns through interaction with environment
- Uses reward/penalty system
- Applications: Game Playing, Robotics, Autonomous Vehicles

## Applications
AI has transformative applications across industries:
- Healthcare: Medical diagnosis, drug discovery
- Finance: Fraud detection, algorithmic trading
- Transportation: Autonomous vehicles, route optimization
- Technology: Recommendation systems, virtual assistants
""",

            "data_science.txt": """
Data Science: The Interdisciplinary Field

Data science combines statistics, computer science, and domain expertise to extract
insights from data. It involves the entire data lifecycle from collection to
actionable insights.

Key Components:
1. Data Collection and Storage
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis
4. Statistical Modeling
5. Machine Learning
6. Data Visualization
7. Communication of Results

Tools and Technologies:
- Programming: Python, R, SQL
- Libraries: Pandas, NumPy, Scikit-learn, TensorFlow
- Visualization: Matplotlib, Seaborn, Plotly, Tableau
- Big Data: Spark, Hadoop, Kafka
- Cloud Platforms: AWS, Azure, Google Cloud

The Data Science Process:
1. Problem Definition
2. Data Acquisition
3. Data Exploration
4. Data Preparation
5. Model Building
6. Model Evaluation
7. Deployment and Monitoring

Career Paths:
- Data Scientist
- Data Analyst
- Machine Learning Engineer
- Data Engineer
- Business Intelligence Analyst
""",

            "cloud_computing.md": """
# Cloud Computing Fundamentals

## What is Cloud Computing?
Cloud computing delivers computing services over the internet, including servers,
storage, databases, networking, software, and analytics.

## Service Models

### Infrastructure as a Service (IaaS)
- Virtual machines, storage, networks
- Examples: AWS EC2, Azure VMs, Google Compute Engine
- Use cases: Development environments, backup solutions

### Platform as a Service (PaaS)
- Development platforms and tools
- Examples: AWS Lambda, Azure App Service, Google App Engine
- Use cases: Application development, API hosting

### Software as a Service (SaaS)
- Complete applications over the internet
- Examples: Office 365, Salesforce, Google Workspace
- Use cases: Email, CRM, productivity tools

## Deployment Models

### Public Cloud
- Services offered over public internet
- Owned by cloud service providers
- Cost-effective, scalable

### Private Cloud
- Dedicated to single organization
- Enhanced security and control
- Higher costs

### Hybrid Cloud
- Combination of public and private
- Flexibility and optimization
- Complex management

## Benefits
- Cost Efficiency: Pay-as-you-use model
- Scalability: Scale resources up or down
- Accessibility: Access from anywhere
- Reliability: High availability and disaster recovery
- Security: Enterprise-grade security measures

## Challenges
- Security and Privacy concerns
- Vendor lock-in
- Internet dependency
- Compliance requirements
"""
        }

        # Write documents to files
        for filename, content in documents.items():
            with open(sample_dir / filename, "w", encoding="utf-8") as f:
                f.write(content.strip())

        return sample_dir

    async def run_complete_pipeline(self):
        """Run the complete document processing and vectorization pipeline."""
        print("\n" + "="*80)
        print("COMPLETE DOCUMENT PROCESSING AND VECTORIZATION PIPELINE")
        print("="*80)

        # Step 1: Create sample documents
        print("\n1. Creating sample documents...")
        sample_dir = self.create_sample_documents()
        document_files = list(sample_dir.glob("*"))
        print(f"   Created {len(document_files)} sample documents")

        # Step 2: Process documents
        print("\n2. Processing documents...")
        all_chunks = []
        processing_results = []

        for doc_file in document_files:
            print(f"   Processing: {doc_file.name}")
            result = self.document_processor.process_file(doc_file)
            processing_results.append(result)
            all_chunks.extend(result.chunks)

            print(f"     - Text length: {len(result.get_text())} chars")
            print(f"     - Chunks created: {len(result.chunks)}")
            print(f"     - Processing time: {result.processing_time:.2f}s")

        print(f"\n   Total chunks across all documents: {len(all_chunks)}")

        # Step 3: Generate embeddings
        print("\n3. Generating embeddings...")
        vectorization_result = await self.batch_processor.process_chunks_async(all_chunks)

        print(f"   - Total items processed: {vectorization_result.total_items}")
        print(f"   - Successful embeddings: {vectorization_result.successful_items}")
        print(f"   - Cached results: {vectorization_result.cached_items}")
        print(f"   - Processing time: {vectorization_result.total_time:.2f}s")
        print(f"   - Average time per item: {vectorization_result.average_time_per_item:.3f}s")

        if vectorization_result.cache_hit_rate:
            print(f"   - Cache hit rate: {vectorization_result.cache_hit_rate:.1%}")

        # Step 4: Create vector records
        print("\n4. Creating vector store records...")
        vector_records = []
        successful_embeddings = vectorization_result.get_successful_results()

        for i, (chunk, embedding_result) in enumerate(zip(all_chunks, successful_embeddings)):
            if embedding_result is not None:
                # Get document metadata from processing results
                doc_metadata = None
                for proc_result in processing_results:
                    if chunk in proc_result.chunks:
                        doc_metadata = proc_result.get_metadata()
                        break

                if doc_metadata:
                    vector_record = self.vector_schema.create_record(
                        chunk_metadata=chunk.metadata,
                        document_metadata=doc_metadata,
                        vector=embedding_result.embedding,
                        embedding_model=embedding_result.model_name,
                        processing_time=embedding_result.processing_time
                    )
                    vector_records.append(vector_record)

        print(f"   Created {len(vector_records)} vector records")

        # Step 5: Demonstrate vector record usage
        print("\n5. Vector record analysis...")
        if vector_records:
            # Analyze vector records
            total_text_length = sum(len(record.text) for record in vector_records)
            avg_text_length = total_text_length / len(vector_records)

            # Group by document
            doc_groups = {}
            for record in vector_records:
                doc_id = record.document_id
                if doc_id not in doc_groups:
                    doc_groups[doc_id] = []
                doc_groups[doc_id].append(record)

            print(f"   - Average chunk length: {avg_text_length:.0f} characters")
            print(f"   - Documents processed: {len(doc_groups)}")

            for doc_id, records in doc_groups.items():
                print(f"     * {doc_id}: {len(records)} chunks")

            # Show sample record
            sample_record = vector_records[0]
            print(f"\n   Sample record:")
            print(f"     - ID: {sample_record.id}")
            print(f"     - Document: {sample_record.document_title}")
            print(f"     - Chunk index: {sample_record.chunk_index}")
            print(f"     - Text preview: {sample_record.text[:100]}...")
            print(f"     - Vector dimension: {len(sample_record.vector)}")
            print(f"     - Embedding model: {sample_record.embedding_model}")

        # Step 6: Performance summary
        print("\n6. Performance Summary...")
        total_processing_time = sum(r.processing_time for r in processing_results)
        total_vectorization_time = vectorization_result.total_time
        total_pipeline_time = total_processing_time + total_vectorization_time

        print(f"   - Document processing time: {total_processing_time:.2f}s")
        print(f"   - Vectorization time: {total_vectorization_time:.2f}s")
        print(f"   - Total pipeline time: {total_pipeline_time:.2f}s")
        print(f"   - Documents per second: {len(document_files) / total_pipeline_time:.2f}")
        print(f"   - Chunks per second: {len(all_chunks) / total_pipeline_time:.2f}")

        # Step 7: Cache statistics
        cache_stats = self.batch_processor.get_cache_stats()
        if cache_stats:
            print(f"\n7. Cache Statistics...")
            print(f"   - Cache hits: {cache_stats['hits']}")
            print(f"   - Cache misses: {cache_stats['misses']}")
            print(f"   - Hit rate: {cache_stats['hit_rate']:.1%}")
            print(f"   - Cache size: {cache_stats['size']} items")

        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*80)

        return {
            "processing_results": processing_results,
            "vectorization_result": vectorization_result,
            "vector_records": vector_records,
            "performance": {
                "total_time": total_pipeline_time,
                "processing_time": total_processing_time,
                "vectorization_time": total_vectorization_time
            }
        }

async def main():
    """Main demonstration function."""
    try:
        # Create and run pipeline demo
        demo = CompletePipelineDemo()
        results = await demo.run_complete_pipeline()

        print(f"\nDemo completed successfully!")
        print(f"Processed {len(results['processing_results'])} documents")
        print(f"Created {len(results['vector_records'])} vector records")

    except Exception as e:
        print(f"\nError during pipeline demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
