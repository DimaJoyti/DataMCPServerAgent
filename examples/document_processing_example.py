"""
Example demonstrating document processing pipeline with parsing, chunking, and metadata extraction.
"""

import asyncio
import logging
import os
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

from src.data_pipeline.document_processing import (
    DocumentProcessor,
    DocumentProcessingConfig,
    ParsingConfig,
    ChunkingConfig
)


def create_sample_documents():
    """Create sample documents for testing."""
    sample_dir = Path("data/sample_documents")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample text file
    text_content = """
    # Introduction to Machine Learning
    
    Machine learning is a subset of artificial intelligence (AI) that provides systems 
    the ability to automatically learn and improve from experience without being 
    explicitly programmed.
    
    ## Types of Machine Learning
    
    ### Supervised Learning
    Supervised learning uses labeled training data to learn a mapping function from 
    input variables to output variables. Common algorithms include:
    
    - Linear Regression
    - Decision Trees
    - Random Forest
    - Support Vector Machines
    
    ### Unsupervised Learning
    Unsupervised learning finds hidden patterns in data without labeled examples. 
    Key techniques include:
    
    - Clustering (K-means, Hierarchical)
    - Dimensionality Reduction (PCA, t-SNE)
    - Association Rules
    
    ### Reinforcement Learning
    Reinforcement learning involves an agent learning to make decisions by taking 
    actions in an environment to maximize cumulative reward.
    
    ## Applications
    
    Machine learning has numerous applications across industries:
    
    1. **Healthcare**: Medical diagnosis, drug discovery, personalized treatment
    2. **Finance**: Fraud detection, algorithmic trading, credit scoring
    3. **Technology**: Recommendation systems, natural language processing, computer vision
    4. **Transportation**: Autonomous vehicles, route optimization, predictive maintenance
    
    ## Conclusion
    
    Machine learning continues to evolve and transform how we solve complex problems 
    across various domains. Understanding its fundamentals is crucial for leveraging 
    its potential effectively.
    """
    
    with open(sample_dir / "ml_introduction.txt", "w", encoding="utf-8") as f:
        f.write(text_content)
    
    # Create sample markdown file
    markdown_content = """---
title: "Data Pipeline Architecture"
author: "DataMCP Team"
date: "2024-01-15"
tags: ["data", "pipeline", "architecture"]
---

# Data Pipeline Architecture

## Overview

A data pipeline is a series of data processing steps that move data from one or more 
sources to a destination where it can be stored and analyzed.

## Components

### Data Ingestion
- **Batch Ingestion**: Processing data in large chunks at scheduled intervals
- **Stream Ingestion**: Processing data in real-time as it arrives
- **Hybrid Approach**: Combining both batch and stream processing

### Data Transformation
Data transformation involves:
1. Cleaning and validation
2. Format conversion
3. Aggregation and summarization
4. Enrichment with additional data

### Data Storage
- **Data Lakes**: Store raw data in its native format
- **Data Warehouses**: Store structured, processed data
- **Operational Databases**: Support real-time applications

## Best Practices

> "The goal is to turn data into information, and information into insight." - Carly Fiorina

### Design Principles
- **Scalability**: Handle increasing data volumes
- **Reliability**: Ensure data quality and consistency
- **Monitoring**: Track pipeline performance and health
- **Security**: Protect sensitive data throughout the pipeline

### Implementation Tips
```python
# Example pipeline configuration
pipeline_config = {
    "source": "s3://data-bucket/raw/",
    "destination": "postgresql://warehouse/",
    "batch_size": 10000,
    "schedule": "0 2 * * *"  # Daily at 2 AM
}
```

## Conclusion

Effective data pipeline architecture is crucial for modern data-driven organizations. 
It enables reliable, scalable, and efficient data processing workflows.
"""
    
    with open(sample_dir / "pipeline_architecture.md", "w", encoding="utf-8") as f:
        f.write(markdown_content)
    
    return sample_dir


def demonstrate_basic_processing():
    """Demonstrate basic document processing."""
    print("\n" + "="*60)
    print("BASIC DOCUMENT PROCESSING DEMO")
    print("="*60)
    
    # Create sample documents
    sample_dir = create_sample_documents()
    
    # Initialize document processor with default configuration
    processor = DocumentProcessor()
    
    # Process text file
    print("\n1. Processing text file...")
    text_file = sample_dir / "ml_introduction.txt"
    result = processor.process_file(text_file)
    
    print(f"   - Document ID: {result.document_id}")
    print(f"   - Text length: {len(result.get_text())} characters")
    print(f"   - Number of chunks: {len(result.chunks)}")
    print(f"   - Processing time: {result.processing_time:.2f} seconds")
    print(f"   - Status: {result.processing_status}")
    
    if result.chunks:
        print(f"   - First chunk preview: {result.chunks[0].text[:100]}...")
    
    # Process markdown file
    print("\n2. Processing markdown file...")
    md_file = sample_dir / "pipeline_architecture.md"
    result = processor.process_file(md_file)
    
    print(f"   - Document ID: {result.document_id}")
    print(f"   - Text length: {len(result.get_text())} characters")
    print(f"   - Number of chunks: {len(result.chunks)}")
    print(f"   - Processing time: {result.processing_time:.2f} seconds")
    print(f"   - Metadata title: {result.get_metadata().title}")
    print(f"   - Metadata author: {result.get_metadata().author}")
    print(f"   - Metadata keywords: {result.get_metadata().keywords}")


def demonstrate_custom_configuration():
    """Demonstrate document processing with custom configuration."""
    print("\n" + "="*60)
    print("CUSTOM CONFIGURATION DEMO")
    print("="*60)
    
    # Create custom configuration
    parsing_config = ParsingConfig(
        extract_metadata=True,
        normalize_whitespace=True,
        preserve_formatting=False
    )
    
    chunking_config = ChunkingConfig(
        chunk_size=500,  # Smaller chunks
        chunk_overlap=100,
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
    
    # Initialize processor with custom configuration
    processor = DocumentProcessor(processing_config)
    
    # Process with custom configuration
    sample_dir = Path("data/sample_documents")
    text_file = sample_dir / "ml_introduction.txt"
    
    print("\n1. Processing with custom configuration...")
    result = processor.process_file(text_file)
    
    print(f"   - Number of chunks: {len(result.chunks)}")
    print(f"   - Average chunk size: {sum(len(chunk.text) for chunk in result.chunks) / len(result.chunks):.0f} chars")
    
    # Show chunk details
    print("\n2. Chunk details:")
    for i, chunk in enumerate(result.chunks[:3]):  # Show first 3 chunks
        print(f"   Chunk {i+1}:")
        print(f"     - Length: {len(chunk.text)} characters")
        print(f"     - Word count: {chunk.metadata.word_count}")
        print(f"     - Preview: {chunk.text[:80]}...")


def demonstrate_content_processing():
    """Demonstrate processing content directly (without files)."""
    print("\n" + "="*60)
    print("CONTENT PROCESSING DEMO")
    print("="*60)
    
    # Sample content
    content = """
    Artificial Intelligence and Machine Learning
    
    Artificial Intelligence (AI) is the simulation of human intelligence in machines 
    that are programmed to think and learn like humans. Machine Learning (ML) is a 
    subset of AI that enables machines to learn from data without explicit programming.
    
    Key Concepts:
    - Neural Networks: Inspired by biological neural networks
    - Deep Learning: Multi-layered neural networks
    - Natural Language Processing: Understanding human language
    - Computer Vision: Interpreting visual information
    
    Applications include autonomous vehicles, medical diagnosis, financial trading, 
    and recommendation systems.
    """
    
    processor = DocumentProcessor()
    
    print("\n1. Processing text content...")
    result = processor.process_content(
        content=content,
        document_id="ai_ml_overview",
        document_type="text",
        title="AI and ML Overview",
        author="Example Author"
    )
    
    print(f"   - Document ID: {result.document_id}")
    print(f"   - Text length: {len(result.get_text())} characters")
    print(f"   - Number of chunks: {len(result.chunks)}")
    print(f"   - Title: {result.get_metadata().title}")
    print(f"   - Author: {result.get_metadata().author}")


def demonstrate_chunking_strategies():
    """Demonstrate different chunking strategies."""
    print("\n" + "="*60)
    print("CHUNKING STRATEGIES DEMO")
    print("="*60)
    
    sample_text = """
    Data science is an interdisciplinary field that uses scientific methods, processes, 
    algorithms and systems to extract knowledge and insights from structured and 
    unstructured data. Data science is related to data mining, machine learning and big data.
    
    Data science is a "concept to unify statistics, data analysis, informatics, and their 
    related methods" in order to "understand and analyze actual phenomena" with data. 
    It uses techniques and theories drawn from many fields within the context of mathematics, 
    statistics, computer science, information science, and domain knowledge.
    
    The term "data science" has been traced back to 1974, when Peter Naur proposed it as 
    an alternative name for computer science. In 1996, the International Federation of 
    Classification Societies became the first conference to specifically feature data science.
    """
    
    strategies = ["text", "semantic", "adaptive"]
    
    for strategy in strategies:
        print(f"\n{strategy.upper()} CHUNKING:")
        
        try:
            chunking_config = ChunkingConfig(
                chunk_size=300,
                chunk_overlap=50,
                strategy=strategy
            )
            
            processing_config = DocumentProcessingConfig(
                chunking_config=chunking_config
            )
            
            processor = DocumentProcessor(processing_config)
            result = processor.process_content(
                content=sample_text,
                document_id=f"data_science_{strategy}",
                document_type="text"
            )
            
            print(f"   - Number of chunks: {len(result.chunks)}")
            for i, chunk in enumerate(result.chunks):
                print(f"   - Chunk {i+1}: {len(chunk.text)} chars - {chunk.text[:60]}...")
                
        except Exception as e:
            print(f"   - Strategy '{strategy}' not available: {e}")


def demonstrate_metadata_extraction():
    """Demonstrate metadata extraction and enrichment."""
    print("\n" + "="*60)
    print("METADATA EXTRACTION DEMO")
    print("="*60)
    
    # Process a document and show detailed metadata
    sample_dir = Path("data/sample_documents")
    md_file = sample_dir / "pipeline_architecture.md"
    
    processor = DocumentProcessor()
    result = processor.process_file(md_file)
    
    metadata = result.get_metadata()
    
    print("\n1. Basic Metadata:")
    print(f"   - Document ID: {metadata.document_id}")
    print(f"   - Document Type: {metadata.document_type}")
    print(f"   - Title: {metadata.title}")
    print(f"   - Author: {metadata.author}")
    print(f"   - Keywords: {metadata.keywords}")
    print(f"   - Language: {metadata.language}")
    
    print("\n2. Content Statistics:")
    print(f"   - Character Count: {metadata.character_count}")
    print(f"   - Word Count: {metadata.word_count}")
    print(f"   - Sentence Count: {metadata.sentence_count}")
    print(f"   - Paragraph Count: {metadata.paragraph_count}")
    
    print("\n3. Processing Information:")
    print(f"   - Processing Status: {metadata.processing_status}")
    print(f"   - Processing Time: {metadata.processing_time}")
    print(f"   - Processed At: {metadata.processed_at}")
    
    print("\n4. Custom Fields:")
    for key, value in metadata.custom_fields.items():
        print(f"   - {key}: {value}")


def main():
    """Main demonstration function."""
    print("DOCUMENT PROCESSING PIPELINE DEMONSTRATION")
    print("=" * 80)
    
    try:
        # Run demonstrations
        demonstrate_basic_processing()
        demonstrate_custom_configuration()
        demonstrate_content_processing()
        demonstrate_chunking_strategies()
        demonstrate_metadata_extraction()
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("="*80)
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
