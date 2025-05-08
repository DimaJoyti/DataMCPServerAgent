"""
Enhanced example of using distributed memory with Redis or MongoDB.
This example demonstrates the use of the DistributedMemoryManager for scalable memory operations.
"""

import asyncio
import os
import sys
import time
import random
import logging
from typing import Dict, List, Any, Optional

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.distributed_memory_manager import DistributedMemoryManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_basic_operations(memory_manager: DistributedMemoryManager) -> None:
    """Test basic memory operations.
    
    Args:
        memory_manager: Distributed memory manager
    """
    logger.info("Testing basic memory operations...")
    
    # Test entity operations
    entity_type = "test_entity"
    entity_id = f"entity_{int(time.time())}"
    entity_data = {
        "name": "Test Entity",
        "description": "This is a test entity",
        "attributes": {
            "attr1": "value1",
            "attr2": 42,
            "attr3": True
        },
        "tags": ["test", "example", "distributed_memory"]
    }
    
    # Save entity
    await memory_manager.save_entity(entity_type, entity_id, entity_data)
    logger.info(f"Saved entity: {entity_type}:{entity_id}")
    
    # Load entity
    loaded_entity = await memory_manager.load_entity(entity_type, entity_id)
    logger.info(f"Loaded entity: {loaded_entity}")
    
    # Test conversation history
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you for asking!"},
        {"role": "user", "content": "Tell me about distributed memory."},
        {"role": "assistant", "content": "Distributed memory is a memory architecture where memory is physically distributed across multiple storage nodes but logically shared."}
    ]
    
    # Save conversation history
    await memory_manager.save_conversation_history(messages)
    logger.info("Saved conversation history")
    
    # Load conversation history
    loaded_history = await memory_manager.load_conversation_history()
    logger.info(f"Loaded conversation history: {len(loaded_history)} messages")
    
    # Test tool usage
    tool_name = "search_tool"
    args = {"query": "distributed memory architecture"}
    result = "Example search results for distributed memory architecture"
    
    # Save tool usage
    await memory_manager.save_tool_usage(tool_name, args, result)
    logger.info(f"Saved tool usage for {tool_name}")
    
    # Delete entity
    deleted = await memory_manager.delete_entity(entity_type, entity_id)
    logger.info(f"Deleted entity {entity_type}:{entity_id}: {deleted}")


async def test_cache_performance(memory_manager: DistributedMemoryManager) -> None:
    """Test cache performance.
    
    Args:
        memory_manager: Distributed memory manager
    """
    logger.info("Testing cache performance...")
    
    # Create test entities
    entity_type = "performance_test"
    num_entities = 10
    
    # Create entities
    for i in range(num_entities):
        entity_id = f"perf_entity_{i}"
        entity_data = {
            "name": f"Performance Test Entity {i}",
            "value": random.randint(1, 1000),
            "timestamp": time.time()
        }
        
        await memory_manager.save_entity(entity_type, entity_id, entity_data)
    
    logger.info(f"Created {num_entities} test entities")
    
    # Test cache performance
    logger.info("Testing cache hits...")
    
    # First access (cache misses)
    start_time = time.time()
    for i in range(num_entities):
        entity_id = f"perf_entity_{i}"
        await memory_manager.load_entity(entity_type, entity_id, use_cache=True)
    
    first_access_time = time.time() - start_time
    logger.info(f"First access time (cache misses): {first_access_time:.4f} seconds")
    
    # Second access (cache hits)
    start_time = time.time()
    for i in range(num_entities):
        entity_id = f"perf_entity_{i}"
        await memory_manager.load_entity(entity_type, entity_id, use_cache=True)
    
    second_access_time = time.time() - start_time
    logger.info(f"Second access time (cache hits): {second_access_time:.4f} seconds")
    logger.info(f"Cache performance improvement: {(first_access_time / second_access_time):.2f}x faster")
    
    # Get cache metrics
    metrics = memory_manager.get_metrics()
    logger.info(f"Cache hits: {metrics['cache_hits']}")
    logger.info(f"Cache misses: {metrics['cache_misses']}")
    
    # Clear cache
    memory_manager.clear_cache()
    logger.info("Cleared cache")
    
    # Clean up test entities
    for i in range(num_entities):
        entity_id = f"perf_entity_{i}"
        await memory_manager.delete_entity(entity_type, entity_id)
    
    logger.info(f"Deleted {num_entities} test entities")


async def test_error_handling(memory_manager: DistributedMemoryManager) -> None:
    """Test error handling.
    
    Args:
        memory_manager: Distributed memory manager
    """
    logger.info("Testing error handling...")
    
    # Test ping
    is_accessible = await memory_manager.backend.ping()
    logger.info(f"Backend is accessible: {is_accessible}")
    
    if not is_accessible:
        logger.error("Backend is not accessible, skipping error handling tests")
        return
    
    # Test invalid operations
    try:
        # Try to load a non-existent entity
        entity = await memory_manager.load_entity("non_existent_type", "non_existent_id")
        logger.info(f"Load non-existent entity result: {entity}")
    except Exception as e:
        logger.error(f"Error loading non-existent entity: {e}")
    
    # Get memory summary
    summary = await memory_manager.get_memory_summary()
    logger.info("Memory Summary:")
    logger.info(summary)


async def run_example():
    """Run the distributed memory example."""
    logger.info("Running enhanced distributed memory example...")
    
    # Choose the backend type (redis or mongodb)
    backend_type = input("Choose backend type (redis or mongodb): ").strip().lower()
    
    if backend_type not in ["redis", "mongodb"]:
        logger.warning("Invalid backend type. Using redis as default.")
        backend_type = "redis"
    
    # Create the distributed memory manager
    if backend_type == "redis":
        # Redis configuration
        host = input("Redis host (default: localhost): ").strip() or "localhost"
        port = int(input("Redis port (default: 6379): ").strip() or "6379")
        db = int(input("Redis database (default: 0): ").strip() or "0")
        password = input("Redis password (default: none): ").strip() or None
        
        config = {
            "host": host,
            "port": port,
            "db": db,
            "password": password,
            "prefix": "datamcp_test:"
        }
    else:
        # MongoDB configuration
        connection_string = input("MongoDB connection string (default: mongodb://localhost:27017/): ").strip() or "mongodb://localhost:27017/"
        database_name = input("MongoDB database name (default: datamcp_memory): ").strip() or "datamcp_memory"
        
        config = {
            "connection_string": connection_string,
            "database_name": database_name
        }
    
    # Create memory manager
    memory_manager = DistributedMemoryManager(
        memory_type=backend_type,
        config=config,
        namespace="test"
    )
    
    # Run tests
    try:
        # Test basic operations
        await test_basic_operations(memory_manager)
        
        # Test cache performance
        await test_cache_performance(memory_manager)
        
        # Test error handling
        await test_error_handling(memory_manager)
        
        logger.info("All tests completed successfully!")
    except Exception as e:
        logger.error(f"Error during tests: {e}")
    finally:
        # Get final metrics
        metrics = memory_manager.get_metrics()
        logger.info(f"Final metrics: {metrics}")


if __name__ == "__main__":
    asyncio.run(run_example())
