"""
Example of using distributed memory with Redis or MongoDB.
"""

import asyncio
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory.distributed_memory import DistributedMemoryFactory

async def run_example():
    """Run the distributed memory example."""
    print("Running distributed memory example...")

    # Choose the backend type (redis or mongodb)
    backend_type = input("Choose backend type (redis or mongodb): ").strip().lower()

    if backend_type not in ["redis", "mongodb"]:
        print("Invalid backend type. Using redis as default.")
        backend_type = "redis"

    # Create the distributed memory backend
    if backend_type == "redis":
        # Redis configuration
        host = input("Redis host (default: localhost): ").strip() or "localhost"
        port = int(input("Redis port (default: 6379): ").strip() or "6379")
        db = int(input("Redis database (default: 0): ").strip() or "0")
        password = input("Redis password (default: none): ").strip() or None

        memory_backend = DistributedMemoryFactory.create_memory_backend(
            "redis",
            host=host,
            port=port,
            db=db,
            password=password
        )
    else:
        # MongoDB configuration
        connection_string = input("MongoDB connection string (default: mongodb://localhost:27017/): ").strip() or "mongodb://localhost:27017/"
        database_name = input("MongoDB database name (default: datamcp_memory): ").strip() or "datamcp_memory"

        memory_backend = DistributedMemoryFactory.create_memory_backend(
            "mongodb",
            connection_string=connection_string,
            database_name=database_name
        )

    # Example operations
    print("\nPerforming example operations...")

    # Save an entity
    await memory_backend.save_entity(
        "product",
        "product123",
        {
            "name": "Example Product",
            "price": 99.99,
            "description": "This is an example product"
        }
    )
    print("Saved entity: product123")

    # Load the entity
    entity = await memory_backend.load_entity("product", "product123")
    print(f"Loaded entity: {entity}")

    # Save conversation history
    await memory_backend.save_conversation_history([
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you for asking!"}
    ])
    print("Saved conversation history")

    # Load conversation history
    history = await memory_backend.load_conversation_history()
    print(f"Loaded conversation history: {len(history)} messages")

    # Save tool usage
    await memory_backend.save_tool_usage(
        "search_tool",
        {"query": "example search"},
        "Example search results"
    )
    print("Saved tool usage")

    # Get memory summary
    summary = await memory_backend.get_memory_summary()
    print("\nMemory Summary:")
    print(summary)

if __name__ == "__main__":
    asyncio.run(run_example())
