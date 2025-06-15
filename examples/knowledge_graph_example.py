"""
Example script demonstrating knowledge graph integration for better context understanding.
"""

import asyncio
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.memory.distributed_memory_manager import DistributedMemoryManager
from src.memory.knowledge_graph_integration import KnowledgeGraphIntegration
from src.memory.memory_persistence import MemoryDatabase

# Load environment variables
load_dotenv()

# Initialize model
model = ChatAnthropic(model=os.getenv("MODEL_NAME", "claude-3-5-sonnet-20240620"))

async def setup_memory_systems():
    """Set up memory systems with knowledge graph integration."""
    # Initialize memory database
    db_path = os.getenv("MEMORY_DB_PATH", "knowledge_graph_example.db")
    db = MemoryDatabase(db_path)

    # Initialize distributed memory manager
    memory_type = os.getenv("MEMORY_TYPE", "sqlite")
    memory_config = {}

    if memory_type == "redis":
        memory_config = {
            "host": os.getenv("REDIS_HOST", "localhost"),
            "port": int(os.getenv("REDIS_PORT", "6379")),
            "db": int(os.getenv("REDIS_DB", "0")),
            "password": os.getenv("REDIS_PASSWORD", None),
            "prefix": "datamcp_kg_example:"
        }
    elif memory_type == "mongodb":
        memory_config = {
            "connection_string": os.getenv("MONGODB_URI", "mongodb://localhost:27017/"),
            "database_name": os.getenv("MONGODB_DB", "agent_memory")
        }

    memory_manager = DistributedMemoryManager(
        memory_type=memory_type,
        config=memory_config,
        namespace="kg_example"
    )

    # Initialize knowledge graph integration
    kg_integration = KnowledgeGraphIntegration(
        memory_manager=memory_manager,
        db=db,
        model=model,
        namespace="kg_example"
    )

    return memory_manager, kg_integration

async def simulate_conversation(memory_manager, kg_integration):
    """Simulate a conversation with knowledge graph integration."""
    print("=== Simulating Conversation with Knowledge Graph Integration ===")

    conversation_id = "kg_example_conversation"

    # First message
    print("\n--- User Message 1 ---")
    user_message1 = {
        "role": "user",
        "content": "I'm interested in learning about distributed memory systems for my agent architecture."
    }
    print(user_message1["content"])

    # Save message to memory
    await memory_manager.save_conversation_message(user_message1, conversation_id)

    # Generate response
    system_prompt = """You are a helpful AI assistant with expertise in agent architectures and distributed systems."""

    response1 = await model.ainvoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message1["content"])
        ]
    )

    assistant_message1 = {
        "role": "assistant",
        "content": response1.content
    }
    print("\n--- Assistant Response 1 ---")
    print(assistant_message1["content"][:300] + "...")

    # Save message to memory
    await memory_manager.save_conversation_message(assistant_message1, conversation_id)

    # Second message
    print("\n--- User Message 2 ---")
    user_message2 = {
        "role": "user",
        "content": "What are the advantages of using Redis for distributed memory compared to MongoDB?"
    }
    print(user_message2["content"])

    # Save message to memory
    await memory_manager.save_conversation_message(user_message2, conversation_id)

    # Get context from knowledge graph
    context = await kg_integration.get_context_for_request(user_message2["content"])

    # Generate response with context
    context_str = ""
    if context["entities"]:
        context_str = "Relevant context:\n"
        for entity in context["entities"]:
            context_str += f"- {entity['type']}: {entity['properties'].get('name', '')}\n"
            for key, value in entity['properties'].items():
                if key not in ['name', 'timestamp', 'source_type']:
                    context_str += f"  - {key}: {value}\n"

    response2 = await model.ainvoke(
        [
            SystemMessage(content=system_prompt + "\n\n" + context_str if context_str else system_prompt),
            HumanMessage(content=user_message1["content"]),
            AIMessage(content=assistant_message1["content"]),
            HumanMessage(content=user_message2["content"])
        ]
    )

    assistant_message2 = {
        "role": "assistant",
        "content": response2.content
    }
    print("\n--- Assistant Response 2 ---")
    print(assistant_message2["content"][:300] + "...")

    # Save message to memory
    await memory_manager.save_conversation_message(assistant_message2, conversation_id)

    # Third message
    print("\n--- User Message 3 ---")
    user_message3 = {
        "role": "user",
        "content": "How can I implement a knowledge graph to improve context understanding in my agent?"
    }
    print(user_message3["content"])

    # Save message to memory
    await memory_manager.save_conversation_message(user_message3, conversation_id)

    # Get context from knowledge graph
    context = await kg_integration.get_context_for_request(user_message3["content"])

    # Generate response with context
    context_str = ""
    if context["entities"]:
        context_str = "Relevant context:\n"
        for entity in context["entities"]:
            context_str += f"- {entity['type']}: {entity['properties'].get('name', '')}\n"
            for key, value in entity['properties'].items():
                if key not in ['name', 'timestamp', 'source_type']:
                    context_str += f"  - {key}: {value}\n"

    response3 = await model.ainvoke(
        [
            SystemMessage(content=system_prompt + "\n\n" + context_str if context_str else system_prompt),
            HumanMessage(content=user_message1["content"]),
            AIMessage(content=assistant_message1["content"]),
            HumanMessage(content=user_message2["content"]),
            AIMessage(content=assistant_message2["content"]),
            HumanMessage(content=user_message3["content"])
        ]
    )

    assistant_message3 = {
        "role": "assistant",
        "content": response3.content
    }
    print("\n--- Assistant Response 3 ---")
    print(assistant_message3["content"][:300] + "...")

    # Save message to memory
    await memory_manager.save_conversation_message(assistant_message3, conversation_id)

    # Get knowledge graph summary
    kg_summary = await kg_integration.get_knowledge_graph_summary()
    print("\n=== Knowledge Graph Summary ===")
    print(f"Total Nodes: {kg_summary['total_nodes']}")
    print(f"Total Edges: {kg_summary['total_edges']}")
    print("\nNode Types:")
    for node_type, count in kg_summary['node_types'].items():
        print(f"- {node_type}: {count}")
    print("\nEdge Types:")
    for edge_type, count in kg_summary['edge_types'].items():
        print(f"- {edge_type}: {count}")

async def main():
    """Main function."""
    # Set up memory systems
    memory_manager, kg_integration = await setup_memory_systems()

    # Simulate conversation
    await simulate_conversation(memory_manager, kg_integration)

if __name__ == "__main__":
    asyncio.run(main())
