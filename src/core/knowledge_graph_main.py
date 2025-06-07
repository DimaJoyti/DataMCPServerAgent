"""
Knowledge graph agent entry point.
This module provides the main entry point for the knowledge graph agent.
"""

import asyncio
import logging
import os
import sys
from typing import Dict, Any, Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.memory.distributed_memory_manager import DistributedMemoryManager
from src.memory.knowledge_graph_integration import KnowledgeGraphIntegration
from src.memory.memory_persistence import MemoryDatabase
from src.utils.error_handlers import format_error_for_user

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

async def setup_knowledge_graph_agent():
    """Set up the knowledge graph agent.

    Returns:
        Tuple of (model, memory_manager, kg_integration)
    """
    try:
        # Initialize model
        model = ChatAnthropic(model=os.getenv("MODEL_NAME", "claude-3-5-sonnet-20240620"))

        # Initialize memory database
        db_path = os.getenv("MEMORY_DB_PATH", "knowledge_graph_agent.db")
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
                "prefix": "datamcp_kg:"
            }
        elif memory_type == "mongodb":
            memory_config = {
                "connection_string": os.getenv("MONGODB_URI", "mongodb://localhost:27017/"),
                "database_name": os.getenv("MONGODB_DB", "agent_memory")
            }

        memory_manager = DistributedMemoryManager(
            memory_type=memory_type,
            config=memory_config,
            namespace="knowledge_graph"
        )

        # Initialize knowledge graph integration
        kg_integration = KnowledgeGraphIntegration(
            memory_manager=memory_manager,
            db=db,
            model=model,
            namespace="knowledge_graph"
        )

        return model, memory_manager, kg_integration
    except Exception as e:
        error_message = format_error_for_user(e)
        logger.error(f"Failed to set up knowledge graph agent: {error_message}")
        raise

async def chat_with_knowledge_graph_agent():
    """Chat with the knowledge graph agent.

    This function provides an interactive chat interface with the knowledge graph agent.
    """
    try:
        # Set up agent
        model, memory_manager, kg_integration = await setup_knowledge_graph_agent()

        # Initialize conversation
        conversation_id = f"knowledge_graph_{int(asyncio.get_event_loop().time())}"
        history = []

        # System prompt
        system_prompt = """You are a helpful AI assistant with knowledge graph capabilities.
        You can extract entities and relationships from text, store them in a knowledge graph,
        and use the knowledge graph to provide more relevant and contextual responses.

        Special commands:
        - !kg_summary: Get a summary of the knowledge graph
        - !kg_context <query>: Get context from the knowledge graph for a query
        - !kg_entity <type> <id>: Get context for an entity from the knowledge graph
        - !kg_sparql <query>: Execute a SPARQL query on the knowledge graph
        - !help: Show this help message
        - !exit: Exit the chat
        """

        print("Welcome to the Knowledge Graph Agent!")
        print("Type '!help' for a list of commands or '!exit' to exit.")
        print("=" * 50)

        while True:
            # Get user input
            user_input = input("\nYou: ")

            # Check for special commands
            if user_input.lower() == "!exit":
                print("Goodbye!")
                break
            elif user_input.lower() == "!help":
                print("\nSpecial commands:")
                print("- !kg_summary: Get a summary of the knowledge graph")
                print("- !kg_context <query>: Get context from the knowledge graph for a query")
                print("- !kg_entity <type> <id>: Get context for an entity from the knowledge graph")
                print("- !kg_sparql <query>: Execute a SPARQL query on the knowledge graph")
                print("- !help: Show this help message")
                print("- !exit: Exit the chat")
                continue
            elif user_input.lower() == "!kg_summary":
                # Get knowledge graph summary
                summary = await kg_integration.get_knowledge_graph_summary()
                print("\nKnowledge Graph Summary:")
                print(f"Total Nodes: {summary['total_nodes']}")
                print(f"Total Edges: {summary['total_edges']}")
                print("\nNode Types:")
                for node_type, count in summary['node_types'].items():
                    print(f"- {node_type}: {count}")
                print("\nEdge Types:")
                for edge_type, count in summary['edge_types'].items():
                    print(f"- {edge_type}: {count}")
                continue
            elif user_input.lower().startswith("!kg_context "):
                # Get context from knowledge graph
                query = user_input[12:].strip()
                context = await kg_integration.get_context_for_request(query)
                print("\nContext from Knowledge Graph:")
                print(f"Found {len(context['entities'])} entities and {len(context['relationships'])} relationships")
                print("\nEntities:")
                for entity in context['entities']:
                    print(f"- {entity['type']}: {entity['properties'].get('name', '')}")
                print("\nRelationships:")
                for relationship in context['relationships']:
                    print(f"- {relationship['source']} -> {relationship['type']} -> {relationship['target']}")
                continue
            elif user_input.lower().startswith("!kg_entity "):
                # Get entity context
                parts = user_input[11:].strip().split(" ", 1)
                if len(parts) != 2:
                    print("Invalid command format. Use: !kg_entity <type> <id>")
                    continue
                entity_type, entity_id = parts
                context = await kg_integration.get_entity_context(entity_type, entity_id)
                print("\nEntity Context:")
                if context['entity']:
                    print(f"Entity: {context['entity']['type']} - {context['entity']['properties'].get('name', '')}")
                    print(f"Properties: {context['entity']['properties']}")
                    print(f"\nNeighbors: {len(context['neighbors'])}")
                    for neighbor in context['neighbors']:
                        print(f"- {neighbor['type']}: {neighbor['properties'].get('name', '')}")
                    print(f"\nRelationships: {len(context['relationships'])}")
                    for relationship in context['relationships']:
                        print(f"- {relationship['source']} -> {relationship['type']} -> {relationship['target']}")
                else:
                    print("Entity not found")
                continue
            elif user_input.lower().startswith("!kg_sparql "):
                # Execute SPARQL query
                query = user_input[11:].strip()
                results = await kg_integration.execute_sparql_query(query)
                print("\nSPARQL Query Results:")
                print(f"Found {len(results)} results")
                for i, result in enumerate(results):
                    print(f"\nResult {i + 1}:")
                    for key, value in result.items():
                        print(f"- {key}: {value}")
                continue

            # Save user message
            user_message = {
                "role": "user",
                "content": user_input
            }
            history.append(user_message)
            await memory_manager.save_conversation_message(user_message, conversation_id)

            # Get context from knowledge graph
            context = await kg_integration.get_context_for_request(user_input)

            # Generate response with context
            context_str = ""
            if context["entities"]:
                context_str = "Relevant context from knowledge graph:\n"
                for entity in context["entities"]:
                    context_str += f"- {entity['type']}: {entity['properties'].get('name', '')}\n"
                    for key, value in entity['properties'].items():
                        if key not in ['name', 'timestamp', 'source_type']:
                            context_str += f"  - {key}: {value}\n"

            # Create messages for model
            messages = [SystemMessage(content=system_prompt + "\n\n" + context_str if context_str else system_prompt)]

            # Add conversation history
            for message in history:
                if message["role"] == "user":
                    messages.append(HumanMessage(content=message["content"]))
                else:
                    messages.append(AIMessage(content=message["content"]))

            # Generate response
            response = await model.ainvoke(messages)

            # Save assistant message
            assistant_message = {
                "role": "assistant",
                "content": response.content
            }
            history.append(assistant_message)
            await memory_manager.save_conversation_message(assistant_message, conversation_id)

            # Print response
            print(f"\nAssistant: {response.content}")
    except Exception as e:
        error_message = format_error_for_user(e)
        logger.error(f"Error in chat with knowledge graph agent: {error_message}")
        print(f"\nError: {error_message}")

if __name__ == "__main__":
    asyncio.run(chat_with_knowledge_graph_agent())
