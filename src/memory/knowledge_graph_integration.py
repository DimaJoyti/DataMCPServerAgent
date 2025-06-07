"""
Knowledge graph integration for DataMCPServerAgent.
This module integrates the knowledge graph with the distributed memory manager.
"""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from langchain_anthropic import ChatAnthropic

from src.memory.distributed_memory_manager import DistributedMemoryManager
from src.memory.knowledge_graph import KnowledgeGraph
from src.memory.knowledge_graph_manager import KnowledgeGraphManager
from src.memory.memory_persistence import MemoryDatabase
from src.utils.error_handlers import format_error_for_user

# Configure logging
logger = logging.getLogger(__name__)

class KnowledgeGraphIntegration:
    """Integration of knowledge graph with distributed memory manager."""

    def __init__(
        self,
        memory_manager: DistributedMemoryManager,
        db: MemoryDatabase,
        model: Optional[ChatAnthropic] = None,
        namespace: str = "datamcp"
    ):
        """Initialize the knowledge graph integration.

        Args:
            memory_manager: Distributed memory manager
            db: Memory database for persistence
            model: Language model for entity extraction and relationship identification
            namespace: Namespace for the knowledge graph
        """
        self.memory_manager = memory_manager
        self.db = db
        self.model = model
        self.namespace = namespace

        # Initialize knowledge graph manager
        self.kg_manager = KnowledgeGraphManager(memory_manager, db, model, namespace)

        # Register event handlers
        self._register_event_handlers()

    def _register_event_handlers(self) -> None:
        """Register event handlers for distributed memory events."""
        # Override save_entity method
        original_save_entity = self.memory_manager.save_entity

        async def save_entity_with_kg(
            entity_type: str,
            entity_id: str,
            entity_data: Dict[str, Any],
            cache: bool = True
        ) -> None:
            # Call original method
            await original_save_entity(entity_type, entity_id, entity_data, cache)

            # Process entity for knowledge graph
            try:
                await self.kg_manager.process_entity(entity_type, entity_id, entity_data)
            except Exception as e:
                error_message = format_error_for_user(e)
                logger.error(f"Failed to process entity for knowledge graph: {error_message}")

        # Override save_conversation_message method
        original_save_conversation_message = self.memory_manager.save_conversation_message

        async def save_conversation_message_with_kg(
            message: Dict[str, Any],
            conversation_id: str = "default"
        ) -> None:
            # Call original method
            await original_save_conversation_message(message, conversation_id)

            # Process conversation message for knowledge graph
            try:
                await self.kg_manager.process_conversation_message(message, conversation_id)
            except Exception as e:
                error_message = format_error_for_user(e)
                logger.error(f"Failed to process conversation message for knowledge graph: {error_message}")

        # Override save_tool_usage method
        original_save_tool_usage = self.memory_manager.save_tool_usage

        async def save_tool_usage_with_kg(
            tool_name: str,
            args: Dict[str, Any],
            result: Any
        ) -> None:
            # Call original method
            await original_save_tool_usage(tool_name, args, result)

            # Process tool usage for knowledge graph
            try:
                await self.kg_manager.process_tool_usage(tool_name, args, result)
            except Exception as e:
                error_message = format_error_for_user(e)
                logger.error(f"Failed to process tool usage for knowledge graph: {error_message}")

        # Replace methods
        self.memory_manager.save_entity = save_entity_with_kg
        self.memory_manager.save_conversation_message = save_conversation_message_with_kg
        self.memory_manager.save_tool_usage = save_tool_usage_with_kg

    async def get_context_for_request(
        self,
        request: str,
        max_entities: int = 10,
        max_relationships: int = 20
    ) -> Dict[str, Any]:
        """Get relevant context from the knowledge graph for a request.

        Args:
            request: User request
            max_entities: Maximum number of entities to return
            max_relationships: Maximum number of relationships to return

        Returns:
            Relevant context
        """
        return await self.kg_manager.get_context_for_request(
            request, max_entities, max_relationships
        )

    async def get_knowledge_graph_summary(self) -> Dict[str, Any]:
        """Get a summary of the knowledge graph.

        Returns:
            Knowledge graph summary
        """
        try:
            # Get node counts by type
            node_types = {}
            for node_type in self.kg_manager.entity_type_mapping.values():
                nodes = self.kg_manager.knowledge_graph.get_nodes_by_type(node_type)
                if nodes:
                    node_types[node_type] = len(nodes)

            # Get edge counts by type
            edge_types = {}
            for edge_type in self.kg_manager.relationship_mapping.values():
                edges = self.kg_manager.knowledge_graph.get_edges(edge_type=edge_type)
                if edges:
                    edge_types[edge_type] = len(edges)

            # Get total counts
            total_nodes = sum(node_types.values())
            total_edges = sum(edge_types.values())

            return {
                "total_nodes": total_nodes,
                "total_edges": total_edges,
                "node_types": node_types,
                "edge_types": edge_types
            }
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to get knowledge graph summary: {error_message}")
            return {
                "total_nodes": 0,
                "total_edges": 0,
                "node_types": {},
                "edge_types": {}
            }

    async def execute_sparql_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a SPARQL query on the knowledge graph.

        Args:
            query: SPARQL query

        Returns:
            Query results
        """
        return self.kg_manager.knowledge_graph.execute_sparql_query(query)

    async def get_entity_context(
        self,
        entity_type: str,
        entity_id: str,
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """Get context for an entity from the knowledge graph.

        Args:
            entity_type: Entity type
            entity_id: Entity ID
            max_depth: Maximum depth of relationships to include

        Returns:
            Entity context
        """
        try:
            # Get node ID
            node_id = f"{entity_type}_{entity_id}"

            # Get node
            node = self.kg_manager.knowledge_graph.get_node(node_id)
            if not node:
                return {"entity": None, "neighbors": [], "relationships": []}

            # Get neighbors
            neighbors = []
            for depth in range(1, max_depth + 1):
                # Get neighbors at current depth
                current_neighbors = self.kg_manager.knowledge_graph.get_neighbors(
                    node_id, direction="both"
                )

                # Add to neighbors list
                neighbors.extend(current_neighbors)

                # Get next level of neighbors
                if depth < max_depth:
                    for neighbor in current_neighbors:
                        neighbor_neighbors = self.kg_manager.knowledge_graph.get_neighbors(
                            neighbor["id"], direction="both"
                        )
                        neighbors.extend(neighbor_neighbors)

            # Deduplicate neighbors
            seen_ids = set()
            unique_neighbors = []
            for neighbor in neighbors:
                if neighbor["id"] not in seen_ids:
                    seen_ids.add(neighbor["id"])
                    unique_neighbors.append(neighbor)

            # Get relationships between entity and neighbors
            relationships = []
            for neighbor in unique_neighbors:
                # Get relationships from entity to neighbor
                outgoing = self.kg_manager.knowledge_graph.get_edges(
                    source_id=node_id, target_id=neighbor["id"]
                )
                relationships.extend(outgoing)

                # Get relationships from neighbor to entity
                incoming = self.kg_manager.knowledge_graph.get_edges(
                    source_id=neighbor["id"], target_id=node_id
                )
                relationships.extend(incoming)

            return {
                "entity": node,
                "neighbors": unique_neighbors,
                "relationships": relationships
            }
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to get entity context: {error_message}")
            return {"entity": None, "neighbors": [], "relationships": []}
