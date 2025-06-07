"""
Knowledge graph manager for DataMCPServerAgent.
This module integrates the knowledge graph with the distributed memory manager.
"""

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from src.memory.distributed_memory_manager import DistributedMemoryManager
from src.memory.knowledge_graph import KnowledgeGraph
from src.memory.memory_persistence import MemoryDatabase
from src.utils.error_handlers import format_error_for_user

# Configure logging
logger = logging.getLogger(__name__)

class KnowledgeGraphManager:
    """Manager for integrating knowledge graph with distributed memory."""

    def __init__(
        self,
        memory_manager: DistributedMemoryManager,
        db: MemoryDatabase,
        model: Optional[ChatAnthropic] = None,
        namespace: str = "datamcp",
    ):
        """Initialize the knowledge graph manager.

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

        # Initialize knowledge graph
        self.knowledge_graph = KnowledgeGraph(db, namespace)

        # Entity type mapping
        self.entity_type_mapping = {
            "person": "Person",
            "organization": "Organization",
            "location": "Location",
            "product": "Product",
            "event": "Event",
            "concept": "Concept",
            "document": "Document",
            "website": "Website",
            "tool": "Tool",
            "query": "Query",
            "response": "Response",
        }

        # Relationship type mapping
        self.relationship_mapping = {
            "has_property": "hasProperty",
            "related_to": "relatedTo",
            "part_of": "partOf",
            "has_part": "hasPart",
            "created_by": "createdBy",
            "created": "created",
            "located_in": "locatedIn",
            "contains": "contains",
            "mentioned_in": "mentionedIn",
            "mentions": "mentions",
            "used_by": "usedBy",
            "uses": "uses",
            "preceded_by": "precededBy",
            "follows": "follows",
            "similar_to": "similarTo",
            "same_as": "sameAs",
            "instance_of": "instanceOf",
            "has_instance": "hasInstance",
        }

    async def extract_entities_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities from text using the language model.

        Args:
            text: Text to extract entities from

        Returns:
            List of extracted entities
        """
        if not self.model:
            logger.warning("No language model provided for entity extraction")
            return []

        try:
            # Create prompt for entity extraction
            system_prompt = """
            Extract entities from the provided text. For each entity, identify:
            1. Entity type (person, organization, location, product, event, concept, document, website, tool, query, response)
            2. Entity name or identifier
            3. Properties (key-value pairs)

            Format your response as a JSON array of objects, each with 'type', 'name', and 'properties' fields.
            Example:
            [
                {
                    "type": "person",
                    "name": "John Smith",
                    "properties": {
                        "age": 35,
                        "occupation": "software engineer"
                    }
                },
                {
                    "type": "organization",
                    "name": "Acme Corp",
                    "properties": {
                        "industry": "technology",
                        "founded": 2005
                    }
                }
            ]

            Only extract entities that are clearly identifiable in the text. If no entities are found, return an empty array.
            """

            # Generate response
            response = await self.model.ainvoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=text)]
            )

            # Extract JSON from response
            json_match = re.search(r"\[.*\]", response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                entities = json.loads(json_str)
                return entities

            return []
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to extract entities from text: {error_message}")
            return []

    async def identify_relationships(
        self, entities: List[Dict[str, Any]], text: str
    ) -> List[Dict[str, Any]]:
        """Identify relationships between entities using the language model.

        Args:
            entities: List of entities
            text: Original text

        Returns:
            List of relationships
        """
        if not self.model or not entities or len(entities) < 2:
            return []

        try:
            # Create prompt for relationship identification
            system_prompt = """
            Identify relationships between the provided entities based on the text. For each relationship, identify:
            1. Source entity (by name)
            2. Target entity (by name)
            3. Relationship type (choose from: has_property, related_to, part_of, has_part, created_by, created, located_in, contains, mentioned_in, mentions, used_by, uses, preceded_by, follows, similar_to, same_as, instance_of, has_instance)
            4. Properties (key-value pairs)

            Format your response as a JSON array of objects, each with 'source', 'target', 'type', and 'properties' fields.
            Example:
            [
                {
                    "source": "John Smith",
                    "target": "Acme Corp",
                    "type": "part_of",
                    "properties": {
                        "role": "employee",
                        "since": 2018
                    }
                },
                {
                    "source": "Acme Corp",
                    "target": "New York",
                    "type": "located_in",
                    "properties": {
                        "headquarters": true
                    }
                }
            ]

            Only identify relationships that are clearly stated or strongly implied in the text. If no relationships are found, return an empty array.
            """

            # Create entity list for context
            entity_context = "\n".join(
                [
                    f"{i + 1}. {entity['name']} (Type: {entity['type']})"
                    for i, entity in enumerate(entities)
                ]
            )

            # Generate response
            response = await self.model.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(
                        content=f"Text: {text}\n\nEntities:\n{entity_context}"
                    ),
                ]
            )

            # Extract JSON from response
            json_match = re.search(r"\[.*\]", response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                relationships = json.loads(json_str)
                return relationships

            return []
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to identify relationships: {error_message}")
            return []

    async def add_entities_to_graph(
        self, entities: List[Dict[str, Any]]
    ) -> Dict[str, str]:
        """Add entities to the knowledge graph.

        Args:
            entities: List of entities

        Returns:
            Mapping of entity names to node IDs
        """
        entity_id_mapping = {}

        for entity in entities:
            try:
                # Map entity type
                entity_type = entity.get("type", "").lower()
                mapped_type = self.entity_type_mapping.get(entity_type, "Entity")

                # Add entity to graph
                node_id = self.knowledge_graph.add_node(
                    node_type=mapped_type,
                    properties={
                        "name": entity.get("name", ""),
                        "source_type": entity_type,
                        **entity.get("properties", {}),
                    },
                )

                # Add to mapping
                entity_id_mapping[entity.get("name", "")] = node_id
            except Exception as e:
                error_message = format_error_for_user(e)
                logger.error(f"Failed to add entity to graph: {error_message}")

        return entity_id_mapping

    async def add_relationships_to_graph(
        self, relationships: List[Dict[str, Any]], entity_id_mapping: Dict[str, str]
    ) -> None:
        """Add relationships to the knowledge graph.

        Args:
            relationships: List of relationships
            entity_id_mapping: Mapping of entity names to node IDs
        """
        for relationship in relationships:
            try:
                source_name = relationship.get("source", "")
                target_name = relationship.get("target", "")

                # Skip if source or target not in mapping
                if (
                    source_name not in entity_id_mapping
                    or target_name not in entity_id_mapping
                ):
                    continue

                source_id = entity_id_mapping[source_name]
                target_id = entity_id_mapping[target_name]

                # Map relationship type
                rel_type = relationship.get("type", "").lower()
                mapped_type = self.relationship_mapping.get(rel_type, "relatedTo")

                # Add relationship to graph
                self.knowledge_graph.add_edge(
                    source_id=source_id,
                    target_id=target_id,
                    edge_type=mapped_type,
                    properties=relationship.get("properties", {}),
                )
            except Exception as e:
                error_message = format_error_for_user(e)
                logger.error(f"Failed to add relationship to graph: {error_message}")

    async def process_text(self, text: str, context_id: str = "") -> Dict[str, Any]:
        """Process text to extract entities and relationships and add them to the knowledge graph.

        Args:
            text: Text to process
            context_id: Optional context ID for grouping related entities

        Returns:
            Processing results
        """
        try:
            # Extract entities
            entities = await self.extract_entities_from_text(text)

            if not entities:
                return {"entities": [], "relationships": []}

            # Identify relationships
            relationships = await self.identify_relationships(entities, text)

            # Add entities to graph
            entity_id_mapping = await self.add_entities_to_graph(entities)

            # Add relationships to graph
            await self.add_relationships_to_graph(relationships, entity_id_mapping)

            # If context ID provided, create a context node and link entities to it
            if context_id:
                context_node_id = self.knowledge_graph.add_node(
                    node_type="Context",
                    properties={"context_id": context_id, "timestamp": time.time()},
                )

                # Link entities to context
                for entity_name, entity_id in entity_id_mapping.items():
                    self.knowledge_graph.add_edge(
                        source_id=context_node_id,
                        target_id=entity_id,
                        edge_type="contains",
                        properties={"source": "text_processing"},
                    )

            return {
                "entities": [
                    {**entity, "id": entity_id_mapping.get(entity.get("name", ""), "")}
                    for entity in entities
                ],
                "relationships": relationships,
            }
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to process text: {error_message}")
            return {"entities": [], "relationships": []}

    async def process_conversation_message(
        self, message: Dict[str, Any], conversation_id: str
    ) -> Dict[str, Any]:
        """Process a conversation message to extract entities and relationships.

        Args:
            message: Conversation message
            conversation_id: Conversation ID

        Returns:
            Processing results
        """
        try:
            # Extract text from message
            text = message.get("content", "")
            role = message.get("role", "")

            # Process text
            results = await self.process_text(text, context_id=conversation_id)

            # Create message node
            message_id = str(time.time())
            message_node_id = self.knowledge_graph.add_node(
                node_type="Message",
                properties={
                    "content": text,
                    "role": role,
                    "conversation_id": conversation_id,
                    "timestamp": time.time(),
                },
                node_id=f"message_{message_id}",
            )

            # Link message to conversation
            conversation_nodes = self.knowledge_graph.search_nodes(
                property_filters={"conversation_id": conversation_id},
                node_types=["Conversation"],
            )

            conversation_node_id = None
            if conversation_nodes:
                conversation_node_id = conversation_nodes[0]["id"]
            else:
                # Create conversation node if it doesn't exist
                conversation_node_id = self.knowledge_graph.add_node(
                    node_type="Conversation",
                    properties={
                        "conversation_id": conversation_id,
                        "start_time": time.time(),
                    },
                    node_id=f"conversation_{conversation_id}",
                )

            # Link message to conversation
            self.knowledge_graph.add_edge(
                source_id=conversation_node_id,
                target_id=message_node_id,
                edge_type="contains",
                properties={"order": message_id},
            )

            # Link entities to message
            for entity in results.get("entities", []):
                entity_id = entity.get("id", "")
                if entity_id:
                    self.knowledge_graph.add_edge(
                        source_id=message_node_id,
                        target_id=entity_id,
                        edge_type="mentions",
                        properties={},
                    )

            return {
                **results,
                "message_id": message_node_id,
                "conversation_id": conversation_node_id,
            }
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to process conversation message: {error_message}")
            return {"entities": [], "relationships": []}

    async def process_entity(
        self, entity_type: str, entity_id: str, entity_data: Dict[str, Any]
    ) -> str:
        """Process an entity from distributed memory and add it to the knowledge graph.

        Args:
            entity_type: Entity type
            entity_id: Entity ID
            entity_data: Entity data

        Returns:
            Node ID in the knowledge graph
        """
        try:
            # Map entity type
            mapped_type = self.entity_type_mapping.get(entity_type.lower(), "Entity")

            # Extract properties
            properties = {
                "entity_type": entity_type,
                "entity_id": entity_id,
                "name": entity_data.get("name", entity_id),
                **{k: v for k, v in entity_data.items() if k != "_timestamp"},
            }

            # Add entity to graph
            node_id = self.knowledge_graph.add_node(
                node_type=mapped_type,
                properties=properties,
                node_id=f"{entity_type}_{entity_id}",
            )

            return node_id
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to process entity: {error_message}")
            return ""

    async def process_tool_usage(
        self, tool_name: str, args: Dict[str, Any], result: Any
    ) -> Dict[str, Any]:
        """Process tool usage and add it to the knowledge graph.

        Args:
            tool_name: Tool name
            args: Tool arguments
            result: Tool result

        Returns:
            Processing results
        """
        try:
            # Create tool node if it doesn't exist
            tool_nodes = self.knowledge_graph.search_nodes(
                property_filters={"name": tool_name}, node_types=["Tool"]
            )

            tool_node_id = None
            if tool_nodes:
                tool_node_id = tool_nodes[0]["id"]
            else:
                tool_node_id = self.knowledge_graph.add_node(
                    node_type="Tool",
                    properties={"name": tool_name, "timestamp": time.time()},
                    node_id=f"tool_{tool_name}",
                )

            # Create usage node
            usage_id = str(time.time())
            usage_node_id = self.knowledge_graph.add_node(
                node_type="ToolUsage",
                properties={
                    "tool_name": tool_name,
                    "args": args,
                    "result": result
                    if isinstance(result, (str, int, float, bool))
                    else str(result),
                    "timestamp": time.time(),
                },
                node_id=f"tool_usage_{usage_id}",
            )

            # Link usage to tool
            self.knowledge_graph.add_edge(
                source_id=usage_node_id,
                target_id=tool_node_id,
                edge_type="uses",
                properties={},
            )

            # Process result text if it's a string
            result_entities = []
            result_relationships = []
            if isinstance(result, str):
                # Process result text
                processing_results = await self.process_text(
                    result, context_id=usage_id
                )
                result_entities = processing_results.get("entities", [])
                result_relationships = processing_results.get("relationships", [])

                # Link entities to usage
                for entity in result_entities:
                    entity_id = entity.get("id", "")
                    if entity_id:
                        self.knowledge_graph.add_edge(
                            source_id=usage_node_id,
                            target_id=entity_id,
                            edge_type="produced",
                            properties={},
                        )

            return {
                "tool_id": tool_node_id,
                "usage_id": usage_node_id,
                "entities": result_entities,
                "relationships": result_relationships,
            }
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to process tool usage: {error_message}")
            return {"entities": [], "relationships": []}

    async def get_context_for_request(
        self, request: str, max_entities: int = 10, max_relationships: int = 20
    ) -> Dict[str, Any]:
        """Get relevant context from the knowledge graph for a request.

        Args:
            request: User request
            max_entities: Maximum number of entities to return
            max_relationships: Maximum number of relationships to return

        Returns:
            Relevant context
        """
        try:
            # Extract entities from request
            request_entities = await self.extract_entities_from_text(request)

            if not request_entities:
                return {"entities": [], "relationships": []}

            # Add request entities to graph
            entity_id_mapping = await self.add_entities_to_graph(request_entities)

            # Find related entities in the graph
            related_entities = []
            for entity_name, entity_id in entity_id_mapping.items():
                # Get neighbors
                neighbors = self.knowledge_graph.get_neighbors(
                    entity_id, direction="both"
                )
                related_entities.extend(neighbors)

            # Deduplicate and limit
            seen_ids = set()
            unique_entities = []
            for entity in related_entities:
                if entity["id"] not in seen_ids:
                    seen_ids.add(entity["id"])
                    unique_entities.append(entity)

            # Sort by relevance (for now, just use timestamp as a proxy)
            sorted_entities = sorted(
                unique_entities,
                key=lambda x: x.get("properties", {}).get("timestamp", 0),
                reverse=True,
            )

            # Limit to max_entities
            top_entities = sorted_entities[:max_entities]

            # Get relationships between top entities
            relationships = []
            entity_ids = [entity["id"] for entity in top_entities]
            for i, source_id in enumerate(entity_ids):
                for target_id in entity_ids[i + 1 :]:
                    # Find paths between entities
                    path = self.knowledge_graph.find_path(source_id, target_id)
                    if path:
                        relationships.extend(path)

            # Limit to max_relationships
            top_relationships = relationships[:max_relationships]

            return {"entities": top_entities, "relationships": top_relationships}
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to get context for request: {error_message}")
            return {"entities": [], "relationships": []}
