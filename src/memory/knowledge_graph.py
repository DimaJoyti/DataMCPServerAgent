"""
Knowledge graph module for DataMCPServerAgent.
This module provides a graph-based representation of entities and their relationships.
"""

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

# Try to import networkx, make it optional
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

# Try to import rdflib, make it optional
try:
    from rdflib import RDF, Graph, Literal, Namespace, URIRef
    from rdflib.namespace import FOAF
    RDFLIB_AVAILABLE = True
except ImportError:
    RDFLIB_AVAILABLE = False
    RDF = Graph = Literal = Namespace = URIRef = FOAF = None

from src.memory.memory_persistence import MemoryDatabase
from src.utils.error_handlers import format_error_for_user

# Configure logging
logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """Knowledge graph for representing entities and their relationships."""

    def __init__(self, db: MemoryDatabase, namespace: str = "datamcp"):
        """Initialize the knowledge graph.

        Args:
            db: Memory database for persistence
            namespace: Namespace for the knowledge graph
        """
        self.db = db
        self.namespace = namespace

        # Initialize RDF graph (if available)
        if RDFLIB_AVAILABLE:
            self.rdf_graph = Graph()
            # Define namespaces
            self.ns = Namespace(f"http://{namespace}.org/")
            self.rdf_graph.bind("datamcp", self.ns)
            self.rdf_graph.bind("foaf", FOAF)
        else:
            self.rdf_graph = None
            self.ns = None
            logger.warning("RDFLib not available. RDF operations will be disabled.")

        # Initialize NetworkX graph for in-memory operations (if available)
        if NETWORKX_AVAILABLE:
            self.nx_graph = nx.MultiDiGraph()
        else:
            self.nx_graph = None
            logger.warning("NetworkX not available. Some graph operations will be limited.")

        # Load existing graph from database
        self._load_graph()

    def _load_graph(self) -> None:
        """Load the knowledge graph from the database."""
        try:
            # Load nodes
            nodes = self.db.execute("SELECT * FROM knowledge_graph_nodes").fetchall()

            for node_id, node_type, properties, timestamp in nodes:
                properties_dict = json.loads(properties)
                if self.nx_graph is not None:
                    self.nx_graph.add_node(
                        node_id,
                        node_type=node_type,
                        properties=properties_dict,
                        timestamp=timestamp,
                    )

                # Add to RDF graph (if available)
                if self.rdf_graph is not None and self.ns is not None:
                    node_uri = URIRef(f"{self.ns}{node_id}")
                    self.rdf_graph.add(
                        (node_uri, RDF.type, URIRef(f"{self.ns}{node_type}"))
                    )

                for prop, value in properties_dict.items():
                    if isinstance(value, str):
                        self.rdf_graph.add(
                            (node_uri, URIRef(f"{self.ns}{prop}"), Literal(value))
                        )
                    elif isinstance(value, (int, float)):
                        self.rdf_graph.add(
                            (node_uri, URIRef(f"{self.ns}{prop}"), Literal(value))
                        )
                    elif isinstance(value, bool):
                        self.rdf_graph.add(
                            (node_uri, URIRef(f"{self.ns}{prop}"), Literal(value))
                        )
                    elif isinstance(value, dict):
                        self.rdf_graph.add(
                            (
                                node_uri,
                                URIRef(f"{self.ns}{prop}"),
                                Literal(json.dumps(value)),
                            )
                        )

            # Load edges
            edges = self.db.execute("SELECT * FROM knowledge_graph_edges").fetchall()

            for source_id, target_id, edge_type, properties, timestamp in edges:
                properties_dict = json.loads(properties)
                self.nx_graph.add_edge(
                    source_id,
                    target_id,
                    edge_type=edge_type,
                    properties=properties_dict,
                    timestamp=timestamp,
                )

                # Add to RDF graph
                source_uri = URIRef(f"{self.ns}{source_id}")
                target_uri = URIRef(f"{self.ns}{target_id}")
                edge_uri = URIRef(f"{self.ns}{edge_type}")

                self.rdf_graph.add((source_uri, edge_uri, target_uri))

                # Add edge properties
                for prop, value in properties_dict.items():
                    edge_prop_uri = URIRef(f"{self.ns}edge_{edge_type}_{prop}")
                    if isinstance(value, str):
                        self.rdf_graph.add((edge_uri, edge_prop_uri, Literal(value)))
                    elif isinstance(value, (int, float)):
                        self.rdf_graph.add((edge_uri, edge_prop_uri, Literal(value)))
                    elif isinstance(value, bool):
                        self.rdf_graph.add((edge_uri, edge_prop_uri, Literal(value)))

            logger.info(
                f"Loaded knowledge graph with {len(nodes)} nodes and {len(edges)} edges"
            )
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to load knowledge graph: {error_message}")

            # Initialize tables if they don't exist
            self._initialize_tables()

    def _initialize_tables(self) -> None:
        """Initialize the knowledge graph tables in the database."""
        try:
            # Create nodes table
            self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_graph_nodes (
                    node_id TEXT PRIMARY KEY,
                    node_type TEXT NOT NULL,
                    properties TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
                """
            )

            # Create edges table
            self.db.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_graph_edges (
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    edge_type TEXT NOT NULL,
                    properties TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    PRIMARY KEY (source_id, target_id, edge_type)
                )
                """
            )

            # Create index on node type
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_node_type ON knowledge_graph_nodes (node_type)"
            )

            # Create index on edge type
            self.db.execute(
                "CREATE INDEX IF NOT EXISTS idx_edge_type ON knowledge_graph_edges (edge_type)"
            )

            logger.info("Initialized knowledge graph tables")
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(
                f"Failed to initialize knowledge graph tables: {error_message}"
            )
            raise

    def add_node(
        self, node_type: str, properties: Dict[str, Any], node_id: Optional[str] = None
    ) -> str:
        """Add a node to the knowledge graph.

        Args:
            node_type: Type of node
            properties: Node properties
            node_id: Optional node ID (generated if not provided)

        Returns:
            Node ID
        """
        try:
            # Generate node ID if not provided
            if node_id is None:
                node_id = str(uuid.uuid4())

            # Add node to NetworkX graph (if available)
            if self.nx_graph is not None:
                self.nx_graph.add_node(
                    node_id,
                    node_type=node_type,
                    properties=properties,
                    timestamp=time.time(),
                )

            # Add node to RDF graph (if available)
            if self.rdf_graph is not None and self.ns is not None:
                node_uri = URIRef(f"{self.ns}{node_id}")
                self.rdf_graph.add((node_uri, RDF.type, URIRef(f"{self.ns}{node_type}")))

                for prop, value in properties.items():
                    if isinstance(value, str):
                        self.rdf_graph.add(
                            (node_uri, URIRef(f"{self.ns}{prop}"), Literal(value))
                        )
                    elif isinstance(value, (int, float)):
                        self.rdf_graph.add(
                            (node_uri, URIRef(f"{self.ns}{prop}"), Literal(value))
                        )
                    elif isinstance(value, bool):
                        self.rdf_graph.add(
                            (node_uri, URIRef(f"{self.ns}{prop}"), Literal(value))
                        )
                    elif isinstance(value, dict):
                        self.rdf_graph.add(
                            (
                                node_uri,
                                URIRef(f"{self.ns}{prop}"),
                                Literal(json.dumps(value)),
                            )
                        )

            # Save node to database
            self.db.execute(
                """
                INSERT OR REPLACE INTO knowledge_graph_nodes
                (node_id, node_type, properties, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (node_id, node_type, json.dumps(properties), time.time()),
            )

            logger.debug(f"Added node {node_id} of type {node_type}")
            return node_id
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to add node: {error_message}")
            raise

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        properties: Dict[str, Any] = {},
    ) -> None:
        """Add an edge to the knowledge graph.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            edge_type: Type of edge
            properties: Edge properties
        """
        try:
            # Add edge to NetworkX graph (if available)
            if self.nx_graph is not None:
                self.nx_graph.add_edge(
                    source_id,
                    target_id,
                    edge_type=edge_type,
                    properties=properties,
                    timestamp=time.time(),
                )

            # Add edge to RDF graph (if available)
            if self.rdf_graph is not None and self.ns is not None:
                source_uri = URIRef(f"{self.ns}{source_id}")
                target_uri = URIRef(f"{self.ns}{target_id}")
                edge_uri = URIRef(f"{self.ns}{edge_type}")

                self.rdf_graph.add((source_uri, edge_uri, target_uri))

                # Add edge properties
                for prop, value in properties.items():
                    edge_prop_uri = URIRef(f"{self.ns}edge_{edge_type}_{prop}")
                    if isinstance(value, str):
                        self.rdf_graph.add((edge_uri, edge_prop_uri, Literal(value)))
                    elif isinstance(value, (int, float)):
                        self.rdf_graph.add((edge_uri, edge_prop_uri, Literal(value)))
                    elif isinstance(value, bool):
                        self.rdf_graph.add((edge_uri, edge_prop_uri, Literal(value)))

            # Save edge to database
            self.db.execute(
                """
                INSERT OR REPLACE INTO knowledge_graph_edges
                (source_id, target_id, edge_type, properties, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (source_id, target_id, edge_type, json.dumps(properties), time.time()),
            )

            logger.debug(
                f"Added edge from {source_id} to {target_id} of type {edge_type}"
            )
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to add edge: {error_message}")
            raise

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get a node from the knowledge graph.

        Args:
            node_id: Node ID

        Returns:
            Node data or None if not found
        """
        try:
            if self.nx_graph is not None and node_id in self.nx_graph:
                node_data = self.nx_graph.nodes[node_id]
                return {
                    "id": node_id,
                    "type": node_data["node_type"],
                    "properties": node_data["properties"],
                    "timestamp": node_data["timestamp"],
                }

            # Fallback to database query if NetworkX not available
            result = self.db.execute(
                "SELECT node_type, properties, timestamp FROM knowledge_graph_nodes WHERE node_id = ?",
                (node_id,)
            ).fetchone()

            if result:
                node_type, properties_json, timestamp = result
                return {
                    "id": node_id,
                    "type": node_type,
                    "properties": json.loads(properties_json),
                    "timestamp": timestamp,
                }

            return None
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to get node {node_id}: {error_message}")
            return None

    def get_nodes_by_type(self, node_type: str) -> List[Dict[str, Any]]:
        """Get nodes of a specific type from the knowledge graph.

        Args:
            node_type: Node type

        Returns:
            List of nodes
        """
        try:
            nodes = []

            if self.nx_graph is not None:
                # Use NetworkX if available
                for node_id, node_data in self.nx_graph.nodes(data=True):
                    if node_data.get("node_type") == node_type:
                        nodes.append(
                            {
                                "id": node_id,
                                "type": node_data["node_type"],
                                "properties": node_data["properties"],
                                "timestamp": node_data["timestamp"],
                            }
                        )
            else:
                # Fallback to database query
                results = self.db.execute(
                    "SELECT node_id, properties, timestamp FROM knowledge_graph_nodes WHERE node_type = ?",
                    (node_type,)
                ).fetchall()

                for node_id, properties_json, timestamp in results:
                    nodes.append(
                        {
                            "id": node_id,
                            "type": node_type,
                            "properties": json.loads(properties_json),
                            "timestamp": timestamp,
                        }
                    )

            return nodes
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to get nodes of type {node_type}: {error_message}")
            return []

    def get_edges(
        self,
        source_id: Optional[str] = None,
        target_id: Optional[str] = None,
        edge_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get edges from the knowledge graph.

        Args:
            source_id: Optional source node ID filter
            target_id: Optional target node ID filter
            edge_type: Optional edge type filter

        Returns:
            List of edges
        """
        try:
            if self.nx_graph is None:
                # Fallback to database query if NetworkX not available
                logger.warning("NetworkX not available, using database fallback for get_edges")
                return []

            edges = []

            # Get all edges
            for u, v, edge_data in self.nx_graph.edges(data=True):
                # Apply filters
                if source_id is not None and u != source_id:
                    continue
                if target_id is not None and v != target_id:
                    continue
                if edge_type is not None and edge_data.get("edge_type") != edge_type:
                    continue

                edges.append(
                    {
                        "source": u,
                        "target": v,
                        "type": edge_data["edge_type"],
                        "properties": edge_data["properties"],
                        "timestamp": edge_data["timestamp"],
                    }
                )

            return edges
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to get edges: {error_message}")
            return []

    def get_neighbors(
        self, node_id: str, edge_type: Optional[str] = None, direction: str = "outgoing"
    ) -> List[Dict[str, Any]]:
        """Get neighbors of a node in the knowledge graph.

        Args:
            node_id: Node ID
            edge_type: Optional edge type filter
            direction: Direction of edges ("outgoing", "incoming", or "both")

        Returns:
            List of neighbor nodes
        """
        try:
            if self.nx_graph is None:
                logger.warning("NetworkX not available, cannot get neighbors")
                return []

            neighbors = []

            if direction == "outgoing" or direction == "both":
                for _, neighbor_id, edge_data in self.nx_graph.out_edges(
                    node_id, data=True
                ):
                    if (
                        edge_type is not None
                        and edge_data.get("edge_type") != edge_type
                    ):
                        continue

                    neighbor_data = self.nx_graph.nodes[neighbor_id]
                    neighbors.append(
                        {
                            "id": neighbor_id,
                            "type": neighbor_data["node_type"],
                            "properties": neighbor_data["properties"],
                            "edge_type": edge_data["edge_type"],
                            "edge_properties": edge_data["properties"],
                            "direction": "outgoing",
                        }
                    )

            if direction == "incoming" or direction == "both":
                for neighbor_id, _, edge_data in self.nx_graph.in_edges(
                    node_id, data=True
                ):
                    if (
                        edge_type is not None
                        and edge_data.get("edge_type") != edge_type
                    ):
                        continue

                    neighbor_data = self.nx_graph.nodes[neighbor_id]
                    neighbors.append(
                        {
                            "id": neighbor_id,
                            "type": neighbor_data["node_type"],
                            "properties": neighbor_data["properties"],
                            "edge_type": edge_data["edge_type"],
                            "edge_properties": edge_data["properties"],
                            "direction": "incoming",
                        }
                    )

            return neighbors
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to get neighbors of node {node_id}: {error_message}")
            return []

    def search_nodes(
        self,
        property_filters: Dict[str, Any] = {},
        node_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for nodes in the knowledge graph.

        Args:
            property_filters: Property filters (key-value pairs)
            node_types: Optional list of node types to filter by

        Returns:
            List of matching nodes
        """
        try:
            if self.nx_graph is None:
                logger.warning("NetworkX not available, cannot search nodes")
                return []

            matching_nodes = []

            for node_id, node_data in self.nx_graph.nodes(data=True):
                # Filter by node type
                if (
                    node_types is not None
                    and node_data.get("node_type") not in node_types
                ):
                    continue

                # Filter by properties
                properties = node_data.get("properties", {})
                match = True

                for key, value in property_filters.items():
                    if key not in properties or properties[key] != value:
                        match = False
                        break

                if match:
                    matching_nodes.append(
                        {
                            "id": node_id,
                            "type": node_data["node_type"],
                            "properties": properties,
                            "timestamp": node_data["timestamp"],
                        }
                    )

            return matching_nodes
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to search nodes: {error_message}")
            return []

    def execute_sparql_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a SPARQL query on the knowledge graph.

        Args:
            query: SPARQL query

        Returns:
            Query results
        """
        try:
            if self.rdf_graph is None:
                logger.warning("RDFLib not available, cannot execute SPARQL query")
                return []

            results = []

            # Execute query
            query_results = self.rdf_graph.query(query)

            # Convert results to dictionaries
            for row in query_results:
                result = {}
                for var in query_results.vars:
                    value = row[var]
                    if value is not None:
                        if isinstance(value, URIRef):
                            # Extract the local name from the URI
                            uri_str = str(value)
                            if uri_str.startswith(f"http://{self.namespace}.org/"):
                                result[var] = uri_str.split("/")[-1]
                            else:
                                result[var] = uri_str
                        else:
                            result[var] = value
                results.append(result)

            return results
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to execute SPARQL query: {error_message}")
            return []

    def find_path(
        self, source_id: str, target_id: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Find a path between two nodes in the knowledge graph.

        Args:
            source_id: Source node ID
            target_id: Target node ID
            max_depth: Maximum path depth

        Returns:
            Path as a list of edges, or None if no path exists
        """
        try:
            if self.nx_graph is None or not NETWORKX_AVAILABLE:
                logger.warning("NetworkX not available, cannot find path")
                return None

            # Check if nodes exist
            if source_id not in self.nx_graph or target_id not in self.nx_graph:
                return None

            # Find shortest path
            try:
                path = nx.shortest_path(
                    self.nx_graph, source=source_id, target=target_id
                )

                # Convert path to edges
                edges = []
                for i in range(len(path) - 1):
                    u, v = path[i], path[i + 1]
                    edge_data = self.nx_graph.get_edge_data(u, v)

                    # There might be multiple edges between the same nodes
                    # We'll take the first one for simplicity
                    first_edge = list(edge_data.values())[0]

                    edges.append(
                        {
                            "source": u,
                            "target": v,
                            "type": first_edge["edge_type"],
                            "properties": first_edge["properties"],
                        }
                    )

                return edges
            except nx.NetworkXNoPath:
                return None
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to find path: {error_message}")
            return None

    def get_subgraph(
        self, node_ids: List[str], include_neighbors: bool = False
    ) -> Dict[str, Any]:
        """Get a subgraph of the knowledge graph.

        Args:
            node_ids: List of node IDs
            include_neighbors: Whether to include neighbors of the specified nodes

        Returns:
            Subgraph as a dictionary of nodes and edges
        """
        try:
            if self.nx_graph is None:
                logger.warning("NetworkX not available, cannot get subgraph")
                return {"nodes": [], "edges": []}

            # Get nodes
            nodes = []
            node_set = set(node_ids)

            # Include neighbors if requested
            if include_neighbors:
                for node_id in node_ids:
                    for neighbor in self.get_neighbors(node_id, direction="both"):
                        node_set.add(neighbor["id"])

            # Get node data
            for node_id in node_set:
                node_data = self.get_node(node_id)
                if node_data:
                    nodes.append(node_data)

            # Get edges between nodes
            edges = []
            for u in node_set:
                for v in node_set:
                    if u != v:
                        edge_data = self.nx_graph.get_edge_data(u, v)
                        if edge_data:
                            # There might be multiple edges between the same nodes
                            for key, data in edge_data.items():
                                edges.append(
                                    {
                                        "source": u,
                                        "target": v,
                                        "type": data["edge_type"],
                                        "properties": data["properties"],
                                    }
                                )

            return {"nodes": nodes, "edges": edges}
        except Exception as e:
            error_message = format_error_for_user(e)
            logger.error(f"Failed to get subgraph: {error_message}")
            return {"nodes": [], "edges": []}
