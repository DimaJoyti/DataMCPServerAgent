"""
Collaborative knowledge base for DataMCPServerAgent.
This module provides mechanisms for storing and retrieving shared knowledge between agents.
"""

import time
from typing import Any, Dict, List, Optional

from src.memory.memory_persistence import MemoryDatabase


class CollaborativeKnowledgeBase:
    """Knowledge base for collaborative learning between agents."""

    def __init__(self, db: MemoryDatabase):
        """Initialize the collaborative knowledge base.

        Args:
            db: Memory database for persistence
        """
        self.db = db
        self._initialize_tables()

    def _initialize_tables(self) -> None:
        """Initialize the database tables for collaborative knowledge."""
        # Create knowledge items table
        self.db.execute(
            """
        CREATE TABLE IF NOT EXISTS knowledge_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            confidence REAL NOT NULL,
            domain TEXT NOT NULL,
            source_agent TEXT,
            timestamp REAL NOT NULL
        )
        """
        )

        # Create knowledge applicability table
        self.db.execute(
            """
        CREATE TABLE IF NOT EXISTS knowledge_applicability (
            knowledge_id INTEGER NOT NULL,
            agent_type TEXT NOT NULL,
            PRIMARY KEY (knowledge_id, agent_type),
            FOREIGN KEY (knowledge_id) REFERENCES knowledge_items (id)
        )
        """
        )

        # Create knowledge prerequisites table
        self.db.execute(
            """
        CREATE TABLE IF NOT EXISTS knowledge_prerequisites (
            knowledge_id INTEGER NOT NULL,
            prerequisite TEXT NOT NULL,
            PRIMARY KEY (knowledge_id, prerequisite),
            FOREIGN KEY (knowledge_id) REFERENCES knowledge_items (id)
        )
        """
        )

        # Create knowledge transfers table
        self.db.execute(
            """
        CREATE TABLE IF NOT EXISTS knowledge_transfers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_agent TEXT NOT NULL,
            target_agent TEXT NOT NULL,
            knowledge_id INTEGER NOT NULL,
            success BOOLEAN NOT NULL,
            timestamp REAL NOT NULL,
            FOREIGN KEY (knowledge_id) REFERENCES knowledge_items (id)
        )
        """
        )

        # Create agent knowledge table
        self.db.execute(
            """
        CREATE TABLE IF NOT EXISTS agent_knowledge (
            agent_name TEXT NOT NULL,
            knowledge_id INTEGER NOT NULL,
            proficiency REAL NOT NULL,
            last_used REAL,
            PRIMARY KEY (agent_name, knowledge_id),
            FOREIGN KEY (knowledge_id) REFERENCES knowledge_items (id)
        )
        """
        )

    def store_knowledge(self, knowledge: Dict[str, Any], source_agent: Optional[str] = None) -> int:
        """Store knowledge in the knowledge base.

        Args:
            knowledge: Knowledge to store
            source_agent: Source agent name

        Returns:
            ID of the stored knowledge
        """
        # Extract knowledge items
        knowledge_items = knowledge.get("knowledge_items", [])
        if not knowledge_items:
            knowledge_items = [knowledge.get("content", "")]

        # Extract confidence
        confidence = knowledge.get("confidence", 50)
        if isinstance(confidence, dict):
            confidence = sum(confidence.values()) / len(confidence) if confidence else 50

        # Extract domain
        domain = knowledge.get("domain", "general")

        # Extract applicability
        applicability = knowledge.get("applicability", ["all"])

        # Extract prerequisites
        prerequisites = knowledge.get("prerequisites", [])

        # Store each knowledge item
        knowledge_ids = []
        for item in knowledge_items:
            # Insert knowledge item
            self.db.execute(
                """
                INSERT INTO knowledge_items (content, confidence, domain, source_agent, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (item, confidence, domain, source_agent, time.time()),
            )

            # Get the ID of the inserted knowledge item
            knowledge_id = self.db.execute("SELECT last_insert_rowid()").fetchone()[0]
            knowledge_ids.append(knowledge_id)

            # Insert applicability
            for agent_type in applicability:
                self.db.execute(
                    """
                    INSERT INTO knowledge_applicability (knowledge_id, agent_type)
                    VALUES (?, ?)
                    """,
                    (knowledge_id, agent_type),
                )

            # Insert prerequisites
            for prerequisite in prerequisites:
                self.db.execute(
                    """
                    INSERT INTO knowledge_prerequisites (knowledge_id, prerequisite)
                    VALUES (?, ?)
                    """,
                    (knowledge_id, prerequisite),
                )

        return knowledge_ids[0] if knowledge_ids else -1

    def get_knowledge(self, knowledge_id: int) -> Dict[str, Any]:
        """Get knowledge from the knowledge base.

        Args:
            knowledge_id: ID of the knowledge to get

        Returns:
            Knowledge
        """
        # Get knowledge item
        knowledge_item = self.db.execute(
            """
            SELECT content, confidence, domain, source_agent, timestamp
            FROM knowledge_items
            WHERE id = ?
            """,
            (knowledge_id,),
        ).fetchone()

        if not knowledge_item:
            return {}

        content, confidence, domain, source_agent, timestamp = knowledge_item

        # Get applicability
        applicability = self.db.execute(
            """
            SELECT agent_type
            FROM knowledge_applicability
            WHERE knowledge_id = ?
            """,
            (knowledge_id,),
        ).fetchall()

        # Get prerequisites
        prerequisites = self.db.execute(
            """
            SELECT prerequisite
            FROM knowledge_prerequisites
            WHERE knowledge_id = ?
            """,
            (knowledge_id,),
        ).fetchall()

        return {
            "id": knowledge_id,
            "content": content,
            "confidence": confidence,
            "domain": domain,
            "source_agent": source_agent,
            "timestamp": timestamp,
            "applicability": [a[0] for a in applicability],
            "prerequisites": [p[0] for p in prerequisites],
        }

    def get_applicable_knowledge(
        self, agent_type: str, domain: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get knowledge applicable to a specific agent type.

        Args:
            agent_type: Agent type
            domain: Optional domain filter

        Returns:
            List of applicable knowledge items
        """
        # Build query
        query = """
        SELECT k.id
        FROM knowledge_items k
        JOIN knowledge_applicability a ON k.id = a.knowledge_id
        WHERE a.agent_type IN (?, 'all')
        """
        params = [agent_type]

        if domain:
            query += " AND k.domain = ?"
            params.append(domain)

        # Get knowledge IDs
        knowledge_ids = self.db.execute(query, params).fetchall()

        # Get knowledge items
        knowledge_items = []
        for (knowledge_id,) in knowledge_ids:
            knowledge_items.append(self.get_knowledge(knowledge_id))

        return knowledge_items

    def record_knowledge_transfer(
        self, source_agent: str, target_agent: str, knowledge_id: int, success: bool
    ) -> None:
        """Record a knowledge transfer between agents.

        Args:
            source_agent: Source agent name
            target_agent: Target agent name
            knowledge_id: ID of the transferred knowledge
            success: Whether the transfer was successful
        """
        self.db.execute(
            """
            INSERT INTO knowledge_transfers (source_agent, target_agent, knowledge_id, success, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (source_agent, target_agent, knowledge_id, success, time.time()),
        )

    def assign_knowledge_to_agent(
        self, agent_name: str, knowledge_id: int, proficiency: float = 0.5
    ) -> None:
        """Assign knowledge to an agent.

        Args:
            agent_name: Agent name
            knowledge_id: ID of the knowledge
            proficiency: Proficiency level (0.0 to 1.0)
        """
        # Check if the agent already has this knowledge
        existing = self.db.execute(
            """
            SELECT proficiency
            FROM agent_knowledge
            WHERE agent_name = ? AND knowledge_id = ?
            """,
            (agent_name, knowledge_id),
        ).fetchone()

        if existing:
            # Update proficiency
            self.db.execute(
                """
                UPDATE agent_knowledge
                SET proficiency = ?, last_used = ?
                WHERE agent_name = ? AND knowledge_id = ?
                """,
                (max(existing[0], proficiency), time.time(), agent_name, knowledge_id),
            )
        else:
            # Insert new knowledge
            self.db.execute(
                """
                INSERT INTO agent_knowledge (agent_name, knowledge_id, proficiency, last_used)
                VALUES (?, ?, ?, ?)
                """,
                (agent_name, knowledge_id, proficiency, time.time()),
            )

    def get_agent_knowledge(
        self, agent_name: str, min_proficiency: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Get knowledge assigned to an agent.

        Args:
            agent_name: Agent name
            min_proficiency: Minimum proficiency level

        Returns:
            List of knowledge items
        """
        # Get knowledge IDs
        knowledge_ids = self.db.execute(
            """
            SELECT knowledge_id, proficiency
            FROM agent_knowledge
            WHERE agent_name = ? AND proficiency >= ?
            """,
            (agent_name, min_proficiency),
        ).fetchall()

        # Get knowledge items
        knowledge_items = []
        for knowledge_id, proficiency in knowledge_ids:
            knowledge = self.get_knowledge(knowledge_id)
            knowledge["proficiency"] = proficiency
            knowledge_items.append(knowledge)

        return knowledge_items

    def update_agent_proficiency(
        self, agent_name: str, knowledge_id: int, proficiency_delta: float
    ) -> None:
        """Update an agent's proficiency with a knowledge item.

        Args:
            agent_name: Agent name
            knowledge_id: ID of the knowledge
            proficiency_delta: Change in proficiency
        """
        # Get current proficiency
        current = self.db.execute(
            """
            SELECT proficiency
            FROM agent_knowledge
            WHERE agent_name = ? AND knowledge_id = ?
            """,
            (agent_name, knowledge_id),
        ).fetchone()

        if current:
            # Update proficiency
            new_proficiency = max(0.0, min(1.0, current[0] + proficiency_delta))
            self.db.execute(
                """
                UPDATE agent_knowledge
                SET proficiency = ?, last_used = ?
                WHERE agent_name = ? AND knowledge_id = ?
                """,
                (new_proficiency, time.time(), agent_name, knowledge_id),
            )

    def get_knowledge_transfer_history(
        self, source_agent: Optional[str] = None, target_agent: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get history of knowledge transfers.

        Args:
            source_agent: Optional source agent filter
            target_agent: Optional target agent filter

        Returns:
            List of knowledge transfers
        """
        # Build query
        query = """
        SELECT source_agent, target_agent, knowledge_id, success, timestamp
        FROM knowledge_transfers
        WHERE 1=1
        """
        params = []

        if source_agent:
            query += " AND source_agent = ?"
            params.append(source_agent)

        if target_agent:
            query += " AND target_agent = ?"
            params.append(target_agent)

        # Get transfers
        transfers = self.db.execute(query, params).fetchall()

        # Format transfers
        transfer_history = []
        for source, target, knowledge_id, success, timestamp in transfers:
            knowledge = self.get_knowledge(knowledge_id)
            transfer_history.append(
                {
                    "source_agent": source,
                    "target_agent": target,
                    "knowledge": knowledge,
                    "success": bool(success),
                    "timestamp": timestamp,
                }
            )

        return transfer_history

    def get_agent_knowledge_stats(self, agent_name: str) -> Dict[str, Any]:
        """Get statistics about an agent's knowledge.

        Args:
            agent_name: Agent name

        Returns:
            Knowledge statistics
        """
        # Get total knowledge count
        total_count = self.db.execute(
            """
            SELECT COUNT(*)
            FROM agent_knowledge
            WHERE agent_name = ?
            """,
            (agent_name,),
        ).fetchone()[0]

        # Get average proficiency
        avg_proficiency = (
            self.db.execute(
                """
            SELECT AVG(proficiency)
            FROM agent_knowledge
            WHERE agent_name = ?
            """,
                (agent_name,),
            ).fetchone()[0]
            or 0.0
        )

        # Get domain distribution
        domains = self.db.execute(
            """
            SELECT k.domain, COUNT(*)
            FROM agent_knowledge a
            JOIN knowledge_items k ON a.knowledge_id = k.id
            WHERE a.agent_name = ?
            GROUP BY k.domain
            """,
            (agent_name,),
        ).fetchall()

        # Get source distribution
        sources = self.db.execute(
            """
            SELECT k.source_agent, COUNT(*)
            FROM agent_knowledge a
            JOIN knowledge_items k ON a.knowledge_id = k.id
            WHERE a.agent_name = ?
            GROUP BY k.source_agent
            """,
            (agent_name,),
        ).fetchall()

        return {
            "total_knowledge": total_count,
            "average_proficiency": avg_proficiency,
            "domain_distribution": {d: c for d, c in domains},
            "source_distribution": {s if s else "unknown": c for s, c in sources},
        }


# Factory function to create collaborative knowledge base
def create_collaborative_knowledge_base(db: MemoryDatabase) -> CollaborativeKnowledgeBase:
    """Create a collaborative knowledge base.

    Args:
        db: Memory database for persistence

    Returns:
        Collaborative knowledge base
    """
    return CollaborativeKnowledgeBase(db)
