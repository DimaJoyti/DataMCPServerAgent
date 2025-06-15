"""
Advanced memory persistence module for DataMCPServerAgent.
This module extends the basic memory persistence with support for advanced reinforcement learning.
"""

import json
import sqlite3
import time
from typing import Any, Dict, List, Optional

from src.memory.memory_persistence import MemoryDatabase


class AdvancedMemoryDatabase(MemoryDatabase):
    """Extended database for persisting advanced agent memory."""

    def __init__(self, db_path: str = "agent_memory.db"):
        """Initialize the advanced memory database.

        Args:
            db_path: Path to the SQLite database file
        """
        super().__init__(db_path)
        self._initialize_advanced_db()

    def _initialize_advanced_db(self) -> None:
        """Initialize the advanced database schema if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Deep RL weights
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS drl_weights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            weights TEXT NOT NULL,
            last_updated REAL NOT NULL,
            UNIQUE(agent_name)
        )
        """
        )

        # Multi-objective Q-tables
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS mo_q_tables (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            objective TEXT NOT NULL,
            q_table TEXT NOT NULL,
            last_updated REAL NOT NULL,
            UNIQUE(agent_name, objective)
        )
        """
        )

        # Multi-objective agent rewards
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS agent_mo_rewards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            total_reward REAL NOT NULL,
            objective_rewards TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
        """
        )

        # Agent decisions
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS agent_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            state TEXT NOT NULL,
            selected_action TEXT NOT NULL,
            q_values TEXT NOT NULL,
            reward TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
        """
        )

        conn.commit()
        conn.close()

    def save_drl_weights(self, agent_name: str, weights: Dict[str, Any]) -> None:
        """Save deep RL weights to the database.

        Args:
            agent_name: Name of the agent
            weights: Neural network weights
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO drl_weights
            (agent_name, weights, last_updated)
            VALUES (?, ?, ?)
            """,
            (agent_name, json.dumps(weights), time.time()),
        )

        conn.commit()
        conn.close()

    def get_drl_weights(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get deep RL weights from the database.

        Args:
            agent_name: Name of the agent

        Returns:
            Neural network weights or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT weights FROM drl_weights WHERE agent_name = ?",
            (agent_name,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return json.loads(row[0])
        return None

    def save_mo_q_tables(
        self, agent_name: str, mo_q_tables: Dict[str, Dict[str, Dict[str, float]]]
    ) -> None:
        """Save multi-objective Q-tables to the database.

        Args:
            agent_name: Name of the agent
            mo_q_tables: Dictionary of Q-tables by objective
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Delete existing entries for this agent
        cursor.execute(
            "DELETE FROM mo_q_tables WHERE agent_name = ?",
            (agent_name,),
        )

        # Insert new entries
        for objective, q_table in mo_q_tables.items():
            cursor.execute(
                """
                INSERT INTO mo_q_tables
                (agent_name, objective, q_table, last_updated)
                VALUES (?, ?, ?, ?)
                """,
                (agent_name, objective, json.dumps(q_table), time.time()),
            )

        conn.commit()
        conn.close()

    def get_mo_q_tables(self, agent_name: str) -> Optional[Dict[str, Dict[str, Dict[str, float]]]]:
        """Get multi-objective Q-tables from the database.

        Args:
            agent_name: Name of the agent

        Returns:
            Dictionary of Q-tables by objective or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT objective, q_table FROM mo_q_tables WHERE agent_name = ?",
            (agent_name,),
        )

        rows = cursor.fetchall()
        conn.close()

        if rows:
            return {objective: json.loads(q_table) for objective, q_table in rows}
        return None

    def save_agent_multi_objective_reward(
        self, agent_name: str, total_reward: float, objective_rewards: Dict[str, float]
    ) -> None:
        """Save multi-objective reward to the database.

        Args:
            agent_name: Name of the agent
            total_reward: Total reward value
            objective_rewards: Dictionary of rewards by objective
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO agent_mo_rewards
            (agent_name, total_reward, objective_rewards, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (agent_name, total_reward, json.dumps(objective_rewards), time.time()),
        )

        conn.commit()
        conn.close()

    def get_agent_multi_objective_rewards(
        self, agent_name: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get multi-objective rewards for an agent.

        Args:
            agent_name: Name of the agent
            limit: Maximum number of rewards to return

        Returns:
            List of multi-objective rewards
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT total_reward, objective_rewards, timestamp
            FROM agent_mo_rewards
            WHERE agent_name = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (agent_name, limit),
        )

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "total_reward": total_reward,
                "objective_rewards": json.loads(objective_rewards),
                "timestamp": timestamp,
            }
            for total_reward, objective_rewards, timestamp in rows
        ]

    def save_agent_decision(
        self,
        agent_name: str,
        decision: Dict[str, Any],
    ) -> None:
        """Save agent decision to the database.

        Args:
            agent_name: Name of the agent
            decision: Decision data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO agent_decisions
            (agent_name, state, selected_action, q_values, reward, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                agent_name,
                decision["state"],
                decision["selected_action"],
                json.dumps(decision["q_values"]),
                json.dumps(decision["reward"]),
                time.time(),
            ),
        )

        conn.commit()
        conn.close()

    def get_agent_decisions(self, agent_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get decisions for an agent.

        Args:
            agent_name: Name of the agent
            limit: Maximum number of decisions to return

        Returns:
            List of decisions
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT state, selected_action, q_values, reward, timestamp
            FROM agent_decisions
            WHERE agent_name = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (agent_name, limit),
        )

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "state": state,
                "selected_action": selected_action,
                "q_values": json.loads(q_values),
                "reward": json.loads(reward),
                "timestamp": timestamp,
            }
            for state, selected_action, q_values, reward, timestamp in rows
        ]
