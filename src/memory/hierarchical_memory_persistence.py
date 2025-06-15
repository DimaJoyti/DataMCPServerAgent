"""
Hierarchical memory persistence module for DataMCPServerAgent.
This module extends the advanced memory persistence with support for hierarchical reinforcement learning.
"""

import json
import sqlite3
import time
from typing import Any, Dict, List, Optional

from src.memory.advanced_memory_persistence import AdvancedMemoryDatabase


class HierarchicalMemoryDatabase(AdvancedMemoryDatabase):
    """Extended database for persisting hierarchical agent memory."""

    def __init__(self, db_path: str = "agent_memory.db"):
        """Initialize the hierarchical memory database.

        Args:
            db_path: Path to the SQLite database file
        """
        super().__init__(db_path)
        self._initialize_hierarchical_db()

    def _initialize_hierarchical_db(self) -> None:
        """Initialize the hierarchical database schema if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Options table for storing temporally extended actions
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS options (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            option_id TEXT NOT NULL,
            option_name TEXT NOT NULL,
            initiation_set TEXT NOT NULL,
            termination_condition TEXT NOT NULL,
            policy TEXT NOT NULL,
            last_updated REAL NOT NULL,
            UNIQUE(agent_name, option_id)
        )
        """
        )

        # Hierarchical Q-tables for storing Q-values at different levels
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS hierarchical_q_tables (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            level INTEGER NOT NULL,
            q_table TEXT NOT NULL,
            last_updated REAL NOT NULL,
            UNIQUE(agent_name, level)
        )
        """
        )

        # Subtask history for tracking subtask performance
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS subtask_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            task_id TEXT NOT NULL,
            parent_task_id TEXT,
            subtask_name TEXT NOT NULL,
            state TEXT NOT NULL,
            action TEXT NOT NULL,
            reward REAL NOT NULL,
            success INTEGER NOT NULL,
            start_time REAL NOT NULL,
            end_time REAL NOT NULL,
            metadata TEXT
        )
        """
        )

        # Task decomposition history
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS task_decomposition (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            task_id TEXT NOT NULL,
            parent_task_id TEXT,
            task_name TEXT NOT NULL,
            subtasks TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
        """
        )

        conn.commit()
        conn.close()

    def save_option(
        self,
        agent_name: str,
        option_id: str,
        option_name: str,
        initiation_set: Dict[str, Any],
        termination_condition: Dict[str, Any],
        policy: Dict[str, Any],
    ) -> None:
        """Save an option to the database.

        Args:
            agent_name: Name of the agent
            option_id: Unique identifier for the option
            option_name: Human-readable name for the option
            initiation_set: Conditions for initiating the option
            termination_condition: Conditions for terminating the option
            policy: Policy for executing the option
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO options
            (agent_name, option_id, option_name, initiation_set, termination_condition, policy, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent_name,
                option_id,
                option_name,
                json.dumps(initiation_set),
                json.dumps(termination_condition),
                json.dumps(policy),
                time.time(),
            ),
        )

        conn.commit()
        conn.close()

    def get_option(self, agent_name: str, option_id: str) -> Optional[Dict[str, Any]]:
        """Get an option from the database.

        Args:
            agent_name: Name of the agent
            option_id: Unique identifier for the option

        Returns:
            Option data or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT option_name, initiation_set, termination_condition, policy, last_updated
            FROM options
            WHERE agent_name = ? AND option_id = ?
            """,
            (agent_name, option_id),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            option_name, initiation_set, termination_condition, policy, last_updated = row
            return {
                "option_id": option_id,
                "option_name": option_name,
                "initiation_set": json.loads(initiation_set),
                "termination_condition": json.loads(termination_condition),
                "policy": json.loads(policy),
                "last_updated": last_updated,
            }
        return None

    def get_all_options(self, agent_name: str) -> List[Dict[str, Any]]:
        """Get all options for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            List of options
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT option_id, option_name, initiation_set, termination_condition, policy, last_updated
            FROM options
            WHERE agent_name = ?
            """,
            (agent_name,),
        )

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "option_id": option_id,
                "option_name": option_name,
                "initiation_set": json.loads(initiation_set),
                "termination_condition": json.loads(termination_condition),
                "policy": json.loads(policy),
                "last_updated": last_updated,
            }
            for option_id, option_name, initiation_set, termination_condition, policy, last_updated in rows
        ]

    def save_hierarchical_q_table(
        self, agent_name: str, level: int, q_table: Dict[str, Dict[str, float]]
    ) -> None:
        """Save a hierarchical Q-table to the database.

        Args:
            agent_name: Name of the agent
            level: Hierarchy level (0 for top level)
            q_table: Q-table to save
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO hierarchical_q_tables
            (agent_name, level, q_table, last_updated)
            VALUES (?, ?, ?, ?)
            """,
            (agent_name, level, json.dumps(q_table), time.time()),
        )

        conn.commit()
        conn.close()

    def get_hierarchical_q_table(
        self, agent_name: str, level: int
    ) -> Optional[Dict[str, Dict[str, float]]]:
        """Get a hierarchical Q-table from the database.

        Args:
            agent_name: Name of the agent
            level: Hierarchy level (0 for top level)

        Returns:
            Q-table or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT q_table
            FROM hierarchical_q_tables
            WHERE agent_name = ? AND level = ?
            """,
            (agent_name, level),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return json.loads(row[0])
        return None

    def save_subtask_execution(
        self,
        agent_name: str,
        task_id: str,
        parent_task_id: Optional[str],
        subtask_name: str,
        state: str,
        action: str,
        reward: float,
        success: bool,
        start_time: float,
        end_time: float,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save subtask execution to the database.

        Args:
            agent_name: Name of the agent
            task_id: Unique identifier for the task
            parent_task_id: Identifier for the parent task (None for top-level tasks)
            subtask_name: Name of the subtask
            state: State in which the subtask was executed
            action: Action taken
            reward: Reward received
            success: Whether the subtask was successful
            start_time: Start time of the subtask
            end_time: End time of the subtask
            metadata: Additional metadata
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO subtask_history
            (agent_name, task_id, parent_task_id, subtask_name, state, action, reward, success, start_time, end_time, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                agent_name,
                task_id,
                parent_task_id,
                subtask_name,
                state,
                action,
                reward,
                1 if success else 0,
                start_time,
                end_time,
                json.dumps(metadata or {}),
            ),
        )

        conn.commit()
        conn.close()

    def get_subtask_history(
        self, agent_name: str, task_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get subtask execution history.

        Args:
            agent_name: Name of the agent
            task_id: Optional task ID to filter by
            limit: Maximum number of records to return

        Returns:
            List of subtask execution records
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if task_id:
            cursor.execute(
                """
                SELECT task_id, parent_task_id, subtask_name, state, action, reward, success, start_time, end_time, metadata
                FROM subtask_history
                WHERE agent_name = ? AND task_id = ?
                ORDER BY start_time DESC
                LIMIT ?
                """,
                (agent_name, task_id, limit),
            )
        else:
            cursor.execute(
                """
                SELECT task_id, parent_task_id, subtask_name, state, action, reward, success, start_time, end_time, metadata
                FROM subtask_history
                WHERE agent_name = ?
                ORDER BY start_time DESC
                LIMIT ?
                """,
                (agent_name, limit),
            )

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "task_id": task_id,
                "parent_task_id": parent_task_id,
                "subtask_name": subtask_name,
                "state": state,
                "action": action,
                "reward": reward,
                "success": bool(success),
                "start_time": start_time,
                "end_time": end_time,
                "duration": end_time - start_time,
                "metadata": json.loads(metadata),
            }
            for task_id, parent_task_id, subtask_name, state, action, reward, success, start_time, end_time, metadata in rows
        ]

    def save_task_decomposition(
        self,
        agent_name: str,
        task_id: str,
        parent_task_id: Optional[str],
        task_name: str,
        subtasks: List[Dict[str, Any]],
    ) -> None:
        """Save task decomposition to the database.

        Args:
            agent_name: Name of the agent
            task_id: Unique identifier for the task
            parent_task_id: Identifier for the parent task (None for top-level tasks)
            task_name: Name of the task
            subtasks: List of subtasks
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO task_decomposition
            (agent_name, task_id, parent_task_id, task_name, subtasks, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                agent_name,
                task_id,
                parent_task_id,
                task_name,
                json.dumps(subtasks),
                time.time(),
            ),
        )

        conn.commit()
        conn.close()

    def get_task_decomposition(self, agent_name: str, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task decomposition from the database.

        Args:
            agent_name: Name of the agent
            task_id: Unique identifier for the task

        Returns:
            Task decomposition or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT parent_task_id, task_name, subtasks, timestamp
            FROM task_decomposition
            WHERE agent_name = ? AND task_id = ?
            """,
            (agent_name, task_id),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            parent_task_id, task_name, subtasks, timestamp = row
            return {
                "task_id": task_id,
                "parent_task_id": parent_task_id,
                "task_name": task_name,
                "subtasks": json.loads(subtasks),
                "timestamp": timestamp,
            }
        return None
