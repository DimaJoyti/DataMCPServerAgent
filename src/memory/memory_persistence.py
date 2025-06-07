"""
Memory persistence module for DataMCPServerAgent.
This module provides database integration for persisting agent memory between sessions.
"""

import json
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import aiofiles
except ImportError:
    print("Warning: aiofiles package not found. Installing...")
    import subprocess

    subprocess.check_call(["pip", "install", "aiofiles"])
    import aiofiles

class MemoryDatabase:
    """Database for persisting agent memory."""

    def __init__(self, db_path: str = "agent_memory.db"):
        """Initialize the memory database.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize the database schema if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create conversation history table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
        """)

        # Create tool usage history table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS tool_usage_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tool_name TEXT NOT NULL,
            args TEXT NOT NULL,
            result TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
        """)

        # Create entity memory table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS entity_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_type TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            data TEXT NOT NULL,
            last_updated REAL NOT NULL,
            UNIQUE(entity_type, entity_id)
        )
        """)

        # Create reinforcement learning tables

        # Q-table for Q-learning
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS q_tables (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            q_table TEXT NOT NULL,
            last_updated REAL NOT NULL,
            UNIQUE(agent_name)
        )
        """)

        # Policy parameters for policy gradient
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS policy_params (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            policy_params TEXT NOT NULL,
            last_updated REAL NOT NULL,
            UNIQUE(agent_name)
        )
        """)

        # Deep RL weights
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS drl_weights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            weights TEXT NOT NULL,
            last_updated REAL NOT NULL,
            UNIQUE(agent_name)
        )
        """)

        # Multi-objective Q-tables
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS mo_q_tables (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            objective TEXT NOT NULL,
            q_table TEXT NOT NULL,
            last_updated REAL NOT NULL,
            UNIQUE(agent_name, objective)
        )
        """)

        # Agent rewards
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_rewards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            reward REAL NOT NULL,
            reward_components TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
        """)

        # Multi-objective agent rewards
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_mo_rewards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            total_reward REAL NOT NULL,
            objective_rewards TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
        """)

        # Agent decisions
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            state TEXT NOT NULL,
            selected_action TEXT NOT NULL,
            q_values TEXT NOT NULL,
            reward TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
        """)

        # Agent interactions for batch learning
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS agent_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            request TEXT NOT NULL,
            response TEXT NOT NULL,
            feedback TEXT,
            timestamp REAL NOT NULL
        )
        """)

        # Create tool performance table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS tool_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tool_name TEXT NOT NULL,
            success INTEGER NOT NULL,
            execution_time REAL NOT NULL,
            timestamp REAL NOT NULL
        )
        """)

        # Create learning feedback table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS learning_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            feedback_type TEXT NOT NULL,
            feedback_data TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
        """)

        # Create advanced reasoning tables
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS reasoning_chains (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chain_id TEXT NOT NULL UNIQUE,
            goal TEXT NOT NULL,
            initial_context TEXT NOT NULL,
            start_time REAL NOT NULL
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS reasoning_steps (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chain_id TEXT NOT NULL,
            step_id TEXT NOT NULL,
            step_type TEXT NOT NULL,
            content TEXT NOT NULL,
            confidence REAL NOT NULL,
            dependencies TEXT NOT NULL,
            timestamp REAL NOT NULL,
            evidence TEXT NOT NULL,
            alternatives TEXT NOT NULL,
            FOREIGN KEY (chain_id) REFERENCES reasoning_chains (chain_id)
        )
        """)

        # Create planning tables
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_id TEXT NOT NULL UNIQUE,
            goal TEXT NOT NULL,
            actions TEXT NOT NULL,
            initial_state TEXT NOT NULL,
            goal_state TEXT NOT NULL,
            metadata TEXT NOT NULL,
            created_at REAL NOT NULL
        )
        """)

        # Create meta-reasoning tables
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS meta_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id TEXT NOT NULL UNIQUE,
            strategy TEXT NOT NULL,
            decision TEXT NOT NULL,
            rationale TEXT NOT NULL,
            confidence REAL NOT NULL,
            expected_impact TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
        """)

        # Create reflection tables
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS reflection_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL UNIQUE,
            trigger_event TEXT NOT NULL,
            focus_areas TEXT NOT NULL,
            insights TEXT NOT NULL,
            conclusions TEXT NOT NULL,
            improvement_plan TEXT NOT NULL,
            metadata TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
        """)

        conn.commit()
        conn.close()

    def save_conversation_history(self, messages: List[Dict[str, str]]) -> None:
        """Save conversation history to the database.

        Args:
            messages: List of messages to save
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Clear existing history
        cursor.execute("DELETE FROM conversation_history")

        # Insert new history
        for message in messages:
            cursor.execute(
                "INSERT INTO conversation_history (role, content, timestamp) VALUES (?, ?, ?)",
                (message["role"], message["content"], time.time()),
            )

        conn.commit()
        conn.close()

    def load_conversation_history(self) -> List[Dict[str, str]]:
        """Load conversation history from the database.

        Returns:
            List of messages
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT role, content FROM conversation_history ORDER BY id")
        rows = cursor.fetchall()

        conn.close()

        return [{"role": role, "content": content} for role, content in rows]

    def save_tool_usage(
        self, tool_name: str, args: Dict[str, Any], result: Any
    ) -> None:
        """Save tool usage to the database.

        Args:
            tool_name: Name of the tool used
            args: Arguments passed to the tool
            result: Result returned by the tool
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO tool_usage_history (tool_name, args, result, timestamp) VALUES (?, ?, ?, ?)",
            (tool_name, json.dumps(args), json.dumps(str(result)), time.time()),
        )

        conn.commit()
        conn.close()

    def load_tool_usage(
        self, tool_name: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Load tool usage history from the database.

        Args:
            tool_name: Name of tool to get history for, or None for all tools

        Returns:
            Tool usage history
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if tool_name:
            cursor.execute(
                "SELECT tool_name, args, result, timestamp FROM tool_usage_history WHERE tool_name = ? ORDER BY timestamp",
                (tool_name,),
            )
        else:
            cursor.execute(
                "SELECT tool_name, args, result, timestamp FROM tool_usage_history ORDER BY timestamp"
            )

        rows = cursor.fetchall()
        conn.close()

        result = {}
        for tool, args, res, timestamp in rows:
            if tool not in result:
                result[tool] = []

            result[tool].append(
                {
                    "args": json.loads(args),
                    "result": json.loads(res),
                    "timestamp": timestamp,
                }
            )

        return result

    def save_entity(
        self, entity_type: str, entity_id: str, data: Dict[str, Any]
    ) -> None:
        """Save an entity to the database.

        Args:
            entity_type: Type of entity
            entity_id: Entity identifier
            data: Entity data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO entity_memory
            (entity_type, entity_id, data, last_updated)
            VALUES (?, ?, ?, ?)
            """,
            (entity_type, entity_id, json.dumps(data), time.time()),
        )

        conn.commit()
        conn.close()

    def load_entity(self, entity_type: str, entity_id: str) -> Optional[Dict[str, Any]]:
        """Load an entity from the database.

        Args:
            entity_type: Type of entity
            entity_id: Entity identifier

        Returns:
            Entity data or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT data FROM entity_memory WHERE entity_type = ? AND entity_id = ?",
            (entity_type, entity_id),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return json.loads(row[0])
        return None

    def load_entities_by_type(self, entity_type: str) -> Dict[str, Dict[str, Any]]:
        """Load all entities of a specific type from the database.

        Args:
            entity_type: Type of entities to retrieve

        Returns:
            Dictionary of entities by ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT entity_id, data FROM entity_memory WHERE entity_type = ?",
            (entity_type,),
        )

        rows = cursor.fetchall()
        conn.close()

        return {entity_id: json.loads(data) for entity_id, data in rows}

    def save_tool_performance(
        self, tool_name: str, success: bool, execution_time: float
    ) -> None:
        """Save tool performance metrics to the database.

        Args:
            tool_name: Name of the tool
            success: Whether the tool execution was successful
            execution_time: Time taken to execute the tool in seconds
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO tool_performance (tool_name, success, execution_time, timestamp) VALUES (?, ?, ?, ?)",
            (tool_name, 1 if success else 0, execution_time, time.time()),
        )

        conn.commit()
        conn.close()

    def get_tool_performance(self, tool_name: str) -> Dict[str, Any]:
        """Get performance metrics for a tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Performance metrics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT
                COUNT(*) as total_uses,
                SUM(success) as successful_uses,
                AVG(execution_time) as avg_execution_time,
                MIN(execution_time) as min_execution_time,
                MAX(execution_time) as max_execution_time
            FROM tool_performance
            WHERE tool_name = ?
            """,
            (tool_name,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            total_uses, successful_uses, avg_time, min_time, max_time = row

            if total_uses > 0:
                success_rate = (successful_uses / total_uses) * 100
            else:
                success_rate = 0

            return {
                "total_uses": total_uses,
                "successful_uses": successful_uses,
                "success_rate": success_rate,
                "avg_execution_time": avg_time,
                "min_execution_time": min_time,
                "max_execution_time": max_time,
            }

        return {
            "total_uses": 0,
            "successful_uses": 0,
            "success_rate": 0,
            "avg_execution_time": 0,
            "min_execution_time": 0,
            "max_execution_time": 0,
        }

    def save_learning_feedback(
        self, agent_name: str, feedback_type: str, feedback_data: Dict[str, Any]
    ) -> None:
        """Save learning feedback to the database.

        Args:
            agent_name: Name of the agent
            feedback_type: Type of feedback (e.g., 'user_feedback', 'self_evaluation')
            feedback_data: Feedback data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO learning_feedback (agent_name, feedback_type, feedback_data, timestamp) VALUES (?, ?, ?, ?)",
            (agent_name, feedback_type, json.dumps(feedback_data), time.time()),
        )

        conn.commit()
        conn.close()

    def get_learning_feedback(
        self, agent_name: Optional[str] = None, feedback_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get learning feedback from the database.

        Args:
            agent_name: Name of the agent, or None for all agents
            feedback_type: Type of feedback, or None for all types

        Returns:
            List of feedback entries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT agent_name, feedback_type, feedback_data, timestamp FROM learning_feedback"
        params = []

        if agent_name or feedback_type:
            query += " WHERE"

            if agent_name:
                query += " agent_name = ?"
                params.append(agent_name)

                if feedback_type:
                    query += " AND"

            if feedback_type:
                query += " feedback_type = ?"
                params.append(feedback_type)

        query += " ORDER BY timestamp DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "agent_name": agent_name,
                "feedback_type": feedback_type,
                "feedback_data": json.loads(feedback_data),
                "timestamp": timestamp,
            }
            for agent_name, feedback_type, feedback_data, timestamp in rows
        ]

    def save_q_table(
        self, agent_name: str, q_table: Dict[str, Dict[str, float]]
    ) -> None:
        """Save a Q-table to the database.

        Args:
            agent_name: Name of the agent
            q_table: Q-table to save
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO q_tables
            (agent_name, q_table, last_updated)
            VALUES (?, ?, ?)
            """,
            (agent_name, json.dumps(q_table), time.time()),
        )

        conn.commit()
        conn.close()

    def get_q_table(self, agent_name: str) -> Optional[Dict[str, Dict[str, float]]]:
        """Get a Q-table from the database.

        Args:
            agent_name: Name of the agent

        Returns:
            Q-table or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT q_table FROM q_tables WHERE agent_name = ?",
            (agent_name,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return json.loads(row[0])
        return None

    def save_agent_reward(
        self, agent_name: str, reward: float, reward_components: Dict[str, float]
    ) -> None:
        """Save agent reward to the database.

        Args:
            agent_name: Name of the agent
            reward: Total reward
            reward_components: Reward components
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO agent_rewards (agent_name, reward, reward_components, timestamp) VALUES (?, ?, ?, ?)",
            (agent_name, reward, json.dumps(reward_components), time.time()),
        )

        conn.commit()
        conn.close()

    def get_agent_rewards(
        self, agent_name: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get agent rewards from the database.

        Args:
            agent_name: Name of the agent
            limit: Maximum number of rewards to retrieve

        Returns:
            List of reward entries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT reward, reward_components, timestamp FROM agent_rewards WHERE agent_name = ? ORDER BY timestamp DESC LIMIT ?",
            (agent_name, limit),
        )

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "reward": reward,
                "components": json.loads(reward_components),
                "timestamp": timestamp,
            }
            for reward, reward_components, timestamp in rows
        ]

    def save_tool_selection(
        self, query: str, selected_tools: List[Dict[str, Any]]
    ) -> None:
        """Save tool selection to the database.

        Args:
            query: Research query
            selected_tools: Selected tools with arguments and reasons
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO learning_feedback (agent_name, feedback_type, feedback_data, timestamp) VALUES (?, ?, ?, ?)",
            (
                "tool_selector",
                "tool_selection",
                json.dumps({"query": query, "selected_tools": selected_tools}),
                time.time(),
            ),
        )

        conn.commit()
        conn.close()

    def get_entity_types(self) -> List[str]:
        """Get all entity types in the database.

        Returns:
            List of entity types
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT DISTINCT entity_type FROM entity_memory")
        rows = cursor.fetchall()
        conn.close()

        return [row[0] for row in rows]

    def get_tool_names(self) -> List[str]:
        """Get all tool names in the database.

        Returns:
            List of tool names
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT DISTINCT tool_name FROM tool_usage_history")
        rows = cursor.fetchall()
        conn.close()

        return [row[0] for row in rows]

    def save_q_table(
        self, agent_name: str, q_table: Dict[str, Dict[str, float]]
    ) -> None:
        """Save Q-table to the database.

        Args:
            agent_name: Name of the agent
            q_table: Q-table to save
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO q_tables
            (agent_name, q_table, last_updated)
            VALUES (?, ?, ?)
            """,
            (agent_name, json.dumps(q_table), time.time()),
        )

        conn.commit()
        conn.close()

    def get_q_table(self, agent_name: str) -> Optional[Dict[str, Dict[str, float]]]:
        """Get Q-table from the database.

        Args:
            agent_name: Name of the agent

        Returns:
            Q-table or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT q_table FROM q_tables WHERE agent_name = ?",
            (agent_name,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return json.loads(row[0])
        return None

    def save_policy_params(
        self, agent_name: str, policy_params: Dict[str, List[float]]
    ) -> None:
        """Save policy parameters to the database.

        Args:
            agent_name: Name of the agent
            policy_params: Policy parameters to save
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO policy_params
            (agent_name, policy_params, last_updated)
            VALUES (?, ?, ?)
            """,
            (agent_name, json.dumps(policy_params), time.time()),
        )

        conn.commit()
        conn.close()

    def get_policy_params(self, agent_name: str) -> Optional[Dict[str, List[float]]]:
        """Get policy parameters from the database.

        Args:
            agent_name: Name of the agent

        Returns:
            Policy parameters or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT policy_params FROM policy_params WHERE agent_name = ?",
            (agent_name,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return json.loads(row[0])
        return None

    def save_agent_reward(
        self, agent_name: str, reward: float, reward_components: Dict[str, float]
    ) -> None:
        """Save agent reward to the database.

        Args:
            agent_name: Name of the agent
            reward: Reward value
            reward_components: Reward components
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO agent_rewards (agent_name, reward, reward_components, timestamp) VALUES (?, ?, ?, ?)",
            (agent_name, reward, json.dumps(reward_components), time.time()),
        )

        conn.commit()
        conn.close()

    def get_agent_rewards(
        self, agent_name: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get agent rewards from the database.

        Args:
            agent_name: Name of the agent
            limit: Maximum number of rewards to return

        Returns:
            List of rewards
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT reward, reward_components, timestamp FROM agent_rewards WHERE agent_name = ? ORDER BY timestamp DESC LIMIT ?",
            (agent_name, limit),
        )

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "reward": reward,
                "components": json.loads(components),
                "timestamp": timestamp,
            }
            for reward, components, timestamp in rows
        ]

    def save_agent_interaction(
        self,
        agent_name: str,
        request: str,
        response: str,
        feedback: Optional[str] = None,
    ) -> None:
        """Save agent interaction to the database.

        Args:
            agent_name: Name of the agent
            request: User request
            response: Agent response
            feedback: User feedback (optional)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO agent_interactions (agent_name, request, response, feedback, timestamp) VALUES (?, ?, ?, ?, ?)",
            (agent_name, request, response, feedback, time.time()),
        )

        conn.commit()
        conn.close()

    def get_agent_interactions(
        self, agent_name: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get agent interactions from the database.

        Args:
            agent_name: Name of the agent
            limit: Maximum number of interactions to return

        Returns:
            List of interactions
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT request, response, feedback, timestamp FROM agent_interactions WHERE agent_name = ? ORDER BY timestamp DESC LIMIT ?",
            (agent_name, limit),
        )

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "request": request,
                "response": response,
                "feedback": feedback,
                "timestamp": timestamp,
            }
            for request, response, feedback, timestamp in rows
        ]

    def get_memory_summary(self) -> str:
        """Generate a summary of the memory contents.

        Returns:
            Summary string
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get conversation history count
        cursor.execute("SELECT COUNT(*) FROM conversation_history")
        conversation_count = cursor.fetchone()[0]

        # Get tool usage counts
        cursor.execute(
            "SELECT tool_name, COUNT(*) FROM tool_usage_history GROUP BY tool_name"
        )
        tool_usage = cursor.fetchall()

        # Get entity counts by type
        cursor.execute(
            "SELECT entity_type, COUNT(*) FROM entity_memory GROUP BY entity_type"
        )
        entity_counts = cursor.fetchall()

        # Get feedback counts
        cursor.execute(
            "SELECT feedback_type, COUNT(*) FROM learning_feedback GROUP BY feedback_type"
        )
        feedback_counts = cursor.fetchall()

        # Get RL agent counts
        cursor.execute("SELECT COUNT(*) FROM q_tables")
        q_table_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM policy_params")
        policy_params_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM agent_rewards")
        rewards_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM agent_interactions")
        interactions_count = cursor.fetchone()[0]

        conn.close()

        # Format the summary
        summary = "## Memory Summary\n\n"

        # Conversation summary
        summary += "### Conversation History\n"
        summary += f"- {conversation_count} messages in history\n\n"

        # Tool usage summary
        summary += "### Tool Usage\n"
        for tool_name, count in tool_usage:
            summary += f"- {tool_name}: {count} uses\n"
        summary += "\n"

        # Entity memory summary
        summary += "### Entities in Memory\n"
        for entity_type, count in entity_counts:
            summary += f"- {entity_type}: {count} entities\n"
        summary += "\n"

        # Feedback summary
        summary += "### Learning Feedback\n"
        for feedback_type, count in feedback_counts:
            summary += f"- {feedback_type}: {count} entries\n"
        summary += "\n"

        # RL summary
        summary += "### Reinforcement Learning\n"
        summary += f"- Q-tables: {q_table_count}\n"
        summary += f"- Policy parameters: {policy_params_count}\n"
        summary += f"- Rewards: {rewards_count}\n"
        summary += f"- Interactions: {interactions_count}\n"

        return summary

class FileBackedMemoryDatabase:
    """File-backed database for persisting agent memory."""

    def __init__(self, base_dir: str = "agent_memory"):
        """Initialize the file-backed memory database.

        Args:
            base_dir: Base directory for storing memory files
        """
        self.base_dir = Path(base_dir)
        self._initialize_dirs()

    def _initialize_dirs(self) -> None:
        """Initialize the directory structure."""
        # Create base directory
        os.makedirs(self.base_dir, exist_ok=True)

        # Create subdirectories for different types of data
        os.makedirs(self.base_dir / "conversation", exist_ok=True)
        os.makedirs(self.base_dir / "tool_usage", exist_ok=True)
        os.makedirs(self.base_dir / "entities", exist_ok=True)
        os.makedirs(self.base_dir / "performance", exist_ok=True)
        os.makedirs(self.base_dir / "learning", exist_ok=True)

    async def save_conversation_history(self, messages: List[Dict[str, str]]) -> None:
        """Save conversation history to files.

        Args:
            messages: List of messages to save
        """
        # Save the entire conversation history to a single file
        history_file = self.base_dir / "conversation" / "history.json"

        async with aiofiles.open(history_file, "w") as f:
            await f.write(json.dumps(messages, indent=2))

    async def load_conversation_history(self) -> List[Dict[str, str]]:
        """Load conversation history from files.

        Returns:
            List of messages
        """
        history_file = self.base_dir / "conversation" / "history.json"

        if not history_file.exists():
            return []

        async with aiofiles.open(history_file, "r") as f:
            content = await f.read()
            return json.loads(content)

    async def save_tool_usage(
        self, tool_name: str, args: Dict[str, Any], result: Any
    ) -> None:
        """Save tool usage to files.

        Args:
            tool_name: Name of the tool used
            args: Arguments passed to the tool
            result: Result returned by the tool
        """
        # Create tool directory if it doesn't exist
        tool_dir = self.base_dir / "tool_usage" / tool_name
        os.makedirs(tool_dir, exist_ok=True)

        # Create a timestamped file for this usage
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        usage_file = tool_dir / f"{timestamp}.json"

        usage_data = {
            "tool_name": tool_name,
            "args": args,
            "result": str(result),
            "timestamp": time.time(),
        }

        async with aiofiles.open(usage_file, "w") as f:
            await f.write(json.dumps(usage_data, indent=2))

    async def load_tool_usage(
        self, tool_name: Optional[str] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Load tool usage history from files.

        Args:
            tool_name: Name of tool to get history for, or None for all tools

        Returns:
            Tool usage history
        """
        result = {}

        if tool_name:
            # Load usage for a specific tool
            tool_dir = self.base_dir / "tool_usage" / tool_name

            if not tool_dir.exists():
                return {tool_name: []}

            result[tool_name] = []

            for usage_file in tool_dir.glob("*.json"):
                async with aiofiles.open(usage_file, "r") as f:
                    content = await f.read()
                    usage_data = json.loads(content)
                    result[tool_name].append(
                        {
                            "args": usage_data["args"],
                            "result": usage_data["result"],
                            "timestamp": usage_data["timestamp"],
                        }
                    )
        else:
            # Load usage for all tools
            tool_usage_dir = self.base_dir / "tool_usage"

            if not tool_usage_dir.exists():
                return {}

            for tool_dir in tool_usage_dir.iterdir():
                if tool_dir.is_dir():
                    tool = tool_dir.name
                    result[tool] = []

                    for usage_file in tool_dir.glob("*.json"):
                        async with aiofiles.open(usage_file, "r") as f:
                            content = await f.read()
                            usage_data = json.loads(content)
                            result[tool].append(
                                {
                                    "args": usage_data["args"],
                                    "result": usage_data["result"],
                                    "timestamp": usage_data["timestamp"],
                                }
                            )

        return result

    async def get_entity_types(self) -> List[str]:
        """Get all entity types in the file system.

        Returns:
            List of entity types
        """
        entities_dir = self.base_dir / "entities"

        if not entities_dir.exists():
            return []

        # Get all subdirectories in the entities directory
        return [d.name for d in entities_dir.iterdir() if d.is_dir()]

    async def get_tool_names(self) -> List[str]:
        """Get all tool names in the file system.

        Returns:
            List of tool names
        """
        tool_usage_dir = self.base_dir / "tool_usage"

        if not tool_usage_dir.exists():
            return []

        # Get all subdirectories in the tool_usage directory
        return [d.name for d in tool_usage_dir.iterdir() if d.is_dir()]

    async def get_memory_summary(self) -> str:
        """Generate a summary of the memory contents.

        Returns:
            Summary string
        """
        summary = "## Memory Summary\n\n"

        # Conversation summary
        history_file = self.base_dir / "conversation" / "history.json"
        if history_file.exists():
            async with aiofiles.open(history_file, "r") as f:
                content = await f.read()
                messages = json.loads(content)
                summary += "### Conversation History\n"
                summary += f"- {len(messages)} messages in history\n\n"
        else:
            summary += "### Conversation History\n"
            summary += "- 0 messages in history\n\n"

        # Tool usage summary
        tool_usage_dir = self.base_dir / "tool_usage"
        if tool_usage_dir.exists():
            summary += "### Tool Usage\n"
            for tool_dir in tool_usage_dir.iterdir():
                if tool_dir.is_dir():
                    tool_name = tool_dir.name
                    usage_count = len(list(tool_dir.glob("*.json")))
                    summary += f"- {tool_name}: {usage_count} uses\n"
            summary += "\n"
        else:
            summary += "### Tool Usage\n"
            summary += "- No tool usage recorded\n\n"

        # Entity memory summary
        entities_dir = self.base_dir / "entities"
        if entities_dir.exists():
            summary += "### Entities in Memory\n"
            for entity_type_dir in entities_dir.iterdir():
                if entity_type_dir.is_dir():
                    entity_type = entity_type_dir.name
                    entity_count = len(list(entity_type_dir.glob("*.json")))
                    summary += f"- {entity_type}: {entity_count} entities\n"
            summary += "\n"
        else:
            summary += "### Entities in Memory\n"
            summary += "- No entities recorded\n\n"

        # Learning feedback summary
        learning_dir = self.base_dir / "learning"
        if learning_dir.exists():
            feedback_types = {}
            for feedback_file in learning_dir.glob("*.json"):
                async with aiofiles.open(feedback_file, "r") as f:
                    content = await f.read()
                    feedback_data = json.loads(content)
                    feedback_type = feedback_data.get("feedback_type", "unknown")
                    if feedback_type not in feedback_types:
                        feedback_types[feedback_type] = 0
                    feedback_types[feedback_type] += 1

            summary += "### Learning Feedback\n"
            for feedback_type, count in feedback_types.items():
                summary += f"- {feedback_type}: {count} entries\n"
        else:
            summary += "### Learning Feedback\n"
            summary += "- No feedback recorded\n"

        return summary

    # Advanced reasoning methods
    async def save_reasoning_chain(self, chain_id: str, chain_data: Dict[str, Any]) -> None:
        """Save a reasoning chain to the database.

        Args:
            chain_id: Unique identifier for the reasoning chain
            chain_data: Chain data including goal and initial context
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO reasoning_chains
            (chain_id, goal, initial_context, start_time)
            VALUES (?, ?, ?, ?)
            """,
            (
                chain_id,
                chain_data["goal"],
                json.dumps(chain_data["initial_context"]),
                chain_data["start_time"]
            )
        )

        conn.commit()
        conn.close()

    async def save_reasoning_step(self, chain_id: str, step_data: Dict[str, Any]) -> None:
        """Save a reasoning step to the database.

        Args:
            chain_id: ID of the reasoning chain
            step_data: Step data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO reasoning_steps
            (chain_id, step_id, step_type, content, confidence, dependencies, timestamp, evidence, alternatives)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                chain_id,
                step_data["step_id"],
                step_data["step_type"],
                step_data["content"],
                step_data["confidence"],
                json.dumps(step_data["dependencies"]),
                step_data["timestamp"],
                json.dumps(step_data["evidence"]),
                json.dumps(step_data["alternatives"])
            )
        )

        conn.commit()
        conn.close()

    # Planning methods
    async def save_plan(self, plan_id: str, plan_data: Dict[str, Any]) -> None:
        """Save a plan to the database.

        Args:
            plan_id: Unique identifier for the plan
            plan_data: Plan data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO plans
            (plan_id, goal, actions, initial_state, goal_state, metadata, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plan_id,
                plan_data["goal"],
                json.dumps(plan_data["actions"]),
                json.dumps(plan_data["initial_state"]),
                json.dumps(plan_data["goal_state"]),
                json.dumps(plan_data["metadata"]),
                time.time()
            )
        )

        conn.commit()
        conn.close()

    # Meta-reasoning methods
    async def save_meta_decision(self, decision_data: Dict[str, Any]) -> None:
        """Save a meta-reasoning decision to the database.

        Args:
            decision_data: Decision data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO meta_decisions
            (decision_id, strategy, decision, rationale, confidence, expected_impact, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision_data["decision_id"],
                decision_data["strategy"],
                decision_data["decision"],
                decision_data["rationale"],
                decision_data["confidence"],
                json.dumps(decision_data["expected_impact"]),
                decision_data["timestamp"]
            )
        )

        conn.commit()
        conn.close()

    # Reflection methods
    async def save_reflection_session(self, session_id: str, session_data: Dict[str, Any]) -> None:
        """Save a reflection session to the database.

        Args:
            session_id: Unique identifier for the reflection session
            session_data: Session data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO reflection_sessions
            (session_id, trigger_event, focus_areas, insights, conclusions, improvement_plan, metadata, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                session_data["trigger_event"],
                json.dumps(session_data["focus_areas"]),
                json.dumps(session_data["insights"]),
                json.dumps(session_data["conclusions"]),
                json.dumps(session_data["improvement_plan"]),
                json.dumps(session_data["metadata"]),
                time.time()
            )
        )

        conn.commit()
        conn.close()

    # Additional helper methods for the new systems
    async def get_entity_types(self) -> List[str]:
        """Get all entity types in the database.

        Returns:
            List of entity types
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT DISTINCT entity_type FROM entity_memory")
        rows = cursor.fetchall()
        conn.close()

        return [row[0] for row in rows]

    async def get_tool_names(self) -> List[str]:
        """Get all tool names in the database.

        Returns:
            List of tool names
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT DISTINCT tool_name FROM tool_usage_history")
        rows = cursor.fetchall()
        conn.close()

        return [row[0] for row in rows]

    def get_agent_rewards(self, agent_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent rewards for an agent.

        Args:
            agent_name: Name of the agent
            limit: Maximum number of rewards to return

        Returns:
            List of recent rewards
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT reward, reward_components, timestamp
            FROM agent_rewards
            WHERE agent_name = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (agent_name, limit)
        )

        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "reward": reward,
                "reward_components": json.loads(components),
                "timestamp": timestamp
            }
            for reward, components, timestamp in rows
        ]
