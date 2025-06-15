"""
Agent metrics module for DataMCPServerAgent.
This module provides mechanisms for tracking and analyzing agent performance.
"""

import json
import time
from typing import Any, Dict, List, Optional, Tuple

from src.memory.memory_persistence import MemoryDatabase


class AgentPerformanceTracker:
    """Tracker for agent performance metrics."""

    def __init__(self, db: MemoryDatabase):
        """Initialize the agent performance tracker.

        Args:
            db: Memory database for persistence
        """
        self.db = db
        self._initialize_tables()

    def _initialize_tables(self) -> None:
        """Initialize the database tables for performance tracking."""
        # Create agent performance table
        self.db.execute(
            """
        CREATE TABLE IF NOT EXISTS agent_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            agent_name TEXT NOT NULL,
            task_id TEXT,
            success BOOLEAN NOT NULL,
            execution_time REAL NOT NULL,
            timestamp REAL NOT NULL
        )
        """
        )

        # Create agent metrics table
        self.db.execute(
            """
        CREATE TABLE IF NOT EXISTS agent_metrics (
            agent_name TEXT NOT NULL,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            timestamp REAL NOT NULL,
            PRIMARY KEY (agent_name, metric_name, timestamp)
        )
        """
        )

        # Create collaborative metrics table
        self.db.execute(
            """
        CREATE TABLE IF NOT EXISTS collaborative_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT NOT NULL,
            metric_value REAL NOT NULL,
            agents TEXT NOT NULL,
            timestamp REAL NOT NULL
        )
        """
        )

    def record_agent_execution(
        self, agent_name: str, success: bool, execution_time: float, task_id: Optional[str] = None
    ) -> None:
        """Record an agent execution.

        Args:
            agent_name: Agent name
            success: Whether the execution was successful
            execution_time: Execution time in seconds
            task_id: Optional task ID
        """
        self.db.execute(
            """
            INSERT INTO agent_performance (agent_name, task_id, success, execution_time, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (agent_name, task_id, success, execution_time, time.time()),
        )

    def record_agent_metric(self, agent_name: str, metric_name: str, metric_value: float) -> None:
        """Record an agent metric.

        Args:
            agent_name: Agent name
            metric_name: Metric name
            metric_value: Metric value
        """
        self.db.execute(
            """
            INSERT INTO agent_metrics (agent_name, metric_name, metric_value, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (agent_name, metric_name, metric_value, time.time()),
        )

    def record_collaborative_metric(
        self, metric_name: str, metric_value: float, agents: List[str]
    ) -> None:
        """Record a collaborative metric.

        Args:
            metric_name: Metric name
            metric_value: Metric value
            agents: List of agent names involved
        """
        self.db.execute(
            """
            INSERT INTO collaborative_metrics (metric_name, metric_value, agents, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (metric_name, metric_value, json.dumps(agents), time.time()),
        )

    def get_agent_success_rate(self, agent_name: str, time_window: Optional[float] = None) -> float:
        """Get an agent's success rate.

        Args:
            agent_name: Agent name
            time_window: Optional time window in seconds

        Returns:
            Success rate (0.0 to 1.0)
        """
        # Build query
        query = """
        SELECT COUNT(*), SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END)
        FROM agent_performance
        WHERE agent_name = ?
        """
        params = [agent_name]

        if time_window:
            query += " AND timestamp >= ?"
            params.append(time.time() - time_window)

        # Get counts
        result = self.db.execute(query, params).fetchone()

        if not result or result[0] == 0:
            return 0.0

        total_count, success_count = result
        return success_count / total_count

    def get_agent_average_execution_time(
        self, agent_name: str, time_window: Optional[float] = None
    ) -> float:
        """Get an agent's average execution time.

        Args:
            agent_name: Agent name
            time_window: Optional time window in seconds

        Returns:
            Average execution time in seconds
        """
        # Build query
        query = """
        SELECT AVG(execution_time)
        FROM agent_performance
        WHERE agent_name = ?
        """
        params = [agent_name]

        if time_window:
            query += " AND timestamp >= ?"
            params.append(time.time() - time_window)

        # Get average
        result = self.db.execute(query, params).fetchone()

        if not result or result[0] is None:
            return 0.0

        return result[0]

    def get_agent_metric_history(
        self, agent_name: str, metric_name: str, limit: int = 10
    ) -> List[Tuple[float, float]]:
        """Get an agent's metric history.

        Args:
            agent_name: Agent name
            metric_name: Metric name
            limit: Maximum number of records to return

        Returns:
            List of (timestamp, value) tuples
        """
        # Get metric history
        history = self.db.execute(
            """
            SELECT timestamp, metric_value
            FROM agent_metrics
            WHERE agent_name = ? AND metric_name = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (agent_name, metric_name, limit),
        ).fetchall()

        return [(ts, val) for ts, val in history]

    def get_agent_performance_summary(
        self, agent_name: str, time_window: Optional[float] = None
    ) -> Dict[str, Any]:
        """Get a summary of an agent's performance.

        Args:
            agent_name: Agent name
            time_window: Optional time window in seconds

        Returns:
            Performance summary
        """
        # Get success rate
        success_rate = self.get_agent_success_rate(agent_name, time_window)

        # Get average execution time
        avg_execution_time = self.get_agent_average_execution_time(agent_name, time_window)

        # Get execution count
        query = """
        SELECT COUNT(*)
        FROM agent_performance
        WHERE agent_name = ?
        """
        params = [agent_name]

        if time_window:
            query += " AND timestamp >= ?"
            params.append(time.time() - time_window)

        execution_count = self.db.execute(query, params).fetchone()[0]

        # Get recent metrics
        metrics = self.db.execute(
            """
            SELECT metric_name, AVG(metric_value)
            FROM agent_metrics
            WHERE agent_name = ?
            GROUP BY metric_name
            """,
            (agent_name,),
        ).fetchall()

        return {
            "agent_name": agent_name,
            "success_rate": success_rate,
            "average_execution_time": avg_execution_time,
            "execution_count": execution_count,
            "metrics": {name: value for name, value in metrics},
        }

    def get_collaborative_performance(
        self, agents: List[str], time_window: Optional[float] = None
    ) -> Dict[str, Any]:
        """Get collaborative performance metrics for a group of agents.

        Args:
            agents: List of agent names
            time_window: Optional time window in seconds

        Returns:
            Collaborative performance metrics
        """
        # Get collaborative metrics
        query = """
        SELECT metric_name, AVG(metric_value)
        FROM collaborative_metrics
        WHERE agents LIKE ?
        """
        params = [f"%{json.dumps(agents)[1:-1]}%"]

        if time_window:
            query += " AND timestamp >= ?"
            params.append(time.time() - time_window)

        query += " GROUP BY metric_name"

        metrics = self.db.execute(query, params).fetchall()

        # Get individual agent performance
        agent_performance = {}
        for agent in agents:
            agent_performance[agent] = self.get_agent_performance_summary(agent, time_window)

        return {
            "agents": agents,
            "collaborative_metrics": {name: value for name, value in metrics},
            "individual_performance": agent_performance,
        }

    def compare_agents(
        self, agent_names: List[str], time_window: Optional[float] = None
    ) -> Dict[str, Any]:
        """Compare performance between multiple agents.

        Args:
            agent_names: List of agent names to compare
            time_window: Optional time window in seconds

        Returns:
            Comparison results
        """
        # Get performance summaries
        summaries = {}
        for agent_name in agent_names:
            summaries[agent_name] = self.get_agent_performance_summary(agent_name, time_window)

        # Calculate relative performance
        if len(agent_names) > 1:
            # Success rate comparison
            success_rates = {name: summary["success_rate"] for name, summary in summaries.items()}
            max_success_rate = max(success_rates.values()) if success_rates else 0.0

            # Execution time comparison
            execution_times = {
                name: summary["average_execution_time"] for name, summary in summaries.items()
            }
            min_execution_time = min(execution_times.values()) if execution_times else 0.0

            # Calculate relative performance scores
            relative_performance = {}
            for name in agent_names:
                success_score = (
                    success_rates[name] / max_success_rate if max_success_rate > 0 else 0.0
                )
                time_score = (
                    min_execution_time / execution_times[name] if execution_times[name] > 0 else 0.0
                )
                relative_performance[name] = (success_score + time_score) / 2
        else:
            relative_performance = {agent_names[0]: 1.0} if agent_names else {}

        return {"agent_summaries": summaries, "relative_performance": relative_performance}


class MultiAgentPerformanceAnalyzer:
    """Analyzer for multi-agent performance."""

    def __init__(self, db: MemoryDatabase, performance_tracker: AgentPerformanceTracker):
        """Initialize the multi-agent performance analyzer.

        Args:
            db: Memory database for persistence
            performance_tracker: Agent performance tracker
        """
        self.db = db
        self.performance_tracker = performance_tracker

    def analyze_agent_synergy(self, agents: List[str]) -> Dict[str, Any]:
        """Analyze synergy between agents.

        Args:
            agents: List of agent names

        Returns:
            Synergy analysis
        """
        # Get collaborative performance
        collaborative = self.performance_tracker.get_collaborative_performance(agents)

        # Get individual performance
        individual = {
            agent: self.performance_tracker.get_agent_performance_summary(agent) for agent in agents
        }

        # Calculate synergy metrics
        synergy_metrics = {}

        # Success rate synergy
        individual_success_rates = [perf["success_rate"] for perf in individual.values()]
        avg_individual_success = (
            sum(individual_success_rates) / len(individual_success_rates)
            if individual_success_rates
            else 0.0
        )

        collaborative_success = 0.0
        if (
            "collaborative_metrics" in collaborative
            and "success_rate" in collaborative["collaborative_metrics"]
        ):
            collaborative_success = collaborative["collaborative_metrics"]["success_rate"]

        synergy_metrics["success_rate_synergy"] = collaborative_success - avg_individual_success

        # Execution time synergy
        individual_times = [perf["average_execution_time"] for perf in individual.values()]
        avg_individual_time = (
            sum(individual_times) / len(individual_times) if individual_times else 0.0
        )

        collaborative_time = 0.0
        if (
            "collaborative_metrics" in collaborative
            and "execution_time" in collaborative["collaborative_metrics"]
        ):
            collaborative_time = collaborative["collaborative_metrics"]["execution_time"]

        synergy_metrics["execution_time_synergy"] = avg_individual_time - collaborative_time

        return {
            "agents": agents,
            "synergy_metrics": synergy_metrics,
            "collaborative_performance": collaborative,
            "individual_performance": individual,
        }

    def identify_optimal_agent_combinations(
        self, all_agents: List[str], max_combination_size: int = 3
    ) -> List[Dict[str, Any]]:
        """Identify optimal combinations of agents.

        Args:
            all_agents: List of all available agents
            max_combination_size: Maximum combination size

        Returns:
            List of optimal agent combinations
        """
        from itertools import combinations

        # Generate all possible combinations
        all_combinations = []
        for size in range(2, min(max_combination_size + 1, len(all_agents) + 1)):
            all_combinations.extend(combinations(all_agents, size))

        # Analyze each combination
        combination_results = []
        for combo in all_combinations:
            agents = list(combo)
            synergy = self.analyze_agent_synergy(agents)

            # Calculate overall synergy score
            success_synergy = synergy["synergy_metrics"].get("success_rate_synergy", 0.0)
            time_synergy = synergy["synergy_metrics"].get("execution_time_synergy", 0.0)

            # Normalize time synergy (higher is better)
            normalized_time_synergy = time_synergy / 10.0 if time_synergy > 0 else 0.0

            overall_score = success_synergy + normalized_time_synergy

            combination_results.append(
                {
                    "agents": agents,
                    "synergy_score": overall_score,
                    "success_rate_synergy": success_synergy,
                    "execution_time_synergy": time_synergy,
                }
            )

        # Sort by synergy score
        combination_results.sort(key=lambda x: x["synergy_score"], reverse=True)

        return combination_results

    def analyze_learning_impact(
        self, agent_name: str, before_timestamp: float, after_timestamp: float
    ) -> Dict[str, Any]:
        """Analyze the impact of learning on agent performance.

        Args:
            agent_name: Agent name
            before_timestamp: Timestamp before learning
            after_timestamp: Timestamp after learning

        Returns:
            Learning impact analysis
        """
        # Get performance before learning
        query_before = """
        SELECT COUNT(*), SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END), AVG(execution_time)
        FROM agent_performance
        WHERE agent_name = ? AND timestamp < ? AND timestamp >= ?
        """
        before_window_start = before_timestamp - (after_timestamp - before_timestamp)
        before_result = self.db.execute(
            query_before, (agent_name, before_timestamp, before_window_start)
        ).fetchone()

        # Get performance after learning
        query_after = """
        SELECT COUNT(*), SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END), AVG(execution_time)
        FROM agent_performance
        WHERE agent_name = ? AND timestamp >= ? AND timestamp < ?
        """
        after_window_end = after_timestamp + (after_timestamp - before_timestamp)
        after_result = self.db.execute(
            query_after, (agent_name, after_timestamp, after_window_end)
        ).fetchone()

        # Calculate metrics
        before_count, before_success, before_time = before_result if before_result else (0, 0, 0)
        after_count, after_success, after_time = after_result if after_result else (0, 0, 0)

        before_success_rate = before_success / before_count if before_count > 0 else 0.0
        after_success_rate = after_success / after_count if after_count > 0 else 0.0

        success_rate_change = after_success_rate - before_success_rate
        execution_time_change = before_time - after_time if before_time and after_time else 0.0

        return {
            "agent_name": agent_name,
            "before_timestamp": before_timestamp,
            "after_timestamp": after_timestamp,
            "before_metrics": {
                "execution_count": before_count,
                "success_rate": before_success_rate,
                "average_execution_time": before_time,
            },
            "after_metrics": {
                "execution_count": after_count,
                "success_rate": after_success_rate,
                "average_execution_time": after_time,
            },
            "changes": {
                "success_rate_change": success_rate_change,
                "execution_time_change": execution_time_change,
            },
        }


# Factory function to create agent performance tracker
def create_agent_performance_tracker(db: MemoryDatabase) -> AgentPerformanceTracker:
    """Create an agent performance tracker.

    Args:
        db: Memory database for persistence

    Returns:
        Agent performance tracker
    """
    return AgentPerformanceTracker(db)


# Factory function to create multi-agent performance analyzer
def create_multi_agent_performance_analyzer(
    db: MemoryDatabase, performance_tracker: AgentPerformanceTracker
) -> MultiAgentPerformanceAnalyzer:
    """Create a multi-agent performance analyzer.

    Args:
        db: Memory database for persistence
        performance_tracker: Agent performance tracker

    Returns:
        Multi-agent performance analyzer
    """
    return MultiAgentPerformanceAnalyzer(db, performance_tracker)
