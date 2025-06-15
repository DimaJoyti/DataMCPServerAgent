"""
Advanced memory systems for reinforcement learning in DataMCPServerAgent.
This module implements sophisticated memory mechanisms including episodic memory,
working memory, and long-term memory consolidation.
"""

import json
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from src.memory.memory_persistence import MemoryDatabase


@dataclass
class EpisodicMemory:
    """Represents an episodic memory entry."""

    memory_id: str
    timestamp: float
    state: np.ndarray
    action: int
    reward: float
    context: Dict[str, Any]
    importance: float = 1.0
    access_count: int = 0
    last_access: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "memory_id": self.memory_id,
            "timestamp": self.timestamp,
            "state": self.state.tolist() if isinstance(self.state, np.ndarray) else self.state,
            "action": self.action,
            "reward": self.reward,
            "context": self.context,
            "importance": self.importance,
            "access_count": self.access_count,
            "last_access": self.last_access,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodicMemory":
        """Create from dictionary."""
        data["state"] = np.array(data["state"]) if isinstance(data["state"], list) else data["state"]
        return cls(**data)


class NeuralEpisodicControl:
    """Neural Episodic Control for fast learning from few examples."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        memory_capacity: int = 10000,
        k_neighbors: int = 50,
        learning_rate: float = 0.1,
    ):
        """Initialize Neural Episodic Control.
        
        Args:
            state_dim: State space dimension
            action_dim: Action space dimension
            memory_capacity: Maximum memory capacity
            k_neighbors: Number of neighbors for retrieval
            learning_rate: Learning rate for value updates
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory_capacity = memory_capacity
        self.k_neighbors = k_neighbors
        self.learning_rate = learning_rate

        # Episodic memory for each action
        self.episodic_memories = {action: [] for action in range(action_dim)}

        # State encoder (simple linear for now, can be enhanced)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_encoder.to(self.device)

    def encode_state(self, state: np.ndarray) -> np.ndarray:
        """Encode state using neural network.
        
        Args:
            state: Input state
            
        Returns:
            Encoded state
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            encoded = self.state_encoder(state_tensor)
            return encoded.cpu().numpy()

    def add_memory(self, state: np.ndarray, action: int, reward: float, context: Dict[str, Any]):
        """Add memory to episodic control.
        
        Args:
            state: State
            action: Action taken
            reward: Reward received
            context: Additional context
        """
        encoded_state = self.encode_state(state)

        memory = EpisodicMemory(
            memory_id=f"{action}_{len(self.episodic_memories[action])}_{time.time()}",
            timestamp=time.time(),
            state=encoded_state,
            action=action,
            reward=reward,
            context=context,
        )

        # Add to action-specific memory
        self.episodic_memories[action].append(memory)

        # Maintain capacity
        if len(self.episodic_memories[action]) > self.memory_capacity // self.action_dim:
            # Remove oldest memory
            self.episodic_memories[action].pop(0)

    def retrieve_value(self, state: np.ndarray, action: int) -> float:
        """Retrieve value estimate for state-action pair.
        
        Args:
            state: Query state
            action: Query action
            
        Returns:
            Estimated value
        """
        if not self.episodic_memories[action]:
            return 0.0

        encoded_state = self.encode_state(state)

        # Compute similarities to all memories for this action
        similarities = []
        for memory in self.episodic_memories[action]:
            similarity = self._compute_similarity(encoded_state, memory.state)
            similarities.append((similarity, memory))

        # Sort by similarity and take top k
        similarities.sort(key=lambda x: x[0], reverse=True)
        top_k = similarities[:min(self.k_neighbors, len(similarities))]

        if not top_k:
            return 0.0

        # Weighted average of rewards
        total_weight = 0
        weighted_value = 0

        for similarity, memory in top_k:
            weight = similarity
            weighted_value += weight * memory.reward
            total_weight += weight

            # Update access statistics
            memory.access_count += 1
            memory.last_access = time.time()

        if total_weight == 0:
            return 0.0

        return weighted_value / total_weight

    def _compute_similarity(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """Compute similarity between two states.
        
        Args:
            state1: First state
            state2: Second state
            
        Returns:
            Similarity score
        """
        # Cosine similarity
        dot_product = np.dot(state1, state2)
        norm1 = np.linalg.norm(state1)
        norm2 = np.linalg.norm(state2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory statistics.
        
        Returns:
            Memory statistics
        """
        total_memories = sum(len(memories) for memories in self.episodic_memories.values())

        action_counts = {
            action: len(memories)
            for action, memories in self.episodic_memories.items()
        }

        # Average access counts
        all_memories = []
        for memories in self.episodic_memories.values():
            all_memories.extend(memories)

        avg_access_count = np.mean([m.access_count for m in all_memories]) if all_memories else 0

        return {
            "total_memories": total_memories,
            "action_counts": action_counts,
            "avg_access_count": avg_access_count,
            "memory_utilization": total_memories / self.memory_capacity,
        }


class WorkingMemory:
    """Working memory for maintaining current context and goals."""

    def __init__(self, capacity: int = 10):
        """Initialize working memory.
        
        Args:
            capacity: Maximum capacity of working memory
        """
        self.capacity = capacity
        self.items = []
        self.attention_weights = []

    def add_item(self, item: Dict[str, Any], importance: float = 1.0):
        """Add item to working memory.
        
        Args:
            item: Item to add
            importance: Importance weight
        """
        if len(self.items) >= self.capacity:
            # Remove least important item
            min_idx = np.argmin(self.attention_weights)
            self.items.pop(min_idx)
            self.attention_weights.pop(min_idx)

        self.items.append(item)
        self.attention_weights.append(importance)

    def get_context(self) -> Dict[str, Any]:
        """Get current context from working memory.
        
        Returns:
            Aggregated context
        """
        if not self.items:
            return {}

        # Weight items by attention
        total_weight = sum(self.attention_weights)
        if total_weight == 0:
            return {}

        # Aggregate context
        context = {}
        for item, weight in zip(self.items, self.attention_weights):
            normalized_weight = weight / total_weight
            for key, value in item.items():
                if key not in context:
                    context[key] = 0
                if isinstance(value, (int, float)):
                    context[key] += value * normalized_weight

        return context

    def update_attention(self, query: Dict[str, Any]):
        """Update attention weights based on query relevance.
        
        Args:
            query: Query to match against
        """
        for i, item in enumerate(self.items):
            # Simple relevance scoring
            relevance = 0
            for key, value in query.items():
                if key in item:
                    if isinstance(value, str) and isinstance(item[key], str):
                        # Text similarity
                        common_words = set(value.lower().split()) & set(item[key].lower().split())
                        relevance += len(common_words)
                    elif isinstance(value, (int, float)) and isinstance(item[key], (int, float)):
                        # Numerical similarity
                        relevance += 1.0 / (1.0 + abs(value - item[key]))

            self.attention_weights[i] = max(0.1, relevance)  # Minimum attention


class LongTermMemoryConsolidation:
    """Long-term memory consolidation for important experiences."""

    def __init__(self, db: MemoryDatabase, consolidation_threshold: float = 0.8):
        """Initialize long-term memory consolidation.
        
        Args:
            db: Memory database
            consolidation_threshold: Threshold for consolidation
        """
        self.db = db
        self.consolidation_threshold = consolidation_threshold

        # Create table for consolidated memories
        self._create_consolidated_memory_table()

    def _create_consolidated_memory_table(self):
        """Create table for consolidated memories."""
        with sqlite3.connect(self.db.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS consolidated_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    importance REAL NOT NULL,
                    consolidation_score REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0
                )
            """)
            conn.commit()

    def consolidate_episodic_memories(self, episodic_memories: List[EpisodicMemory]):
        """Consolidate episodic memories into long-term memory.
        
        Args:
            episodic_memories: List of episodic memories to consolidate
        """
        # Group memories by similarity
        memory_clusters = self._cluster_memories(episodic_memories)

        for cluster in memory_clusters:
            if len(cluster) >= 3:  # Need multiple similar experiences
                consolidated_memory = self._create_consolidated_memory(cluster)
                self._store_consolidated_memory(consolidated_memory)

    def _cluster_memories(self, memories: List[EpisodicMemory]) -> List[List[EpisodicMemory]]:
        """Cluster similar memories together.
        
        Args:
            memories: List of memories to cluster
            
        Returns:
            List of memory clusters
        """
        clusters = []
        used_memories = set()

        for i, memory in enumerate(memories):
            if i in used_memories:
                continue

            cluster = [memory]
            used_memories.add(i)

            for j, other_memory in enumerate(memories[i+1:], i+1):
                if j in used_memories:
                    continue

                # Check similarity
                similarity = self._compute_memory_similarity(memory, other_memory)
                if similarity > 0.7:  # Similarity threshold
                    cluster.append(other_memory)
                    used_memories.add(j)

            clusters.append(cluster)

        return clusters

    def _compute_memory_similarity(self, memory1: EpisodicMemory, memory2: EpisodicMemory) -> float:
        """Compute similarity between two memories.
        
        Args:
            memory1: First memory
            memory2: Second memory
            
        Returns:
            Similarity score
        """
        # State similarity
        state_sim = np.dot(memory1.state, memory2.state) / (
            np.linalg.norm(memory1.state) * np.linalg.norm(memory2.state)
        )

        # Action similarity
        action_sim = 1.0 if memory1.action == memory2.action else 0.0

        # Context similarity
        context_sim = 0.0
        if "request" in memory1.context and "request" in memory2.context:
            words1 = set(memory1.context["request"].lower().split())
            words2 = set(memory2.context["request"].lower().split())
            if words1 and words2:
                context_sim = len(words1 & words2) / len(words1 | words2)

        # Combined similarity
        return 0.5 * state_sim + 0.3 * action_sim + 0.2 * context_sim

    def _create_consolidated_memory(self, cluster: List[EpisodicMemory]) -> Dict[str, Any]:
        """Create consolidated memory from cluster.
        
        Args:
            cluster: Cluster of similar memories
            
        Returns:
            Consolidated memory
        """
        # Average state
        avg_state = np.mean([m.state for m in cluster], axis=0)

        # Most common action
        actions = [m.action for m in cluster]
        most_common_action = max(set(actions), key=actions.count)

        # Average reward
        avg_reward = np.mean([m.reward for m in cluster])

        # Aggregate context
        contexts = [m.context for m in cluster]
        common_context = {}
        for context in contexts:
            for key, value in context.items():
                if key not in common_context:
                    common_context[key] = []
                common_context[key].append(value)

        # Calculate importance
        importance = np.mean([m.importance for m in cluster])

        # Calculate consolidation score
        consolidation_score = len(cluster) / 10.0  # More similar experiences = higher score

        return {
            "memory_type": "consolidated_episodic",
            "content": {
                "state": avg_state.tolist(),
                "action": most_common_action,
                "reward": avg_reward,
                "context": common_context,
                "cluster_size": len(cluster),
            },
            "importance": importance,
            "consolidation_score": consolidation_score,
        }

    def _store_consolidated_memory(self, memory: Dict[str, Any]):
        """Store consolidated memory in database.
        
        Args:
            memory: Consolidated memory to store
        """
        with sqlite3.connect(self.db.db_path) as conn:
            conn.execute("""
                INSERT INTO consolidated_memories 
                (memory_type, content, importance, consolidation_score)
                VALUES (?, ?, ?, ?)
            """, (
                memory["memory_type"],
                json.dumps(memory["content"]),
                memory["importance"],
                memory["consolidation_score"],
            ))
            conn.commit()

    def retrieve_consolidated_memories(
        self,
        query_state: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant consolidated memories.
        
        Args:
            query_state: Query state
            top_k: Number of memories to retrieve
            
        Returns:
            List of relevant consolidated memories
        """
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.execute("""
                SELECT content, importance, consolidation_score 
                FROM consolidated_memories 
                WHERE memory_type = 'consolidated_episodic'
                ORDER BY consolidation_score DESC, importance DESC
                LIMIT ?
            """, (top_k * 2,))  # Get more than needed for filtering

            results = cursor.fetchall()

        # Filter by state similarity
        relevant_memories = []
        for content_json, importance, consolidation_score in results:
            content = json.loads(content_json)
            memory_state = np.array(content["state"])

            # Compute similarity
            similarity = np.dot(query_state, memory_state) / (
                np.linalg.norm(query_state) * np.linalg.norm(memory_state)
            )

            if similarity > 0.5:  # Similarity threshold
                relevant_memories.append({
                    "content": content,
                    "importance": importance,
                    "consolidation_score": consolidation_score,
                    "similarity": similarity,
                })

        # Sort by combined score and return top k
        relevant_memories.sort(
            key=lambda x: x["similarity"] * x["consolidation_score"],
            reverse=True
        )

        return relevant_memories[:top_k]


class AdvancedRLMemorySystem:
    """Advanced memory system combining multiple memory types."""

    def __init__(
        self,
        db: MemoryDatabase,
        state_dim: int,
        action_dim: int,
        episodic_capacity: int = 10000,
        working_memory_capacity: int = 10,
    ):
        """Initialize advanced RL memory system.
        
        Args:
            db: Memory database
            state_dim: State space dimension
            action_dim: Action space dimension
            episodic_capacity: Episodic memory capacity
            working_memory_capacity: Working memory capacity
        """
        self.db = db
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Initialize memory components
        self.episodic_control = NeuralEpisodicControl(
            state_dim, action_dim, episodic_capacity
        )
        self.working_memory = WorkingMemory(working_memory_capacity)
        self.consolidation = LongTermMemoryConsolidation(db)

        # Memory integration weights
        self.episodic_weight = 0.4
        self.working_memory_weight = 0.3
        self.consolidated_weight = 0.3

    def add_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        context: Dict[str, Any]
    ):
        """Add experience to memory system.
        
        Args:
            state: State
            action: Action taken
            reward: Reward received
            context: Additional context
        """
        # Add to episodic control
        self.episodic_control.add_memory(state, action, reward, context)

        # Add to working memory
        self.working_memory.add_item({
            "state": state.tolist(),
            "action": action,
            "reward": reward,
            "context": context,
        }, importance=abs(reward))

    def get_value_estimate(self, state: np.ndarray, action: int) -> float:
        """Get integrated value estimate from all memory systems.
        
        Args:
            state: Query state
            action: Query action
            
        Returns:
            Integrated value estimate
        """
        # Episodic control value
        episodic_value = self.episodic_control.retrieve_value(state, action)

        # Working memory context
        working_context = self.working_memory.get_context()
        working_value = working_context.get("reward", 0.0)

        # Consolidated memory value
        consolidated_memories = self.consolidation.retrieve_consolidated_memories(state)
        consolidated_value = 0.0
        if consolidated_memories:
            # Average reward from relevant consolidated memories
            relevant_rewards = [
                m["content"]["reward"] for m in consolidated_memories
                if m["content"]["action"] == action
            ]
            if relevant_rewards:
                consolidated_value = np.mean(relevant_rewards)

        # Integrate values
        integrated_value = (
            self.episodic_weight * episodic_value +
            self.working_memory_weight * working_value +
            self.consolidated_weight * consolidated_value
        )

        return integrated_value

    def consolidate_memories(self):
        """Trigger memory consolidation process."""
        # Get all episodic memories
        all_episodic_memories = []
        for action_memories in self.episodic_control.episodic_memories.values():
            all_episodic_memories.extend(action_memories)

        # Consolidate important memories
        important_memories = [
            m for m in all_episodic_memories
            if m.importance > 0.5 and m.access_count > 2
        ]

        if important_memories:
            self.consolidation.consolidate_episodic_memories(important_memories)

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics.
        
        Returns:
            Memory statistics from all systems
        """
        episodic_stats = self.episodic_control.get_memory_statistics()

        working_memory_stats = {
            "working_memory_items": len(self.working_memory.items),
            "working_memory_capacity": self.working_memory.capacity,
        }

        # Consolidated memory stats
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*), AVG(importance), AVG(consolidation_score)
                FROM consolidated_memories
            """)
            count, avg_importance, avg_consolidation = cursor.fetchone()

        consolidated_stats = {
            "consolidated_memories": count or 0,
            "avg_importance": avg_importance or 0.0,
            "avg_consolidation_score": avg_consolidation or 0.0,
        }

        return {
            "episodic": episodic_stats,
            "working_memory": working_memory_stats,
            "consolidated": consolidated_stats,
        }
