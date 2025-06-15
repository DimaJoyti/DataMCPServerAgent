"""
Enhanced state representation for reinforcement learning in DataMCPServerAgent.
This module provides advanced state encoding techniques for better RL performance.
"""

from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.memory.memory_persistence import MemoryDatabase


class TextEmbeddingEncoder:
    """Text embedding-based state encoder using sentence transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", max_length: int = 512):
        """Initialize text embedding encoder.
        
        Args:
            model_name: Name of the sentence transformer model
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.encoder = SentenceTransformer(model_name)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding vector.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        # Truncate text if too long
        if len(text) > self.max_length:
            text = text[:self.max_length]

        embedding = self.encoder.encode(text, convert_to_numpy=True)
        return embedding

    def encode_conversation(self, messages: List[Dict[str, Any]]) -> np.ndarray:
        """Encode conversation history to embedding.
        
        Args:
            messages: List of conversation messages
            
        Returns:
            Conversation embedding
        """
        # Combine messages into single text
        text_parts = []
        for msg in messages[-10:]:  # Last 10 messages
            if isinstance(msg, dict):
                role = msg.get("role", "")
                content = msg.get("content", "")
                text_parts.append(f"{role}: {content}")
            else:
                text_parts.append(str(msg))

        conversation_text = " ".join(text_parts)
        return self.encode_text(conversation_text)


class ContextualStateEncoder:
    """Contextual state encoder that combines multiple information sources."""

    def __init__(
        self,
        text_encoder: Optional[TextEmbeddingEncoder] = None,
        include_temporal: bool = True,
        include_performance: bool = True,
        include_user_profile: bool = True,
    ):
        """Initialize contextual state encoder.
        
        Args:
            text_encoder: Text embedding encoder
            include_temporal: Whether to include temporal features
            include_performance: Whether to include performance features
            include_user_profile: Whether to include user profile features
        """
        self.text_encoder = text_encoder or TextEmbeddingEncoder()
        self.include_temporal = include_temporal
        self.include_performance = include_performance
        self.include_user_profile = include_user_profile

        # Feature dimensions
        self.text_dim = self.text_encoder.embedding_dim
        self.temporal_dim = 10 if include_temporal else 0
        self.performance_dim = 15 if include_performance else 0
        self.user_profile_dim = 20 if include_user_profile else 0

        self.total_dim = (
            self.text_dim + self.temporal_dim +
            self.performance_dim + self.user_profile_dim
        )

    def extract_temporal_features(self, context: Dict[str, Any]) -> np.ndarray:
        """Extract temporal features from context.
        
        Args:
            context: Context dictionary
            
        Returns:
            Temporal feature vector
        """
        import datetime

        features = []

        # Current time features
        now = datetime.datetime.now()
        features.append(now.hour / 24.0)  # Hour of day
        features.append(now.weekday() / 7.0)  # Day of week
        features.append(now.month / 12.0)  # Month of year

        # Conversation timing
        history = context.get("history", [])
        if history:
            # Time since last message (normalized)
            features.append(min(1.0, len(history) / 100.0))
        else:
            features.append(0.0)

        # Session length
        session_length = context.get("session_length", 0)
        features.append(min(1.0, session_length / 3600.0))  # Normalized to hours

        # Pad to fixed size
        while len(features) < self.temporal_dim:
            features.append(0.0)

        return np.array(features[:self.temporal_dim], dtype=np.float32)

    def extract_performance_features(
        self, context: Dict[str, Any], db: MemoryDatabase
    ) -> np.ndarray:
        """Extract performance features from context and database.
        
        Args:
            context: Context dictionary
            db: Memory database
            
        Returns:
            Performance feature vector
        """
        features = []

        # Recent success rate
        recent_rewards = context.get("recent_rewards", [])
        if recent_rewards:
            success_rate = sum(1 for r in recent_rewards if r > 0) / len(recent_rewards)
            avg_reward = np.mean(recent_rewards)
        else:
            success_rate = 0.5  # Neutral
            avg_reward = 0.0

        features.extend([success_rate, avg_reward])

        # Response time features
        recent_times = context.get("recent_response_times", [])
        if recent_times:
            avg_time = np.mean(recent_times)
            std_time = np.std(recent_times)
        else:
            avg_time = 1.0  # Default
            std_time = 0.0

        features.extend([min(1.0, avg_time / 10.0), min(1.0, std_time / 5.0)])

        # Tool usage patterns
        tool_usage = context.get("tool_usage_counts", {})
        total_usage = sum(tool_usage.values()) if tool_usage else 1

        # Most used tools (top 5)
        sorted_tools = sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)[:5]
        for i in range(5):
            if i < len(sorted_tools):
                features.append(sorted_tools[i][1] / total_usage)
            else:
                features.append(0.0)

        # Error rate
        error_count = context.get("recent_error_count", 0)
        total_requests = context.get("recent_request_count", 1)
        error_rate = error_count / total_requests
        features.append(error_rate)

        # User satisfaction (if available)
        satisfaction = context.get("user_satisfaction", 0.5)
        features.append(satisfaction)

        # Task complexity (estimated)
        request = context.get("request", "")
        complexity = min(1.0, len(request.split()) / 50.0)  # Based on word count
        features.append(complexity)

        # Pad to fixed size
        while len(features) < self.performance_dim:
            features.append(0.0)

        return np.array(features[:self.performance_dim], dtype=np.float32)

    def extract_user_profile_features(self, context: Dict[str, Any]) -> np.ndarray:
        """Extract user profile features from context.
        
        Args:
            context: Context dictionary
            
        Returns:
            User profile feature vector
        """
        features = []

        user_profile = context.get("user_profile", {})

        # User preferences
        preferences = user_profile.get("preferences", {})
        features.append(preferences.get("verbosity", 0.5))  # 0=concise, 1=verbose
        features.append(preferences.get("technical_level", 0.5))  # 0=basic, 1=expert
        features.append(preferences.get("response_speed", 0.5))  # 0=thorough, 1=fast

        # User behavior patterns
        behavior = user_profile.get("behavior", {})
        features.append(behavior.get("avg_session_length", 0.5))
        features.append(behavior.get("question_complexity", 0.5))
        features.append(behavior.get("follow_up_rate", 0.5))

        # User expertise in different domains
        expertise = user_profile.get("expertise", {})
        domains = ["technology", "business", "science", "arts", "general"]
        for domain in domains:
            features.append(expertise.get(domain, 0.5))

        # User interaction style
        interaction = user_profile.get("interaction_style", {})
        features.append(interaction.get("politeness", 0.5))
        features.append(interaction.get("directness", 0.5))
        features.append(interaction.get("patience", 0.5))

        # Recent activity
        activity = user_profile.get("recent_activity", {})
        features.append(activity.get("frequency", 0.5))  # How often user interacts
        features.append(activity.get("consistency", 0.5))  # Consistency of requests

        # Satisfaction history
        satisfaction_history = user_profile.get("satisfaction_history", [])
        if satisfaction_history:
            avg_satisfaction = np.mean(satisfaction_history[-10:])  # Last 10 interactions
        else:
            avg_satisfaction = 0.5
        features.append(avg_satisfaction)

        # Pad to fixed size
        while len(features) < self.user_profile_dim:
            features.append(0.5)  # Neutral default

        return np.array(features[:self.user_profile_dim], dtype=np.float32)

    async def encode_state(
        self, context: Dict[str, Any], db: MemoryDatabase
    ) -> np.ndarray:
        """Encode complete state from context.
        
        Args:
            context: Context dictionary
            db: Memory database
            
        Returns:
            Complete state vector
        """
        features = []

        # Text features
        request = context.get("request", "")
        history = context.get("history", [])

        # Encode current request
        request_embedding = self.text_encoder.encode_text(request)
        features.append(request_embedding)

        # Encode conversation history
        if history:
            history_embedding = self.text_encoder.encode_conversation(history)
        else:
            history_embedding = np.zeros(self.text_dim, dtype=np.float32)
        features.append(history_embedding)

        # Combine request and history embeddings (average)
        text_features = (request_embedding + history_embedding) / 2

        # Temporal features
        if self.include_temporal:
            temporal_features = self.extract_temporal_features(context)
            features.append(temporal_features)

        # Performance features
        if self.include_performance:
            performance_features = self.extract_performance_features(context, db)
            features.append(performance_features)

        # User profile features
        if self.include_user_profile:
            user_features = self.extract_user_profile_features(context)
            features.append(user_features)

        # Concatenate all features
        state_vector = np.concatenate([
            text_features,
            temporal_features if self.include_temporal else np.array([]),
            performance_features if self.include_performance else np.array([]),
            user_features if self.include_user_profile else np.array([])
        ])

        return state_vector.astype(np.float32)


class GraphStateEncoder:
    """Graph-based state encoder for relational information."""

    def __init__(self, embedding_dim: int = 128):
        """Initialize graph state encoder.
        
        Args:
            embedding_dim: Dimension of node embeddings
        """
        self.embedding_dim = embedding_dim
        self.entity_embeddings = {}
        self.relation_embeddings = {}

    def encode_knowledge_graph_state(
        self, entities: List[Dict[str, Any]], relations: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Encode knowledge graph state.
        
        Args:
            entities: List of entities
            relations: List of relations
            
        Returns:
            Graph state encoding
        """
        # Simple graph encoding - can be enhanced with GNNs
        entity_features = []

        for entity in entities[:10]:  # Limit to top 10 entities
            entity_type = entity.get("type", "unknown")
            entity_importance = entity.get("importance", 0.5)

            # Create simple entity encoding
            encoding = [entity_importance]

            # Add type encoding (one-hot for common types)
            common_types = ["person", "organization", "location", "concept", "tool"]
            for t in common_types:
                encoding.append(1.0 if entity_type == t else 0.0)

            entity_features.extend(encoding)

        # Pad or truncate to fixed size
        target_size = self.embedding_dim
        if len(entity_features) < target_size:
            entity_features.extend([0.0] * (target_size - len(entity_features)))
        else:
            entity_features = entity_features[:target_size]

        return np.array(entity_features, dtype=np.float32)
