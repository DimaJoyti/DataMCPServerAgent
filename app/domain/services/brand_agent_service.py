"""
Brand Agent domain services.
Contains business logic for brand agent management, deployment, and optimization.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.core.logging import LoggerMixin, get_logger
from app.domain.models.base import BusinessRuleError, DomainService, ValidationError
from app.domain.models.brand_agent import (
    BrandAgent,
    BrandAgentConfiguration,
    BrandAgentCreated,
    BrandAgentDeployed,
    BrandAgentMetrics,
    BrandAgentType,
    BrandKnowledge,
    BrandPersonality,
    ConversationChannel,
    ConversationMessage,
    ConversationSession,
    ConversationStarted,
    ConversationEnded,
    KnowledgeType,
)

logger = get_logger(__name__)


class BrandAgentService(DomainService, LoggerMixin):
    """Core brand agent management service."""

    async def create_brand_agent(
        self,
        name: str,
        brand_id: str,
        agent_type: BrandAgentType,
        owner_id: str,
        personality: Optional[BrandPersonality] = None,
        configuration: Optional[BrandAgentConfiguration] = None,
        description: Optional[str] = None,
    ) -> BrandAgent:
        """Create a new brand agent."""
        self.logger.info(f"Creating brand agent: {name} for brand {brand_id}")

        # Validate brand agent name uniqueness within brand
        agent_repo = self.get_repository("brand_agent")
        existing_agents = await agent_repo.list(brand_id=brand_id, name=name)
        if existing_agents:
            raise ValidationError(f"Brand agent with name '{name}' already exists for this brand")

        # Create brand agent
        agent = BrandAgent(
            name=name,
            brand_id=brand_id,
            agent_type=agent_type,
            owner_id=owner_id,
            personality=personality or BrandPersonality(),
            configuration=configuration or BrandAgentConfiguration(),
            description=description,
        )

        # Save agent
        saved_agent = await agent_repo.save(agent)

        # Raise domain event
        event = BrandAgentCreated(
            agent_id=saved_agent.id,
            brand_id=brand_id,
            agent_type=agent_type,
            owner_id=owner_id,
        )
        await self.publish_event(event)

        self.logger.info(f"Brand agent created successfully: {saved_agent.id}")
        return saved_agent

    async def deploy_agent_to_channel(
        self, agent_id: str, channel: ConversationChannel
    ) -> BrandAgent:
        """Deploy brand agent to a specific channel."""
        agent_repo = self.get_repository("brand_agent")
        agent = await agent_repo.get_by_id(agent_id)

        if not agent:
            raise ValidationError(f"Brand agent not found: {agent_id}")

        if not agent.is_active:
            raise BusinessRuleError("Cannot deploy inactive brand agent")

        # Deploy to channel
        agent.deploy_to_channel(channel)
        saved_agent = await agent_repo.save(agent)

        # Raise domain event
        event = BrandAgentDeployed(
            agent_id=agent_id,
            channel=channel,
            deployed_at=datetime.now(timezone.utc),
        )
        await self.publish_event(event)

        self.logger.info(f"Brand agent {agent_id} deployed to channel {channel}")
        return saved_agent

    async def update_agent_personality(
        self, agent_id: str, personality: BrandPersonality
    ) -> BrandAgent:
        """Update brand agent personality."""
        agent_repo = self.get_repository("brand_agent")
        agent = await agent_repo.get_by_id(agent_id)

        if not agent:
            raise ValidationError(f"Brand agent not found: {agent_id}")

        agent.update_personality(personality)
        saved_agent = await agent_repo.save(agent)

        self.logger.info(f"Updated personality for brand agent {agent_id}")
        return saved_agent

    async def add_knowledge_to_agent(
        self, agent_id: str, knowledge_id: str
    ) -> BrandAgent:
        """Add knowledge item to brand agent."""
        agent_repo = self.get_repository("brand_agent")
        knowledge_repo = self.get_repository("brand_knowledge")

        agent = await agent_repo.get_by_id(agent_id)
        if not agent:
            raise ValidationError(f"Brand agent not found: {agent_id}")

        knowledge = await knowledge_repo.get_by_id(knowledge_id)
        if not knowledge:
            raise ValidationError(f"Knowledge item not found: {knowledge_id}")

        agent.add_knowledge_item(knowledge_id)
        saved_agent = await agent_repo.save(agent)

        self.logger.info(f"Added knowledge {knowledge_id} to agent {agent_id}")
        return saved_agent

    async def get_agent_performance_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get performance summary for a brand agent."""
        agent_repo = self.get_repository("brand_agent")
        agent = await agent_repo.get_by_id(agent_id)

        if not agent:
            raise ValidationError(f"Brand agent not found: {agent_id}")

        return {
            "agent_id": agent_id,
            "agent_name": agent.name,
            "success_rate": agent.success_rate,
            "total_conversations": agent.metrics.total_conversations,
            "average_satisfaction": agent.metrics.user_satisfaction_avg,
            "escalation_rate": agent.metrics.escalation_rate,
            "average_response_time": agent.metrics.average_response_time_ms,
            "is_performing_well": agent.is_performing_well,
            "deployment_status": {
                "is_deployed": agent.is_deployed,
                "channels": agent.deployment_channels,
            },
        }

    async def get_brand_agents_summary(self, brand_id: str) -> Dict[str, Any]:
        """Get summary of all brand agents for a brand."""
        agent_repo = self.get_repository("brand_agent")
        agents = await agent_repo.list(brand_id=brand_id)

        total_agents = len(agents)
        active_agents = len([a for a in agents if a.is_active])
        deployed_agents = len([a for a in agents if a.is_deployed])

        total_conversations = sum(a.metrics.total_conversations for a in agents)
        avg_satisfaction = (
            sum(a.metrics.user_satisfaction_avg for a in agents) / total_agents
            if total_agents > 0 else 0
        )

        return {
            "brand_id": brand_id,
            "total_agents": total_agents,
            "active_agents": active_agents,
            "deployed_agents": deployed_agents,
            "total_conversations": total_conversations,
            "average_satisfaction": avg_satisfaction,
            "agents": [
                {
                    "id": agent.id,
                    "name": agent.name,
                    "type": agent.agent_type,
                    "is_active": agent.is_active,
                    "is_deployed": agent.is_deployed,
                    "success_rate": agent.success_rate,
                }
                for agent in agents
            ],
        }


class KnowledgeService(DomainService, LoggerMixin):
    """Service for managing brand knowledge."""

    async def create_knowledge_item(
        self,
        title: str,
        content: str,
        knowledge_type: KnowledgeType,
        brand_id: str,
        tags: Optional[List[str]] = None,
        priority: int = 1,
        source_url: Optional[str] = None,
    ) -> BrandKnowledge:
        """Create a new knowledge item."""
        self.logger.info(f"Creating knowledge item: {title}")

        knowledge = BrandKnowledge(
            title=title,
            content=content,
            knowledge_type=knowledge_type,
            tags=tags or [],
            priority=priority,
            source_url=source_url,
        )

        # Add brand_id to metadata
        knowledge.metadata["brand_id"] = brand_id

        knowledge_repo = self.get_repository("brand_knowledge")
        saved_knowledge = await knowledge_repo.save(knowledge)

        self.logger.info(f"Knowledge item created: {saved_knowledge.id}")
        return saved_knowledge

    async def update_knowledge_content(
        self, knowledge_id: str, new_content: str
    ) -> BrandKnowledge:
        """Update knowledge item content."""
        knowledge_repo = self.get_repository("brand_knowledge")
        knowledge = await knowledge_repo.get_by_id(knowledge_id)

        if not knowledge:
            raise ValidationError(f"Knowledge item not found: {knowledge_id}")

        knowledge.update_content(new_content)
        saved_knowledge = await knowledge_repo.save(knowledge)

        self.logger.info(f"Updated knowledge item: {knowledge_id}")
        return saved_knowledge

    async def search_knowledge(
        self, brand_id: str, query: str, knowledge_type: Optional[KnowledgeType] = None
    ) -> List[BrandKnowledge]:
        """Search knowledge items by query."""
        knowledge_repo = self.get_repository("brand_knowledge")
        
        # Build search criteria
        criteria = {"metadata.brand_id": brand_id}
        if knowledge_type:
            criteria["knowledge_type"] = knowledge_type

        # Get all knowledge items for the brand
        all_knowledge = await knowledge_repo.list(**criteria)

        # Simple text search (in a real implementation, use proper search engine)
        query_lower = query.lower()
        matching_knowledge = [
            k for k in all_knowledge
            if (query_lower in k.title.lower() or 
                query_lower in k.content.lower() or
                any(query_lower in tag.lower() for tag in k.tags))
        ]

        # Sort by priority
        matching_knowledge.sort(key=lambda x: x.priority, reverse=True)

        return matching_knowledge


class ConversationService(DomainService, LoggerMixin):
    """Service for managing conversations."""

    async def start_conversation(
        self,
        agent_id: str,
        channel: ConversationChannel,
        user_id: Optional[str] = None,
        session_token: Optional[str] = None,
    ) -> ConversationSession:
        """Start a new conversation session."""
        # Validate agent exists and is deployed to channel
        agent_repo = self.get_repository("brand_agent")
        agent = await agent_repo.get_by_id(agent_id)

        if not agent:
            raise ValidationError(f"Brand agent not found: {agent_id}")

        if not agent.is_active:
            raise BusinessRuleError("Cannot start conversation with inactive agent")

        if channel not in agent.deployment_channels:
            raise BusinessRuleError(f"Agent not deployed to channel: {channel}")

        # Create session
        session = ConversationSession(
            brand_agent_id=agent_id,
            user_id=user_id,
            session_token=session_token or str(uuid4()),
            channel=channel,
        )

        session_repo = self.get_repository("conversation_session")
        saved_session = await session_repo.save(session)

        # Raise domain event
        event = ConversationStarted(
            session_id=saved_session.id,
            agent_id=agent_id,
            channel=channel,
            user_id=user_id,
        )
        await self.publish_event(event)

        self.logger.info(f"Started conversation session: {saved_session.id}")
        return saved_session

    async def add_message_to_conversation(
        self,
        session_id: str,
        sender_type: str,
        content: str,
        message_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationMessage:
        """Add a message to a conversation."""
        session_repo = self.get_repository("conversation_session")
        session = await session_repo.get_by_id(session_id)

        if not session:
            raise ValidationError(f"Conversation session not found: {session_id}")

        if session.status != "active":
            raise BusinessRuleError("Cannot add message to inactive session")

        # Create message
        message = ConversationMessage(
            session_id=session_id,
            sender_type=sender_type,
            content=content,
            message_type=message_type,
            metadata=metadata or {},
        )

        message_repo = self.get_repository("conversation_message")
        saved_message = await message_repo.save(message)

        # Update session
        session.add_message()
        await session_repo.save(session)

        self.logger.info(f"Added message to session {session_id}")
        return saved_message

    async def end_conversation(
        self, session_id: str, satisfaction_rating: Optional[int] = None
    ) -> ConversationSession:
        """End a conversation session."""
        session_repo = self.get_repository("conversation_session")
        session = await session_repo.get_by_id(session_id)

        if not session:
            raise ValidationError(f"Conversation session not found: {session_id}")

        # Calculate duration
        duration_seconds = int(
            (datetime.now(timezone.utc) - session.started_at).total_seconds()
        )

        # End session
        session.end_session(satisfaction_rating)
        saved_session = await session_repo.save(session)

        # Raise domain event
        event = ConversationEnded(
            session_id=session_id,
            agent_id=session.brand_agent_id,
            duration_seconds=duration_seconds,
            message_count=session.message_count,
            user_satisfaction=satisfaction_rating,
        )
        await self.publish_event(event)

        self.logger.info(f"Ended conversation session: {session_id}")
        return saved_session
