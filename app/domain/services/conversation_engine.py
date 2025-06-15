"""
Conversation Engine - Core service for real-time conversation processing.
Handles message processing, AI response generation, and conversation management.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.core.logging import LoggerMixin, get_logger
from app.domain.models.base import DomainService
from app.domain.models.brand_agent import BrandAgent, ConversationChannel
from app.domain.models.conversation import (
    ConversationMessage,
    ConversationStatus,
    IntentType,
    LiveConversation,
    MessageAnalysis,
    MessageContext,
    MessageSent,
    MessageStatus,
    MessageType,
    SentimentType,
)

logger = get_logger(__name__)


class ConversationEngine(DomainService, LoggerMixin):
    """Core conversation processing engine."""

    def __init__(self):
        super().__init__()
        self._active_conversations: Dict[str, LiveConversation] = {}
        self._message_processors: List[callable] = []
        self._response_generators: List[callable] = []

    async def start_conversation(
        self,
        brand_agent_id: str,
        channel: ConversationChannel,
        user_id: Optional[str] = None,
        session_token: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None,
    ) -> LiveConversation:
        """Start a new conversation."""
        self.logger.info(f"Starting conversation with agent {brand_agent_id}")

        # Get brand agent
        agent_repo = self.get_repository("brand_agent")
        agent = await agent_repo.get_by_id(brand_agent_id)
        if not agent:
            raise ValueError(f"Brand agent not found: {brand_agent_id}")

        if not agent.is_active:
            raise ValueError("Cannot start conversation with inactive agent")

        if channel not in agent.deployment_channels:
            raise ValueError(f"Agent not deployed to channel: {channel}")

        # Create conversation
        conversation = LiveConversation(
            brand_agent_id=brand_agent_id,
            user_id=user_id,
            session_token=session_token or str(uuid4()),
            channel=channel,
            context=initial_context or {},
        )

        # Save conversation
        conversation_repo = self.get_repository("live_conversation")
        saved_conversation = await conversation_repo.save(conversation)

        # Add to active conversations
        self._active_conversations[saved_conversation.id] = saved_conversation

        # Send welcome message if configured
        await self._send_welcome_message(saved_conversation, agent)

        self.logger.info(f"Conversation started: {saved_conversation.id}")
        return saved_conversation

    async def process_user_message(
        self,
        conversation_id: str,
        content: str,
        message_type: MessageType = MessageType.TEXT,
        context: Optional[MessageContext] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ConversationMessage:
        """Process a user message and generate AI response."""
        self.logger.info(f"Processing user message in conversation {conversation_id}")

        # Get conversation
        conversation = await self._get_active_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Active conversation not found: {conversation_id}")

        # Create user message
        user_message = ConversationMessage(
            conversation_id=conversation_id,
            sender_type="user",
            sender_id=conversation.user_id,
            content=content,
            message_type=message_type,
            context=context,
            metadata=metadata or {},
        )

        # Save user message
        message_repo = self.get_repository("conversation_message")
        saved_user_message = await message_repo.save(user_message)

        # Add to conversation
        conversation.add_message(saved_user_message.id)

        # Process message through pipeline
        await self._process_message_pipeline(saved_user_message, conversation)

        # Generate AI response
        ai_response = await self._generate_ai_response(saved_user_message, conversation)

        # Update conversation
        conversation_repo = self.get_repository("live_conversation")
        await conversation_repo.save(conversation)

        # Publish event
        event = MessageSent(
            conversation_id=conversation_id,
            message_id=saved_user_message.id,
            sender_type="user",
            message_type=message_type,
            content_preview=content[:100],
        )
        await self.publish_event(event)

        self.logger.info(f"User message processed: {saved_user_message.id}")
        return saved_user_message

    async def _generate_ai_response(
        self, user_message: ConversationMessage, conversation: LiveConversation
    ) -> ConversationMessage:
        """Generate AI response to user message."""
        start_time = datetime.now()

        # Get brand agent
        agent_repo = self.get_repository("brand_agent")
        agent = await agent_repo.get_by_id(conversation.brand_agent_id)

        # Build context for AI
        context = await self._build_ai_context(user_message, conversation, agent)

        # Generate response using AI service
        response_content = await self._call_ai_service(context, agent)

        # Create AI message
        ai_message = ConversationMessage(
            conversation_id=conversation.id,
            sender_type="agent",
            sender_id=conversation.brand_agent_id,
            content=response_content,
            message_type=MessageType.TEXT,
            status=MessageStatus.SENT,
            response_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
        )

        # Save AI message
        message_repo = self.get_repository("conversation_message")
        saved_ai_message = await message_repo.save(ai_message)

        # Add to conversation
        conversation.add_message(saved_ai_message.id)

        # Publish event
        event = MessageSent(
            conversation_id=conversation.id,
            message_id=saved_ai_message.id,
            sender_type="agent",
            message_type=MessageType.TEXT,
            content_preview=response_content[:100],
        )
        await self.publish_event(event)

        return saved_ai_message

    async def _build_ai_context(
        self, user_message: ConversationMessage, conversation: LiveConversation, agent: BrandAgent
    ) -> Dict[str, Any]:
        """Build context for AI response generation."""
        # Get recent conversation history
        recent_messages = await self._get_recent_messages(conversation.id, limit=10)

        # Get relevant knowledge
        knowledge_items = await self._get_relevant_knowledge(
            user_message.content, agent.knowledge_items
        )

        # Build context
        context = {
            "agent": {
                "name": agent.name,
                "type": agent.agent_type,
                "personality": agent.personality.dict(),
                "configuration": agent.configuration.dict(),
            },
            "conversation": {
                "id": conversation.id,
                "channel": conversation.channel,
                "context": conversation.context,
                "duration_seconds": conversation.duration_seconds,
            },
            "user_message": {
                "content": user_message.content,
                "type": user_message.message_type,
                "analysis": user_message.analysis.dict() if user_message.analysis else None,
            },
            "conversation_history": [
                {
                    "sender": msg.sender_type,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                }
                for msg in recent_messages
            ],
            "knowledge_base": [
                {
                    "title": item.get("title", ""),
                    "content": item.get("content", ""),
                    "type": item.get("type", ""),
                }
                for item in knowledge_items
            ],
        }

        return context

    async def _call_ai_service(self, context: Dict[str, Any], agent: BrandAgent) -> str:
        """Call AI service to generate response."""
        # This is a placeholder for AI service integration
        # In a real implementation, this would call OpenAI, Claude, or other LLM

        personality = agent.personality
        agent_name = agent.name

        # Build prompt based on personality
        system_prompt = self._build_system_prompt(personality, agent_name)
        user_prompt = self._build_user_prompt(context)

        # Mock AI response for now
        mock_responses = [
            f"Hello! I'm {agent_name}, and I'm here to help you. How can I assist you today?",
            "Thank you for your question. Let me help you with that.",
            "I understand your concern. Here's what I can do to help:",
            "That's a great question! Based on our information, here's what I recommend:",
            "I'm happy to assist you with this. Let me provide you with the details:",
        ]

        # Simple response selection based on content
        if "hello" in context["user_message"]["content"].lower():
            response = f"Hello! I'm {agent_name}. How can I help you today?"
        elif "help" in context["user_message"]["content"].lower():
            response = "I'm here to help! What specific assistance do you need?"
        elif "thank" in context["user_message"]["content"].lower():
            response = "You're welcome! Is there anything else I can help you with?"
        else:
            response = f"Thank you for your message. As {agent_name}, I'm here to assist you. Could you please provide more details about what you need help with?"

        # Add personality touches
        if personality.emoji_usage:
            response += " 😊"

        if personality.custom_phrases:
            response += f" {personality.custom_phrases[0]}"

        return response

    def _build_system_prompt(self, personality, agent_name: str) -> str:
        """Build system prompt for AI."""
        traits_str = ", ".join(personality.traits)

        return f"""You are {agent_name}, an AI assistant with the following characteristics:
        
Personality Traits: {traits_str}
Communication Tone: {personality.tone}
Communication Style: {personality.communication_style}
Response Length: {personality.response_length}
Formality Level: {personality.formality_level}
Use Emojis: {personality.emoji_usage}

Always respond in character, maintaining these personality traits throughout the conversation.
Be helpful, accurate, and maintain the specified tone and style.
"""

    def _build_user_prompt(self, context: Dict[str, Any]) -> str:
        """Build user prompt with context."""
        user_message = context["user_message"]["content"]
        history = context["conversation_history"]
        knowledge = context["knowledge_base"]

        prompt = f"User message: {user_message}\n\n"

        if history:
            prompt += "Recent conversation:\n"
            for msg in history[-5:]:  # Last 5 messages
                prompt += f"{msg['sender']}: {msg['content']}\n"
            prompt += "\n"

        if knowledge:
            prompt += "Relevant knowledge:\n"
            for item in knowledge[:3]:  # Top 3 relevant items
                prompt += f"- {item['title']}: {item['content'][:200]}...\n"
            prompt += "\n"

        prompt += (
            "Please respond appropriately based on your personality and the available information."
        )

        return prompt

    async def _process_message_pipeline(
        self, message: ConversationMessage, conversation: LiveConversation
    ) -> None:
        """Process message through analysis pipeline."""
        # Analyze message sentiment and intent
        analysis = await self._analyze_message(message)
        message.add_analysis(analysis)

        # Check for escalation triggers
        await self._check_escalation_triggers(message, conversation, analysis)

        # Update conversation metrics
        await self._update_conversation_metrics(conversation, message, analysis)

    async def _analyze_message(self, message: ConversationMessage) -> MessageAnalysis:
        """Analyze message for sentiment, intent, etc."""
        content = message.content.lower()

        # Simple sentiment analysis (in real implementation, use ML models)
        sentiment = SentimentType.NEUTRAL
        confidence = 0.7

        if any(word in content for word in ["angry", "frustrated", "terrible", "awful", "hate"]):
            sentiment = SentimentType.NEGATIVE
            confidence = 0.8
        elif any(word in content for word in ["happy", "great", "excellent", "love", "amazing"]):
            sentiment = SentimentType.POSITIVE
            confidence = 0.8
        elif any(word in content for word in ["confused", "don't understand", "unclear"]):
            sentiment = SentimentType.CONFUSED
            confidence = 0.9

        # Simple intent classification
        intent = IntentType.GENERAL_CHAT
        if any(word in content for word in ["help", "support", "problem", "issue"]):
            intent = IntentType.SUPPORT
        elif any(word in content for word in ["buy", "purchase", "price", "cost"]):
            intent = IntentType.SALES_INQUIRY
        elif any(word in content for word in ["complaint", "complain", "wrong", "error"]):
            intent = IntentType.COMPLAINT
        elif "?" in content:
            intent = IntentType.QUESTION

        # Extract simple keywords
        keywords = [word for word in content.split() if len(word) > 3][:5]

        return MessageAnalysis(
            sentiment=sentiment,
            intent=intent,
            confidence=confidence,
            keywords=keywords,
            language="en",  # Default to English
            toxicity_score=0.0,  # Placeholder
        )

    async def _check_escalation_triggers(
        self,
        message: ConversationMessage,
        conversation: LiveConversation,
        analysis: MessageAnalysis,
    ) -> None:
        """Check if message triggers escalation."""
        escalation_triggers = []

        # Check sentiment-based escalation
        if analysis.sentiment == SentimentType.NEGATIVE and analysis.confidence > 0.8:
            escalation_triggers.append("negative_sentiment")

        # Check for specific keywords
        agent_repo = self.get_repository("brand_agent")
        agent = await agent_repo.get_by_id(conversation.brand_agent_id)

        if agent and agent.configuration.escalation_triggers:
            content_lower = message.content.lower()
            for trigger in agent.configuration.escalation_triggers:
                if trigger.lower() in content_lower:
                    escalation_triggers.append(f"keyword_{trigger}")

        # Add to conversation metrics
        conversation.metrics.escalation_triggers.extend(escalation_triggers)

    async def _update_conversation_metrics(
        self,
        conversation: LiveConversation,
        message: ConversationMessage,
        analysis: MessageAnalysis,
    ) -> None:
        """Update conversation metrics."""
        if analysis.sentiment:
            # Convert sentiment to score
            sentiment_score = {
                SentimentType.POSITIVE: 1.0,
                SentimentType.SATISFIED: 0.8,
                SentimentType.NEUTRAL: 0.5,
                SentimentType.CONFUSED: 0.3,
                SentimentType.FRUSTRATED: 0.2,
                SentimentType.NEGATIVE: 0.0,
            }.get(analysis.sentiment, 0.5)

            conversation.metrics.sentiment_scores.append(sentiment_score)

    async def _get_active_conversation(self, conversation_id: str) -> Optional[LiveConversation]:
        """Get active conversation by ID."""
        if conversation_id in self._active_conversations:
            return self._active_conversations[conversation_id]

        # Try to load from repository
        conversation_repo = self.get_repository("live_conversation")
        conversation = await conversation_repo.get_by_id(conversation_id)

        if conversation and conversation.is_active():
            self._active_conversations[conversation_id] = conversation
            return conversation

        return None

    async def _get_recent_messages(
        self, conversation_id: str, limit: int = 10
    ) -> List[ConversationMessage]:
        """Get recent messages from conversation."""
        message_repo = self.get_repository("conversation_message")
        # This would be implemented based on your repository pattern
        # For now, return empty list
        return []

    async def _get_relevant_knowledge(
        self, query: str, knowledge_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Get relevant knowledge items for the query."""
        # This would integrate with your RAG system
        # For now, return empty list
        return []

    async def _send_welcome_message(
        self, conversation: LiveConversation, agent: BrandAgent
    ) -> None:
        """Send welcome message when conversation starts."""
        welcome_text = agent.configuration.auto_responses.get(
            "greeting", f"Hello! I'm {agent.name}. How can I help you today?"
        )

        welcome_message = ConversationMessage(
            conversation_id=conversation.id,
            sender_type="agent",
            sender_id=agent.id,
            content=welcome_text,
            message_type=MessageType.SYSTEM,
            status=MessageStatus.SENT,
        )

        message_repo = self.get_repository("conversation_message")
        saved_message = await message_repo.save(welcome_message)
        conversation.add_message(saved_message.id)

    async def end_conversation(
        self,
        conversation_id: str,
        reason: str = "user_ended",
        user_satisfaction: Optional[int] = None,
    ) -> LiveConversation:
        """End a conversation."""
        conversation = await self._get_active_conversation(conversation_id)
        if not conversation:
            raise ValueError(f"Active conversation not found: {conversation_id}")

        # Update status
        conversation.update_status(ConversationStatus.CLOSED, reason)

        # Add satisfaction rating
        if user_satisfaction:
            conversation.metrics.user_satisfaction = user_satisfaction

        # Remove from active conversations
        if conversation_id in self._active_conversations:
            del self._active_conversations[conversation_id]

        # Save conversation
        conversation_repo = self.get_repository("live_conversation")
        await conversation_repo.save(conversation)

        self.logger.info(f"Conversation ended: {conversation_id}")
        return conversation

    async def get_conversation_status(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """Get conversation status and metrics."""
        conversation = await self._get_active_conversation(conversation_id)
        if not conversation:
            return None

        return {
            "id": conversation.id,
            "status": conversation.status,
            "started_at": conversation.started_at.isoformat(),
            "duration_seconds": conversation.duration_seconds,
            "message_count": conversation.metrics.message_count,
            "last_activity": conversation.last_activity_at.isoformat(),
            "participants": conversation.participants,
        }
