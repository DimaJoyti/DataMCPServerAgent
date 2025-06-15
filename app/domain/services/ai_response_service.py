"""
AI Response Service - Handles AI response generation with personality and context.
Integrates with various LLM providers and manages response quality.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from app.core.logging import LoggerMixin, get_logger
from app.domain.models.base import DomainService
from app.domain.models.brand_agent import BrandAgent, BrandPersonality
from app.domain.models.conversation import ConversationMessage, LiveConversation

logger = get_logger(__name__)


class AIProvider:
    """Base class for AI providers."""

    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config

    async def generate_response(
        self, system_prompt: str, user_prompt: str, context: Dict[str, Any]
    ) -> str:
        """Generate AI response."""
        raise NotImplementedError


class OpenAIProvider(AIProvider):
    """OpenAI GPT provider."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("openai", config)
        self.api_key = config.get("api_key")
        self.model = config.get("model", "gpt-3.5-turbo")
        self.max_tokens = config.get("max_tokens", 500)
        self.temperature = config.get("temperature", 0.7)

    async def generate_response(
        self, system_prompt: str, user_prompt: str, context: Dict[str, Any]
    ) -> str:
        """Generate response using OpenAI API."""
        # Mock implementation - replace with actual OpenAI API call
        await asyncio.sleep(0.1)  # Simulate API call

        # Simple response based on context
        agent_name = context.get("agent", {}).get("name", "Assistant")
        user_message = context.get("user_message", {}).get("content", "")

        if "hello" in user_message.lower():
            return f"Hello! I'm {agent_name}. How can I help you today?"
        elif "help" in user_message.lower():
            return "I'm here to help! What specific assistance do you need?"
        elif "thank" in user_message.lower():
            return "You're welcome! Is there anything else I can help you with?"
        else:
            return f"Thank you for your message. As {agent_name}, I'm here to assist you. How can I help?"


class ClaudeProvider(AIProvider):
    """Anthropic Claude provider."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("claude", config)
        self.api_key = config.get("api_key")
        self.model = config.get("model", "claude-3-sonnet-20240229")
        self.max_tokens = config.get("max_tokens", 500)

    async def generate_response(
        self, system_prompt: str, user_prompt: str, context: Dict[str, Any]
    ) -> str:
        """Generate response using Claude API."""
        # Mock implementation - replace with actual Claude API call
        await asyncio.sleep(0.1)  # Simulate API call

        agent_name = context.get("agent", {}).get("name", "Assistant")
        return f"Hello from {agent_name}! I'm powered by Claude and ready to help you."


class LocalLLMProvider(AIProvider):
    """Local LLM provider (e.g., Ollama, local models)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__("local", config)
        self.endpoint = config.get("endpoint", "http://localhost:11434")
        self.model = config.get("model", "llama2")

    async def generate_response(
        self, system_prompt: str, user_prompt: str, context: Dict[str, Any]
    ) -> str:
        """Generate response using local LLM."""
        # Mock implementation - replace with actual local LLM call
        await asyncio.sleep(0.2)  # Simulate local processing

        agent_name = context.get("agent", {}).get("name", "Assistant")
        return f"Greetings! I'm {agent_name}, running on a local AI model. How may I assist you?"


class AIResponseService(DomainService, LoggerMixin):
    """Service for generating AI responses with personality and context."""

    def __init__(self):
        super().__init__()
        self.providers: Dict[str, AIProvider] = {}
        self.default_provider = "openai"
        self._setup_providers()

    def _setup_providers(self):
        """Setup AI providers."""
        # OpenAI provider
        openai_config = {
            "api_key": "your-openai-api-key",  # Should come from environment
            "model": "gpt-3.5-turbo",
            "max_tokens": 500,
            "temperature": 0.7,
        }
        self.providers["openai"] = OpenAIProvider(openai_config)

        # Claude provider
        claude_config = {
            "api_key": "your-claude-api-key",  # Should come from environment
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 500,
        }
        self.providers["claude"] = ClaudeProvider(claude_config)

        # Local LLM provider
        local_config = {
            "endpoint": "http://localhost:11434",
            "model": "llama2",
        }
        self.providers["local"] = LocalLLMProvider(local_config)

    async def generate_response(
        self,
        user_message: ConversationMessage,
        conversation: LiveConversation,
        agent: BrandAgent,
        context: Optional[Dict[str, Any]] = None,
        provider_name: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Generate AI response with personality and context."""
        start_time = datetime.now()

        # Select provider
        provider = self.providers.get(provider_name or self.default_provider)
        if not provider:
            raise ValueError(f"AI provider not found: {provider_name}")

        # Build prompts
        system_prompt = self._build_system_prompt(agent)
        user_prompt = self._build_user_prompt(user_message, conversation, context or {})

        # Build full context
        full_context = await self._build_full_context(user_message, conversation, agent, context)

        try:
            # Generate response
            response = await provider.generate_response(system_prompt, user_prompt, full_context)

            # Post-process response
            processed_response = self._post_process_response(response, agent)

            # Calculate metrics
            generation_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            metadata = {
                "provider": provider.name,
                "model": getattr(provider, "model", "unknown"),
                "generation_time_ms": generation_time_ms,
                "system_prompt_length": len(system_prompt),
                "user_prompt_length": len(user_prompt),
                "response_length": len(processed_response),
            }

            self.logger.info(
                f"AI response generated in {generation_time_ms}ms using {provider.name}"
            )
            return processed_response, metadata

        except Exception as e:
            self.logger.error(f"Failed to generate AI response: {e}")
            # Fallback response
            fallback_response = self._get_fallback_response(agent)
            metadata = {
                "provider": "fallback",
                "error": str(e),
                "generation_time_ms": int((datetime.now() - start_time).total_seconds() * 1000),
            }
            return fallback_response, metadata

    def _build_system_prompt(self, agent: BrandAgent) -> str:
        """Build system prompt based on agent personality."""
        personality = agent.personality

        # Base prompt
        prompt = f"""You are {agent.name}, a {agent.agent_type.replace('_', ' ')} AI assistant.

PERSONALITY PROFILE:
- Traits: {', '.join(personality.traits)}
- Communication Tone: {personality.tone}
- Communication Style: {personality.communication_style}
- Response Length: {personality.response_length}
- Formality Level: {personality.formality_level}
- Use Emojis: {'Yes' if personality.emoji_usage else 'No'}

BEHAVIOR GUIDELINES:
1. Always maintain your personality traits in responses
2. Match the specified tone and communication style
3. Keep responses at the {personality.response_length} length level
4. Use {personality.formality_level} language
5. {'Include appropriate emojis' if personality.emoji_usage else 'Do not use emojis'}

CUSTOM PHRASES:
"""

        if personality.custom_phrases:
            for phrase in personality.custom_phrases:
                prompt += f"- {phrase}\n"

        prompt += f"""
AGENT CONFIGURATION:
- Maximum response length: {agent.configuration.max_response_length} characters
- Supported channels: {', '.join(agent.configuration.supported_channels)}
- Business hours: {json.dumps(agent.configuration.business_hours)}

Remember to stay in character and provide helpful, accurate responses while maintaining your personality.
"""

        return prompt

    def _build_user_prompt(
        self,
        user_message: ConversationMessage,
        conversation: LiveConversation,
        context: Dict[str, Any],
    ) -> str:
        """Build user prompt with conversation context."""
        prompt = f"CURRENT USER MESSAGE:\n{user_message.content}\n\n"

        # Add message analysis if available
        if user_message.analysis:
            analysis = user_message.analysis
            prompt += "MESSAGE ANALYSIS:\n"
            prompt += f"- Sentiment: {analysis.sentiment}\n"
            prompt += f"- Intent: {analysis.intent}\n"
            prompt += f"- Confidence: {analysis.confidence:.2f}\n"
            if analysis.keywords:
                prompt += f"- Keywords: {', '.join(analysis.keywords)}\n"
            prompt += "\n"

        # Add conversation context
        prompt += "CONVERSATION CONTEXT:\n"
        prompt += f"- Channel: {conversation.channel}\n"
        prompt += f"- Duration: {conversation.duration_seconds} seconds\n"
        prompt += f"- Message count: {conversation.metrics.message_count}\n"

        if conversation.context:
            prompt += f"- Additional context: {json.dumps(conversation.context)}\n"

        prompt += "\n"

        # Add conversation history if available
        if "conversation_history" in context:
            history = context["conversation_history"]
            if history:
                prompt += "RECENT CONVERSATION HISTORY:\n"
                for msg in history[-5:]:  # Last 5 messages
                    prompt += f"{msg['sender']}: {msg['content']}\n"
                prompt += "\n"

        # Add knowledge base information
        if "knowledge_base" in context:
            knowledge = context["knowledge_base"]
            if knowledge:
                prompt += "RELEVANT KNOWLEDGE:\n"
                for item in knowledge[:3]:  # Top 3 relevant items
                    prompt += f"- {item['title']}: {item['content'][:200]}...\n"
                prompt += "\n"

        prompt += "Please respond appropriately based on your personality, the conversation context, and available information."

        return prompt

    async def _build_full_context(
        self,
        user_message: ConversationMessage,
        conversation: LiveConversation,
        agent: BrandAgent,
        additional_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build comprehensive context for AI generation."""
        # Get conversation history
        conversation_history = await self._get_conversation_history(conversation.id)

        # Get relevant knowledge
        knowledge_items = await self._get_relevant_knowledge(
            user_message.content, agent.knowledge_items
        )

        context = {
            "agent": {
                "id": agent.id,
                "name": agent.name,
                "type": agent.agent_type,
                "personality": agent.personality.dict(),
                "configuration": agent.configuration.dict(),
            },
            "conversation": {
                "id": conversation.id,
                "channel": conversation.channel,
                "status": conversation.status,
                "duration_seconds": conversation.duration_seconds,
                "message_count": conversation.metrics.message_count,
                "context": conversation.context,
            },
            "user_message": {
                "content": user_message.content,
                "type": user_message.message_type,
                "timestamp": user_message.timestamp.isoformat(),
                "analysis": user_message.analysis.dict() if user_message.analysis else None,
            },
            "conversation_history": conversation_history,
            "knowledge_base": knowledge_items,
            **additional_context,
        }

        return context

    def _post_process_response(self, response: str, agent: BrandAgent) -> str:
        """Post-process AI response based on agent configuration."""
        # Trim to max length
        max_length = agent.configuration.max_response_length
        if len(response) > max_length:
            response = response[:max_length].rsplit(" ", 1)[0] + "..."

        # Add custom phrases if configured
        personality = agent.personality
        if personality.custom_phrases and not any(
            phrase in response for phrase in personality.custom_phrases
        ):
            # Randomly add a custom phrase
            import random

            if random.random() < 0.3:  # 30% chance
                phrase = random.choice(personality.custom_phrases)
                response += f" {phrase}"

        # Ensure emoji usage matches configuration
        if not personality.emoji_usage:
            # Remove emojis if not allowed
            import re

            response = re.sub(r"[^\w\s.,!?-]", "", response)

        return response.strip()

    def _get_fallback_response(self, agent: BrandAgent) -> str:
        """Get fallback response when AI generation fails."""
        fallback_responses = [
            f"I'm {agent.name}, and I'm here to help you. Could you please rephrase your question?",
            "I apologize, but I'm having trouble processing your request right now. Please try again.",
            "Thank you for your patience. I'm experiencing some technical difficulties. How else can I assist you?",
            "I want to make sure I give you the best answer. Could you provide a bit more detail about what you need?",
        ]

        import random

        response = random.choice(fallback_responses)

        # Add emoji if configured
        if agent.personality.emoji_usage:
            response += " 😊"

        return response

    async def _get_conversation_history(
        self, conversation_id: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get conversation history for context."""
        # This would integrate with your message repository
        # For now, return empty list
        return []

    async def _get_relevant_knowledge(
        self, query: str, knowledge_ids: List[str]
    ) -> List[Dict[str, Any]]:
        """Get relevant knowledge items using RAG."""
        # This would integrate with your RAG system
        # For now, return empty list
        return []

    async def analyze_response_quality(
        self, response: str, user_message: ConversationMessage, agent: BrandAgent
    ) -> Dict[str, Any]:
        """Analyze the quality of generated response."""
        analysis = {
            "length": len(response),
            "word_count": len(response.split()),
            "personality_match": self._check_personality_match(response, agent.personality),
            "appropriateness": self._check_appropriateness(response),
            "helpfulness": self._check_helpfulness(response, user_message),
        }

        # Overall quality score
        scores = [
            analysis["personality_match"],
            analysis["appropriateness"],
            analysis["helpfulness"],
        ]
        analysis["overall_quality"] = sum(scores) / len(scores)

        return analysis

    def _check_personality_match(self, response: str, personality: BrandPersonality) -> float:
        """Check how well response matches personality."""
        score = 0.5  # Base score

        # Check tone
        if personality.tone == "friendly" and any(
            word in response.lower() for word in ["happy", "glad", "pleased"]
        ):
            score += 0.2
        elif personality.tone == "professional" and not any(
            word in response.lower() for word in ["hey", "yo", "sup"]
        ):
            score += 0.2

        # Check emoji usage
        has_emoji = any(char for char in response if ord(char) > 127)
        if personality.emoji_usage == has_emoji:
            score += 0.2

        # Check formality
        if personality.formality_level == "formal" and not any(
            word in response.lower() for word in ["gonna", "wanna", "yeah"]
        ):
            score += 0.1

        return min(1.0, score)

    def _check_appropriateness(self, response: str) -> float:
        """Check if response is appropriate."""
        # Simple checks for inappropriate content
        inappropriate_words = ["hate", "stupid", "idiot", "damn"]
        if any(word in response.lower() for word in inappropriate_words):
            return 0.3

        return 0.9  # Default high score

    def _check_helpfulness(self, response: str, user_message: ConversationMessage) -> float:
        """Check if response is helpful."""
        # Simple heuristics for helpfulness
        if len(response) < 10:
            return 0.3  # Too short

        if "I don't know" in response and "let me" not in response.lower():
            return 0.4  # Not helpful without offering alternatives

        if any(word in response.lower() for word in ["help", "assist", "support", "can", "will"]):
            return 0.8  # Offers help

        return 0.6  # Default moderate score
