"""
Agent Communication System

Provides inter-agent communication capabilities including message passing,
event broadcasting, and coordination protocols.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from pydantic import BaseModel, Field


class MessageType(str, Enum):
    """Types of messages that can be sent between agents."""
    
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    KNOWLEDGE_SHARE = "knowledge_share"
    STATUS_QUERY = "status_query"
    STATUS_RESPONSE = "status_response"
    COLLABORATION_INVITE = "collaboration_invite"
    COLLABORATION_RESPONSE = "collaboration_response"
    EVENT_NOTIFICATION = "event_notification"
    HEARTBEAT = "heartbeat"
    ERROR_REPORT = "error_report"


class MessagePriority(str, Enum):
    """Message priority levels."""
    
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class AgentMessage(BaseModel):
    """Message structure for inter-agent communication."""
    
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str
    recipient_id: Optional[str] = None  # None for broadcast messages
    message_type: MessageType
    priority: MessagePriority = MessagePriority.NORMAL
    data: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    requires_response: bool = False
    correlation_id: Optional[str] = None  # For request-response correlation


@dataclass
class MessageHandler:
    """Message handler configuration."""
    
    handler_func: Callable[[AgentMessage], Any]
    message_types: Set[MessageType]
    priority_filter: Optional[MessagePriority] = None
    sender_filter: Optional[Set[str]] = None


class MessageBus:
    """
    Central message bus for agent communication.
    
    Provides:
    - Message routing and delivery
    - Topic-based subscriptions
    - Message persistence and replay
    - Load balancing and failover
    """
    
    def __init__(self):
        """Initialize the message bus."""
        self.subscribers: Dict[str, List[MessageHandler]] = {}
        self.topic_subscribers: Dict[str, Set[str]] = {}
        self.message_history: List[AgentMessage] = []
        self.pending_responses: Dict[str, asyncio.Future] = {}
        self.logger = logging.getLogger("message_bus")
        
    async def publish(
        self,
        message: AgentMessage,
        topic: Optional[str] = None,
    ) -> Optional[Any]:
        """
        Publish a message to the bus.
        
        Args:
            message: The message to publish
            topic: Optional topic for topic-based routing
            
        Returns:
            Response if message requires one, None otherwise
        """
        self.logger.debug(f"Publishing message {message.message_id} from {message.sender_id}")
        
        # Store message in history
        self.message_history.append(message)
        
        # Handle direct messages
        if message.recipient_id:
            await self._deliver_direct_message(message)
        else:
            # Handle broadcast or topic-based messages
            if topic:
                await self._deliver_topic_message(message, topic)
            else:
                await self._deliver_broadcast_message(message)
                
        # Handle response requirement
        if message.requires_response:
            return await self._wait_for_response(message)
            
        return None
        
    async def subscribe(
        self,
        agent_id: str,
        handler: MessageHandler,
    ) -> None:
        """Subscribe an agent to receive messages."""
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
            
        self.subscribers[agent_id].append(handler)
        self.logger.info(f"Agent {agent_id} subscribed to message types: {handler.message_types}")
        
    async def subscribe_topic(
        self,
        agent_id: str,
        topic: str,
    ) -> None:
        """Subscribe an agent to a topic."""
        if topic not in self.topic_subscribers:
            self.topic_subscribers[topic] = set()
            
        self.topic_subscribers[topic].add(agent_id)
        self.logger.info(f"Agent {agent_id} subscribed to topic: {topic}")
        
    async def unsubscribe(
        self,
        agent_id: str,
        message_types: Optional[Set[MessageType]] = None,
    ) -> None:
        """Unsubscribe an agent from messages."""
        if agent_id not in self.subscribers:
            return
            
        if message_types:
            # Remove specific handlers
            self.subscribers[agent_id] = [
                handler for handler in self.subscribers[agent_id]
                if not handler.message_types.intersection(message_types)
            ]
        else:
            # Remove all handlers
            del self.subscribers[agent_id]
            
        self.logger.info(f"Agent {agent_id} unsubscribed")
        
    async def send_response(
        self,
        original_message: AgentMessage,
        response_data: Dict[str, Any],
        sender_id: str,
    ) -> None:
        """Send a response to a message."""
        response_message = AgentMessage(
            sender_id=sender_id,
            recipient_id=original_message.sender_id,
            message_type=MessageType.TASK_RESPONSE,
            data=response_data,
            correlation_id=original_message.message_id,
        )
        
        # Resolve pending response future
        if original_message.message_id in self.pending_responses:
            future = self.pending_responses.pop(original_message.message_id)
            if not future.done():
                future.set_result(response_data)
                
        await self.publish(response_message)
        
    async def _deliver_direct_message(self, message: AgentMessage) -> None:
        """Deliver a message to a specific recipient."""
        recipient_id = message.recipient_id
        
        if recipient_id not in self.subscribers:
            self.logger.warning(f"No subscribers found for agent {recipient_id}")
            return
            
        for handler in self.subscribers[recipient_id]:
            if await self._should_handle_message(handler, message):
                try:
                    await self._invoke_handler(handler, message)
                except Exception as e:
                    self.logger.error(f"Error in message handler: {e}")
                    
    async def _deliver_topic_message(self, message: AgentMessage, topic: str) -> None:
        """Deliver a message to topic subscribers."""
        if topic not in self.topic_subscribers:
            self.logger.warning(f"No subscribers found for topic {topic}")
            return
            
        for agent_id in self.topic_subscribers[topic]:
            if agent_id in self.subscribers:
                for handler in self.subscribers[agent_id]:
                    if await self._should_handle_message(handler, message):
                        try:
                            await self._invoke_handler(handler, message)
                        except Exception as e:
                            self.logger.error(f"Error in message handler: {e}")
                            
    async def _deliver_broadcast_message(self, message: AgentMessage) -> None:
        """Deliver a message to all subscribers."""
        for agent_id, handlers in self.subscribers.items():
            # Don't send to sender
            if agent_id == message.sender_id:
                continue
                
            for handler in handlers:
                if await self._should_handle_message(handler, message):
                    try:
                        await self._invoke_handler(handler, message)
                    except Exception as e:
                        self.logger.error(f"Error in message handler: {e}")
                        
    async def _should_handle_message(
        self,
        handler: MessageHandler,
        message: AgentMessage,
    ) -> bool:
        """Check if a handler should process a message."""
        # Check message type
        if message.message_type not in handler.message_types:
            return False
            
        # Check priority filter
        if handler.priority_filter and message.priority != handler.priority_filter:
            return False
            
        # Check sender filter
        if handler.sender_filter and message.sender_id not in handler.sender_filter:
            return False
            
        return True
        
    async def _invoke_handler(
        self,
        handler: MessageHandler,
        message: AgentMessage,
    ) -> None:
        """Invoke a message handler."""
        if asyncio.iscoroutinefunction(handler.handler_func):
            await handler.handler_func(message)
        else:
            handler.handler_func(message)
            
    async def _wait_for_response(
        self,
        message: AgentMessage,
        timeout: float = 30.0,
    ) -> Any:
        """Wait for a response to a message."""
        future = asyncio.Future()
        self.pending_responses[message.message_id] = future
        
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            self.pending_responses.pop(message.message_id, None)
            raise
            
    def get_message_history(
        self,
        agent_id: Optional[str] = None,
        message_type: Optional[MessageType] = None,
        limit: int = 100,
    ) -> List[AgentMessage]:
        """Get message history with optional filtering."""
        messages = self.message_history
        
        if agent_id:
            messages = [
                msg for msg in messages
                if msg.sender_id == agent_id or msg.recipient_id == agent_id
            ]
            
        if message_type:
            messages = [msg for msg in messages if msg.message_type == message_type]
            
        return messages[-limit:]


class AgentCommunicationHub:
    """
    High-level communication hub for semantic agents.
    
    Provides simplified interfaces for common communication patterns.
    """
    
    def __init__(self, message_bus: MessageBus):
        """Initialize the communication hub."""
        self.message_bus = message_bus
        self.logger = logging.getLogger("communication_hub")
        
    async def request_task_execution(
        self,
        requester_id: str,
        target_agent_id: str,
        task_description: str,
        task_data: Optional[Dict[str, Any]] = None,
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """Request another agent to execute a task."""
        message = AgentMessage(
            sender_id=requester_id,
            recipient_id=target_agent_id,
            message_type=MessageType.TASK_REQUEST,
            priority=MessagePriority.NORMAL,
            data={
                "task_description": task_description,
                "task_data": task_data or {},
            },
            requires_response=True,
        )
        
        try:
            response = await self.message_bus.publish(message)
            return response
        except asyncio.TimeoutError:
            self.logger.error(f"Task request to {target_agent_id} timed out")
            raise
            
    async def share_knowledge(
        self,
        sender_id: str,
        knowledge_data: Dict[str, Any],
        target_agents: Optional[List[str]] = None,
        topic: Optional[str] = None,
    ) -> None:
        """Share knowledge with other agents."""
        message = AgentMessage(
            sender_id=sender_id,
            message_type=MessageType.KNOWLEDGE_SHARE,
            priority=MessagePriority.NORMAL,
            data=knowledge_data,
        )
        
        if target_agents:
            for agent_id in target_agents:
                message.recipient_id = agent_id
                await self.message_bus.publish(message)
        else:
            await self.message_bus.publish(message, topic=topic)
            
    async def query_agent_status(
        self,
        requester_id: str,
        target_agent_id: str,
    ) -> Dict[str, Any]:
        """Query the status of another agent."""
        message = AgentMessage(
            sender_id=requester_id,
            recipient_id=target_agent_id,
            message_type=MessageType.STATUS_QUERY,
            requires_response=True,
        )
        
        response = await self.message_bus.publish(message)
        return response
        
    async def broadcast_event(
        self,
        sender_id: str,
        event_type: str,
        event_data: Dict[str, Any],
        topic: Optional[str] = None,
    ) -> None:
        """Broadcast an event to interested agents."""
        message = AgentMessage(
            sender_id=sender_id,
            message_type=MessageType.EVENT_NOTIFICATION,
            priority=MessagePriority.NORMAL,
            data={
                "event_type": event_type,
                "event_data": event_data,
            },
        )
        
        await self.message_bus.publish(message, topic=topic)
