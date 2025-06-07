"""
Request models for the API.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class ChatRequest(BaseModel):
    """Request model for chat interactions."""

    message: str = Field(..., description="The message to send to the agent")
    session_id: Optional[str] = Field(None, description="Session ID for continuing a conversation")
    agent_mode: Optional[str] = Field(None, description="Agent mode to use")
    user_id: Optional[str] = Field(None, description="User ID for personalized responses")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for the agent")
    stream: Optional[bool] = Field(False, description="Whether to stream the response")

class AgentRequest(BaseModel):
    """Request model for agent operations."""

    agent_mode: str = Field(..., description="Agent mode to use")
    config: Optional[Dict[str, Any]] = Field(None, description="Agent configuration")

class MemoryRequest(BaseModel):
    """Request model for memory operations."""

    session_id: str = Field(..., description="Session ID for the memory")
    query: Optional[str] = Field(None, description="Query for retrieving memory")
    memory_item: Optional[Dict[str, Any]] = Field(None, description="Memory item to store")
    memory_backend: Optional[str] = Field(None, description="Memory backend to use")

class ToolRequest(BaseModel):
    """Request model for tool operations."""

    tool_name: str = Field(..., description="Name of the tool to use")
    tool_input: Dict[str, Any] = Field(..., description="Input for the tool")
    session_id: Optional[str] = Field(None, description="Session ID for the tool operation")
    agent_mode: Optional[str] = Field(None, description="Agent mode to use for the tool operation")

class FeedbackRequest(BaseModel):
    """Request model for feedback."""

    session_id: str = Field(..., description="Session ID for the feedback")
    message_id: str = Field(..., description="Message ID for the feedback")
    rating: int = Field(..., description="Rating (1-5)")
    feedback_text: Optional[str] = Field(None, description="Feedback text")
    feedback_type: str = Field("user", description="Type of feedback (user, self)")

class WebhookRequest(BaseModel):
    """Request model for webhooks."""

    event_type: str = Field(..., description="Type of event")
    payload: Dict[str, Any] = Field(..., description="Event payload")
    signature: Optional[str] = Field(None, description="Webhook signature for verification")
