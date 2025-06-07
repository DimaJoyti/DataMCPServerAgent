"""
Response models for the API.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime

class ChatResponse(BaseModel):
    """Response model for chat interactions."""

    message_id: str = Field(..., description="Unique ID for the message")
    response: str = Field(..., description="Response from the agent")
    session_id: str = Field(..., description="Session ID for the conversation")
    created_at: datetime = Field(..., description="Timestamp for the response")
    agent_mode: str = Field(..., description="Agent mode used for the response")
    tool_usage: Optional[List[Dict[str, Any]]] = Field(None, description="Tools used in generating the response")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Sources used in generating the response")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the response")

class ChatStreamResponse(BaseModel):
    """Response model for streaming chat interactions."""

    message_id: str = Field(..., description="Unique ID for the message")
    chunk: str = Field(..., description="Chunk of the response")
    session_id: str = Field(..., description="Session ID for the conversation")
    created_at: datetime = Field(..., description="Timestamp for the chunk")
    is_final: bool = Field(False, description="Whether this is the final chunk")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the chunk")

class AgentResponse(BaseModel):
    """Response model for agent operations."""

    agent_id: str = Field(..., description="Unique ID for the agent")
    agent_mode: str = Field(..., description="Agent mode")
    status: str = Field(..., description="Status of the agent")
    capabilities: List[str] = Field(..., description="Capabilities of the agent")
    created_at: datetime = Field(..., description="Timestamp for agent creation")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the agent")

class MemoryResponse(BaseModel):
    """Response model for memory operations."""

    session_id: str = Field(..., description="Session ID for the memory")
    memory_items: List[Dict[str, Any]] = Field(..., description="Memory items")
    memory_backend: str = Field(..., description="Memory backend used")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the memory")

class ToolResponse(BaseModel):
    """Response model for tool operations."""

    tool_name: str = Field(..., description="Name of the tool used")
    tool_output: Any = Field(..., description="Output from the tool")
    execution_time: float = Field(..., description="Time taken to execute the tool (in seconds)")
    status: str = Field(..., description="Status of the tool operation")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the tool operation")

class FeedbackResponse(BaseModel):
    """Response model for feedback."""

    feedback_id: str = Field(..., description="Unique ID for the feedback")
    session_id: str = Field(..., description="Session ID for the feedback")
    message_id: str = Field(..., description="Message ID for the feedback")
    status: str = Field(..., description="Status of the feedback")
    created_at: datetime = Field(..., description="Timestamp for the feedback")

class ErrorResponse(BaseModel):
    """Response model for errors."""

    error_code: str = Field(..., description="Error code")
    error_message: str = Field(..., description="Error message")
    error_details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")

class HealthResponse(BaseModel):
    """Response model for health checks."""

    status: str = Field(..., description="Status of the API")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Timestamp for the health check")
    components: Dict[str, str] = Field(..., description="Status of individual components")
    uptime: float = Field(..., description="API uptime in seconds")
