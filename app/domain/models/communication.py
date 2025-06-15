"""
Communication domain models.
Defines models for email, WebRTC, and other communication channels.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import Field, field_validator

from .base import BaseEntity, BaseValueObject, ValidationError


class EmailStatus(str, Enum):
    """Email delivery status."""

    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    BOUNCED = "bounced"
    OPENED = "opened"
    CLICKED = "clicked"


class CallStatus(str, Enum):
    """Call status enumeration."""

    PENDING = "pending"
    RINGING = "ringing"
    ACTIVE = "active"
    ON_HOLD = "on_hold"
    ENDED = "ended"
    FAILED = "failed"


class MediaType(str, Enum):
    """Media type enumeration."""

    AUDIO = "audio"
    VIDEO = "video"
    SCREEN_SHARE = "screen_share"


class ApprovalStatus(str, Enum):
    """Approval workflow status."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"


class EmailTemplate(BaseValueObject):
    """Email template value object."""

    template_id: str = Field(description="Template unique identifier")
    name: str = Field(description="Template name")
    subject: str = Field(description="Email subject template")
    html_content: str = Field(description="HTML content template")
    text_content: str = Field(description="Text content template")
    variables: List[str] = Field(default_factory=list, description="Template variables")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v):
        if not v or not v.strip():
            raise ValidationError("Template name cannot be empty")
        return v.strip()


class EmailMessage(BaseEntity):
    """Email message entity."""

    to_email: str = Field(description="Recipient email address")
    from_email: str = Field(description="Sender email address")
    subject: str = Field(description="Email subject")
    html_content: str = Field(description="HTML content")
    text_content: str = Field(description="Text content")
    status: EmailStatus = Field(default=EmailStatus.PENDING, description="Email status")
    sent_at: Optional[datetime] = Field(default=None, description="When email was sent")
    delivered_at: Optional[datetime] = Field(default=None, description="When email was delivered")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("to_email", "from_email")
    @classmethod
    def validate_email(cls, v):
        if not v or "@" not in v:
            raise ValidationError("Invalid email address")
        return v.lower().strip()


class MediaStream(BaseValueObject):
    """Media stream configuration."""

    stream_id: str = Field(description="Stream unique identifier")
    media_type: MediaType = Field(description="Type of media")
    enabled: bool = Field(default=True, description="Whether stream is enabled")
    codec: str = Field(description="Media codec")
    bitrate: int = Field(description="Stream bitrate")
    resolution: Optional[str] = Field(default=None, description="Video resolution")

    @field_validator("bitrate")
    @classmethod
    def validate_bitrate(cls, v):
        if v <= 0:
            raise ValidationError("Bitrate must be positive")
        return v


class CallParticipant(BaseValueObject):
    """Call participant information."""

    participant_id: str = Field(description="Participant unique identifier")
    user_id: str = Field(description="User identifier")
    display_name: str = Field(description="Display name")
    is_agent: bool = Field(default=False, description="Whether participant is an AI agent")
    joined_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="When participant joined"
    )
    left_at: Optional[datetime] = Field(default=None, description="When participant left")
    media_streams: List[MediaStream] = Field(
        default_factory=list, description="Participant media streams"
    )
    is_muted: bool = Field(default=False, description="Whether participant is muted")
    is_video_enabled: bool = Field(default=False, description="Whether video is enabled")


class CallSession(BaseEntity):
    """Call session entity."""

    agent_id: str = Field(description="Associated agent ID")
    status: CallStatus = Field(default=CallStatus.PENDING, description="Call status")
    participants: List[CallParticipant] = Field(
        default_factory=list, description="Call participants"
    )
    started_at: Optional[datetime] = Field(default=None, description="When call started")
    ended_at: Optional[datetime] = Field(default=None, description="When call ended")
    duration_seconds: Optional[int] = Field(default=None, description="Call duration")
    recording_enabled: bool = Field(default=False, description="Whether call is recorded")
    recording_url: Optional[str] = Field(default=None, description="Recording URL")
    transcript: Optional[str] = Field(default=None, description="Call transcript")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @property
    def is_active(self) -> bool:
        """Check if call is active."""
        return self.status == CallStatus.ACTIVE

    @property
    def participant_count(self) -> int:
        """Get number of participants."""
        return len(self.participants)


class ApprovalRequest(BaseEntity):
    """Approval request entity for human-in-the-loop workflows."""

    agent_id: str = Field(description="Requesting agent ID")
    task_id: str = Field(description="Associated task ID")
    title: str = Field(description="Approval request title")
    description: str = Field(description="Approval request description")
    data: Dict[str, Any] = Field(default_factory=dict, description="Request data")
    approver_email: str = Field(description="Approver email address")
    status: ApprovalStatus = Field(default=ApprovalStatus.PENDING, description="Approval status")
    expires_at: datetime = Field(description="When request expires")
    approved_at: Optional[datetime] = Field(default=None, description="When request was approved")
    approved_by: Optional[str] = Field(default=None, description="Who approved the request")
    rejection_reason: Optional[str] = Field(default=None, description="Reason for rejection")

    @field_validator("approver_email")
    @classmethod
    def validate_approver_email(cls, v):
        if not v or "@" not in v:
            raise ValidationError("Invalid approver email address")
        return v.lower().strip()

    @property
    def is_expired(self) -> bool:
        """Check if approval request is expired."""
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def is_pending(self) -> bool:
        """Check if approval request is pending."""
        return self.status == ApprovalStatus.PENDING and not self.is_expired
