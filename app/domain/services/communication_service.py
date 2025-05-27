"""
Communication domain services.
Contains business logic for email, WebRTC, and other communication channels.
"""

from datetime import datetime, timezone

from app.core.logging import LoggerMixin, get_logger
from app.domain.models.base import DomainService
from app.domain.models.communication import CallSession, CallStatus, EmailMessage, EmailStatus

logger = get_logger(__name__)


class EmailService(DomainService, LoggerMixin):
    """Email communication service."""

    async def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: str,
        from_email: str = "noreply@datamcp.com",
    ) -> EmailMessage:
        """Send an email message."""
        self.logger.info(f"Sending email to {to_email}")

        # Create email message
        email = EmailMessage(
            to_email=to_email,
            from_email=from_email,
            subject=subject,
            html_content=html_content,
            text_content=text_content,
            status=EmailStatus.PENDING,
        )

        # Save email
        email_repo = self.get_repository("email")
        saved_email = await email_repo.save(email)

        # Simulate sending
        saved_email.status = EmailStatus.SENT
        saved_email.sent_at = datetime.now(timezone.utc)
        await email_repo.save(saved_email)

        self.logger.info(f"Email sent successfully: {saved_email.id}")
        return saved_email


class WebRTCService(DomainService, LoggerMixin):
    """WebRTC communication service."""

    async def create_call_session(
        self, agent_id: str, recording_enabled: bool = True
    ) -> CallSession:
        """Create a new call session."""
        self.logger.info(f"Creating call session for agent {agent_id}")

        # Create call session
        call = CallSession(
            agent_id=agent_id, status=CallStatus.PENDING, recording_enabled=recording_enabled
        )

        # Save call session
        call_repo = self.get_repository("call")
        saved_call = await call_repo.save(call)

        self.logger.info(f"Call session created: {saved_call.id}")
        return saved_call
