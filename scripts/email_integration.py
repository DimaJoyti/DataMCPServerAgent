"""
Email API Integration for Human-in-the-Loop workflows.
Supports Cloudflare Email Workers, SendGrid, and Mailgun.
"""

import logging
import smtplib
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailProvider(Enum):
    """Supported email providers."""
    CLOUDFLARE = "cloudflare"
    SENDGRID = "sendgrid"
    MAILGUN = "mailgun"
    SMTP = "smtp"

class EmailStatus(Enum):
    """Email delivery status."""
    PENDING = "pending"
    SENT = "sent"
    DELIVERED = "delivered"
    FAILED = "failed"
    BOUNCED = "bounced"
    OPENED = "opened"
    CLICKED = "clicked"

class ApprovalStatus(Enum):
    """Approval workflow status."""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"

@dataclass
class EmailTemplate:
    """Email template structure."""
    template_id: str
    name: str
    subject: str
    html_content: str
    text_content: str
    variables: List[str]
    created_at: datetime
    updated_at: datetime

@dataclass
class EmailMessage:
    """Email message structure."""
    message_id: str
    to_email: str
    from_email: str
    subject: str
    html_content: str
    text_content: str
    status: EmailStatus
    provider: EmailProvider
    created_at: datetime
    sent_at: Optional[datetime]
    delivered_at: Optional[datetime]
    metadata: Dict[str, Any]

@dataclass
class ApprovalRequest:
    """Approval request structure for human-in-the-loop workflows."""
    request_id: str
    agent_id: str
    task_id: str
    title: str
    description: str
    data: Dict[str, Any]
    approver_email: str
    status: ApprovalStatus
    created_at: datetime
    expires_at: datetime
    approved_at: Optional[datetime]
    approved_by: Optional[str]
    rejection_reason: Optional[str]

class EmailIntegration:
    """Email integration for human-in-the-loop workflows."""

    def __init__(self, default_provider: EmailProvider = EmailProvider.SMTP):
        self.default_provider = default_provider
        self.email_templates: Dict[str, EmailTemplate] = {}
        self.email_messages: Dict[str, EmailMessage] = {}
        self.approval_requests: Dict[str, ApprovalRequest] = {}

        # Email provider configurations
        self.provider_configs = {
            EmailProvider.SMTP: {
                "host": "smtp.gmail.com",
                "port": 587,
                "username": "",
                "password": "",
                "use_tls": True
            },
            EmailProvider.SENDGRID: {
                "api_key": "",
                "from_email": "noreply@yourdomain.com"
            },
            EmailProvider.MAILGUN: {
                "api_key": "",
                "domain": "yourdomain.com",
                "from_email": "noreply@yourdomain.com"
            },
            EmailProvider.CLOUDFLARE: {
                "worker_url": "https://your-email-worker.your-subdomain.workers.dev",
                "api_key": "",
                "from_email": "noreply@yourdomain.com"
            }
        }

        self._initialize_default_templates()
        logger.info(f"Email Integration initialized with provider: {default_provider.value}")

    def _initialize_default_templates(self):
        """Initialize default email templates."""
        # Approval request template
        approval_template = EmailTemplate(
            template_id="approval_request",
            name="Approval Request",
            subject="Action Required: {task_title}",
            html_content="""
            <html>
            <body>
                <h2>Approval Required</h2>
                <p>Hello,</p>
                <p>An AI agent requires your approval for the following action:</p>

                <div style="background-color: #f5f5f5; padding: 15px; margin: 20px 0; border-radius: 5px;">
                    <h3>{task_title}</h3>
                    <p><strong>Description:</strong> {task_description}</p>
                    <p><strong>Agent:</strong> {agent_id}</p>
                    <p><strong>Task ID:</strong> {task_id}</p>
                </div>

                <div style="margin: 30px 0;">
                    <a href="{approval_url}" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; margin-right: 10px;">Approve</a>
                    <a href="{rejection_url}" style="background-color: #f44336; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px;">Reject</a>
                </div>

                <p><small>This request expires on {expires_at}</small></p>

                <hr>
                <p><small>This is an automated message from your AI Agent system.</small></p>
            </body>
            </html>
            """,
            text_content="""
            Approval Required

            Hello,

            An AI agent requires your approval for the following action:

            Task: {task_title}
            Description: {task_description}
            Agent: {agent_id}
            Task ID: {task_id}

            To approve: {approval_url}
            To reject: {rejection_url}

            This request expires on {expires_at}

            This is an automated message from your AI Agent system.
            """,
            variables=["task_title", "task_description", "agent_id", "task_id", "approval_url", "rejection_url", "expires_at"],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )

        self.email_templates[approval_template.template_id] = approval_template

        # Task completion notification template
        completion_template = EmailTemplate(
            template_id="task_completion",
            name="Task Completion Notification",
            subject="Task Completed: {task_title}",
            html_content="""
            <html>
            <body>
                <h2>Task Completed</h2>
                <p>Hello,</p>
                <p>The following task has been completed:</p>

                <div style="background-color: #e8f5e8; padding: 15px; margin: 20px 0; border-radius: 5px;">
                    <h3>{task_title}</h3>
                    <p><strong>Status:</strong> {status}</p>
                    <p><strong>Agent:</strong> {agent_id}</p>
                    <p><strong>Completed at:</strong> {completed_at}</p>
                </div>

                <p><strong>Results:</strong></p>
                <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 5px;">{results}</pre>

                <hr>
                <p><small>This is an automated message from your AI Agent system.</small></p>
            </body>
            </html>
            """,
            text_content="""
            Task Completed

            Hello,

            The following task has been completed:

            Task: {task_title}
            Status: {status}
            Agent: {agent_id}
            Completed at: {completed_at}

            Results:
            {results}

            This is an automated message from your AI Agent system.
            """,
            variables=["task_title", "status", "agent_id", "completed_at", "results"],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )

        self.email_templates[completion_template.template_id] = completion_template

    # ==================== EMAIL TEMPLATE MANAGEMENT ====================

    async def create_email_template(self, name: str, subject: str, html_content: str,
                                  text_content: str, variables: List[str]) -> str:
        """Create a new email template."""
        try:
            template_id = f"template_{uuid.uuid4().hex[:12]}"
            current_time = datetime.now(timezone.utc)

            template = EmailTemplate(
                template_id=template_id,
                name=name,
                subject=subject,
                html_content=html_content,
                text_content=text_content,
                variables=variables,
                created_at=current_time,
                updated_at=current_time
            )

            self.email_templates[template_id] = template
            logger.info(f"Created email template: {template_id}")
            return template_id

        except Exception as e:
            logger.error(f"Error creating email template: {e}")
            raise

    async def get_email_template(self, template_id: str) -> Optional[EmailTemplate]:
        """Get email template by ID."""
        return self.email_templates.get(template_id)

    async def list_email_templates(self) -> List[EmailTemplate]:
        """List all email templates."""
        return list(self.email_templates.values())

    # ==================== EMAIL SENDING ====================

    async def send_email(self, to_email: str, template_id: str, variables: Dict[str, str],
                        provider: EmailProvider = None) -> str:
        """Send email using specified template and variables."""
        try:
            provider = provider or self.default_provider
            template = self.email_templates.get(template_id)

            if not template:
                raise ValueError(f"Template {template_id} not found")

            # Replace variables in template
            subject = template.subject.format(**variables)
            html_content = template.html_content.format(**variables)
            text_content = template.text_content.format(**variables)

            message_id = f"msg_{uuid.uuid4().hex[:12]}"
            current_time = datetime.now(timezone.utc)

            message = EmailMessage(
                message_id=message_id,
                to_email=to_email,
                from_email=self.provider_configs[provider].get("from_email", "noreply@example.com"),
                subject=subject,
                html_content=html_content,
                text_content=text_content,
                status=EmailStatus.PENDING,
                provider=provider,
                created_at=current_time,
                sent_at=None,
                delivered_at=None,
                metadata={"template_id": template_id, "variables": variables}
            )

            # Send email based on provider
            success = await self._send_via_provider(message, provider)

            if success:
                message.status = EmailStatus.SENT
                message.sent_at = datetime.now(timezone.utc)
                logger.info(f"Email sent successfully: {message_id}")
            else:
                message.status = EmailStatus.FAILED
                logger.error(f"Failed to send email: {message_id}")

            self.email_messages[message_id] = message
            return message_id

        except Exception as e:
            logger.error(f"Error sending email: {e}")
            raise

    async def _send_via_provider(self, message: EmailMessage, provider: EmailProvider) -> bool:
        """Send email via specific provider."""
        try:
            if provider == EmailProvider.SMTP:
                return await self._send_via_smtp(message)
            elif provider == EmailProvider.SENDGRID:
                return await self._send_via_sendgrid(message)
            elif provider == EmailProvider.MAILGUN:
                return await self._send_via_mailgun(message)
            elif provider == EmailProvider.CLOUDFLARE:
                return await self._send_via_cloudflare(message)
            else:
                logger.error(f"Unsupported email provider: {provider}")
                return False

        except Exception as e:
            logger.error(f"Error sending via {provider.value}: {e}")
            return False

    async def _send_via_smtp(self, message: EmailMessage) -> bool:
        """Send email via SMTP."""
        try:
            config = self.provider_configs[EmailProvider.SMTP]

            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = message.subject
            msg['From'] = message.from_email
            msg['To'] = message.to_email

            # Add text and HTML parts
            text_part = MIMEText(message.text_content, 'plain')
            html_part = MIMEText(message.html_content, 'html')

            msg.attach(text_part)
            msg.attach(html_part)

            # Send email
            with smtplib.SMTP(config["host"], config["port"]) as server:
                if config.get("use_tls"):
                    server.starttls()
                if config.get("username") and config.get("password"):
                    server.login(config["username"], config["password"])

                server.send_message(msg)

            return True

        except Exception as e:
            logger.error(f"SMTP send error: {e}")
            return False

    async def _send_via_sendgrid(self, message: EmailMessage) -> bool:
        """Send email via SendGrid API."""
        try:
            # In production, use SendGrid Python SDK
            # import sendgrid
            # from sendgrid.helpers.mail import Mail

            logger.info(f"Simulating SendGrid send for message: {message.message_id}")
            return True

        except Exception as e:
            logger.error(f"SendGrid send error: {e}")
            return False

    async def _send_via_mailgun(self, message: EmailMessage) -> bool:
        """Send email via Mailgun API."""
        try:
            # In production, use Mailgun API
            logger.info(f"Simulating Mailgun send for message: {message.message_id}")
            return True

        except Exception as e:
            logger.error(f"Mailgun send error: {e}")
            return False

    async def _send_via_cloudflare(self, message: EmailMessage) -> bool:
        """Send email via Cloudflare Email Workers."""
        try:
            # In production, call Cloudflare Email Worker
            logger.info(f"Simulating Cloudflare Email Worker send for message: {message.message_id}")
            return True

        except Exception as e:
            logger.error(f"Cloudflare Email Worker send error: {e}")
            return False

    # ==================== APPROVAL WORKFLOWS ====================

    async def create_approval_request(self, agent_id: str, task_id: str, title: str,
                                    description: str, data: Dict[str, Any],
                                    approver_email: str, expires_in_hours: int = 24) -> str:
        """Create an approval request for human-in-the-loop workflow."""
        try:
            request_id = f"approval_{uuid.uuid4().hex[:12]}"
            current_time = datetime.now(timezone.utc)
            expires_at = current_time + timedelta(hours=expires_in_hours)

            approval_request = ApprovalRequest(
                request_id=request_id,
                agent_id=agent_id,
                task_id=task_id,
                title=title,
                description=description,
                data=data,
                approver_email=approver_email,
                status=ApprovalStatus.PENDING,
                created_at=current_time,
                expires_at=expires_at,
                approved_at=None,
                approved_by=None,
                rejection_reason=None
            )

            self.approval_requests[request_id] = approval_request

            # Send approval email
            base_url = "https://your-agent-ui.com"  # Configure this
            approval_url = f"{base_url}/approve/{request_id}?action=approve"
            rejection_url = f"{base_url}/approve/{request_id}?action=reject"

            variables = {
                "task_title": title,
                "task_description": description,
                "agent_id": agent_id,
                "task_id": task_id,
                "approval_url": approval_url,
                "rejection_url": rejection_url,
                "expires_at": expires_at.strftime("%Y-%m-%d %H:%M:%S UTC")
            }

            await self.send_email(approver_email, "approval_request", variables)

            logger.info(f"Created approval request: {request_id}")
            return request_id

        except Exception as e:
            logger.error(f"Error creating approval request: {e}")
            raise

    async def process_approval_response(self, request_id: str, action: str,
                                      approver_email: str, reason: str = None) -> bool:
        """Process approval response (approve/reject)."""
        try:
            approval_request = self.approval_requests.get(request_id)
            if not approval_request:
                logger.error(f"Approval request not found: {request_id}")
                return False

            if approval_request.status != ApprovalStatus.PENDING:
                logger.error(f"Approval request already processed: {request_id}")
                return False

            if datetime.now(timezone.utc) > approval_request.expires_at:
                approval_request.status = ApprovalStatus.EXPIRED
                logger.error(f"Approval request expired: {request_id}")
                return False

            current_time = datetime.now(timezone.utc)

            if action.lower() == "approve":
                approval_request.status = ApprovalStatus.APPROVED
                approval_request.approved_at = current_time
                approval_request.approved_by = approver_email
                logger.info(f"Approval request approved: {request_id}")

            elif action.lower() == "reject":
                approval_request.status = ApprovalStatus.REJECTED
                approval_request.approved_at = current_time
                approval_request.approved_by = approver_email
                approval_request.rejection_reason = reason
                logger.info(f"Approval request rejected: {request_id}")

            else:
                logger.error(f"Invalid action: {action}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error processing approval response: {e}")
            return False

    async def get_approval_status(self, request_id: str) -> Optional[ApprovalRequest]:
        """Get approval request status."""
        return self.approval_requests.get(request_id)

    async def get_pending_approvals(self, approver_email: str = None) -> List[ApprovalRequest]:
        """Get pending approval requests."""
        pending = [req for req in self.approval_requests.values()
                  if req.status == ApprovalStatus.PENDING]

        if approver_email:
            pending = [req for req in pending if req.approver_email == approver_email]

        return pending

# Global instance
email_integration = EmailIntegration()
