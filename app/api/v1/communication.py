"""
Communication API endpoints.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.dependencies import get_current_user, get_email_service
from app.api.models.responses import SuccessResponse
from app.domain.services.communication_service import EmailService

router = APIRouter()

class SendEmailRequest(BaseModel):
    """Request model for sending email."""

    to_email: str = Field(description="Recipient email")
    subject: str = Field(description="Email subject")
    html_content: str = Field(description="HTML content")
    text_content: str = Field(description="Text content")

@router.post("/email/send", response_model=SuccessResponse)
async def send_email(
    request: SendEmailRequest,
    email_service: EmailService = Depends(get_email_service),
    current_user=Depends(get_current_user),
):
    """Send an email."""
    await email_service.send_email(
        to_email=request.to_email,
        subject=request.subject,
        html_content=request.html_content,
        text_content=request.text_content,
    )

    return SuccessResponse(message="Email sent successfully")
