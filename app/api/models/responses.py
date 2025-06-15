"""
API response models.
"""

from typing import Generic, List, Optional, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


class SuccessResponse(BaseModel):
    """Generic success response."""

    success: bool = Field(default=True, description="Whether operation was successful")
    message: str = Field(description="Success message")
    data: Optional[dict] = Field(default=None, description="Optional response data")


class ErrorResponse(BaseModel):
    """Generic error response."""

    success: bool = Field(default=False, description="Whether operation was successful")
    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    details: Optional[dict] = Field(default=None, description="Error details")


class PaginatedResponse(BaseModel, Generic[T]):
    """Paginated response wrapper."""

    items: List[T] = Field(description="List of items")
    total: int = Field(description="Total number of items")
    page: int = Field(description="Current page number")
    size: int = Field(description="Page size")
    pages: int = Field(description="Total number of pages")

    @property
    def has_next(self) -> bool:
        """Check if there are more pages."""
        return self.page < self.pages

    @property
    def has_prev(self) -> bool:
        """Check if there are previous pages."""
        return self.page > 1
