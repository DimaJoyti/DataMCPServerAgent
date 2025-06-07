"""
API request models.
"""

from fastapi import Query
from pydantic import BaseModel, Field

class PaginationParams(BaseModel):
    """Pagination parameters."""

    page: int = Field(default=1, ge=1, description="Page number")
    size: int = Field(default=20, ge=1, le=100, description="Page size")

    @property
    def offset(self) -> int:
        """Calculate offset from page and size."""
        return (self.page - 1) * self.size

    @property
    def limit(self) -> int:
        """Get limit (same as size)."""
        return self.size

def get_pagination_params(
    page: int = Query(1, ge=1, description="Page number"),
    size: int = Query(20, ge=1, le=100, description="Page size"),
) -> PaginationParams:
    """Get pagination parameters from query."""
    return PaginationParams(page=page, size=size)
