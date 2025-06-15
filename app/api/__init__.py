"""
API layer for DataMCPServerAgent.
Contains REST API endpoints, request/response models, and API-specific logic.
"""

# Avoid importing complex dependencies at module level
# Import only when needed to prevent circular dependencies and missing dependencies

__all__ = ["api_router"]


def get_api_router():
    """Get API router - import only when needed."""
    from .v1 import api_router
    return api_router
