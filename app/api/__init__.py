"""
API layer for DataMCPServerAgent.
Contains REST API endpoints, request/response models, and API-specific logic.
"""

from .v1 import api_router

__all__ = ["api_router"]
