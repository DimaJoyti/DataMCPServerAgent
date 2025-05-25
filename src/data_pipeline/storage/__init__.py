"""
Storage Module for the Data Pipeline.

This module provides storage capabilities including:
- Unified data access layer
- Data lake storage
- Time-series storage
- Metadata storage
"""

from .unified_access.data_access_layer import DataAccessLayer

__all__ = [
    "DataAccessLayer",
]
