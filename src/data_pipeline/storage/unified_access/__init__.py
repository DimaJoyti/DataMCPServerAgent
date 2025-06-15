"""
Unified Data Access Layer.

This module provides a unified interface for accessing
various storage systems and data sources.
"""

from .data_access_layer import DataAccessConfig, DataAccessLayer

__all__ = [
    "DataAccessLayer",
    "DataAccessConfig",
]
