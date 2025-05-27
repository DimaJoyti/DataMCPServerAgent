"""
DataMCPServerAgent - Consolidated AI Agent System
================================================

Consolidated AI agent system with single app/ structure and Clean Architecture.

Architecture Overview:
- Single app/ directory structure
- Clean Architecture with DDD principles
- Domain, Application, Infrastructure, API, CLI layers
- Simplified imports and dependencies
- Maintainable and scalable codebase
"""

__version__ = "2.0.0"
__author__ = "DataMCPServerAgent Team"
__description__ = "Consolidated AI Agent System with Clean Architecture"

# Minimal imports to avoid circular dependencies
try:
    from app.core.simple_config import SimpleSettings

    settings: SimpleSettings = SimpleSettings()
except ImportError:
    settings = None  # type: ignore

__all__ = [
    "settings",
]
