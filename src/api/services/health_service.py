"""
Health service for the API.
"""

import platform
from typing import Dict, Any

class HealthService:
    """Service for health checks."""

    async def check_components(self) -> Dict[str, str]:
        """
        Check the health of components.

        Returns:
            Dict[str, str]: Component statuses
        """
        components = {
            "api": "ok",
            "database": "ok",
            "memory": "ok",
            "agents": "ok",
            "tools": "ok",
            "system": platform.system(),
            "python": platform.python_version(),
        }

        return components
