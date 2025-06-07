"""
Authentication middleware for the API.
"""

from typing import Optional

from fastapi import Depends, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN

from ..config import config

# API key header
api_key_header = APIKeyHeader(name=config.api_key_header, auto_error=False)

async def get_api_key(
    api_key: Optional[str] = Security(api_key_header),
) -> Optional[str]:
    """
    Get and validate the API key.

    Args:
        api_key (Optional[str]): API key from the request header

    Returns:
        Optional[str]: Validated API key

    Raises:
        HTTPException: If authentication is enabled and the API key is invalid
    """
    # If authentication is not enabled, return None
    if not config.enable_auth:
        return None

    # If authentication is enabled, validate the API key
    if not api_key:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="API key is required",
        )

    if api_key not in config.api_keys:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return api_key
