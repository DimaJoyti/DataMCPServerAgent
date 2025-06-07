"""
Rate limiting middleware for the API.
"""

import time
from typing import Dict, Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

from ..config import config

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting."""

    def __init__(self, app):
        """
        Initialize the middleware.

        Args:
            app: FastAPI application
        """
        super().__init__(app)
        self.rate_limit_per_minute = config.rate_limit_per_minute
        self.requests = {}  # Dict[str, Dict[str, int]]

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """
        Apply rate limiting.

        Args:
            request (Request): Request object
            call_next (Callable): Next middleware or endpoint

        Returns:
            Response: Response object
        """
        # If rate limiting is not enabled, skip
        if not config.enable_rate_limiting:
            return await call_next(request)

        # Get the client IP
        client_ip = request.client.host

        # Get the current time
        current_time = int(time.time())

        # Clean up old requests
        self._clean_up_old_requests(current_time)

        # Check if the client has exceeded the rate limit
        if self._is_rate_limited(client_ip, current_time):
            # Return a 429 Too Many Requests response
            return Response(
                content="Rate limit exceeded",
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                headers={"Retry-After": "60"},
            )

        # Process the request
        return await call_next(request)

    def _clean_up_old_requests(self, current_time: int) -> None:
        """
        Clean up old requests.

        Args:
            current_time (int): Current time
        """
        # Remove requests older than 1 minute
        for client_ip in list(self.requests.keys()):
            self.requests[client_ip] = {
                timestamp: count
                for timestamp, count in self.requests[client_ip].items()
                if current_time - int(timestamp) < 60
            }

            # Remove the client if there are no requests
            if not self.requests[client_ip]:
                del self.requests[client_ip]

    def _is_rate_limited(self, client_ip: str, current_time: int) -> bool:
        """
        Check if the client has exceeded the rate limit.

        Args:
            client_ip (str): Client IP
            current_time (int): Current time

        Returns:
            bool: Whether the client has exceeded the rate limit
        """
        # Initialize the client if not present
        if client_ip not in self.requests:
            self.requests[client_ip] = {}

        # Initialize the timestamp if not present
        timestamp = str(current_time)
        if timestamp not in self.requests[client_ip]:
            self.requests[client_ip][timestamp] = 0

        # Increment the request count
        self.requests[client_ip][timestamp] += 1

        # Calculate the total requests in the last minute
        total_requests = sum(
            count
            for timestamp, count in self.requests[client_ip].items()
            if current_time - int(timestamp) < 60
        )

        # Check if the total requests exceed the rate limit
        return total_requests > self.rate_limit_per_minute
