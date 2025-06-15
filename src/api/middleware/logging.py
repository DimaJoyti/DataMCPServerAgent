"""
Logging middleware for the API.
"""

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for logging requests and responses."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable,
    ) -> Response:
        """
        Log requests and responses.

        Args:
            request (Request): Request object
            call_next (Callable): Next middleware or endpoint

        Returns:
            Response: Response object
        """
        # Generate a request ID
        request_id = str(uuid.uuid4())

        # Add the request ID to the request state
        request.state.request_id = request_id

        # Log the request
        start_time = time.time()

        # Process the request
        response = await call_next(request)

        # Calculate the processing time
        process_time = time.time() - start_time

        # Add the request ID to the response headers
        response.headers["X-Request-ID"] = request_id

        # Log the response
        print(f"Request {request_id} processed in {process_time:.4f} seconds")

        return response
