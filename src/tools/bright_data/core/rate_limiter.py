"""
Advanced rate limiting and throttling for Bright Data MCP Integration

This module provides sophisticated rate limiting with:
- Token bucket algorithm
- Adaptive throttling
- Per-user/API key limits
- Burst handling
- Queue management
- Metrics and monitoring
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class ThrottleStrategy(Enum):
    """Throttling strategies"""

    FIXED = "fixed"
    ADAPTIVE = "adaptive"
    EXPONENTIAL = "exponential"


@dataclass
class RateLimitInfo:
    """Rate limit information"""

    requests_per_minute: int
    burst_size: int
    current_tokens: float
    last_refill: float
    total_requests: int
    rejected_requests: int
    queue_size: int


@dataclass
class RequestInfo:
    """Request information for tracking"""

    timestamp: float
    user_id: Optional[str]
    endpoint: str
    success: bool
    response_time: Optional[float] = None


class TokenBucket:
    """Token bucket implementation for rate limiting"""

    def __init__(self, capacity: int, refill_rate: float):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.tokens = float(capacity)
        self.last_refill = time.time()
        self._lock = asyncio.Lock()

    async def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from the bucket"""
        async with self._lock:
            now = time.time()

            # Refill tokens based on time elapsed
            time_passed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + time_passed * self.refill_rate)
            self.last_refill = now

            # Check if we have enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True

            return False

    async def wait_for_tokens(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """Wait for tokens to become available"""
        start_time = time.time()

        while True:
            if await self.consume(tokens):
                return True

            if timeout and (time.time() - start_time) >= timeout:
                return False

            # Calculate wait time until next token is available
            async with self._lock:
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.refill_rate
                wait_time = min(wait_time, 1.0)  # Max 1 second wait

            await asyncio.sleep(wait_time)

    def get_available_tokens(self) -> float:
        """Get current number of available tokens"""
        now = time.time()
        time_passed = now - self.last_refill
        return min(self.capacity, self.tokens + time_passed * self.refill_rate)


class AdaptiveThrottler:
    """Adaptive throttling based on response times and error rates"""

    def __init__(self, target_response_time: float = 1.0, error_threshold: float = 0.1):
        self.target_response_time = target_response_time
        self.error_threshold = error_threshold
        self.response_times: deque = deque(maxlen=100)
        self.error_count = 0
        self.total_requests = 0
        self.throttle_factor = 1.0
        self.min_throttle = 0.1
        self.max_throttle = 10.0

    def record_request(self, response_time: Optional[float], success: bool) -> None:
        """Record request metrics"""
        self.total_requests += 1

        if response_time is not None:
            self.response_times.append(response_time)

        if not success:
            self.error_count += 1

        self._update_throttle_factor()

    def _update_throttle_factor(self) -> None:
        """Update throttle factor based on metrics"""
        if len(self.response_times) < 10:
            return

        avg_response_time = sum(self.response_times) / len(self.response_times)
        error_rate = self.error_count / self.total_requests if self.total_requests > 0 else 0

        # Increase throttling if response time is high or error rate is high
        if avg_response_time > self.target_response_time or error_rate > self.error_threshold:
            self.throttle_factor = min(self.throttle_factor * 1.2, self.max_throttle)
        else:
            # Decrease throttling if performance is good
            self.throttle_factor = max(self.throttle_factor * 0.95, self.min_throttle)

    def get_delay(self) -> float:
        """Get current delay based on throttle factor"""
        return self.throttle_factor

    def reset(self) -> None:
        """Reset throttler state"""
        self.response_times.clear()
        self.error_count = 0
        self.total_requests = 0
        self.throttle_factor = 1.0


class RateLimiter:
    """Advanced rate limiter with multiple strategies and monitoring"""

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 10,
        strategy: ThrottleStrategy = ThrottleStrategy.ADAPTIVE,
    ):
        self.logger = logging.getLogger(__name__)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.strategy = strategy

        # Token buckets per user/key
        self.buckets: Dict[str, TokenBucket] = {}
        self.global_bucket = TokenBucket(burst_size, requests_per_minute / 60.0)

        # Adaptive throttling
        self.throttlers: Dict[str, AdaptiveThrottler] = {}
        self.global_throttler = AdaptiveThrottler()

        # Request tracking
        self.request_history: deque = deque(maxlen=10000)
        self.user_stats: Dict[str, RateLimitInfo] = defaultdict(self._create_rate_limit_info)

        # Queue for waiting requests
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self.queue_processors = []

        # Metrics
        self.total_requests = 0
        self.rejected_requests = 0
        self.queued_requests = 0

    def _create_rate_limit_info(self) -> RateLimitInfo:
        """Create default rate limit info"""
        return RateLimitInfo(
            requests_per_minute=self.requests_per_minute,
            burst_size=self.burst_size,
            current_tokens=self.burst_size,
            last_refill=time.time(),
            total_requests=0,
            rejected_requests=0,
            queue_size=0,
        )

    def _get_bucket(self, user_id: str) -> TokenBucket:
        """Get or create token bucket for user"""
        if user_id not in self.buckets:
            self.buckets[user_id] = TokenBucket(self.burst_size, self.requests_per_minute / 60.0)
        return self.buckets[user_id]

    def _get_throttler(self, user_id: str) -> AdaptiveThrottler:
        """Get or create throttler for user"""
        if user_id not in self.throttlers:
            self.throttlers[user_id] = AdaptiveThrottler()
        return self.throttlers[user_id]

    async def acquire(
        self, user_id: str = "default", endpoint: str = "default", timeout: Optional[float] = None
    ) -> bool:
        """Acquire permission to make a request"""
        self.total_requests += 1

        # Check global rate limit first
        if not await self.global_bucket.consume():
            self.rejected_requests += 1
            self.user_stats[user_id].rejected_requests += 1
            return False

        # Check user-specific rate limit
        user_bucket = self._get_bucket(user_id)
        if not await user_bucket.consume():
            self.rejected_requests += 1
            self.user_stats[user_id].rejected_requests += 1
            return False

        # Apply adaptive throttling if enabled
        if self.strategy == ThrottleStrategy.ADAPTIVE:
            throttler = self._get_throttler(user_id)
            delay = throttler.get_delay()
            if delay > 0:
                await asyncio.sleep(delay)

        # Update stats
        self.user_stats[user_id].total_requests += 1
        self.user_stats[user_id].current_tokens = user_bucket.get_available_tokens()

        # Record request
        request_info = RequestInfo(
            timestamp=time.time(), user_id=user_id, endpoint=endpoint, success=True
        )
        self.request_history.append(request_info)

        return True

    async def acquire_with_wait(
        self, user_id: str = "default", endpoint: str = "default", timeout: Optional[float] = 30.0
    ) -> bool:
        """Acquire permission with waiting if necessary"""
        # Try immediate acquisition first
        if await self.acquire(user_id, endpoint):
            return True

        # If immediate acquisition fails, wait for tokens
        user_bucket = self._get_bucket(user_id)
        global_available = await self.global_bucket.wait_for_tokens(1, timeout)
        user_available = await user_bucket.wait_for_tokens(1, timeout)

        if global_available and user_available:
            # Update stats
            self.user_stats[user_id].total_requests += 1
            self.user_stats[user_id].current_tokens = user_bucket.get_available_tokens()

            # Record request
            request_info = RequestInfo(
                timestamp=time.time(), user_id=user_id, endpoint=endpoint, success=True
            )
            self.request_history.append(request_info)

            return True

        return False

    def record_response(self, user_id: str, response_time: Optional[float], success: bool) -> None:
        """Record response metrics for adaptive throttling"""
        if self.strategy == ThrottleStrategy.ADAPTIVE:
            throttler = self._get_throttler(user_id)
            throttler.record_request(response_time, success)

            # Also record for global throttler
            self.global_throttler.record_request(response_time, success)

        # Update request history
        if self.request_history:
            last_request = self.request_history[-1]
            if last_request.user_id == user_id:
                last_request.response_time = response_time
                last_request.success = success

    def get_rate_limit_status(self, user_id: str = "default") -> RateLimitInfo:
        """Get current rate limit status for user"""
        if user_id in self.user_stats:
            info = self.user_stats[user_id]
            if user_id in self.buckets:
                info.current_tokens = self.buckets[user_id].get_available_tokens()
            return info

        return self._create_rate_limit_info()

    def get_global_stats(self) -> Dict[str, Any]:
        """Get global rate limiting statistics"""
        recent_requests = [
            r for r in self.request_history if time.time() - r.timestamp < 3600  # Last hour
        ]

        successful_requests = sum(1 for r in recent_requests if r.success)
        failed_requests = len(recent_requests) - successful_requests

        avg_response_time = None
        response_times = [r.response_time for r in recent_requests if r.response_time is not None]
        if response_times:
            avg_response_time = sum(response_times) / len(response_times)

        return {
            "total_requests": self.total_requests,
            "rejected_requests": self.rejected_requests,
            "queued_requests": self.queued_requests,
            "recent_requests": len(recent_requests),
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": successful_requests / len(recent_requests) if recent_requests else 0,
            "average_response_time": avg_response_time,
            "active_users": len(self.buckets),
            "global_tokens_available": self.global_bucket.get_available_tokens(),
            "throttle_factor": (
                self.global_throttler.throttle_factor
                if self.strategy == ThrottleStrategy.ADAPTIVE
                else 1.0
            ),
        }

    def reset_user_limits(self, user_id: str) -> None:
        """Reset rate limits for a specific user"""
        if user_id in self.buckets:
            del self.buckets[user_id]
        if user_id in self.throttlers:
            del self.throttlers[user_id]
        if user_id in self.user_stats:
            del self.user_stats[user_id]

    def update_limits(self, requests_per_minute: int, burst_size: int) -> None:
        """Update rate limiting parameters"""
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size

        # Update global bucket
        self.global_bucket = TokenBucket(burst_size, requests_per_minute / 60.0)

        # Clear existing buckets to apply new limits
        self.buckets.clear()
        self.user_stats.clear()

    def clear_history(self) -> None:
        """Clear request history and reset counters"""
        self.request_history.clear()
        self.total_requests = 0
        self.rejected_requests = 0
        self.queued_requests = 0

        # Reset throttlers
        for throttler in self.throttlers.values():
            throttler.reset()
        self.global_throttler.reset()
