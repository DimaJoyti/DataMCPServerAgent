"""
Error handling utilities for Bright Data MCP operations.
These functions help detect, classify, and recover from common errors.
"""

import asyncio
import re
from typing import Any, Callable, List, Optional

import aiohttp


class MCPError(Exception):
    """Base exception for MCP-related errors."""

    def __init__(self, message: str, error_type: str, recovery_suggestion: Optional[str] = None):
        """Initialize the MCPError.

        Args:
            message: Error message
            error_type: Type of error (e.g., 'connection', 'auth', 'rate_limit')
            recovery_suggestion: Suggested recovery action
        """
        self.message = message
        self.error_type = error_type
        self.recovery_suggestion = recovery_suggestion
        super().__init__(self.message)


class ConnectionError(MCPError):
    """Error for connection issues with MCP services."""

    def __init__(self, message: str, recovery_suggestion: Optional[str] = None):
        """Initialize the ConnectionError.

        Args:
            message: Error message
            recovery_suggestion: Suggested recovery action
        """
        super().__init__(
            message,
            "connection",
            recovery_suggestion or "Check your internet connection and Bright Data service status.",
        )


class AuthenticationError(MCPError):
    """Error for authentication issues with MCP services."""

    def __init__(self, message: str, recovery_suggestion: Optional[str] = None):
        """Initialize the AuthenticationError.

        Args:
            message: Error message
            recovery_suggestion: Suggested recovery action
        """
        super().__init__(
            message,
            "authentication",
            recovery_suggestion
            or "Verify your API_TOKEN, BROWSER_AUTH, and WEB_UNLOCKER_ZONE environment variables.",
        )


class RateLimitError(MCPError):
    """Error for rate limiting issues with MCP services."""

    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        recovery_suggestion: Optional[str] = None,
    ):
        """Initialize the RateLimitError.

        Args:
            message: Error message
            retry_after: Seconds to wait before retrying
            recovery_suggestion: Suggested recovery action
        """
        self.retry_after = retry_after
        super().__init__(
            message,
            "rate_limit",
            recovery_suggestion or f"Wait {retry_after or 60} seconds before retrying.",
        )


class WebsiteError(MCPError):
    """Error for issues with target websites."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        recovery_suggestion: Optional[str] = None,
    ):
        """Initialize the WebsiteError.

        Args:
            message: Error message
            status_code: HTTP status code if available
            recovery_suggestion: Suggested recovery action
        """
        self.status_code = status_code
        super().__init__(
            message, "website", recovery_suggestion or "Try a different URL or website."
        )


class ContentExtractionError(MCPError):
    """Error for issues with content extraction."""

    def __init__(self, message: str, recovery_suggestion: Optional[str] = None):
        """Initialize the ContentExtractionError.

        Args:
            message: Error message
            recovery_suggestion: Suggested recovery action
        """
        super().__init__(
            message,
            "content_extraction",
            recovery_suggestion or "Try a different extraction method or URL.",
        )


async def with_retry(
    func: Callable,
    *args,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    **kwargs,
) -> Any:
    """Execute a function with exponential backoff retry logic.

    Args:
        func: Function to execute
        *args: Positional arguments for the function
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries in seconds
        **kwargs: Keyword arguments for the function

    Returns:
        Result of the function

    Raises:
        Exception: If all retry attempts fail
    """
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func(*args, **kwargs)
        except RateLimitError as e:
            last_exception = e
            # Use the retry_after value if provided, otherwise use exponential backoff
            delay = e.retry_after if e.retry_after else min(base_delay * (2**attempt), max_delay)
            if attempt < max_retries:
                await asyncio.sleep(delay)
        except (ConnectionError, aiohttp.ClientError) as e:
            last_exception = e
            if attempt < max_retries:
                delay = min(base_delay * (2**attempt), max_delay)
                await asyncio.sleep(delay)
        except Exception as e:
            # Don't retry other types of exceptions
            raise e

    # If we've exhausted all retries, raise the last exception
    raise last_exception


def classify_error(error: Exception) -> MCPError:
    """Classify an exception into a specific MCP error type.

    Args:
        error: The exception to classify

    Returns:
        A specific MCPError subclass
    """
    error_message = str(error)

    # Check for authentication errors
    if any(
        term in error_message.lower()
        for term in ["auth", "unauthorized", "forbidden", "token", "credentials"]
    ):
        return AuthenticationError(error_message)

    # Check for rate limit errors
    if any(
        term in error_message.lower() for term in ["rate limit", "too many requests", "throttle"]
    ):
        # Try to extract retry-after information
        retry_after_match = re.search(r"retry after (\d+)", error_message.lower())
        retry_after = int(retry_after_match.group(1)) if retry_after_match else None
        return RateLimitError(error_message, retry_after)

    # Check for connection errors
    if any(
        term in error_message.lower()
        for term in ["connection", "timeout", "network", "unreachable"]
    ):
        return ConnectionError(error_message)

    # Check for website errors
    if any(
        term in error_message.lower() for term in ["404", "not found", "403", "blocked", "captcha"]
    ):
        # Try to extract status code
        status_code_match = re.search(r"status[_\s]?code[:\s]+(\d+)", error_message.lower())
        status_code = int(status_code_match.group(1)) if status_code_match else None
        return WebsiteError(error_message, status_code)

    # Check for content extraction errors
    if any(term in error_message.lower() for term in ["extract", "parse", "content", "selector"]):
        return ContentExtractionError(error_message)

    # Default to generic MCP error
    return MCPError(error_message, "unknown", "Try a different approach or tool.")


def format_error_for_user(error: Exception) -> str:
    """Format an error into a user-friendly message with recovery suggestions.

    Args:
        error: The exception to format

    Returns:
        Formatted error message
    """
    # Classify the error if it's not already an MCPError
    if not isinstance(error, MCPError):
        error = classify_error(error)

    # Format the error message
    message = f"## Error: {error.error_type.title()}\n\n"
    message += f"{error.message}\n\n"

    if error.recovery_suggestion:
        message += f"**Suggestion**: {error.recovery_suggestion}\n\n"

    # Add specific advice based on error type
    if isinstance(error, AuthenticationError):
        message += "**Check your credentials**:\n"
        message += "- Verify that your Bright Data API credentials are correct\n"
        message += "- Ensure your account is active and has sufficient credits\n"
        message += "- Check if your subscription includes the tools you're trying to use\n"
    elif isinstance(error, RateLimitError):
        message += "**Rate limit reached**:\n"
        message += f"- Wait at least {error.retry_after or 60} seconds before trying again\n"
        message += "- Consider using a different tool or approach in the meantime\n"
        message += "- Break your request into smaller parts\n"
    elif isinstance(error, WebsiteError):
        message += "**Website access issue**:\n"
        message += "- The website might be blocking automated access\n"
        message += "- Try a different URL or website\n"
        message += "- Consider using a different scraping approach\n"
    elif isinstance(error, ContentExtractionError):
        message += "**Content extraction issue**:\n"
        message += "- The website structure might have changed\n"
        message += "- Try a different extraction method\n"
        message += "- Consider using a more general extraction approach\n"

    return message


def suggest_alternative_tools(failed_tool: str) -> List[str]:
    """Suggest alternative tools when a specific tool fails.

    Args:
        failed_tool: Name of the tool that failed

    Returns:
        List of suggested alternative tools
    """
    alternatives = {
        "scrape_as_markdown_Bright_Data": [
            "scrape_as_html_Bright_Data",
            "scraping_browser_get_text_Bright_Data",
            "enhanced_web_scraper",
        ],
        "scrape_as_html_Bright_Data": [
            "scrape_as_markdown_Bright_Data",
            "scraping_browser_get_html_Bright_Data",
            "enhanced_web_scraper",
        ],
        "brave_web_search_Brave": ["enhanced_web_search", "search_engine_Bright_Data"],
        "web_data_amazon_product_Bright_Data": [
            "product_comparison",
            "scrape_as_markdown_Bright_Data",
        ],
        "web_data_instagram_profiles_Bright_Data": [
            "social_media_analyzer",
            "scrape_as_markdown_Bright_Data",
        ],
        "web_data_facebook_posts_Bright_Data": [
            "social_media_analyzer",
            "scrape_as_markdown_Bright_Data",
        ],
        "web_data_x_posts_Bright_Data": ["social_media_analyzer", "scrape_as_markdown_Bright_Data"],
        "enhanced_web_scraper": ["scrape_as_markdown_Bright_Data", "scrape_as_html_Bright_Data"],
        "enhanced_web_search": ["brave_web_search_Brave", "search_engine_Bright_Data"],
        "product_comparison": [
            "web_data_amazon_product_Bright_Data",
            "scrape_as_markdown_Bright_Data",
        ],
        "social_media_analyzer": [
            "web_data_instagram_profiles_Bright_Data",
            "web_data_facebook_posts_Bright_Data",
            "web_data_x_posts_Bright_Data",
            "scrape_as_markdown_Bright_Data",
        ],
    }

    return alternatives.get(
        failed_tool, ["scrape_as_markdown_Bright_Data", "brave_web_search_Brave"]
    )
