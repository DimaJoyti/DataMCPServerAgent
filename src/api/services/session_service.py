"""
Session service for the API.
"""

import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..config import config
from .redis_service import RedisService


class SessionService:
    """Service for session operations."""

    def __init__(self):
        """
        Initialize the session service.
        """
        self.session_store = config.session_store
        self.redis_service = None
        self.memory_sessions = {}

    async def _get_redis_service(self) -> RedisService:
        """
        Get the Redis service.

        Returns:
            RedisService: Redis service
        """
        if self.redis_service is None:
            self.redis_service = RedisService()
            await self.redis_service.connect()
        return self.redis_service

    async def create_session(self) -> str:
        """
        Create a new session.

        Returns:
            str: Session ID
        """
        session_id = str(uuid.uuid4())

        # Initialize session data
        session_data = {
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "data": {},
        }

        # Store session data
        await self.save_session(session_id, session_data)

        return session_id

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get session data.

        Args:
            session_id (str): Session ID

        Returns:
            Optional[Dict[str, Any]]: Session data or None if not found
        """
        if self.session_store == "redis":
            redis_service = await self._get_redis_service()
            return await redis_service.get_session(session_id)
        elif self.session_store == "memory":
            return self.memory_sessions.get(session_id)
        else:
            raise ValueError(f"Unsupported session store: {self.session_store}")

    async def save_session(self, session_id: str, data: Dict[str, Any]) -> None:
        """
        Save session data.

        Args:
            session_id (str): Session ID
            data (Dict[str, Any]): Session data
        """
        # Update last accessed time
        data["last_accessed"] = datetime.now().isoformat()

        if self.session_store == "redis":
            redis_service = await self._get_redis_service()
            await redis_service.save_session(session_id, data)
        elif self.session_store == "memory":
            self.memory_sessions[session_id] = data
        else:
            raise ValueError(f"Unsupported session store: {self.session_store}")

    async def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.

        Args:
            session_id (str): Session ID

        Returns:
            bool: True if the session was deleted, False otherwise
        """
        if self.session_store == "redis":
            redis_service = await self._get_redis_service()
            return await redis_service.delete_session(session_id) > 0
        elif self.session_store == "memory":
            if session_id in self.memory_sessions:
                del self.memory_sessions[session_id]
                return True
            return False
        else:
            raise ValueError(f"Unsupported session store: {self.session_store}")

    async def get_session_data(
        self,
        session_id: str,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Get session data value.

        Args:
            session_id (str): Session ID
            key (str): Data key
            default (Any): Default value if key not found

        Returns:
            Any: Data value or default if not found
        """
        session = await self.get_session(session_id)
        if session and "data" in session:
            return session["data"].get(key, default)
        return default

    async def set_session_data(
        self,
        session_id: str,
        key: str,
        value: Any,
    ) -> None:
        """
        Set session data value.

        Args:
            session_id (str): Session ID
            key (str): Data key
            value (Any): Data value
        """
        session = await self.get_session(session_id)
        if session is None:
            # Create a new session if it doesn't exist
            session = {
                "created_at": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat(),
                "data": {},
            }

        # Update data
        if "data" not in session:
            session["data"] = {}
        session["data"][key] = value

        # Save session
        await self.save_session(session_id, session)

    async def delete_session_data(self, session_id: str, key: str) -> bool:
        """
        Delete session data value.

        Args:
            session_id (str): Session ID
            key (str): Data key

        Returns:
            bool: True if the key was deleted, False otherwise
        """
        session = await self.get_session(session_id)
        if session and "data" in session and key in session["data"]:
            del session["data"][key]
            await self.save_session(session_id, session)
            return True
        return False

    async def get_all_sessions(self) -> List[str]:
        """
        Get all session IDs.

        Returns:
            List[str]: List of session IDs
        """
        if self.session_store == "redis":
            redis_service = await self._get_redis_service()
            sessions = await redis_service.get_sessions()
            return list(sessions)
        elif self.session_store == "memory":
            return list(self.memory_sessions.keys())
        else:
            raise ValueError(f"Unsupported session store: {self.session_store}")

    async def cleanup_expired_sessions(self, max_age_seconds: int = 86400) -> int:
        """
        Clean up expired sessions.

        Args:
            max_age_seconds (int): Maximum age of sessions in seconds

        Returns:
            int: Number of sessions deleted
        """
        now = datetime.now()
        deleted_count = 0

        if self.session_store == "redis":
            redis_service = await self._get_redis_service()
            sessions = await redis_service.get_sessions()

            for session_id in sessions:
                session = await redis_service.get_session(session_id)
                if session:
                    last_accessed = datetime.fromisoformat(session["last_accessed"])
                    age_seconds = (now - last_accessed).total_seconds()

                    if age_seconds > max_age_seconds:
                        await redis_service.delete_session(session_id)
                        deleted_count += 1

        elif self.session_store == "memory":
            for session_id in list(self.memory_sessions.keys()):
                session = self.memory_sessions[session_id]
                last_accessed = datetime.fromisoformat(session["last_accessed"])
                age_seconds = (now - last_accessed).total_seconds()

                if age_seconds > max_age_seconds:
                    del self.memory_sessions[session_id]
                    deleted_count += 1

        else:
            raise ValueError(f"Unsupported session store: {self.session_store}")

        return deleted_count

    async def save_conversation_history(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
    ) -> None:
        """
        Save conversation history.

        Args:
            session_id (str): Session ID
            messages (List[Dict[str, Any]]): List of message dictionaries
        """
        await self.set_session_data(session_id, "conversation_history", messages)

        if self.session_store == "redis" and config.enable_distributed:
            redis_service = await self._get_redis_service()
            await redis_service.save_conversation_history(session_id, messages)

    async def get_conversation_history(
        self,
        session_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get conversation history.

        Args:
            session_id (str): Session ID

        Returns:
            List[Dict[str, Any]]: List of message dictionaries
        """
        history = await self.get_session_data(session_id, "conversation_history", [])

        if not history and self.session_store == "redis" and config.enable_distributed:
            redis_service = await self._get_redis_service()
            history = await redis_service.get_conversation_history(session_id)

        return history

    async def save_tool_usage(
        self,
        session_id: str,
        tool_name: str,
        args: Dict[str, Any],
        result: Any,
    ) -> None:
        """
        Save tool usage.

        Args:
            session_id (str): Session ID
            tool_name (str): Name of the tool
            args (Dict[str, Any]): Tool arguments
            result (Any): Tool result
        """
        # Get existing tool usage
        tool_usage = await self.get_session_data(session_id, "tool_usage", {})

        # Add new usage
        if tool_name not in tool_usage:
            tool_usage[tool_name] = []

        tool_usage[tool_name].append(
            {
                "args": args,
                "result": result,
                "timestamp": time.time(),
            }
        )

        # Save tool usage
        await self.set_session_data(session_id, "tool_usage", tool_usage)

        if self.session_store == "redis" and config.enable_distributed:
            redis_service = await self._get_redis_service()
            await redis_service.save_tool_usage(session_id, tool_name, args, result)

    async def get_tool_usage(
        self,
        session_id: str,
        tool_name: Optional[str] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get tool usage.

        Args:
            session_id (str): Session ID
            tool_name (Optional[str]): Name of tool to get history for, or None for all tools

        Returns:
            Dict[str, List[Dict[str, Any]]]: Dictionary of tool usage by tool name
        """
        tool_usage = await self.get_session_data(session_id, "tool_usage", {})

        if not tool_usage and self.session_store == "redis" and config.enable_distributed:
            redis_service = await self._get_redis_service()
            tool_usage = await redis_service.get_tool_usage(session_id, tool_name)

        if tool_name:
            return {tool_name: tool_usage.get(tool_name, [])}

        return tool_usage
