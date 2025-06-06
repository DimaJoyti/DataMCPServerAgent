"""
MCP Inspector for debugging and monitoring MCP connections and tool usage.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class MCPEventType(Enum):
    CONNECTION_OPENED = "connection_opened"
    CONNECTION_CLOSED = "connection_closed"
    TOOL_CALLED = "tool_called"
    TOOL_RESULT = "tool_result"
    AUTH_CHECK = "auth_check"
    ERROR = "error"

@dataclass
class MCPEvent:
    event_type: MCPEventType
    timestamp: str
    session_id: str
    user_id: Optional[str] = None
    tool_name: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class MCPInspector:
    """Inspector for monitoring MCP connections and tool usage."""
    
    def __init__(self):
        self.events: List[MCPEvent] = []
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.tool_usage_stats: Dict[str, int] = {}
        self.auth_failures: List[MCPEvent] = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("MCPInspector")
    
    def log_event(self, event: MCPEvent):
        """Log an MCP event."""
        self.events.append(event)
        self.logger.info(f"MCP Event: {event.event_type.value} - {event.session_id}")
        
        # Update statistics
        if event.tool_name:
            self.tool_usage_stats[event.tool_name] = self.tool_usage_stats.get(event.tool_name, 0) + 1
        
        if event.event_type == MCPEventType.AUTH_CHECK and event.error:
            self.auth_failures.append(event)
    
    def log_connection_opened(self, session_id: str, user_id: str, metadata: Dict[str, Any] = None):
        """Log when an MCP connection is opened."""
        event = MCPEvent(
            event_type=MCPEventType.CONNECTION_OPENED,
            timestamp=datetime.utcnow().isoformat(),
            session_id=session_id,
            user_id=user_id,
            metadata=metadata or {}
        )
        
        self.active_sessions[session_id] = {
            "user_id": user_id,
            "connected_at": event.timestamp,
            "tools_used": [],
            "metadata": metadata or {}
        }
        
        self.log_event(event)
    
    def log_connection_closed(self, session_id: str, reason: str = None):
        """Log when an MCP connection is closed."""
        event = MCPEvent(
            event_type=MCPEventType.CONNECTION_CLOSED,
            timestamp=datetime.utcnow().isoformat(),
            session_id=session_id,
            metadata={"reason": reason} if reason else None
        )
        
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        
        self.log_event(event)
    
    def log_tool_call(self, session_id: str, tool_name: str, parameters: Dict[str, Any], user_id: str = None):
        """Log when a tool is called."""
        event = MCPEvent(
            event_type=MCPEventType.TOOL_CALLED,
            timestamp=datetime.utcnow().isoformat(),
            session_id=session_id,
            user_id=user_id,
            tool_name=tool_name,
            parameters=parameters
        )
        
        # Update session data
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["tools_used"].append({
                "tool": tool_name,
                "timestamp": event.timestamp,
                "parameters": parameters
            })
        
        self.log_event(event)
    
    def log_tool_result(self, session_id: str, tool_name: str, result: Dict[str, Any], user_id: str = None):
        """Log the result of a tool call."""
        event = MCPEvent(
            event_type=MCPEventType.TOOL_RESULT,
            timestamp=datetime.utcnow().isoformat(),
            session_id=session_id,
            user_id=user_id,
            tool_name=tool_name,
            result=result
        )
        
        self.log_event(event)
    
    def log_auth_check(self, session_id: str, user_id: str, tool_name: str, success: bool, error: str = None):
        """Log an authentication/authorization check."""
        event = MCPEvent(
            event_type=MCPEventType.AUTH_CHECK,
            timestamp=datetime.utcnow().isoformat(),
            session_id=session_id,
            user_id=user_id,
            tool_name=tool_name,
            error=error if not success else None,
            metadata={"success": success}
        )
        
        self.log_event(event)
    
    def log_error(self, session_id: str, error: str, tool_name: str = None, user_id: str = None):
        """Log an error."""
        event = MCPEvent(
            event_type=MCPEventType.ERROR,
            timestamp=datetime.utcnow().isoformat(),
            session_id=session_id,
            user_id=user_id,
            tool_name=tool_name,
            error=error
        )
        
        self.log_event(event)
    
    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific session."""
        return self.active_sessions.get(session_id)
    
    def get_tool_usage_stats(self) -> Dict[str, int]:
        """Get tool usage statistics."""
        return self.tool_usage_stats.copy()
    
    def get_auth_failures(self) -> List[Dict[str, Any]]:
        """Get recent authentication failures."""
        return [asdict(event) for event in self.auth_failures[-10:]]  # Last 10 failures
    
    def get_recent_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent MCP events."""
        return [asdict(event) for event in self.events[-limit:]]
    
    def get_active_sessions_summary(self) -> Dict[str, Any]:
        """Get summary of active sessions."""
        return {
            "total_active": len(self.active_sessions),
            "sessions": {
                session_id: {
                    "user_id": data["user_id"],
                    "connected_at": data["connected_at"],
                    "tools_used_count": len(data["tools_used"])
                }
                for session_id, data in self.active_sessions.items()
            }
        }
    
    def export_events(self, filename: str = None) -> str:
        """Export events to JSON file."""
        if not filename:
            filename = f"mcp_events_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            "export_timestamp": datetime.utcnow().isoformat(),
            "total_events": len(self.events),
            "active_sessions": len(self.active_sessions),
            "tool_usage_stats": self.tool_usage_stats,
            "events": [asdict(event) for event in self.events]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return filename

# Global inspector instance
mcp_inspector = MCPInspector()

# Decorator for automatic tool call logging
def log_tool_call(tool_name: str):
    """Decorator to automatically log tool calls."""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            session_id = kwargs.get('session_id', 'unknown')
            user_id = kwargs.get('user_id', 'unknown')
            
            # Log tool call
            mcp_inspector.log_tool_call(
                session_id=session_id,
                tool_name=tool_name,
                parameters=kwargs,
                user_id=user_id
            )
            
            try:
                result = await func(*args, **kwargs)
                
                # Log successful result
                mcp_inspector.log_tool_result(
                    session_id=session_id,
                    tool_name=tool_name,
                    result={"success": True, "data": result},
                    user_id=user_id
                )
                
                return result
                
            except Exception as e:
                # Log error
                mcp_inspector.log_error(
                    session_id=session_id,
                    error=str(e),
                    tool_name=tool_name,
                    user_id=user_id
                )
                raise
        
        return wrapper
    return decorator
