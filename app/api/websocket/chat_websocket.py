"""
WebSocket handler for real-time chat functionality.
Manages WebSocket connections and real-time message exchange.
"""

import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ValidationError

from app.core.logging import get_logger
from app.domain.models.conversation import ConversationStatus, MessageType
from app.domain.services.conversation_engine import ConversationEngine
from app.domain.services.ai_response_service import AIResponseService

logger = get_logger(__name__)


class WebSocketMessage(BaseModel):
    """WebSocket message structure."""
    
    type: str
    data: Dict[str, Any]
    timestamp: Optional[str] = None
    message_id: Optional[str] = None


class ChatWebSocketManager:
    """Manages WebSocket connections for chat functionality."""
    
    def __init__(self):
        # Active connections: conversation_id -> set of websockets
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # WebSocket to conversation mapping
        self.websocket_conversations: Dict[WebSocket, str] = {}
        # User sessions: session_token -> websocket
        self.user_sessions: Dict[str, WebSocket] = {}
        
        # Services
        self.conversation_engine = ConversationEngine()
        self.ai_response_service = AIResponseService()
    
    async def connect(self, websocket: WebSocket, conversation_id: str, session_token: str):
        """Accept WebSocket connection and register it."""
        await websocket.accept()
        
        # Add to active connections
        if conversation_id not in self.active_connections:
            self.active_connections[conversation_id] = set()
        
        self.active_connections[conversation_id].add(websocket)
        self.websocket_conversations[websocket] = conversation_id
        self.user_sessions[session_token] = websocket
        
        logger.info(f"WebSocket connected for conversation {conversation_id}")
        
        # Send connection confirmation
        await self.send_message(websocket, {
            "type": "connection_established",
            "data": {
                "conversation_id": conversation_id,
                "session_token": session_token,
                "timestamp": datetime.now().isoformat(),
            }
        })
        
        # Send conversation status
        status = await self.conversation_engine.get_conversation_status(conversation_id)
        if status:
            await self.send_message(websocket, {
                "type": "conversation_status",
                "data": status
            })
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection."""
        conversation_id = self.websocket_conversations.get(websocket)
        
        if conversation_id and conversation_id in self.active_connections:
            self.active_connections[conversation_id].discard(websocket)
            
            # Remove conversation if no more connections
            if not self.active_connections[conversation_id]:
                del self.active_connections[conversation_id]
        
        # Clean up mappings
        self.websocket_conversations.pop(websocket, None)
        
        # Remove from user sessions
        session_to_remove = None
        for session_token, ws in self.user_sessions.items():
            if ws == websocket:
                session_to_remove = session_token
                break
        
        if session_to_remove:
            del self.user_sessions[session_to_remove]
        
        logger.info(f"WebSocket disconnected for conversation {conversation_id}")
    
    async def send_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send message to specific WebSocket."""
        try:
            # Add timestamp and message ID if not present
            if "timestamp" not in message:
                message["timestamp"] = datetime.now().isoformat()
            if "message_id" not in message:
                message["message_id"] = str(uuid4())
            
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Failed to send WebSocket message: {e}")
            # Remove disconnected websocket
            self.disconnect(websocket)
    
    async def broadcast_to_conversation(self, conversation_id: str, message: Dict[str, Any]):
        """Broadcast message to all connections in a conversation."""
        if conversation_id in self.active_connections:
            disconnected_websockets = []
            
            for websocket in self.active_connections[conversation_id].copy():
                try:
                    await self.send_message(websocket, message)
                except:
                    disconnected_websockets.append(websocket)
            
            # Clean up disconnected websockets
            for websocket in disconnected_websockets:
                self.disconnect(websocket)
    
    async def handle_message(self, websocket: WebSocket, message_data: str):
        """Handle incoming WebSocket message."""
        try:
            # Parse message
            raw_message = json.loads(message_data)
            message = WebSocketMessage(**raw_message)
            
            conversation_id = self.websocket_conversations.get(websocket)
            if not conversation_id:
                await self.send_error(websocket, "No active conversation")
                return
            
            # Route message based on type
            if message.type == "user_message":
                await self.handle_user_message(websocket, conversation_id, message)
            elif message.type == "typing_start":
                await self.handle_typing_indicator(conversation_id, message, True)
            elif message.type == "typing_stop":
                await self.handle_typing_indicator(conversation_id, message, False)
            elif message.type == "message_read":
                await self.handle_message_read(conversation_id, message)
            elif message.type == "end_conversation":
                await self.handle_end_conversation(websocket, conversation_id, message)
            else:
                await self.send_error(websocket, f"Unknown message type: {message.type}")
        
        except ValidationError as e:
            await self.send_error(websocket, f"Invalid message format: {e}")
        except json.JSONDecodeError:
            await self.send_error(websocket, "Invalid JSON format")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            await self.send_error(websocket, "Internal server error")
    
    async def handle_user_message(self, websocket: WebSocket, conversation_id: str, message: WebSocketMessage):
        """Handle user message."""
        try:
            data = message.data
            content = data.get("content", "").strip()
            
            if not content:
                await self.send_error(websocket, "Message content cannot be empty")
                return
            
            # Send typing indicator for AI
            await self.broadcast_to_conversation(conversation_id, {
                "type": "agent_typing",
                "data": {"is_typing": True}
            })
            
            # Process message through conversation engine
            user_message = await self.conversation_engine.process_user_message(
                conversation_id=conversation_id,
                content=content,
                message_type=MessageType(data.get("message_type", "text")),
                metadata=data.get("metadata", {})
            )
            
            # Broadcast user message to all connections
            await self.broadcast_to_conversation(conversation_id, {
                "type": "message_received",
                "data": {
                    "message_id": user_message.id,
                    "sender_type": "user",
                    "content": content,
                    "message_type": user_message.message_type,
                    "timestamp": user_message.timestamp.isoformat(),
                    "status": user_message.status,
                }
            })
            
            # Stop typing indicator
            await self.broadcast_to_conversation(conversation_id, {
                "type": "agent_typing",
                "data": {"is_typing": False}
            })
            
        except Exception as e:
            logger.error(f"Error processing user message: {e}")
            await self.send_error(websocket, "Failed to process message")
            
            # Stop typing indicator on error
            await self.broadcast_to_conversation(conversation_id, {
                "type": "agent_typing",
                "data": {"is_typing": False}
            })
    
    async def handle_typing_indicator(self, conversation_id: str, message: WebSocketMessage, is_typing: bool):
        """Handle typing indicator."""
        await self.broadcast_to_conversation(conversation_id, {
            "type": "user_typing",
            "data": {
                "is_typing": is_typing,
                "user_id": message.data.get("user_id")
            }
        })
    
    async def handle_message_read(self, conversation_id: str, message: WebSocketMessage):
        """Handle message read receipt."""
        message_id = message.data.get("message_id")
        if message_id:
            # Update message status in database
            # This would be implemented based on your repository pattern
            
            # Broadcast read receipt
            await self.broadcast_to_conversation(conversation_id, {
                "type": "message_read",
                "data": {
                    "message_id": message_id,
                    "read_at": datetime.now().isoformat()
                }
            })
    
    async def handle_end_conversation(self, websocket: WebSocket, conversation_id: str, message: WebSocketMessage):
        """Handle conversation end request."""
        try:
            satisfaction_rating = message.data.get("satisfaction_rating")
            reason = message.data.get("reason", "user_ended")
            
            # End conversation
            conversation = await self.conversation_engine.end_conversation(
                conversation_id=conversation_id,
                reason=reason,
                user_satisfaction=satisfaction_rating
            )
            
            # Broadcast conversation end
            await self.broadcast_to_conversation(conversation_id, {
                "type": "conversation_ended",
                "data": {
                    "conversation_id": conversation_id,
                    "reason": reason,
                    "ended_at": conversation.ended_at.isoformat() if conversation.ended_at else None,
                    "satisfaction_rating": satisfaction_rating
                }
            })
            
            # Close all WebSocket connections for this conversation
            if conversation_id in self.active_connections:
                for ws in self.active_connections[conversation_id].copy():
                    await ws.close()
                    self.disconnect(ws)
        
        except Exception as e:
            logger.error(f"Error ending conversation: {e}")
            await self.send_error(websocket, "Failed to end conversation")
    
    async def send_error(self, websocket: WebSocket, error_message: str):
        """Send error message to WebSocket."""
        await self.send_message(websocket, {
            "type": "error",
            "data": {
                "message": error_message,
                "timestamp": datetime.now().isoformat()
            }
        })
    
    async def send_ai_response(self, conversation_id: str, ai_message):
        """Send AI response to conversation."""
        await self.broadcast_to_conversation(conversation_id, {
            "type": "message_received",
            "data": {
                "message_id": ai_message.id,
                "sender_type": "agent",
                "content": ai_message.content,
                "message_type": ai_message.message_type,
                "timestamp": ai_message.timestamp.isoformat(),
                "status": ai_message.status,
                "response_time_ms": ai_message.response_time_ms,
                "knowledge_sources": ai_message.knowledge_sources,
            }
        })
    
    async def send_system_message(self, conversation_id: str, message: str, message_type: str = "info"):
        """Send system message to conversation."""
        await self.broadcast_to_conversation(conversation_id, {
            "type": "system_message",
            "data": {
                "message": message,
                "message_type": message_type,
                "timestamp": datetime.now().isoformat()
            }
        })
    
    def get_active_conversations(self) -> List[str]:
        """Get list of active conversation IDs."""
        return list(self.active_connections.keys())
    
    def get_connection_count(self, conversation_id: str) -> int:
        """Get number of active connections for a conversation."""
        return len(self.active_connections.get(conversation_id, set()))
    
    async def cleanup_inactive_conversations(self):
        """Clean up inactive conversations."""
        inactive_conversations = []
        
        for conversation_id in self.active_connections.keys():
            # Check if conversation is still active
            status = await self.conversation_engine.get_conversation_status(conversation_id)
            if not status or status.get("status") in [ConversationStatus.CLOSED, ConversationStatus.TIMEOUT]:
                inactive_conversations.append(conversation_id)
        
        # Close connections for inactive conversations
        for conversation_id in inactive_conversations:
            if conversation_id in self.active_connections:
                for websocket in self.active_connections[conversation_id].copy():
                    await websocket.close()
                    self.disconnect(websocket)


# Global WebSocket manager instance
chat_websocket_manager = ChatWebSocketManager()


async def websocket_endpoint(websocket: WebSocket, conversation_id: str, session_token: str):
    """WebSocket endpoint for chat functionality."""
    await chat_websocket_manager.connect(websocket, conversation_id, session_token)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            await chat_websocket_manager.handle_message(websocket, data)
    
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for conversation {conversation_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        chat_websocket_manager.disconnect(websocket)
