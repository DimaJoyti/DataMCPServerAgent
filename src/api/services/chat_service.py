"""
Chat service for the API.
"""

import asyncio
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List, AsyncGenerator

from ..models.response_models import ChatResponse, ChatStreamResponse
from ..config import config

# Import agent functions based on agent mode
from src.core.main import chat_with_agent
from src.core.advanced_main import chat_with_advanced_agent
from src.core.enhanced_main import chat_with_enhanced_agent
from src.core.advanced_enhanced_main import chat_with_advanced_enhanced_agent
from src.core.multi_agent_main import chat_with_multi_agent_learning_system
from src.core.reinforcement_learning_main import chat_with_rl_agent
from src.core.distributed_memory_main import chat_with_distributed_memory_agent
from src.core.knowledge_graph_main import chat_with_knowledge_graph_agent
from src.core.error_recovery_main import chat_with_error_recovery_agent
from src.core.seo_main import chat_with_seo_agent


class ChatService:
    """Service for chat interactions."""
    
    async def process_chat(
        self,
        message: str,
        session_id: str,
        agent_mode: Optional[str] = None,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> ChatResponse:
        """
        Process a chat message.
        
        Args:
            message (str): Message to send to the agent
            session_id (str): Session ID for the conversation
            agent_mode (Optional[str]): Agent mode to use
            user_id (Optional[str]): User ID for personalized responses
            context (Optional[Dict[str, Any]]): Additional context for the agent
            
        Returns:
            ChatResponse: Chat response
        """
        # Use default agent mode if not provided
        agent_mode = agent_mode or config.default_agent_mode
        
        # Validate agent mode
        if agent_mode not in config.available_agent_modes:
            raise ValueError(f"Invalid agent mode: {agent_mode}")
        
        # Get the appropriate chat function based on agent mode
        chat_function = self._get_chat_function(agent_mode)
        
        # Process the message
        response_text = await self._process_message(
            chat_function=chat_function,
            message=message,
            session_id=session_id,
            user_id=user_id,
            context=context,
        )
        
        # Create a response
        response = ChatResponse(
            message_id=str(uuid.uuid4()),
            response=response_text,
            session_id=session_id,
            created_at=datetime.now(),
            agent_mode=agent_mode,
            tool_usage=[],  # In a real implementation, this would be populated with actual tool usage
            sources=[],  # In a real implementation, this would be populated with actual sources
            metadata={
                "user_id": user_id,
                "context": context,
            },
        )
        
        return response
    
    async def stream_chat(
        self,
        message: str,
        session_id: str,
        agent_mode: Optional[str] = None,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Stream a chat response.
        
        Args:
            message (str): Message to send to the agent
            session_id (str): Session ID for the conversation
            agent_mode (Optional[str]): Agent mode to use
            user_id (Optional[str]): User ID for personalized responses
            context (Optional[Dict[str, Any]]): Additional context for the agent
            
        Yields:
            str: Chunks of the response
        """
        # Use default agent mode if not provided
        agent_mode = agent_mode or config.default_agent_mode
        
        # Validate agent mode
        if agent_mode not in config.available_agent_modes:
            raise ValueError(f"Invalid agent mode: {agent_mode}")
        
        # Get the appropriate chat function based on agent mode
        chat_function = self._get_chat_function(agent_mode)
        
        # Generate a message ID
        message_id = str(uuid.uuid4())
        
        # In a real implementation, this would stream chunks from the agent
        # For now, we'll simulate streaming by splitting the response
        response_text = await self._process_message(
            chat_function=chat_function,
            message=message,
            session_id=session_id,
            user_id=user_id,
            context=context,
        )
        
        # Split the response into chunks
        chunks = [response_text[i:i+10] for i in range(0, len(response_text), 10)]
        
        # Stream the chunks
        for i, chunk in enumerate(chunks):
            # Create a streaming response
            stream_response = ChatStreamResponse(
                message_id=message_id,
                chunk=chunk,
                session_id=session_id,
                created_at=datetime.now(),
                is_final=(i == len(chunks) - 1),
                metadata={
                    "user_id": user_id,
                    "context": context,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            )
            
            # Yield the chunk as a server-sent event
            yield f"data: {json.dumps(stream_response.dict())}\n\n"
            
            # Add a small delay to simulate streaming
            await asyncio.sleep(0.1)
        
        # Yield the end of the stream
        yield "data: [DONE]\n\n"
    
    async def get_chat_history(
        self,
        session_id: str,
        limit: int = 10,
        offset: int = 0,
    ) -> List[ChatResponse]:
        """
        Get chat history for a session.
        
        Args:
            session_id (str): Session ID
            limit (int): Maximum number of messages to return
            offset (int): Offset for pagination
            
        Returns:
            List[ChatResponse]: Chat history
        """
        # In a real implementation, this would retrieve chat history from a database
        # For now, we'll return a mock history
        history = []
        
        # Add some mock messages
        for i in range(offset, offset + limit):
            history.append(
                ChatResponse(
                    message_id=str(uuid.uuid4()),
                    response=f"Mock response {i}",
                    session_id=session_id,
                    created_at=datetime.now(),
                    agent_mode=config.default_agent_mode,
                    tool_usage=[],
                    sources=[],
                    metadata={},
                )
            )
        
        return history
    
    async def log_interaction(
        self,
        session_id: str,
        message: str,
        response: ChatResponse,
    ) -> None:
        """
        Log a chat interaction.
        
        Args:
            session_id (str): Session ID
            message (str): User message
            response (ChatResponse): Agent response
        """
        # In a real implementation, this would log the interaction to a database
        # For now, we'll just print it
        print(f"Logged interaction for session {session_id}")
    
    def _get_chat_function(self, agent_mode: str):
        """
        Get the appropriate chat function based on agent mode.
        
        Args:
            agent_mode (str): Agent mode
            
        Returns:
            function: Chat function
        """
        # Map agent modes to chat functions
        chat_functions = {
            "basic": chat_with_agent,
            "advanced": chat_with_advanced_agent,
            "enhanced": chat_with_enhanced_agent,
            "advanced_enhanced": chat_with_advanced_enhanced_agent,
            "multi_agent": chat_with_multi_agent_learning_system,
            "reinforcement_learning": chat_with_rl_agent,
            "distributed_memory": chat_with_distributed_memory_agent,
            "knowledge_graph": chat_with_knowledge_graph_agent,
            "error_recovery": chat_with_error_recovery_agent,
            "seo": chat_with_seo_agent,
        }
        
        # Get the chat function for the agent mode
        chat_function = chat_functions.get(agent_mode)
        
        if not chat_function:
            # Handle research_reports mode separately due to circular imports
            if agent_mode == "research_reports":
                from research_reports_runner import run_research_reports_agent
                return run_research_reports_agent
            
            raise ValueError(f"No chat function found for agent mode: {agent_mode}")
        
        return chat_function
    
    async def _process_message(
        self,
        chat_function,
        message: str,
        session_id: str,
        user_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Process a message using the appropriate chat function.
        
        Args:
            chat_function: Chat function to use
            message (str): Message to send to the agent
            session_id (str): Session ID for the conversation
            user_id (Optional[str]): User ID for personalized responses
            context (Optional[Dict[str, Any]]): Additional context for the agent
            
        Returns:
            str: Response from the agent
        """
        # In a real implementation, this would call the actual chat function
        # For now, we'll return a mock response
        return f"This is a mock response to: {message}"
