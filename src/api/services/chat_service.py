"""
Chat service for the API.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Any, AsyncGenerator, Dict, List, Optional

from src.core.advanced_enhanced_main import chat_with_advanced_enhanced_agent
from src.core.advanced_main import chat_with_advanced_agent
from src.core.distributed_memory_main import chat_with_distributed_memory_agent
from src.core.enhanced_main import chat_with_enhanced_agent
from src.core.error_recovery_main import chat_with_error_recovery_agent
from src.core.knowledge_graph_main import chat_with_knowledge_graph_agent

# Import agent functions based on agent mode
from src.core.main import chat_with_agent
from src.core.multi_agent_main import chat_with_multi_agent_learning_system
from src.core.reinforcement_learning_main import chat_with_rl_agent
from src.core.seo_main import chat_with_seo_agent

from ..config import config
from ..models.response_models import ChatResponse, ChatStreamResponse


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
        try:
            # Use default agent mode if not provided
            agent_mode = agent_mode or config.default_agent_mode

            # Validate agent mode
            if agent_mode not in config.available_agent_modes:
                raise ValueError(f"Invalid agent mode: {agent_mode}")

            # Get the appropriate chat function based on agent mode
            chat_function = self._get_chat_function(agent_mode)

            # Generate a message ID
            message_id = str(uuid.uuid4())

            # Get conversation history from session service
            from .session_service import SessionService

            session_service = SessionService()

            # Get conversation history
            messages = await session_service.get_conversation_history(session_id)

            # If no history, initialize with system message
            if not messages:
                messages = [
                    {
                        "role": "system",
                        "content": "You are an advanced AI assistant with specialized capabilities for web automation and data collection.",
                    }
                ]

            # Add user message to history
            messages.append({"role": "user", "content": message})

            # Save updated history
            await session_service.save_conversation_history(session_id, messages)

            # Process the message with the appropriate chat function
            response_text = await self._process_message(
                chat_function=chat_function,
                message=message,
                session_id=session_id,
                user_id=user_id,
                context=context,
            )

            # Split the response into chunks for simulated streaming
            chunk_size = 20  # Characters per chunk
            chunks = [
                response_text[i : i + chunk_size]
                for i in range(0, len(response_text), chunk_size)
            ]

            # Stream the chunks
            current_text = ""
            for i, chunk in enumerate(chunks):
                current_text += chunk

                # Create a streaming response
                stream_response = ChatStreamResponse(
                    message_id=message_id,
                    chunk=current_text,
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
        except Exception as e:
            # Handle any unexpected errors
            error_message = f"An unexpected error occurred: {str(e)}"

            # Yield the error as a final chunk
            error_response = ChatStreamResponse(
                message_id=str(uuid.uuid4()),
                chunk=error_message,
                session_id=session_id,
                created_at=datetime.now(),
                is_final=True,
                metadata={
                    "user_id": user_id,
                    "context": context,
                    "error": str(e),
                },
            )

            yield f"data: {json.dumps(error_response.dict())}\n\n"

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
        try:
            # Get conversation history from session service
            from .session_service import SessionService

            session_service = SessionService()

            # Get conversation history
            messages = await session_service.get_conversation_history(session_id)

            # Convert messages to ChatResponse objects
            history = []

            # Skip system messages and apply pagination
            user_assistant_messages = [
                msg for msg in messages if msg["role"] in ["user", "assistant"]
            ]
            paginated_messages = user_assistant_messages[offset : offset + limit]

            for i, msg in enumerate(paginated_messages):
                # Generate a deterministic message ID based on content and position
                message_id = str(
                    uuid.uuid5(uuid.NAMESPACE_DNS, f"{session_id}:{i}:{msg['content']}")
                )

                # Create a timestamp (use current time as we don't have real timestamps)
                created_at = datetime.now() - timedelta(
                    minutes=(len(paginated_messages) - i)
                )

                # Create a ChatResponse object
                history.append(
                    ChatResponse(
                        message_id=message_id,
                        response=msg["content"],
                        session_id=session_id,
                        created_at=created_at,
                        agent_mode=config.default_agent_mode,
                        tool_usage=[],  # We don't have real tool usage data
                        sources=[],  # We don't have real source data
                        metadata={
                            "role": msg["role"],
                            "index": offset + i,
                        },
                    )
                )

            return history
        except Exception:
            # If there's an error, return a mock history
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
        try:
            # Get session service
            from .session_service import SessionService

            session_service = SessionService()

            # Get tool service for logging tool usage
            from .tool_service import ToolService

            tool_service = ToolService()

            # Log the interaction
            await session_service.set_session_data(
                session_id=session_id,
                key=f"interaction:{int(datetime.now().timestamp())}",
                value={
                    "message": message,
                    "response": response.response,
                    "timestamp": datetime.now().isoformat(),
                    "agent_mode": response.agent_mode,
                    "message_id": response.message_id,
                },
            )

            # Log tool usage if available
            if response.tool_usage:
                for tool_usage in response.tool_usage:
                    await tool_service.log_tool_usage(
                        session_id=session_id,
                        tool_name=tool_usage.get("tool_name", "unknown"),
                        tool_input=tool_usage.get("tool_input", {}),
                        tool_output=tool_usage.get("tool_output", ""),
                    )

            # Print for debugging
            print(f"Logged interaction for session {session_id}")
        except Exception as e:
            # If there's an error, just print it
            print(f"Error logging interaction for session {session_id}: {str(e)}")

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
        try:
            # Get conversation history from session service
            from .session_service import SessionService

            session_service = SessionService()

            # Get conversation history
            messages = await session_service.get_conversation_history(session_id)

            # If no history, initialize with system message
            if not messages:
                messages = [
                    {
                        "role": "system",
                        "content": "You are an advanced AI assistant with specialized capabilities for web automation and data collection.",
                    }
                ]

            # Add user message to history
            messages.append({"role": "user", "content": message})

            # Save updated history
            await session_service.save_conversation_history(session_id, messages)

            # Process the message with the appropriate chat function
            try:
                # Call the agent with the full message history
                response = await chat_function(messages=messages, session_id=session_id)

                # Extract agent's reply
                if isinstance(response, dict) and "messages" in response:
                    # For ReAct agents that return a dict with messages
                    ai_message = response["messages"][-1]["content"]
                elif isinstance(response, dict) and "response" in response:
                    # For agents that return a dict with response
                    ai_message = response["response"]
                elif isinstance(response, str):
                    # For agents that return a string directly
                    ai_message = response
                else:
                    # Default fallback
                    ai_message = str(response)

                # Add agent's reply to history
                messages.append({"role": "assistant", "content": ai_message})

                # Save updated history
                await session_service.save_conversation_history(session_id, messages)

                return ai_message
            except Exception as e:
                # Handle errors
                error_message = (
                    f"An error occurred while processing your message: {str(e)}"
                )

                # Add error message to history
                messages.append({"role": "assistant", "content": error_message})

                # Save updated history
                await session_service.save_conversation_history(session_id, messages)

                return error_message
        except Exception as e:
            # Fallback to mock response if there's an error
            return f"I'm having trouble processing your request. Error: {str(e)}"
