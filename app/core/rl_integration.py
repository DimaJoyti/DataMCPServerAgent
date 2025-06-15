"""
Reinforcement Learning Integration for DataMCPServerAgent.
This module integrates the advanced RL system with the main application.
"""

import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool

try:
    from app.core.config import get_settings
    Settings = get_settings().__class__
except ImportError:
    class Settings:
        app_name = "DataMCPServerAgent"
        app_version = "2.0.0"
        environment = "development"
        debug = True

try:
    from app.core.logging import get_logger
except ImportError:
    from app.core.simple_logging import get_logger

# Import RL components
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.reinforcement_learning_main import setup_rl_agent
from src.memory.memory_persistence import MemoryDatabase

logger = get_logger(__name__)


class RLMode(str, Enum):
    """Available RL modes."""
    BASIC = "basic"
    ADVANCED = "advanced"
    MULTI_OBJECTIVE = "multi_objective"
    HIERARCHICAL = "hierarchical"
    MODERN_DEEP = "modern_deep"
    RAINBOW = "rainbow"
    MULTI_AGENT = "multi_agent"
    CURRICULUM = "curriculum"
    META_LEARNING = "meta_learning"
    DISTRIBUTED = "distributed"
    SAFE = "safe"
    EXPLAINABLE = "explainable"


@dataclass
class RLConfig:
    """Configuration for RL system."""
    mode: RLMode = RLMode.MODERN_DEEP
    algorithm: str = "dqn"
    state_representation: str = "contextual"

    # Performance settings
    training_enabled: bool = True
    evaluation_episodes: int = 10
    save_frequency: int = 100

    # Safety settings
    safety_enabled: bool = True
    max_resource_usage: float = 0.8
    max_response_time: float = 5.0
    safety_weight: float = 0.5

    # Explainability settings
    explanation_enabled: bool = True
    explanation_methods: List[str] = None

    # Distributed settings
    distributed_workers: int = 4
    parameter_server_host: str = "localhost"
    parameter_server_port: int = 8000

    # Multi-agent settings
    num_agents: int = 3
    cooperation_mode: str = "cooperative"
    communication_enabled: bool = True

    def __post_init__(self):
        if self.explanation_methods is None:
            self.explanation_methods = ["gradient", "permutation"]


class RLSystemManager:
    """Manages the RL system integration."""

    def __init__(self, settings: Settings):
        """Initialize RL system manager.
        
        Args:
            settings: Application settings
        """
        self.settings = settings
        self.config = self._load_rl_config()
        self.rl_agent = None
        self.model = None
        self.db = None
        self.mcp_tools = []

        # Performance tracking
        self.performance_metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "average_response_time": 0.0,
            "average_reward": 0.0,
            "training_episodes": 0,
        }

        # System state
        self.is_initialized = False
        self.is_training = False

        logger.info(f"🤖 RL System Manager initialized with mode: {self.config.mode}")

    def _load_rl_config(self) -> RLConfig:
        """Load RL configuration from environment variables.
        
        Returns:
            RL configuration
        """
        return RLConfig(
            mode=RLMode(os.getenv("RL_MODE", "modern_deep")),
            algorithm=os.getenv("RL_ALGORITHM", "dqn"),
            state_representation=os.getenv("STATE_REPRESENTATION", "contextual"),

            training_enabled=os.getenv("RL_TRAINING_ENABLED", "true").lower() == "true",
            evaluation_episodes=int(os.getenv("RL_EVALUATION_EPISODES", "10")),
            save_frequency=int(os.getenv("RL_SAVE_FREQUENCY", "100")),

            safety_enabled=os.getenv("RL_SAFETY_ENABLED", "true").lower() == "true",
            max_resource_usage=float(os.getenv("SAFE_MAX_RESOURCE_USAGE", "0.8")),
            max_response_time=float(os.getenv("SAFE_MAX_RESPONSE_TIME", "5.0")),
            safety_weight=float(os.getenv("SAFE_WEIGHT", "0.5")),

            explanation_enabled=os.getenv("RL_EXPLANATION_ENABLED", "true").lower() == "true",
            explanation_methods=os.getenv("EXPLAINABLE_METHODS", "gradient,permutation").split(","),

            distributed_workers=int(os.getenv("DISTRIBUTED_WORKERS", "4")),
            parameter_server_host=os.getenv("PARAMETER_SERVER_HOST", "localhost"),
            parameter_server_port=int(os.getenv("PARAMETER_SERVER_PORT", "8000")),

            num_agents=int(os.getenv("MULTI_AGENT_COUNT", "3")),
            cooperation_mode=os.getenv("MULTI_AGENT_MODE", "cooperative"),
            communication_enabled=os.getenv("MULTI_AGENT_COMMUNICATION", "true").lower() == "true",
        )

    async def initialize(self) -> bool:
        """Initialize the RL system.
        
        Returns:
            True if initialization successful
        """
        try:
            logger.info("🚀 Initializing RL system...")

            # Initialize language model
            self.model = ChatAnthropic(
                model="claude-3-5-sonnet-20241022",
                api_key=os.getenv("ANTHROPIC_API_KEY"),
                max_tokens=4000,
            )

            # Initialize database
            db_path = os.getenv("RL_DB_PATH", "rl_agent_memory.db")
            self.db = MemoryDatabase(db_path)

            # Load MCP tools (mock for now)
            self.mcp_tools = await self._load_mcp_tools()

            # Set environment variables for RL system
            self._set_rl_environment_variables()

            # Create RL agent
            self.rl_agent = await setup_rl_agent(self.mcp_tools, self.config.mode.value)

            self.is_initialized = True
            logger.info(f"✅ RL system initialized successfully with {self.config.mode} mode")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize RL system: {e}", exc_info=True)
            return False

    def _set_rl_environment_variables(self):
        """Set environment variables for RL system."""
        env_vars = {
            "RL_MODE": self.config.mode.value,
            "RL_ALGORITHM": self.config.algorithm,
            "STATE_REPRESENTATION": self.config.state_representation,
            "RL_TRAINING_ENABLED": str(self.config.training_enabled).lower(),
            "SAFE_MAX_RESOURCE_USAGE": str(self.config.max_resource_usage),
            "SAFE_MAX_RESPONSE_TIME": str(self.config.max_response_time),
            "SAFE_WEIGHT": str(self.config.safety_weight),
            "EXPLAINABLE_METHODS": ",".join(self.config.explanation_methods),
            "DISTRIBUTED_WORKERS": str(self.config.distributed_workers),
            "MULTI_AGENT_COUNT": str(self.config.num_agents),
            "MULTI_AGENT_MODE": self.config.cooperation_mode,
            "MULTI_AGENT_COMMUNICATION": str(self.config.communication_enabled).lower(),
        }

        for key, value in env_vars.items():
            os.environ[key] = value

    async def _load_mcp_tools(self) -> List[BaseTool]:
        """Load MCP tools for the RL system.
        
        Returns:
            List of MCP tools
        """
        # Mock implementation - in real system, load actual MCP tools
        mock_tools = []

        # Create mock tools
        class MockTool(BaseTool):
            name: str = "mock_tool"
            description: str = "Mock tool for testing"

            def _run(self, query: str) -> str:
                return f"Mock result for: {query}"

            async def _arun(self, query: str) -> str:
                return f"Mock async result for: {query}"

        for i in range(3):
            tool = MockTool()
            tool.name = f"mock_tool_{i}"
            tool.description = f"Mock tool {i} for testing"
            mock_tools.append(tool)

        logger.info(f"📦 Loaded {len(mock_tools)} MCP tools")
        return mock_tools

    async def process_request(
        self,
        request: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Process a request using the RL system.
        
        Args:
            request: User request
            context: Additional context
            
        Returns:
            Processing result
        """
        if not self.is_initialized:
            await self.initialize()

        if not self.rl_agent:
            return {
                "success": False,
                "error": "RL agent not initialized",
                "response": "Sorry, the RL system is not available.",
            }

        start_time = time.time()

        try:
            # Prepare context
            if context is None:
                context = {}

            context.update({
                "timestamp": time.time(),
                "request_id": f"req_{int(time.time() * 1000)}",
                "rl_mode": self.config.mode.value,
            })

            # Process with RL agent
            if hasattr(self.rl_agent, 'process_request'):
                result = await self.rl_agent.process_request(request, [])
            elif hasattr(self.rl_agent, 'process_multi_agent_request'):
                result = await self.rl_agent.process_multi_agent_request(request, [])
            elif hasattr(self.rl_agent, 'train_distributed_episode'):
                result = await self.rl_agent.train_distributed_episode(request, [])
            elif hasattr(self.rl_agent, 'select_safe_action'):
                # For safe RL agents
                import numpy as np
                state = np.random.randn(128).astype(np.float32)
                action, safety_info = await self.rl_agent.select_safe_action(state, context)
                result = {
                    "success": True,
                    "response": f"Selected safe action {action} for: {request}",
                    "action": action,
                    "safety_info": safety_info,
                }
            elif hasattr(self.rl_agent, 'select_action_with_explanation'):
                # For explainable RL agents
                import numpy as np
                state = np.random.randn(128).astype(np.float32)
                action, explanation = await self.rl_agent.select_action_with_explanation(state, context)
                result = {
                    "success": True,
                    "response": f"Selected action {action} for: {request}",
                    "action": action,
                    "explanation": explanation.to_dict(),
                    "reasoning": explanation.get_summary(),
                }
            else:
                # Fallback for other agent types
                result = {
                    "success": True,
                    "response": f"Processed request with {self.config.mode} RL: {request}",
                    "rl_mode": self.config.mode.value,
                }

            # Update performance metrics
            response_time = time.time() - start_time
            self._update_performance_metrics(result, response_time)

            # Add metadata
            result.update({
                "response_time": response_time,
                "rl_mode": self.config.mode.value,
                "timestamp": time.time(),
            })

            return result

        except Exception as e:
            logger.error(f"❌ Error processing request with RL: {e}", exc_info=True)

            response_time = time.time() - start_time
            self._update_performance_metrics({"success": False}, response_time)

            return {
                "success": False,
                "error": str(e),
                "response": "Sorry, I encountered an error processing your request.",
                "response_time": response_time,
                "rl_mode": self.config.mode.value,
            }

    def _update_performance_metrics(self, result: Dict[str, Any], response_time: float):
        """Update performance metrics.
        
        Args:
            result: Processing result
            response_time: Response time in seconds
        """
        self.performance_metrics["total_requests"] += 1

        if result.get("success", False):
            self.performance_metrics["successful_requests"] += 1

        # Update average response time
        total = self.performance_metrics["total_requests"]
        current_avg = self.performance_metrics["average_response_time"]
        self.performance_metrics["average_response_time"] = (
            (current_avg * (total - 1) + response_time) / total
        )

        # Update average reward if available
        if "reward" in result:
            current_reward_avg = self.performance_metrics["average_reward"]
            self.performance_metrics["average_reward"] = (
                (current_reward_avg * (total - 1) + result["reward"]) / total
            )

    async def train_episode(self) -> Dict[str, Any]:
        """Train the RL agent for one episode.
        
        Returns:
            Training metrics
        """
        if not self.config.training_enabled:
            return {"error": "Training is disabled"}

        if not self.rl_agent:
            return {"error": "RL agent not initialized"}

        try:
            self.is_training = True

            # Train based on agent type
            if hasattr(self.rl_agent, 'train_episode'):
                metrics = await self.rl_agent.train_episode()
            elif hasattr(self.rl_agent, 'train_distributed_episode'):
                metrics = await self.rl_agent.train_distributed_episode("Training episode", [])
            else:
                metrics = {"message": "Training not supported for this agent type"}

            self.performance_metrics["training_episodes"] += 1

            # Save model periodically
            if (self.performance_metrics["training_episodes"] % self.config.save_frequency == 0):
                await self.save_model()

            return metrics

        except Exception as e:
            logger.error(f"❌ Error during training: {e}", exc_info=True)
            return {"error": str(e)}
        finally:
            self.is_training = False

    async def save_model(self) -> bool:
        """Save the RL model.
        
        Returns:
            True if save successful
        """
        try:
            if hasattr(self.rl_agent, 'save_model'):
                model_path = f"models/rl_model_{self.config.mode}_{int(time.time())}.pth"
                os.makedirs("models", exist_ok=True)
                self.rl_agent.save_model(model_path)
                logger.info(f"💾 Model saved to {model_path}")
                return True
            else:
                logger.warning("⚠️ Model saving not supported for this agent type")
                return False
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}", exc_info=True)
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get RL system status.
        
        Returns:
            System status
        """
        return {
            "initialized": self.is_initialized,
            "training": self.is_training,
            "mode": self.config.mode.value,
            "algorithm": self.config.algorithm,
            "performance_metrics": self.performance_metrics.copy(),
            "config": {
                "safety_enabled": self.config.safety_enabled,
                "explanation_enabled": self.config.explanation_enabled,
                "training_enabled": self.config.training_enabled,
                "distributed_workers": self.config.distributed_workers,
                "num_agents": self.config.num_agents,
            },
        }

    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report.
        
        Returns:
            Performance report
        """
        metrics = self.performance_metrics

        success_rate = 0.0
        if metrics["total_requests"] > 0:
            success_rate = metrics["successful_requests"] / metrics["total_requests"]

        return {
            "summary": {
                "total_requests": metrics["total_requests"],
                "success_rate": success_rate,
                "average_response_time": metrics["average_response_time"],
                "average_reward": metrics["average_reward"],
                "training_episodes": metrics["training_episodes"],
            },
            "rl_config": {
                "mode": self.config.mode.value,
                "algorithm": self.config.algorithm,
                "state_representation": self.config.state_representation,
            },
            "system_status": {
                "initialized": self.is_initialized,
                "training_active": self.is_training,
                "safety_enabled": self.config.safety_enabled,
                "explanation_enabled": self.config.explanation_enabled,
            },
        }


# Global RL system manager instance
_rl_manager: Optional[RLSystemManager] = None


def get_rl_manager(settings: Optional[Settings] = None) -> RLSystemManager:
    """Get the global RL system manager.
    
    Args:
        settings: Application settings
        
    Returns:
        RL system manager
    """
    global _rl_manager

    if _rl_manager is None:
        if settings is None:
            settings = Settings()
        _rl_manager = RLSystemManager(settings)

    return _rl_manager


async def initialize_rl_system(settings: Optional[Settings] = None) -> bool:
    """Initialize the global RL system.
    
    Args:
        settings: Application settings
        
    Returns:
        True if initialization successful
    """
    manager = get_rl_manager(settings)
    return await manager.initialize()
