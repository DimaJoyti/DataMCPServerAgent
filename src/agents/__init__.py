"""
Agent-related modules for DataMCPServerAgent.

Includes the Infinite Agentic Loop system for sophisticated iterative content generation.
"""

# Import existing agent architecture
try:
    from .agent_architecture import (
        AdvancedAgent,
        CoordinatorAgent,
        create_specialized_sub_agents,
    )
except ImportError:
    pass

# Import reinforcement learning agents
try:
    from .reinforcement_learning import (
        RLCoordinatorAgent,
        create_rl_agent_architecture,
    )
except ImportError:
    pass

# Import infinite loop system
from .infinite_loop import (
    AgentPoolManager,
    ContextMonitor,
    DirectoryAnalyzer,
    InfiniteAgenticLoopOrchestrator,
    SpecificationParser,
    WaveManager,
)

__all__ = [
    # Infinite Agentic Loop System
    "InfiniteAgenticLoopOrchestrator",
    "SpecificationParser",
    "DirectoryAnalyzer",
    "AgentPoolManager",
    "WaveManager",
    "ContextMonitor",
]
