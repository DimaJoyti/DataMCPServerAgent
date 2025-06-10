"""
Infinite Agentic Loop System

This module provides sophisticated infinite generation capabilities with parallel
agent coordination, wave-based execution, and context management.

Features:
- Specification-driven content generation
- Parallel agent coordination and task distribution
- Wave-based infinite execution with context monitoring
- Progressive sophistication across iterations
- Error recovery and quality control
- Resource optimization and state persistence
"""

from .orchestrator import InfiniteAgenticLoopOrchestrator, InfiniteLoopConfig
from .specification_parser import SpecificationParser
from .directory_analyzer import DirectoryAnalyzer
from .agent_pool_manager import AgentPoolManager
from .wave_manager import WaveManager
from .context_monitor import ContextMonitor
from .iteration_generator import IterationGenerator
from .task_assignment_engine import TaskAssignmentEngine
from .progress_tracker import ProgressTracker
from .quality_controller import QualityController
from .parallel_executor import ParallelExecutor, StatePersistence, ErrorRecoveryManager, OutputValidator

__all__ = [
    "InfiniteAgenticLoopOrchestrator",
    "InfiniteLoopConfig",
    "SpecificationParser",
    "DirectoryAnalyzer",
    "AgentPoolManager",
    "WaveManager",
    "ContextMonitor",
    "IterationGenerator",
    "TaskAssignmentEngine",
    "ProgressTracker",
    "QualityController",
    "ParallelExecutor",
    "StatePersistence",
    "ErrorRecoveryManager",
    "OutputValidator",
]
