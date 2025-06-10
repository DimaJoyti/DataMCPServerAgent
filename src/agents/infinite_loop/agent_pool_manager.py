"""
Agent Pool Manager

Manages a pool of parallel agents for executing iteration generation tasks.
Handles agent spawning, task distribution, load balancing, and resource management.
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool

from .iteration_generator import IterationGenerator
from .parallel_executor import ParallelExecutor


@dataclass
class AgentTask:
    """Represents a task for an agent to execute."""
    
    task_id: str
    iteration_number: int
    spec_analysis: Dict[str, Any]
    directory_state: Dict[str, Any]
    innovation_dimension: str
    output_dir: str
    priority: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    assigned_agent_id: Optional[str] = None
    status: str = "pending"  # pending, assigned, running, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None


@dataclass
class AgentInfo:
    """Information about an agent in the pool."""
    
    agent_id: str
    created_at: datetime
    current_task: Optional[str] = None
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    average_execution_time: float = 0.0
    is_busy: bool = False
    last_activity: Optional[datetime] = None


class AgentPoolManager:
    """
    Manages a pool of parallel agents for iteration generation.
    
    Features:
    - Dynamic agent pool sizing based on workload
    - Task queue management and distribution
    - Load balancing across available agents
    - Performance monitoring and optimization
    - Error handling and agent recovery
    - Resource usage tracking
    """
    
    def __init__(
        self,
        model: ChatAnthropic,
        tools: List[BaseTool],
        config: Any,  # InfiniteLoopConfig
    ):
        """Initialize the agent pool manager."""
        self.model = model
        self.tools = tools
        self.config = config
        self.logger = logging.getLogger("agent_pool_manager")
        
        # Agent pool
        self.agents: Dict[str, AgentInfo] = {}
        self.agent_generators: Dict[str, IterationGenerator] = {}
        self.max_agents = config.max_parallel_agents
        
        # Task management
        self.task_queue: List[AgentTask] = []
        self.active_tasks: Dict[str, AgentTask] = {}
        self.completed_tasks: List[AgentTask] = []
        self.failed_tasks: List[AgentTask] = []
        
        # Execution management
        self.parallel_executor = ParallelExecutor(config)
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        
        # Performance tracking
        self.total_tasks_processed = 0
        self.total_execution_time = 0.0
        self.pool_start_time = datetime.now()
    
    async def execute_parallel_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute multiple tasks in parallel using the agent pool.
        
        Args:
            tasks: List of task specifications
            
        Returns:
            List of task results
        """
        self.logger.info(f"Executing {len(tasks)} tasks in parallel")
        
        # Convert task specifications to AgentTask objects
        agent_tasks = []
        for task_spec in tasks:
            agent_task = AgentTask(
                task_id=str(uuid.uuid4()),
                iteration_number=task_spec["iteration_number"],
                spec_analysis=task_spec["spec_analysis"],
                directory_state=task_spec["directory_state"],
                innovation_dimension=task_spec["innovation_dimension"],
                output_dir=task_spec["output_dir"],
            )
            agent_tasks.append(agent_task)
        
        # Execute tasks
        results = await self._execute_tasks_parallel(agent_tasks)
        
        # Convert back to result format
        task_results = []
        for i, result in enumerate(results):
            task_result = {
                "task_id": agent_tasks[i].task_id,
                "iteration_number": agent_tasks[i].iteration_number,
                "success": result.get("success", False),
                "result": result.get("result"),
                "error": result.get("error"),
                "execution_time": result.get("execution_time", 0.0),
                "agent_id": result.get("agent_id"),
            }
            task_results.append(task_result)
        
        self.logger.info(f"Parallel execution completed: {len(task_results)} results")
        return task_results
    
    async def _execute_tasks_parallel(self, tasks: List[AgentTask]) -> List[Dict[str, Any]]:
        """Execute tasks in parallel using available agents."""
        # Ensure we have enough agents
        await self._ensure_agent_capacity(len(tasks))
        
        # Create execution coroutines
        execution_coroutines = []
        for task in tasks:
            coroutine = self._execute_single_task(task)
            execution_coroutines.append(coroutine)
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*execution_coroutines, return_exceptions=True)
        
        # Process results and handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Task {tasks[i].task_id} failed with exception: {result}")
                processed_results.append({
                    "success": False,
                    "error": str(result),
                    "execution_time": 0.0,
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_single_task(self, task: AgentTask) -> Dict[str, Any]:
        """Execute a single task using an available agent."""
        start_time = time.time()
        
        try:
            # Get an available agent
            agent_id = await self._get_available_agent()
            if not agent_id:
                raise RuntimeError("No available agents")
            
            # Assign task to agent
            task.assigned_agent_id = agent_id
            task.status = "assigned"
            self.active_tasks[task.task_id] = task
            
            # Mark agent as busy
            agent_info = self.agents[agent_id]
            agent_info.is_busy = True
            agent_info.current_task = task.task_id
            agent_info.last_activity = datetime.now()
            
            # Execute the task
            task.status = "running"
            generator = self.agent_generators[agent_id]
            
            result = await generator.generate_iteration(
                iteration_number=task.iteration_number,
                spec_analysis=task.spec_analysis,
                directory_state=task.directory_state,
                innovation_dimension=task.innovation_dimension,
                output_dir=task.output_dir,
            )
            
            # Update task status
            execution_time = time.time() - start_time
            task.status = "completed"
            task.result = result
            task.execution_time = execution_time
            
            # Update agent statistics
            agent_info.total_tasks += 1
            agent_info.successful_tasks += 1
            agent_info.average_execution_time = (
                (agent_info.average_execution_time * (agent_info.total_tasks - 1) + execution_time) /
                agent_info.total_tasks
            )
            
            # Clean up
            agent_info.is_busy = False
            agent_info.current_task = None
            del self.active_tasks[task.task_id]
            self.completed_tasks.append(task)
            
            self.logger.debug(f"Task {task.task_id} completed successfully in {execution_time:.2f}s")
            
            return {
                "success": True,
                "result": result,
                "execution_time": execution_time,
                "agent_id": agent_id,
            }
            
        except Exception as e:
            # Handle task failure
            execution_time = time.time() - start_time
            task.status = "failed"
            task.error = str(e)
            task.execution_time = execution_time
            
            # Update agent statistics if agent was assigned
            if task.assigned_agent_id and task.assigned_agent_id in self.agents:
                agent_info = self.agents[task.assigned_agent_id]
                agent_info.total_tasks += 1
                agent_info.failed_tasks += 1
                agent_info.is_busy = False
                agent_info.current_task = None
            
            # Clean up
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            self.failed_tasks.append(task)
            
            self.logger.error(f"Task {task.task_id} failed: {str(e)}")
            
            return {
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "agent_id": task.assigned_agent_id,
            }
    
    async def _ensure_agent_capacity(self, required_agents: int) -> None:
        """Ensure we have enough agents for the required capacity."""
        current_agents = len(self.agents)
        needed_agents = min(required_agents, self.max_agents) - current_agents
        
        if needed_agents > 0:
            self.logger.info(f"Creating {needed_agents} additional agents")
            
            for _ in range(needed_agents):
                await self._create_agent()
    
    async def _create_agent(self) -> str:
        """Create a new agent and add it to the pool."""
        agent_id = f"agent_{uuid.uuid4().hex[:8]}"
        
        # Create agent info
        agent_info = AgentInfo(
            agent_id=agent_id,
            created_at=datetime.now(),
        )
        
        # Create iteration generator for this agent
        generator = IterationGenerator(
            model=self.model,
            tools=self.tools,
            agent_id=agent_id,
        )
        
        # Add to pool
        self.agents[agent_id] = agent_info
        self.agent_generators[agent_id] = generator
        
        self.logger.debug(f"Created agent: {agent_id}")
        return agent_id
    
    async def _get_available_agent(self) -> Optional[str]:
        """Get an available agent from the pool."""
        # Find an idle agent
        for agent_id, agent_info in self.agents.items():
            if not agent_info.is_busy:
                return agent_id
        
        # If no idle agents and we can create more
        if len(self.agents) < self.max_agents:
            return await self._create_agent()
        
        # Wait for an agent to become available (with timeout)
        timeout = 30.0  # 30 seconds
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            for agent_id, agent_info in self.agents.items():
                if not agent_info.is_busy:
                    return agent_id
            
            await asyncio.sleep(0.1)  # Brief pause
        
        return None
    
    async def get_pool_statistics(self) -> Dict[str, Any]:
        """Get statistics about the agent pool."""
        total_tasks = sum(agent.total_tasks for agent in self.agents.values())
        successful_tasks = sum(agent.successful_tasks for agent in self.agents.values())
        failed_tasks = sum(agent.failed_tasks for agent in self.agents.values())
        
        success_rate = (successful_tasks / total_tasks) if total_tasks > 0 else 0.0
        
        avg_execution_time = (
            sum(agent.average_execution_time * agent.total_tasks for agent in self.agents.values()) /
            total_tasks
        ) if total_tasks > 0 else 0.0
        
        return {
            "total_agents": len(self.agents),
            "busy_agents": sum(1 for agent in self.agents.values() if agent.is_busy),
            "idle_agents": sum(1 for agent in self.agents.values() if not agent.is_busy),
            "total_tasks_processed": total_tasks,
            "successful_tasks": successful_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": success_rate,
            "average_execution_time": avg_execution_time,
            "active_tasks": len(self.active_tasks),
            "queued_tasks": len(self.task_queue),
            "uptime_seconds": (datetime.now() - self.pool_start_time).total_seconds(),
        }
    
    async def shutdown(self) -> None:
        """Shutdown the agent pool and clean up resources."""
        self.logger.info("Shutting down agent pool")
        self.is_running = False
        self.shutdown_event.set()
        
        # Wait for active tasks to complete (with timeout)
        timeout = 30.0
        start_time = time.time()
        
        while self.active_tasks and (time.time() - start_time) < timeout:
            await asyncio.sleep(0.1)
        
        # Force cleanup remaining tasks
        for task in self.active_tasks.values():
            task.status = "cancelled"
            if task.assigned_agent_id and task.assigned_agent_id in self.agents:
                self.agents[task.assigned_agent_id].is_busy = False
                self.agents[task.assigned_agent_id].current_task = None
        
        self.active_tasks.clear()
        
        # Shutdown parallel executor
        await self.parallel_executor.shutdown()
        
        self.logger.info(f"Agent pool shutdown complete. Processed {self.total_tasks_processed} total tasks")
