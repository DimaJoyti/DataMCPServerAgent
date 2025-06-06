"""
Semantic Coordinator

Orchestrates multiple semantic agents, manages task distribution,
and ensures optimal resource utilization and performance.
"""

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from .base_semantic_agent import BaseSemanticAgent, SemanticContext
from .communication import (
    AgentCommunicationHub,
    AgentMessage,
    MessageBus,
    MessageHandler,
    MessageType,
)


@dataclass
class TaskAssignment:
    """Task assignment information."""
    
    task_id: str
    agent_id: str
    task_description: str
    priority: int = 1
    estimated_duration: Optional[timedelta] = None
    assigned_at: datetime = field(default_factory=datetime.now)
    status: str = "assigned"  # assigned, running, completed, failed
    result: Optional[Dict[str, Any]] = None


@dataclass
class AgentCapability:
    """Agent capability description."""
    
    capability_name: str
    proficiency_score: float  # 0.0 to 1.0
    specialization_areas: List[str] = field(default_factory=list)
    performance_history: List[float] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


class SemanticCoordinator:
    """
    Coordinates multiple semantic agents for optimal task execution.
    
    Features:
    - Intelligent task routing and load balancing
    - Agent capability assessment and matching
    - Performance monitoring and optimization
    - Collaborative task execution
    - Resource management and scaling
    """
    
    def __init__(
        self,
        coordinator_id: Optional[str] = None,
        message_bus: Optional[MessageBus] = None,
    ):
        """Initialize the semantic coordinator."""
        self.coordinator_id = coordinator_id or str(uuid.uuid4())
        self.message_bus = message_bus or MessageBus()
        self.communication_hub = AgentCommunicationHub(self.message_bus)
        
        # Agent management
        self.registered_agents: Dict[str, BaseSemanticAgent] = {}
        self.agent_capabilities: Dict[str, List[AgentCapability]] = {}
        self.agent_workloads: Dict[str, int] = {}
        self.agent_performance: Dict[str, Dict[str, float]] = {}
        
        # Task management
        self.active_tasks: Dict[str, TaskAssignment] = {}
        self.task_queue: List[Dict[str, Any]] = []
        self.completed_tasks: List[TaskAssignment] = []
        
        # Configuration
        self.max_concurrent_tasks_per_agent = 3
        self.task_timeout = timedelta(minutes=10)
        self.performance_window = timedelta(hours=24)
        
        # State
        self.is_running = False
        self.logger = logging.getLogger("semantic_coordinator")
        
    async def initialize(self) -> None:
        """Initialize the coordinator."""
        self.logger.info("Initializing semantic coordinator")
        
        # Register message handlers
        await self._register_message_handlers()
        
        # Start background tasks
        self.is_running = True
        asyncio.create_task(self._monitor_tasks())
        asyncio.create_task(self._update_agent_metrics())
        
        self.logger.info("Semantic coordinator initialized")
        
    async def shutdown(self) -> None:
        """Shutdown the coordinator."""
        self.logger.info("Shutting down semantic coordinator")
        
        self.is_running = False
        
        # Cancel active tasks
        for task_id in list(self.active_tasks.keys()):
            await self.cancel_task(task_id)
            
        # Shutdown registered agents
        for agent in self.registered_agents.values():
            await agent.shutdown()
            
        self.logger.info("Semantic coordinator shutdown complete")
        
    async def register_agent(self, agent: BaseSemanticAgent) -> None:
        """Register a semantic agent with the coordinator."""
        agent_id = agent.config.agent_id
        
        self.registered_agents[agent_id] = agent
        self.agent_workloads[agent_id] = 0
        self.agent_performance[agent_id] = {}
        
        # Initialize capabilities
        capabilities = []
        for capability_name in agent.config.capabilities:
            capabilities.append(AgentCapability(
                capability_name=capability_name,
                proficiency_score=0.8,  # Default score
                specialization_areas=agent.config.tools,
            ))
            
        self.agent_capabilities[agent_id] = capabilities
        
        # Subscribe agent to coordination messages
        handler = MessageHandler(
            handler_func=agent._handle_task_request,
            message_types={MessageType.TASK_REQUEST},
        )
        await self.message_bus.subscribe(agent_id, handler)
        
        self.logger.info(f"Registered agent: {agent.config.name} ({agent_id})")
        
    async def unregister_agent(self, agent_id: str) -> None:
        """Unregister an agent from the coordinator."""
        if agent_id in self.registered_agents:
            # Cancel agent's active tasks
            agent_tasks = [
                task for task in self.active_tasks.values()
                if task.agent_id == agent_id
            ]
            
            for task in agent_tasks:
                await self.cancel_task(task.task_id)
                
            # Remove agent
            del self.registered_agents[agent_id]
            del self.agent_workloads[agent_id]
            del self.agent_performance[agent_id]
            del self.agent_capabilities[agent_id]
            
            await self.message_bus.unsubscribe(agent_id)
            
            self.logger.info(f"Unregistered agent: {agent_id}")
            
    async def execute_task(
        self,
        task_description: str,
        context: Optional[SemanticContext] = None,
        required_capabilities: Optional[List[str]] = None,
        priority: int = 1,
        collaborative: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute a task using the most suitable agent(s).
        
        Args:
            task_description: Description of the task to execute
            context: Optional semantic context
            required_capabilities: Required agent capabilities
            priority: Task priority (higher = more urgent)
            collaborative: Whether to use multiple agents
            
        Returns:
            Task execution result
        """
        task_id = str(uuid.uuid4())
        
        self.logger.info(f"Executing task {task_id}: {task_description}")
        
        if collaborative:
            return await self._execute_collaborative_task(
                task_id, task_description, context, required_capabilities, priority
            )
        else:
            return await self._execute_single_agent_task(
                task_id, task_description, context, required_capabilities, priority
            )
            
    async def _execute_single_agent_task(
        self,
        task_id: str,
        task_description: str,
        context: Optional[SemanticContext],
        required_capabilities: Optional[List[str]],
        priority: int,
    ) -> Dict[str, Any]:
        """Execute a task using a single agent."""
        # Find the best agent for the task
        best_agent_id = await self._select_best_agent(
            required_capabilities or [],
            task_description,
        )
        
        if not best_agent_id:
            return {
                "success": False,
                "error": "No suitable agent found for the task",
                "task_id": task_id,
            }
            
        # Create task assignment
        assignment = TaskAssignment(
            task_id=task_id,
            agent_id=best_agent_id,
            task_description=task_description,
            priority=priority,
            status="assigned",
        )
        
        self.active_tasks[task_id] = assignment
        self.agent_workloads[best_agent_id] += 1
        
        try:
            # Execute the task
            agent = self.registered_agents[best_agent_id]
            assignment.status = "running"
            
            result = await agent.process_request(task_description, context)
            
            assignment.status = "completed"
            assignment.result = result
            
            # Update performance metrics
            await self._update_agent_performance(best_agent_id, True, task_description)
            
            return {
                "success": True,
                "result": result,
                "task_id": task_id,
                "agent_id": best_agent_id,
            }
            
        except Exception as e:
            assignment.status = "failed"
            assignment.result = {"error": str(e)}
            
            # Update performance metrics
            await self._update_agent_performance(best_agent_id, False, task_description)
            
            self.logger.error(f"Task {task_id} failed: {e}")
            
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id,
                "agent_id": best_agent_id,
            }
            
        finally:
            # Cleanup
            self.agent_workloads[best_agent_id] -= 1
            self.completed_tasks.append(assignment)
            del self.active_tasks[task_id]
            
    async def _execute_collaborative_task(
        self,
        task_id: str,
        task_description: str,
        context: Optional[SemanticContext],
        required_capabilities: Optional[List[str]],
        priority: int,
    ) -> Dict[str, Any]:
        """Execute a task using multiple collaborating agents."""
        # Break down the task into subtasks
        subtasks = await self._decompose_task(task_description, required_capabilities or [])
        
        if not subtasks:
            return await self._execute_single_agent_task(
                task_id, task_description, context, required_capabilities, priority
            )
            
        # Execute subtasks in parallel
        subtask_results = []
        
        for i, subtask in enumerate(subtasks):
            subtask_id = f"{task_id}_sub_{i}"
            
            result = await self._execute_single_agent_task(
                subtask_id,
                subtask["description"],
                context,
                subtask.get("capabilities"),
                priority,
            )
            
            subtask_results.append(result)
            
        # Combine results
        combined_result = await self._combine_subtask_results(
            task_description, subtask_results
        )
        
        return {
            "success": all(r.get("success", False) for r in subtask_results),
            "result": combined_result,
            "task_id": task_id,
            "subtask_results": subtask_results,
            "collaborative": True,
        }
        
    async def _select_best_agent(
        self,
        required_capabilities: List[str],
        task_description: str,
    ) -> Optional[str]:
        """Select the best agent for a task based on capabilities and workload."""
        best_agent_id = None
        best_score = -1.0
        
        for agent_id, agent in self.registered_agents.items():
            # Check if agent is available
            if self.agent_workloads[agent_id] >= self.max_concurrent_tasks_per_agent:
                continue
                
            # Calculate capability score
            capability_score = self._calculate_capability_score(
                agent_id, required_capabilities
            )
            
            # Calculate workload penalty
            workload_penalty = self.agent_workloads[agent_id] * 0.2
            
            # Calculate performance bonus
            performance_bonus = self._get_agent_performance_score(agent_id)
            
            # Calculate total score
            total_score = capability_score + performance_bonus - workload_penalty
            
            if total_score > best_score:
                best_score = total_score
                best_agent_id = agent_id
                
        return best_agent_id
        
    def _calculate_capability_score(
        self,
        agent_id: str,
        required_capabilities: List[str],
    ) -> float:
        """Calculate how well an agent matches required capabilities."""
        if not required_capabilities:
            return 0.5  # Neutral score if no specific requirements
            
        agent_capabilities = self.agent_capabilities.get(agent_id, [])
        
        if not agent_capabilities:
            return 0.0
            
        total_score = 0.0
        matched_capabilities = 0
        
        for required_cap in required_capabilities:
            best_match_score = 0.0
            
            for agent_cap in agent_capabilities:
                if agent_cap.capability_name == required_cap:
                    best_match_score = agent_cap.proficiency_score
                    break
                elif required_cap in agent_cap.specialization_areas:
                    best_match_score = max(best_match_score, agent_cap.proficiency_score * 0.8)
                    
            if best_match_score > 0:
                total_score += best_match_score
                matched_capabilities += 1
                
        if matched_capabilities == 0:
            return 0.0
            
        return total_score / len(required_capabilities)
        
    def _get_agent_performance_score(self, agent_id: str) -> float:
        """Get the recent performance score for an agent."""
        performance_data = self.agent_performance.get(agent_id, {})
        
        if not performance_data:
            return 0.0
            
        # Calculate weighted average of recent performance
        success_rate = performance_data.get("success_rate", 0.5)
        avg_response_time = performance_data.get("avg_response_time", 10.0)
        
        # Normalize response time (lower is better)
        response_time_score = max(0, 1.0 - (avg_response_time / 30.0))
        
        return (success_rate * 0.7) + (response_time_score * 0.3)
        
    async def _decompose_task(
        self,
        task_description: str,
        required_capabilities: List[str],
    ) -> List[Dict[str, Any]]:
        """Decompose a complex task into subtasks."""
        # This is a simplified implementation
        # In practice, this would use more sophisticated task decomposition
        
        if len(required_capabilities) <= 1:
            return []
            
        subtasks = []
        for capability in required_capabilities:
            subtasks.append({
                "description": f"Handle {capability} aspect of: {task_description}",
                "capabilities": [capability],
            })
            
        return subtasks
        
    async def _combine_subtask_results(
        self,
        original_task: str,
        subtask_results: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Combine results from multiple subtasks."""
        # This is a simplified implementation
        # In practice, this would use more sophisticated result combination
        
        combined_data = {}
        errors = []
        
        for result in subtask_results:
            if result.get("success"):
                if "result" in result:
                    combined_data.update(result["result"])
            else:
                errors.append(result.get("error", "Unknown error"))
                
        return {
            "combined_data": combined_data,
            "errors": errors,
            "subtask_count": len(subtask_results),
            "success_count": sum(1 for r in subtask_results if r.get("success")),
        }
        
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel an active task."""
        if task_id not in self.active_tasks:
            return False
            
        assignment = self.active_tasks[task_id]
        assignment.status = "cancelled"
        
        # Decrease agent workload
        self.agent_workloads[assignment.agent_id] -= 1
        
        # Move to completed tasks
        self.completed_tasks.append(assignment)
        del self.active_tasks[task_id]
        
        self.logger.info(f"Cancelled task {task_id}")
        return True
        
    async def _register_message_handlers(self) -> None:
        """Register message handlers for coordination."""
        # This would register handlers for coordination messages
        pass
        
    async def _monitor_tasks(self) -> None:
        """Monitor active tasks for timeouts and issues."""
        while self.is_running:
            try:
                current_time = datetime.now()
                
                for task_id, assignment in list(self.active_tasks.items()):
                    # Check for timeout
                    if (current_time - assignment.assigned_at) > self.task_timeout:
                        self.logger.warning(f"Task {task_id} timed out")
                        await self.cancel_task(task_id)
                        
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Error in task monitoring: {e}")
                await asyncio.sleep(10)
                
    async def _update_agent_metrics(self) -> None:
        """Update agent performance metrics."""
        while self.is_running:
            try:
                # Update metrics for all agents
                for agent_id in self.registered_agents:
                    await self._calculate_agent_metrics(agent_id)
                    
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Error updating agent metrics: {e}")
                await asyncio.sleep(60)
                
    async def _calculate_agent_metrics(self, agent_id: str) -> None:
        """Calculate performance metrics for an agent."""
        # Get recent completed tasks for this agent
        cutoff_time = datetime.now() - self.performance_window
        
        recent_tasks = [
            task for task in self.completed_tasks
            if task.agent_id == agent_id and task.assigned_at > cutoff_time
        ]
        
        if not recent_tasks:
            return
            
        # Calculate success rate
        successful_tasks = [task for task in recent_tasks if task.status == "completed"]
        success_rate = len(successful_tasks) / len(recent_tasks)
        
        # Calculate average response time (simplified)
        avg_response_time = 5.0  # Placeholder
        
        self.agent_performance[agent_id] = {
            "success_rate": success_rate,
            "avg_response_time": avg_response_time,
            "total_tasks": len(recent_tasks),
            "last_updated": datetime.now(),
        }
        
    async def _update_agent_performance(
        self,
        agent_id: str,
        success: bool,
        task_description: str,
    ) -> None:
        """Update agent performance after task completion."""
        # This would update detailed performance metrics
        # For now, we'll just log the result
        self.logger.info(
            f"Agent {agent_id} task {'succeeded' if success else 'failed'}: {task_description}"
        )
        
    def get_coordinator_status(self) -> Dict[str, Any]:
        """Get the current status of the coordinator."""
        return {
            "coordinator_id": self.coordinator_id,
            "is_running": self.is_running,
            "registered_agents": len(self.registered_agents),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "agent_workloads": self.agent_workloads.copy(),
        }
