"""
Cloudflare Workflows with waitForEvent API and Durable Execution.
"""

import asyncio
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from secure_config import config

logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class EventType(Enum):
    USER_INPUT = "user_input"
    TOOL_RESULT = "tool_result"
    TIMEOUT = "timeout"
    EXTERNAL_API = "external_api"
    AGENT_RESPONSE = "agent_response"

@dataclass
class WorkflowEvent:
    event_id: str
    event_type: EventType
    payload: Dict[str, Any]
    timestamp: datetime
    source: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class WorkflowStep:
    step_id: str
    name: str
    function: str
    parameters: Dict[str, Any]
    status: WorkflowStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0

@dataclass
class WorkflowInstance:
    workflow_id: str
    name: str
    status: WorkflowStatus
    steps: List[WorkflowStep]
    events: List[WorkflowEvent]
    context: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    timeout_at: Optional[datetime] = None
    agent_id: Optional[str] = None
    user_id: Optional[str] = None

class DurableWorkflowExecution:
    """
    Simulates Cloudflare Durable Execution for workflows.
    In production, this would be implemented as Cloudflare Workflows.
    """

    def __init__(self):
        self.workflows: Dict[str, WorkflowInstance] = {}
        self.event_handlers: Dict[str, Callable] = {}
        self.step_functions: Dict[str, Callable] = {}
        self.pending_events: Dict[str, List[WorkflowEvent]] = {}

        # Register built-in step functions
        self._register_builtin_functions()

    def _register_builtin_functions(self):
        """Register built-in workflow step functions."""
        self.step_functions.update({
            "wait_for_event": self._wait_for_event,
            "call_tool": self._call_tool,
            "send_message": self._send_message,
            "delay": self._delay,
            "conditional": self._conditional,
            "parallel": self._parallel,
            "retry": self._retry
        })

    async def create_workflow(
        self,
        name: str,
        steps: List[Dict[str, Any]],
        context: Dict[str, Any] = None,
        timeout_seconds: int = None,
        agent_id: str = None,
        user_id: str = None
    ) -> str:
        """Create a new workflow instance."""
        workflow_id = f"workflow_{uuid.uuid4().hex[:12]}"

        # Convert step definitions to WorkflowStep objects
        workflow_steps = []
        for i, step_def in enumerate(steps):
            step = WorkflowStep(
                step_id=f"step_{i}",
                name=step_def.get("name", f"Step {i+1}"),
                function=step_def["function"],
                parameters=step_def.get("parameters", {}),
                status=WorkflowStatus.PENDING
            )
            workflow_steps.append(step)

        # Calculate timeout
        timeout_at = None
        if timeout_seconds:
            timeout_at = datetime.utcnow() + timedelta(seconds=timeout_seconds)
        elif config.workflow.timeout:
            timeout_at = datetime.utcnow() + timedelta(milliseconds=config.workflow.timeout)

        workflow = WorkflowInstance(
            workflow_id=workflow_id,
            name=name,
            status=WorkflowStatus.PENDING,
            steps=workflow_steps,
            events=[],
            context=context or {},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            timeout_at=timeout_at,
            agent_id=agent_id,
            user_id=user_id
        )

        self.workflows[workflow_id] = workflow
        self.pending_events[workflow_id] = []

        logger.info(f"Created workflow {workflow_id}: {name}")
        return workflow_id

    async def start_workflow(self, workflow_id: str) -> bool:
        """Start workflow execution."""
        if workflow_id not in self.workflows:
            return False

        workflow = self.workflows[workflow_id]
        workflow.status = WorkflowStatus.RUNNING
        workflow.updated_at = datetime.utcnow()

        logger.info(f"Starting workflow {workflow_id}")

        # Start execution in background
        asyncio.create_task(self._execute_workflow(workflow_id))
        return True

    async def _execute_workflow(self, workflow_id: str):
        """Execute workflow steps."""
        workflow = self.workflows[workflow_id]

        try:
            for step in workflow.steps:
                if workflow.status != WorkflowStatus.RUNNING:
                    break

                # Check timeout
                if workflow.timeout_at and datetime.utcnow() > workflow.timeout_at:
                    workflow.status = WorkflowStatus.FAILED
                    logger.error(f"Workflow {workflow_id} timed out")
                    break

                await self._execute_step(workflow_id, step)

                # If step is waiting, pause execution
                if step.status == WorkflowStatus.WAITING:
                    workflow.status = WorkflowStatus.WAITING
                    logger.info(f"Workflow {workflow_id} waiting at step {step.step_id}")
                    return

                # If step failed and no retry, fail workflow
                if step.status == WorkflowStatus.FAILED:
                    workflow.status = WorkflowStatus.FAILED
                    logger.error(f"Workflow {workflow_id} failed at step {step.step_id}")
                    return

            # All steps completed
            if workflow.status == WorkflowStatus.RUNNING:
                workflow.status = WorkflowStatus.COMPLETED
                workflow.updated_at = datetime.utcnow()
                logger.info(f"Workflow {workflow_id} completed successfully")

        except Exception as e:
            workflow.status = WorkflowStatus.FAILED
            workflow.updated_at = datetime.utcnow()
            logger.error(f"Workflow {workflow_id} failed with error: {e}")

    async def _execute_step(self, workflow_id: str, step: WorkflowStep):
        """Execute a single workflow step."""
        workflow = self.workflows[workflow_id]

        step.status = WorkflowStatus.RUNNING
        step.started_at = datetime.utcnow()

        try:
            # Get step function
            if step.function not in self.step_functions:
                raise ValueError(f"Unknown step function: {step.function}")

            func = self.step_functions[step.function]

            # Execute step with context
            result = await func(workflow_id, step, workflow.context)

            step.result = result
            step.status = WorkflowStatus.COMPLETED
            step.completed_at = datetime.utcnow()

            logger.info(f"Step {step.step_id} completed in workflow {workflow_id}")

        except Exception as e:
            step.error = str(e)
            step.status = WorkflowStatus.FAILED
            step.completed_at = datetime.utcnow()

            # Retry logic
            if step.retry_count < config.workflow.retry_attempts:
                step.retry_count += 1
                step.status = WorkflowStatus.PENDING
                logger.warning(f"Retrying step {step.step_id} (attempt {step.retry_count})")
                await asyncio.sleep(2 ** step.retry_count)  # Exponential backoff
                await self._execute_step(workflow_id, step)
            else:
                logger.error(f"Step {step.step_id} failed after {step.retry_count} retries: {e}")

    async def _wait_for_event(self, workflow_id: str, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Wait for a specific event (waitForEvent API)."""
        event_type = step.parameters.get("event_type")
        timeout_seconds = step.parameters.get("timeout", 300)
        filter_criteria = step.parameters.get("filter", {})

        logger.info(f"Waiting for event {event_type} in workflow {workflow_id}")

        # Check if event already exists
        matching_event = await self._find_matching_event(workflow_id, event_type, filter_criteria)
        if matching_event:
            return {"event": asdict(matching_event)}

        # Set step to waiting status
        step.status = WorkflowStatus.WAITING

        # Set timeout
        timeout_at = datetime.utcnow() + timedelta(seconds=timeout_seconds)

        # Wait for event with timeout
        while datetime.utcnow() < timeout_at:
            await asyncio.sleep(1)

            matching_event = await self._find_matching_event(workflow_id, event_type, filter_criteria)
            if matching_event:
                step.status = WorkflowStatus.RUNNING
                return {"event": asdict(matching_event)}

        # Timeout reached
        raise TimeoutError(f"Timeout waiting for event {event_type}")

    async def _find_matching_event(self, workflow_id: str, event_type: str, filter_criteria: Dict[str, Any]) -> Optional[WorkflowEvent]:
        """Find matching event in pending events."""
        if workflow_id not in self.pending_events:
            return None

        for event in self.pending_events[workflow_id]:
            if event.event_type.value == event_type:
                # Check filter criteria
                match = True
                for key, value in filter_criteria.items():
                    if key not in event.payload or event.payload[key] != value:
                        match = False
                        break

                if match:
                    # Remove event from pending
                    self.pending_events[workflow_id].remove(event)
                    return event

        return None

    async def _call_tool(self, workflow_id: str, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool/function."""
        tool_name = step.parameters.get("tool_name")
        tool_params = step.parameters.get("parameters", {})

        # Simulate tool call
        logger.info(f"Calling tool {tool_name} in workflow {workflow_id}")

        # In real implementation, this would call the actual tool
        result = {
            "tool_name": tool_name,
            "parameters": tool_params,
            "result": "Tool executed successfully",
            "timestamp": datetime.utcnow().isoformat()
        }

        return result

    async def _send_message(self, workflow_id: str, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message."""
        message = step.parameters.get("message")
        recipient = step.parameters.get("recipient")

        logger.info(f"Sending message to {recipient} in workflow {workflow_id}")

        return {
            "message": message,
            "recipient": recipient,
            "sent_at": datetime.utcnow().isoformat()
        }

    async def _delay(self, workflow_id: str, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Add delay to workflow."""
        seconds = step.parameters.get("seconds", 1)

        logger.info(f"Delaying workflow {workflow_id} for {seconds} seconds")
        await asyncio.sleep(seconds)

        return {"delayed_seconds": seconds}

    async def _conditional(self, workflow_id: str, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Conditional execution."""
        condition = step.parameters.get("condition")

        # Simple condition evaluation (in real implementation, use safe eval)
        result = eval(condition, {"context": context})

        return {"condition": condition, "result": result}

    async def _parallel(self, workflow_id: str, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Parallel execution of sub-steps."""
        sub_steps = step.parameters.get("steps", [])

        tasks = []
        for sub_step in sub_steps:
            task = asyncio.create_task(self._execute_sub_step(workflow_id, sub_step, context))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        return {"parallel_results": results}

    async def _retry(self, workflow_id: str, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Retry logic for failed operations."""
        operation = step.parameters.get("operation")
        max_attempts = step.parameters.get("max_attempts", 3)

        for attempt in range(max_attempts):
            try:
                # Execute operation
                result = await self._execute_operation(operation, context)
                return {"result": result, "attempts": attempt + 1}
            except Exception:
                if attempt == max_attempts - 1:
                    raise
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

        return {"error": "Max retry attempts reached"}

    async def _execute_sub_step(self, workflow_id: str, sub_step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a sub-step in parallel execution."""
        # Simplified sub-step execution
        return {"sub_step": sub_step, "executed_at": datetime.utcnow().isoformat()}

    async def _execute_operation(self, operation: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an operation for retry logic."""
        # Simplified operation execution
        return {"operation": operation, "executed_at": datetime.utcnow().isoformat()}

    async def send_event(self, workflow_id: str, event: WorkflowEvent) -> bool:
        """Send an event to a workflow."""
        if workflow_id not in self.workflows:
            return False

        workflow = self.workflows[workflow_id]
        workflow.events.append(event)

        # Add to pending events if workflow is waiting
        if workflow.status == WorkflowStatus.WAITING:
            self.pending_events[workflow_id].append(event)

            # Resume workflow execution
            asyncio.create_task(self._resume_workflow(workflow_id))

        logger.info(f"Event {event.event_type.value} sent to workflow {workflow_id}")
        return True

    async def _resume_workflow(self, workflow_id: str):
        """Resume workflow execution after receiving an event."""
        workflow = self.workflows[workflow_id]

        if workflow.status == WorkflowStatus.WAITING:
            workflow.status = WorkflowStatus.RUNNING
            await self._execute_workflow(workflow_id)

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a workflow."""
        if workflow_id not in self.workflows:
            return False

        workflow = self.workflows[workflow_id]
        workflow.status = WorkflowStatus.CANCELLED
        workflow.updated_at = datetime.utcnow()

        logger.info(f"Workflow {workflow_id} cancelled")
        return True

    def get_workflow(self, workflow_id: str) -> Optional[WorkflowInstance]:
        """Get workflow instance."""
        return self.workflows.get(workflow_id)

    def list_workflows(self, status: WorkflowStatus = None, user_id: str = None) -> List[WorkflowInstance]:
        """List workflows with optional filtering."""
        workflows = list(self.workflows.values())

        if status:
            workflows = [w for w in workflows if w.status == status]

        if user_id:
            workflows = [w for w in workflows if w.user_id == user_id]

        return workflows

# Global workflow execution engine
workflow_engine = DurableWorkflowExecution()
