"""
Advanced Planning System for DataMCPServerAgent.
This module implements sophisticated planning capabilities including STRIPS-like planning,
temporal planning, contingency planning, and hierarchical task networks (HTN).
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from src.memory.memory_persistence import MemoryDatabase

class ActionType(Enum):
    """Types of planning actions."""
    PRIMITIVE = "primitive"
    COMPOSITE = "composite"
    CONDITIONAL = "conditional"
    TEMPORAL = "temporal"

class PlanStatus(Enum):
    """Status of plan execution."""
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CONTINGENCY = "contingency"

@dataclass
class Condition:
    """Represents a logical condition in planning."""
    predicate: str
    parameters: List[str]
    negated: bool = False

    def __str__(self) -> str:
        pred_str = f"{self.predicate}({', '.join(self.parameters)})"
        return f"¬{pred_str}" if self.negated else pred_str

@dataclass
class Action:
    """Represents a planning action with preconditions and effects."""
    action_id: str
    name: str
    action_type: ActionType
    parameters: List[str]
    preconditions: List[Condition]
    effects: List[Condition]
    duration: float = 1.0
    cost: float = 1.0
    probability: float = 1.0

    def is_applicable(self, state: Set[str]) -> bool:
        """Check if action is applicable in given state."""
        for precond in self.preconditions:
            condition_str = str(precond)
            if precond.negated:
                if condition_str[1:] in state:  # Remove ¬ and check
                    return False
            else:
                if condition_str not in state:
                    return False
        return True

    def apply(self, state: Set[str]) -> Set[str]:
        """Apply action effects to state."""
        new_state = state.copy()

        for effect in self.effects:
            effect_str = str(effect)
            if effect.negated:
                # Remove the positive version
                positive_str = effect_str[1:]  # Remove ¬
                new_state.discard(positive_str)
            else:
                new_state.add(effect_str)

        return new_state

@dataclass
class Plan:
    """Represents a complete plan."""
    plan_id: str
    goal: str
    actions: List[Action]
    initial_state: Set[str]
    goal_state: Set[str]
    status: PlanStatus = PlanStatus.PENDING
    execution_order: List[str] = field(default_factory=list)
    contingencies: Dict[str, List[Action]] = field(default_factory=dict)
    temporal_constraints: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class HTNTask:
    """Represents a Hierarchical Task Network task."""
    task_id: str
    name: str
    is_primitive: bool
    parameters: List[str]
    preconditions: List[Condition]
    subtasks: List['HTNTask'] = field(default_factory=list)
    ordering_constraints: List[Tuple[str, str]] = field(default_factory=list)

class AdvancedPlanningEngine:
    """Advanced planning engine with multiple planning paradigms."""

    def __init__(
        self,
        model: ChatAnthropic,
        db: MemoryDatabase,
        max_plan_length: int = 20,
        planning_timeout: float = 30.0
    ):
        """Initialize the advanced planning engine.

        Args:
            model: Language model for planning
            db: Memory database for persistence
            max_plan_length: Maximum number of actions in a plan
            planning_timeout: Maximum time for planning in seconds
        """
        self.model = model
        self.db = db
        self.max_plan_length = max_plan_length
        self.planning_timeout = planning_timeout
        self.active_plans: Dict[str, Plan] = {}
        self.action_library: Dict[str, Action] = {}
        self.htn_methods: Dict[str, List[HTNTask]] = {}

        # Initialize planning prompts
        self._initialize_prompts()

        # Initialize basic action library
        self._initialize_action_library()

    def _initialize_prompts(self):
        """Initialize planning prompts."""

        # STRIPS planning prompt
        self.strips_planning_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a STRIPS-style planning agent. Your task is to create a sequence of actions to achieve a goal.

For STRIPS planning, consider:
1. Current state (what is true now)
2. Goal state (what should be true)
3. Available actions with preconditions and effects
4. Action ordering and dependencies

Each action has:
- Preconditions: What must be true to execute the action
- Effects: What becomes true/false after execution

Respond with a JSON object containing:
- "plan_actions": Ordered list of action names
- "action_details": Details for each action including parameters
- "state_progression": How state changes after each action
- "plan_rationale": Explanation of the planning strategy
- "estimated_cost": Total estimated cost/time
"""),
            HumanMessage(content="""
Goal: {goal}
Initial state: {initial_state}
Available actions: {available_actions}
Constraints: {constraints}

Create a plan to achieve the goal.
""")
        ])

        # Temporal planning prompt
        self.temporal_planning_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a temporal planning agent. Your task is to create plans with timing constraints and durations.

For temporal planning, consider:
1. Action durations and resource requirements
2. Temporal constraints (before, after, during)
3. Deadline constraints
4. Resource availability over time
5. Parallel execution opportunities

Respond with a JSON object containing:
- "temporal_plan": Actions with start times and durations
- "resource_schedule": Resource usage over time
- "critical_path": Sequence of actions that determines total time
- "parallel_opportunities": Actions that can run in parallel
- "timeline": Complete timeline of plan execution
"""),
            HumanMessage(content="""
Goal: {goal}
Available actions: {available_actions}
Temporal constraints: {temporal_constraints}
Resource constraints: {resource_constraints}
Deadline: {deadline}

Create a temporal plan.
""")
        ])

        # Contingency planning prompt
        self.contingency_planning_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a contingency planning agent. Your task is to create robust plans that handle uncertainty and failures.

For contingency planning, consider:
1. Potential failure points in the main plan
2. Alternative actions for each failure scenario
3. Monitoring points to detect failures
4. Recovery strategies and rollback procedures
5. Risk assessment and mitigation

Respond with a JSON object containing:
- "main_plan": Primary sequence of actions
- "failure_scenarios": Potential failure points and probabilities
- "contingency_actions": Alternative actions for each scenario
- "monitoring_points": Where to check for failures
- "recovery_strategies": How to recover from failures
"""),
            HumanMessage(content="""
Goal: {goal}
Main plan: {main_plan}
Risk factors: {risk_factors}
Failure probabilities: {failure_probabilities}

Create contingency plans for potential failures.
""")
        ])

    def _initialize_action_library(self):
        """Initialize basic action library."""

        # Web search action
        search_action = Action(
            action_id="web_search",
            name="web_search",
            action_type=ActionType.PRIMITIVE,
            parameters=["query"],
            preconditions=[Condition("need_information", ["query"])],
            effects=[Condition("has_information", ["query"])],
            duration=2.0,
            cost=1.0
        )

        # Data analysis action
        analyze_action = Action(
            action_id="analyze_data",
            name="analyze_data",
            action_type=ActionType.PRIMITIVE,
            parameters=["data"],
            preconditions=[Condition("has_information", ["data"])],
            effects=[Condition("has_analysis", ["data"])],
            duration=3.0,
            cost=2.0
        )

        # Report generation action
        report_action = Action(
            action_id="generate_report",
            name="generate_report",
            action_type=ActionType.PRIMITIVE,
            parameters=["analysis"],
            preconditions=[Condition("has_analysis", ["analysis"])],
            effects=[Condition("has_report", ["analysis"])],
            duration=2.0,
            cost=1.5
        )

        self.action_library = {
            "web_search": search_action,
            "analyze_data": analyze_action,
            "generate_report": report_action
        }

    async def create_strips_plan(
        self,
        goal: str,
        initial_state: Set[str],
        goal_conditions: List[Condition],
        constraints: Optional[Dict[str, Any]] = None
    ) -> Plan:
        """Create a STRIPS-style plan.

        Args:
            goal: Goal description
            initial_state: Initial state predicates
            goal_conditions: Goal conditions to achieve
            constraints: Optional planning constraints

        Returns:
            Generated plan
        """
        plan_id = str(uuid.uuid4())

        # Format available actions for the prompt
        available_actions = {}
        for action_name, action in self.action_library.items():
            available_actions[action_name] = {
                "preconditions": [str(p) for p in action.preconditions],
                "effects": [str(e) for e in action.effects],
                "duration": action.duration,
                "cost": action.cost
            }

        # Prepare input for planning
        input_values = {
            "goal": goal,
            "initial_state": list(initial_state),
            "available_actions": json.dumps(available_actions, indent=2),
            "constraints": json.dumps(constraints or {}, indent=2)
        }

        # Get plan from model
        messages = self.strips_planning_prompt.format_messages(**input_values)
        response = await self.model.ainvoke(messages)

        try:
            plan_data = json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback plan
            plan_data = {
                "plan_actions": ["web_search", "analyze_data", "generate_report"],
                "action_details": {},
                "state_progression": [],
                "plan_rationale": response.content,
                "estimated_cost": 10.0
            }

        # Create plan object
        plan_actions = []
        for action_name in plan_data["plan_actions"]:
            if action_name in self.action_library:
                action = self.action_library[action_name]
                plan_actions.append(action)

        goal_state = set(str(c) for c in goal_conditions)

        plan = Plan(
            plan_id=plan_id,
            goal=goal,
            actions=plan_actions,
            initial_state=initial_state,
            goal_state=goal_state,
            metadata={
                "planning_method": "strips",
                "estimated_cost": plan_data.get("estimated_cost", 0),
                "rationale": plan_data.get("plan_rationale", ""),
                "created_at": time.time()
            }
        )

        self.active_plans[plan_id] = plan

        # Save to database
        await self.db.save_plan(plan_id, {
            "goal": goal,
            "actions": [a.name for a in plan_actions],
            "initial_state": list(initial_state),
            "goal_state": list(goal_state),
            "metadata": plan.metadata
        })

        return plan

    async def create_temporal_plan(
        self,
        goal: str,
        available_actions: List[Action],
        temporal_constraints: List[Dict[str, Any]],
        resource_constraints: Dict[str, Any],
        deadline: Optional[float] = None
    ) -> Dict[str, Any]:
        """Create a temporal plan with timing constraints.

        Args:
            goal: Goal description
            available_actions: Available actions
            temporal_constraints: Timing constraints
            resource_constraints: Resource limitations
            deadline: Optional deadline

        Returns:
            Temporal plan
        """
        # Format actions for prompt
        actions_data = {}
        for action in available_actions:
            actions_data[action.name] = {
                "duration": action.duration,
                "cost": action.cost,
                "preconditions": [str(p) for p in action.preconditions],
                "effects": [str(e) for e in action.effects]
            }

        input_values = {
            "goal": goal,
            "available_actions": json.dumps(actions_data, indent=2),
            "temporal_constraints": json.dumps(temporal_constraints, indent=2),
            "resource_constraints": json.dumps(resource_constraints, indent=2),
            "deadline": str(deadline) if deadline else "No deadline"
        }

        messages = self.temporal_planning_prompt.format_messages(**input_values)
        response = await self.model.ainvoke(messages)

        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            return {
                "temporal_plan": [],
                "resource_schedule": {},
                "critical_path": [],
                "parallel_opportunities": [],
                "timeline": response.content
            }

    async def create_contingency_plan(
        self,
        main_plan: Plan,
        risk_factors: List[Dict[str, Any]],
        failure_probabilities: Dict[str, float]
    ) -> Dict[str, Any]:
        """Create contingency plans for potential failures.

        Args:
            main_plan: Main plan to create contingencies for
            risk_factors: Potential risk factors
            failure_probabilities: Probability of each failure

        Returns:
            Contingency plan
        """
        # Format main plan
        main_plan_data = {
            "actions": [a.name for a in main_plan.actions],
            "goal": main_plan.goal,
            "estimated_duration": sum(a.duration for a in main_plan.actions)
        }

        input_values = {
            "goal": main_plan.goal,
            "main_plan": json.dumps(main_plan_data, indent=2),
            "risk_factors": json.dumps(risk_factors, indent=2),
            "failure_probabilities": json.dumps(failure_probabilities, indent=2)
        }

        messages = self.contingency_planning_prompt.format_messages(**input_values)
        response = await self.model.ainvoke(messages)

        try:
            contingency_data = json.loads(response.content)
        except json.JSONDecodeError:
            contingency_data = {
                "main_plan": main_plan_data,
                "failure_scenarios": [],
                "contingency_actions": {},
                "monitoring_points": [],
                "recovery_strategies": [response.content]
            }

        # Update main plan with contingencies
        main_plan.contingencies = contingency_data.get("contingency_actions", {})

        return contingency_data

    async def execute_plan(
        self,
        plan_id: str,
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a plan with monitoring and contingency handling.

        Args:
            plan_id: ID of plan to execute
            execution_context: Context for execution

        Returns:
            Execution results
        """
        if plan_id not in self.active_plans:
            raise ValueError(f"Plan {plan_id} not found")

        plan = self.active_plans[plan_id]
        plan.status = PlanStatus.EXECUTING

        execution_results = []
        current_state = plan.initial_state.copy()

        for i, action in enumerate(plan.actions):
            # Check if action is applicable
            if not action.is_applicable(current_state):
                # Handle failure - check for contingencies
                if action.name in plan.contingencies:
                    contingency_actions = plan.contingencies[action.name]
                    # Execute contingency (simplified)
                    execution_results.append({
                        "action": action.name,
                        "status": "failed",
                        "contingency_used": True,
                        "contingency_actions": contingency_actions
                    })
                else:
                    plan.status = PlanStatus.FAILED
                    return {
                        "plan_id": plan_id,
                        "status": "failed",
                        "failed_at_action": i,
                        "results": execution_results
                    }
            else:
                # Execute action (simplified simulation)
                current_state = action.apply(current_state)
                execution_results.append({
                    "action": action.name,
                    "status": "completed",
                    "new_state": list(current_state)
                })

        # Check if goal is achieved
        goal_achieved = plan.goal_state.issubset(current_state)

        if goal_achieved:
            plan.status = PlanStatus.COMPLETED
        else:
            plan.status = PlanStatus.FAILED

        return {
            "plan_id": plan_id,
            "status": plan.status.value,
            "goal_achieved": goal_achieved,
            "results": execution_results,
            "final_state": list(current_state)
        }

    def validate_plan(self, plan: Plan) -> Dict[str, Any]:
        """Validate a plan for consistency and feasibility.

        Args:
            plan: Plan to validate

        Returns:
            Validation results
        """
        validation_results = {
            "is_valid": True,
            "issues": [],
            "warnings": []
        }

        # Check action sequence validity
        current_state = plan.initial_state.copy()

        for i, action in enumerate(plan.actions):
            if not action.is_applicable(current_state):
                validation_results["is_valid"] = False
                validation_results["issues"].append(
                    f"Action {action.name} at position {i} has unmet preconditions"
                )
            else:
                current_state = action.apply(current_state)

        # Check if goal is achievable
        if not plan.goal_state.issubset(current_state):
            validation_results["is_valid"] = False
            validation_results["issues"].append("Plan does not achieve the goal state")

        # Check plan length
        if len(plan.actions) > self.max_plan_length:
            validation_results["warnings"].append(
                f"Plan length ({len(plan.actions)}) exceeds recommended maximum ({self.max_plan_length})"
            )

        return validation_results
