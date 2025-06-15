"""
Advanced Agent Orchestration System for DataMCPServerAgent.
This module integrates advanced reasoning, meta-reasoning, planning, and reflection
systems into a cohesive orchestrated agent architecture.
"""

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.agents.advanced_planning import AdvancedPlanningEngine, Plan
from src.agents.advanced_reasoning import AdvancedReasoningEngine, ReasoningChain
from src.agents.agent_architecture import (
    AgentMemory,
    CoordinatorAgent,
    create_specialized_sub_agents,
)
from src.agents.meta_reasoning import MetaReasoningEngine
from src.agents.reflection_systems import AdvancedReflectionEngine
from src.memory.memory_persistence import MemoryDatabase
from src.tools.bright_data_tools import create_bright_data_tools
from src.utils.env_config import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrchestrationCoordinator:
    """Advanced coordinator that orchestrates multiple reasoning and planning systems."""

    def __init__(self, model: ChatAnthropic, tools: List[BaseTool], db: MemoryDatabase):
        """Initialize the orchestration coordinator.

        Args:
            model: Language model for coordination
            tools: Available tools
            db: Memory database
        """
        self.model = model
        self.tools = tools
        self.db = db
        self.memory = AgentMemory()

        # Initialize core systems
        self.reasoning_engine = AdvancedReasoningEngine(model, db)
        self.planning_engine = AdvancedPlanningEngine(model, db)
        self.meta_reasoning_engine = MetaReasoningEngine(model, db, self.reasoning_engine)
        self.reflection_engine = AdvancedReflectionEngine(model, db)

        # Create specialized sub-agents
        self.sub_agents = create_specialized_sub_agents(model, tools)

        # Initialize base coordinator
        self.base_coordinator = CoordinatorAgent(
            model=model, sub_agents=self.sub_agents, memory=self.memory
        )

        # Orchestration state
        self.active_reasoning_chains: Dict[str, ReasoningChain] = {}
        self.active_plans: Dict[str, Plan] = {}
        self.orchestration_history: List[Dict[str, Any]] = []

    async def process_request(self, request: str) -> str:
        """Process a user request using the full orchestration system.

        Args:
            request: User request

        Returns:
            Orchestrated response
        """
        start_time = time.time()

        try:
            # Step 1: Meta-reasoning for strategy selection
            logger.info("Step 1: Selecting optimal reasoning strategy")
            strategy_recommendation = await self.meta_reasoning_engine.select_reasoning_strategy(
                problem=request,
                problem_type=self._classify_problem_type(request),
                confidence_requirement=0.8,
            )

            # Step 2: Create reasoning chain based on strategy
            logger.info(
                f"Step 2: Starting reasoning chain with strategy: {strategy_recommendation['recommended_strategy']}"
            )
            reasoning_chain_id = await self.reasoning_engine.start_reasoning_chain(
                goal=request,
                initial_context={
                    "strategy": strategy_recommendation["recommended_strategy"],
                    "user_request": request,
                    "timestamp": start_time,
                },
            )

            # Step 3: Create execution plan
            logger.info("Step 3: Creating execution plan")
            plan = await self._create_execution_plan(request, strategy_recommendation)

            # Step 4: Execute orchestrated reasoning and planning
            logger.info("Step 4: Executing orchestrated reasoning")
            result = await self._execute_orchestrated_reasoning(request, reasoning_chain_id, plan)

            # Step 5: Monitor performance and adapt
            logger.info("Step 5: Monitoring performance")
            await self._monitor_and_adapt(reasoning_chain_id, result)

            # Step 6: Reflect on the process
            logger.info("Step 6: Conducting reflection")
            await self._conduct_reflection(request, result)

            # Record orchestration history
            self.orchestration_history.append(
                {
                    "request": request,
                    "strategy": strategy_recommendation["recommended_strategy"],
                    "reasoning_chain_id": reasoning_chain_id,
                    "plan_id": plan.plan_id if plan else None,
                    "result": result,
                    "duration": time.time() - start_time,
                    "timestamp": start_time,
                }
            )

            return result["response"]

        except Exception as e:
            logger.error(f"Error in orchestration: {str(e)}")

            # Trigger error reflection
            await self.reflection_engine.trigger_reflection(
                trigger_event=f"Orchestration error: {str(e)}", focus_areas=["errors", "strategy"]
            )

            # Fallback to base coordinator
            return await self.base_coordinator.process_request(request)

    async def _create_execution_plan(
        self, request: str, strategy_recommendation: Dict[str, Any]
    ) -> Optional[Plan]:
        """Create an execution plan for the request.

        Args:
            request: User request
            strategy_recommendation: Recommended strategy

        Returns:
            Execution plan
        """
        try:
            # Determine if planning is needed
            if self._requires_planning(request):
                # Extract goal conditions from request
                goal_conditions = self._extract_goal_conditions(request)

                # Create STRIPS plan
                plan = await self.planning_engine.create_strips_plan(
                    goal=request,
                    initial_state=self._get_current_state(),
                    goal_conditions=goal_conditions,
                )

                # Validate plan
                validation = self.planning_engine.validate_plan(plan)
                if not validation["is_valid"]:
                    logger.warning(f"Plan validation failed: {validation['issues']}")
                    return None

                return plan

            return None

        except Exception as e:
            logger.error(f"Error creating execution plan: {str(e)}")
            return None

    async def _execute_orchestrated_reasoning(
        self, request: str, reasoning_chain_id: str, plan: Optional[Plan]
    ) -> Dict[str, Any]:
        """Execute orchestrated reasoning combining multiple systems.

        Args:
            request: User request
            reasoning_chain_id: ID of reasoning chain
            plan: Optional execution plan

        Returns:
            Execution results
        """
        results = {
            "response": "",
            "reasoning_steps": [],
            "plan_execution": None,
            "confidence": 0.0,
            "metadata": {},
        }

        try:
            # Execute plan if available
            if plan:
                plan_result = await self.planning_engine.execute_plan(
                    plan.plan_id, {"request": request}
                )
                results["plan_execution"] = plan_result

            # Continue reasoning chain
            reasoning_steps = []
            max_steps = 10

            for step in range(max_steps):
                reasoning_step = await self.reasoning_engine.continue_reasoning(reasoning_chain_id)
                reasoning_steps.append(reasoning_step)

                # Check if reasoning is complete
                if (
                    reasoning_step.confidence > 0.9
                    or "conclusion" in reasoning_step.content.lower()
                ):
                    break

                # Monitor performance
                if step % 3 == 0:  # Every 3 steps
                    chain = self.reasoning_engine.active_chains[reasoning_chain_id]
                    performance = await self.meta_reasoning_engine.monitor_performance(chain)

                    # Adapt if performance is poor
                    if performance["performance_score"] < 60:
                        await self.meta_reasoning_engine.adapt_strategy(
                            current_performance={
                                "accuracy": performance["performance_score"] / 100
                            },
                            target_performance={"accuracy": 0.8},
                        )

            results["reasoning_steps"] = [step.__dict__ for step in reasoning_steps]

            # Generate final response using base coordinator
            base_response = await self.base_coordinator.process_request(request)
            results["response"] = base_response

            # Calculate overall confidence
            if reasoning_steps:
                avg_confidence = sum(step.confidence for step in reasoning_steps) / len(
                    reasoning_steps
                )
                results["confidence"] = avg_confidence
            else:
                results["confidence"] = 0.5

            results["metadata"] = {
                "reasoning_chain_id": reasoning_chain_id,
                "plan_id": plan.plan_id if plan else None,
                "steps_count": len(reasoning_steps),
                "execution_time": time.time(),
            }

            return results

        except Exception as e:
            logger.error(f"Error in orchestrated reasoning: {str(e)}")
            results["response"] = f"Error in reasoning: {str(e)}"
            results["confidence"] = 0.0
            return results

    async def _monitor_and_adapt(self, reasoning_chain_id: str, result: Dict[str, Any]):
        """Monitor performance and adapt strategies.

        Args:
            reasoning_chain_id: ID of reasoning chain
            result: Execution results
        """
        try:
            # Get reasoning chain
            if reasoning_chain_id in self.reasoning_engine.active_chains:
                chain = self.reasoning_engine.active_chains[reasoning_chain_id]

                # Monitor performance
                performance = await self.meta_reasoning_engine.monitor_performance(chain)

                # Check for errors in reasoning steps
                if result["reasoning_steps"]:
                    error_analysis = await self.meta_reasoning_engine.detect_errors(
                        reasoning_steps=result["reasoning_steps"],
                        context={"request": chain.goal},
                        goal=chain.goal,
                    )

                    # Log any detected errors
                    if error_analysis["errors_detected"]:
                        logger.warning(
                            f"Detected errors in reasoning: {error_analysis['errors_detected']}"
                        )

        except Exception as e:
            logger.error(f"Error in monitoring and adaptation: {str(e)}")

    async def _conduct_reflection(self, request: str, result: Dict[str, Any]):
        """Conduct reflection on the orchestration process.

        Args:
            request: Original request
            result: Execution results
        """
        try:
            # Determine reflection focus based on results
            focus_areas = ["performance"]

            if result["confidence"] < 0.7:
                focus_areas.append("strategy")

            if "error" in result["response"].lower():
                focus_areas.append("errors")

            # Trigger reflection
            reflection_session = await self.reflection_engine.trigger_reflection(
                trigger_event=f"Orchestration completed for: {request[:100]}...",
                focus_areas=focus_areas,
            )

            logger.info(f"Reflection completed with {len(reflection_session.insights)} insights")

        except Exception as e:
            logger.error(f"Error in reflection: {str(e)}")

    def _classify_problem_type(self, request: str) -> str:
        """Classify the type of problem based on request.

        Args:
            request: User request

        Returns:
            Problem type
        """
        request_lower = request.lower()

        if any(word in request_lower for word in ["analyze", "analysis", "examine"]):
            return "analytical"
        elif any(word in request_lower for word in ["plan", "strategy", "organize"]):
            return "planning"
        elif any(word in request_lower for word in ["search", "find", "lookup"]):
            return "information_retrieval"
        elif any(word in request_lower for word in ["compare", "contrast", "evaluate"]):
            return "comparative"
        else:
            return "general"

    def _requires_planning(self, request: str) -> bool:
        """Determine if request requires formal planning.

        Args:
            request: User request

        Returns:
            True if planning is needed
        """
        planning_keywords = [
            "plan",
            "strategy",
            "organize",
            "schedule",
            "coordinate",
            "multi-step",
            "complex",
            "project",
            "workflow",
        ]

        return any(keyword in request.lower() for keyword in planning_keywords)

    def _extract_goal_conditions(self, request: str) -> List:
        """Extract goal conditions from request (simplified).

        Args:
            request: User request

        Returns:
            List of goal conditions
        """
        # Simplified goal extraction
        from src.agents.advanced_planning import Condition

        if "information" in request.lower():
            return [Condition("has_information", ["query"])]
        elif "analysis" in request.lower():
            return [Condition("has_analysis", ["data"])]
        elif "report" in request.lower():
            return [Condition("has_report", ["analysis"])]
        else:
            return [Condition("task_completed", ["request"])]

    def _get_current_state(self) -> set:
        """Get current state for planning.

        Returns:
            Current state predicates
        """
        # Simplified state representation
        return {"agent_ready", "tools_available", "memory_accessible"}


async def chat_with_orchestrated_agent():
    """Main chat function for the orchestrated agent system."""

    # Initialize language model
    model = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.1,
    )

    # Initialize memory database
    db = MemoryDatabase("orchestrated_agent_memory.db")

    # Load MCP tools
    tools = []
    try:
        # Load Bright Data MCP tools
        server_params = StdioServerParameters(
            command="npx",
            args=["-y", "@brightdata/mcp-server-bright-data"],
            env={"BRIGHT_DATA_API_TOKEN": os.getenv("BRIGHT_DATA_API_TOKEN")},
        )

        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                mcp_tools = await load_mcp_tools(session)
                tools.extend(mcp_tools)

                # Create additional tools
                bright_data_tools = create_bright_data_tools()
                tools.extend(bright_data_tools)

                logger.info(f"Loaded {len(tools)} tools")

                # Initialize orchestration coordinator
                coordinator = OrchestrationCoordinator(model, tools, db)

                print("🤖 Advanced Orchestrated Agent System Ready!")
                print(
                    "This system combines advanced reasoning, planning, meta-reasoning, and reflection."
                )
                print("Type 'quit' to exit, 'help' for commands.\n")

                while True:
                    try:
                        user_input = input("You: ").strip()

                        if user_input.lower() in ["quit", "exit"]:
                            break
                        elif user_input.lower() == "help":
                            print("\nAvailable commands:")
                            print("- quit/exit: Exit the system")
                            print("- help: Show this help message")
                            print("- stats: Show orchestration statistics")
                            print("- reflect: Trigger manual reflection")
                            continue
                        elif user_input.lower() == "stats":
                            print("\nOrchestration Statistics:")
                            print(
                                f"- Total requests processed: {len(coordinator.orchestration_history)}"
                            )
                            print(
                                f"- Active reasoning chains: {len(coordinator.active_reasoning_chains)}"
                            )
                            print(f"- Active plans: {len(coordinator.active_plans)}")
                            print(
                                f"- Reflection sessions: {len(coordinator.reflection_engine.reflection_sessions)}"
                            )
                            continue
                        elif user_input.lower() == "reflect":
                            print("Triggering manual reflection...")
                            session = await coordinator.reflection_engine.trigger_reflection(
                                trigger_event="Manual reflection requested",
                                focus_areas=["performance", "strategy", "learning"],
                            )
                            print(f"Reflection completed with {len(session.insights)} insights")
                            continue

                        if not user_input:
                            continue

                        print("\n🧠 Processing with orchestrated reasoning...")

                        # Process request with orchestration
                        response = await coordinator.process_request(user_input)

                        print(f"\n🤖 Agent: {response}\n")

                    except KeyboardInterrupt:
                        print("\n\nGoodbye!")
                        break
                    except Exception as e:
                        logger.error(f"Error in chat loop: {str(e)}")
                        print(f"❌ Error: {str(e)}\n")

    except Exception as e:
        logger.error(f"Failed to initialize orchestrated agent: {str(e)}")
        print(f"❌ Failed to start orchestrated agent: {str(e)}")


if __name__ == "__main__":
    asyncio.run(chat_with_orchestrated_agent())
