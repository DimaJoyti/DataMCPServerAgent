"""
Example demonstrating the Advanced Agent Orchestration System.
This example shows how to use the integrated reasoning, planning, meta-reasoning, and reflection systems.
"""

import asyncio
import os

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from src.core.orchestration_main import OrchestrationCoordinator
from src.memory.memory_persistence import MemoryDatabase
from src.tools.bright_data_tools import create_bright_data_tools

# Load environment variables
load_dotenv()

async def demonstrate_orchestration():
    """Demonstrate the orchestration system with various types of requests."""

    print("🚀 Advanced Agent Orchestration System Demo")
    print("=" * 50)

    # Initialize components
    model = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.1,
    )

    db = MemoryDatabase("orchestration_demo.db")
    tools = create_bright_data_tools()

    # Create orchestration coordinator
    coordinator = OrchestrationCoordinator(model, tools, db)

    # Test scenarios
    test_scenarios = [
        {
            "name": "Complex Analysis Task",
            "request": "Analyze the current state of AI development, compare different approaches, and provide strategic recommendations for the next 5 years.",
            "expected_systems": ["reasoning", "planning", "reflection"]
        },
        {
            "name": "Multi-Step Research Task",
            "request": "Research the top 5 programming languages for 2024, analyze their strengths and weaknesses, and create a comprehensive comparison report.",
            "expected_systems": ["planning", "reasoning", "meta-reasoning"]
        },
        {
            "name": "Problem-Solving Task",
            "request": "I need to optimize my team's productivity. Help me identify bottlenecks, propose solutions, and create an implementation plan.",
            "expected_systems": ["reasoning", "planning", "reflection"]
        },
        {
            "name": "Creative Planning Task",
            "request": "Design a comprehensive marketing strategy for a new AI-powered productivity app targeting remote workers.",
            "expected_systems": ["planning", "reasoning", "meta-reasoning"]
        }
    ]

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📋 Scenario {i}: {scenario['name']}")
        print("-" * 40)
        print(f"Request: {scenario['request']}")
        print(f"Expected systems: {', '.join(scenario['expected_systems'])}")

        try:
            # Process the request
            print("\n🧠 Processing with orchestrated reasoning...")
            response = await coordinator.process_request(scenario['request'])

            print(f"\n✅ Response: {response[:200]}...")

            # Show orchestration statistics
            latest_history = coordinator.orchestration_history[-1] if coordinator.orchestration_history else None
            if latest_history:
                print("\n📊 Orchestration Stats:")
                print(f"   - Strategy used: {latest_history['strategy']}")
                print(f"   - Processing time: {latest_history['duration']:.2f}s")
                print(f"   - Reasoning chain ID: {latest_history['reasoning_chain_id']}")
                if latest_history['plan_id']:
                    print(f"   - Plan ID: {latest_history['plan_id']}")

            # Show system insights
            if coordinator.reflection_engine.reflection_sessions:
                latest_reflection = coordinator.reflection_engine.reflection_sessions[-1]
                print(f"   - Reflection insights: {len(latest_reflection.insights)}")
                print(f"   - Focus areas: {', '.join(latest_reflection.focus_areas)}")

        except Exception as e:
            print(f"❌ Error processing scenario: {str(e)}")

        print("\n" + "="*50)

    # Demonstrate meta-reasoning capabilities
    print("\n🔍 Meta-Reasoning Demonstration")
    print("-" * 40)

    try:
        # Trigger strategy selection
        strategy_rec = await coordinator.meta_reasoning_engine.select_reasoning_strategy(
            problem="How can I improve my decision-making process?",
            problem_type="self_improvement",
            confidence_requirement=0.85
        )

        print(f"Strategy recommendation: {strategy_rec['recommended_strategy']}")
        print(f"Rationale: {strategy_rec['rationale'][:150]}...")
        print(f"Expected effectiveness: {strategy_rec['expected_effectiveness']}%")

    except Exception as e:
        print(f"❌ Error in meta-reasoning demo: {str(e)}")

    # Demonstrate reflection capabilities
    print("\n🪞 Reflection System Demonstration")
    print("-" * 40)

    try:
        # Trigger comprehensive reflection
        reflection_session = await coordinator.reflection_engine.trigger_reflection(
            trigger_event="Demo session completion",
            focus_areas=["performance", "strategy", "learning"]
        )

        print(f"Reflection session ID: {reflection_session.session_id}")
        print(f"Number of insights: {len(reflection_session.insights)}")
        print(f"Focus areas: {', '.join(reflection_session.focus_areas)}")

        if reflection_session.insights:
            print("\nKey insights:")
            for insight in reflection_session.insights[:2]:  # Show first 2 insights
                print(f"   - {insight.reflection_type.value}: {insight.content[:100]}...")

        if reflection_session.improvement_plan:
            print(f"\nImprovement plan generated with {len(reflection_session.improvement_plan.get('immediate_actions', []))} immediate actions")

    except Exception as e:
        print(f"❌ Error in reflection demo: {str(e)}")

    # Show overall system statistics
    print("\n📈 Overall System Statistics")
    print("-" * 40)
    print(f"Total requests processed: {len(coordinator.orchestration_history)}")
    print(f"Active reasoning chains: {len(coordinator.active_reasoning_chains)}")
    print(f"Active plans: {len(coordinator.active_plans)}")
    print(f"Meta-decisions made: {len(coordinator.meta_reasoning_engine.meta_decisions)}")
    print(f"Reflection sessions: {len(coordinator.reflection_engine.reflection_sessions)}")

    # Show cognitive state
    cognitive_state = coordinator.meta_reasoning_engine.cognitive_state
    print("\nCurrent cognitive state:")
    print(f"   - Confidence level: {cognitive_state.confidence_level:.2f}")
    print(f"   - Cognitive load: {cognitive_state.cognitive_load:.2f}")
    print(f"   - Error rate: {cognitive_state.error_rate:.2f}")
    print(f"   - Working memory usage: {cognitive_state.working_memory_usage:.2f}")

    print("\n🎉 Orchestration demo completed!")

async def interactive_orchestration_demo():
    """Interactive demo allowing user to test the orchestration system."""

    print("🤖 Interactive Orchestration Demo")
    print("=" * 40)
    print("Enter requests to see the orchestration system in action.")
    print("Type 'quit' to exit, 'stats' for statistics, 'reflect' to trigger reflection.\n")

    # Initialize components
    model = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        temperature=0.1,
    )

    db = MemoryDatabase("interactive_orchestration.db")
    tools = create_bright_data_tools()

    # Create orchestration coordinator
    coordinator = OrchestrationCoordinator(model, tools, db)

    while True:
        try:
            user_input = input("You: ").strip()

            if user_input.lower() in ['quit', 'exit']:
                break
            elif user_input.lower() == 'stats':
                print("\n📊 System Statistics:")
                print(f"   - Requests processed: {len(coordinator.orchestration_history)}")
                print(f"   - Active reasoning chains: {len(coordinator.active_reasoning_chains)}")
                print(f"   - Meta-decisions: {len(coordinator.meta_reasoning_engine.meta_decisions)}")
                print(f"   - Reflection sessions: {len(coordinator.reflection_engine.reflection_sessions)}")

                cognitive_state = coordinator.meta_reasoning_engine.cognitive_state
                print(f"   - Confidence: {cognitive_state.confidence_level:.2f}")
                print(f"   - Cognitive load: {cognitive_state.cognitive_load:.2f}")
                continue
            elif user_input.lower() == 'reflect':
                print("🪞 Triggering reflection...")
                session = await coordinator.reflection_engine.trigger_reflection(
                    trigger_event="User-requested reflection",
                    focus_areas=["performance", "strategy"]
                )
                print(f"   - Reflection completed with {len(session.insights)} insights")
                continue

            if not user_input:
                continue

            print("\n🧠 Processing with orchestration...")

            # Process request
            response = await coordinator.process_request(user_input)

            print(f"\n🤖 Agent: {response}")

            # Show brief stats
            if coordinator.orchestration_history:
                latest = coordinator.orchestration_history[-1]
                print(f"\n📈 Strategy: {latest['strategy']}, Time: {latest['duration']:.1f}s")

            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {str(e)}\n")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        asyncio.run(interactive_orchestration_demo())
    else:
        asyncio.run(demonstrate_orchestration())
