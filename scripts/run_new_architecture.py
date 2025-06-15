#!/usr/bin/env python3
"""
Demonstration of the new DataMCPServerAgent architecture.
Shows how to work with the new code structure.
"""

import asyncio
import sys
from pathlib import Path

# Add app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import settings
from app.core.logging import get_logger, set_correlation_id
from app.domain.models.agent import Agent, AgentCapability, AgentConfiguration, AgentType
from app.domain.models.task import Task, TaskPriority, TaskType
from app.domain.services.agent_service import AgentService
from app.infrastructure.repositories.base import InMemoryRepository

logger = get_logger(__name__)


async def demonstrate_new_architecture():
    """Demonstration of the new architecture."""

    # Set correlation ID for request tracing
    set_correlation_id("demo_001")

    logger.info("🚀 Demonstration of the new DataMCPServerAgent architecture")

    # 1. Creating an agent with the new domain model
    logger.info("📦 Creating an agent...")

    # Agent configuration
    config = AgentConfiguration(
        max_concurrent_tasks=5,
        timeout_seconds=300,
        memory_limit_mb=512,
        cpu_limit_cores=1.0
    )

    # Agent capabilities
    capabilities = [
        AgentCapability(
            name="data_processing",
            version="1.0.0",
            description="Process and analyze data",
            enabled=True
        ),
        AgentCapability(
            name="email_handling",
            version="1.0.0",
            description="Send and receive emails",
            enabled=True
        )
    ]

    # Creating the agent
    agent = Agent(
        name="demo-analytics-agent",
        agent_type=AgentType.ANALYTICS,
        description="Demonstration analytical agent",
        configuration=config,
        capabilities=capabilities
    )

    logger.info(f"✅ Agent created: {agent.name} (ID: {agent.id})")

    # 2. Demonstration of domain events
    logger.info("📡 Domain events:")
    events = agent.clear_domain_events()
    for event in events:
        logger.info(f"  - {event.event_type}: {event.data}")

    # 3. Creating a task
    logger.info("📋 Creating a task...")

    task = Task(
        name="Customer data analysis",
        task_type=TaskType.DATA_ANALYSIS,
        agent_id=agent.id,
        priority=TaskPriority.HIGH,
        description="Analyze customer data from the last month",
        input_data={
            "dataset": "customers_2024_01",
            "analysis_type": "behavior_patterns",
            "output_format": "json"
        }
    )

    logger.info(f"✅ Task created: {task.name} (ID: {task.id})")

    # 4. Demonstration of business logic
    logger.info("🔄 Executing business logic...")

    # Checking agent capabilities
    if agent.has_capability("data_processing"):
        logger.info("✅ Agent has data processing capability")

        # Changing task status
        task.change_status(task.status.__class__.RUNNING)
        logger.info(f"📊 Task status changed to: {task.status}")

        # Updating progress
        from app.domain.models.task import TaskProgress
        progress = TaskProgress(
            percentage=50.0,
            current_step="Data processing",
            total_steps=4,
            completed_steps=2
        )
        task.update_progress(progress)
        logger.info(f"📈 Task progress: {progress.percentage}%")

        # Completing the task
        result_data = {
            "patterns_found": 15,
            "customer_segments": ["high_value", "regular", "new"],
            "recommendations": [
                "Increase personalization for high_value segment",
                "Improve onboarding for new customers"
            ]
        }
        task.complete_successfully(result_data)
        logger.info("✅ Task successfully completed")

    # 5. Demonstration of scaling
    logger.info("📈 Demonstration of scaling...")

    if agent.is_scalable():
        agent.scale_to(3)
        logger.info(f"🔄 Agent scaled to {agent.desired_instances} instances")

    # 6. Demonstration of repository
    logger.info("💾 Demonstration of repository operations...")

    # Creating in-memory repository for demonstration
    agent_repo = InMemoryRepository()

    # Saving the agent
    saved_agent = await agent_repo.save(agent)
    logger.info("💾 Agent saved in repository")

    # Loading the agent
    loaded_agent = await agent_repo.get_by_id(saved_agent.id)
    if loaded_agent:
        logger.info(f"📖 Agent loaded from repository: {loaded_agent.name}")

    # Searching agents by type
    agents = await agent_repo.list(agent_type=AgentType.ANALYTICS)
    logger.info(f"🔍 Found {len(agents)} analytical agents")

    # 7. Demonstration of domain service
    logger.info("🔧 Demonstration of domain service...")

    # Creating service
    agent_service = AgentService()
    agent_service.register_repository("agent", agent_repo)

    # Finding healthy agents
    healthy_agents = await agent_service.get_healthy_agents()
    logger.info(f"💚 Found {len(healthy_agents)} healthy agents")

    # 8. Configuration demonstration
    logger.info("⚙️ Configuration demonstration...")
    logger.info(f"🌍 Environment: {settings.environment}")
    logger.info(f"🐛 Debug mode: {settings.debug}")
    logger.info(f"📊 Cloudflare enabled: {settings.enable_cloudflare}")
    logger.info(f"📧 Email enabled: {settings.enable_email}")
    logger.info(f"🎥 WebRTC enabled: {settings.enable_webrtc}")

    # 9. Demonstration of validation
    logger.info("✅ Demonstration of validation...")

    try:
        # Attempt to create an agent with invalid data
        invalid_agent = Agent(
            name="",  # Empty name - should trigger an error
            agent_type=AgentType.WORKER
        )
    except Exception as e:
        logger.info(f"🚫 Validation triggered: {type(e).__name__}")

    # 10. Final statistics
    logger.info("📊 Final statistics:")
    logger.info("  - Agents created: 1")
    logger.info("  - Tasks completed: 1")
    logger.info(f"  - Domain events: {len(events)}")
    logger.info("  - Success rate: 100%")

    logger.info("🎉 Demonstration of the new architecture completed successfully!")


async def main():
    """Main function."""
    try:
        await demonstrate_new_architecture()
    except Exception as e:
        logger.error(f"❌ Error during demonstration: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
