"""
Test script for the Enhanced Research Assistant.
This script tests the basic functionality of the Enhanced Research Assistant.
"""

import asyncio
import os
import sys
import time
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from src.agents.enhanced_research_assistant import EnhancedResearchAssistant
from src.agents.research_rl_integration import RLEnhancedResearchAssistant
from src.memory.research_memory_persistence import ResearchMemoryDatabase

async def test_enhanced_research_assistant():
    """Test the Enhanced Research Assistant."""
    print("Testing Enhanced Research Assistant...")

    # Initialize the database
    db_path = "test_research_memory.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    db = ResearchMemoryDatabase(db_path)

    # Initialize the model
    model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

    # Initialize the research assistant
    assistant = EnhancedResearchAssistant(model=model, db_path=db_path)

    # Create a project
    project = assistant.create_project(
        name="Test Project",
        description="Test project for research assistant",
        tags=["test", "research"]
    )

    print(f"Created project: {project.name} (ID: {project.id})")

    # Test a research query
    query = "What are the benefits of exercise?"

    print(f"Researching: {query}")

    response = await assistant.invoke({
        "query": query,
        "project_id": project.id,
        "citation_format": "apa"
    })

    # Parse the response
    import json
    output = json.loads(response["output"])

    print("\n--- Research Results ---")
    print(f"Topic: {output['topic']}")
    print(f"Summary: {output['summary']}")

    print("\nSources:")
    for i, source in enumerate(output["sources"], 1):
        if isinstance(source, dict):
            title = source.get("title", f"Source {i}")
            url = source.get("url", "")

            print(f"{i}. {title}")
            if url:
                print(f"   URL: {url}")
        else:
            print(f"{i}. {source}")

    print(f"\nTools used: {', '.join(output['tools_used'])}")

    if "bibliography" in output:
        print(f"\nBibliography ({output.get('citation_format', 'apa')}):")
        print(output["bibliography"])

    # Test getting the project
    retrieved_project = assistant.get_project(project.id)
    print(f"\nRetrieved project: {retrieved_project.name} (ID: {retrieved_project.id})")
    print(f"Queries: {len(retrieved_project.queries)}")

    # Clean up
    if os.path.exists(db_path):
        os.remove(db_path)

async def test_rl_enhanced_research_assistant():
    """Test the RL-Enhanced Research Assistant."""
    print("\nTesting RL-Enhanced Research Assistant...")

    # Initialize the database
    db_path = "test_rl_research_memory.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    db = ResearchMemoryDatabase(db_path)

    # Initialize the model
    model = ChatAnthropic(model="claude-3-5-sonnet-20240620")

    # Initialize the RL-enhanced research assistant
    assistant = RLEnhancedResearchAssistant(
        model=model,
        db_path=db_path,
        learning_rate=0.1,
        discount_factor=0.9,
        exploration_rate=0.2
    )

    # Create a project
    project = assistant.create_project(
        name="RL Test Project",
        description="Test project for RL-enhanced research assistant",
        tags=["test", "research", "rl"]
    )

    print(f"Created project: {project.name} (ID: {project.id})")

    # Test a research query
    query = "What are the latest advancements in artificial intelligence?"

    print(f"Researching: {query}")

    response = await assistant.invoke({
        "query": query,
        "project_id": project.id,
        "citation_format": "apa"
    })

    # Parse the response
    import json
    output = json.loads(response["output"])

    print("\n--- Research Results ---")
    print(f"Topic: {output['topic']}")
    print(f"Summary: {output['summary']}")

    print("\nSources:")
    for i, source in enumerate(output["sources"], 1):
        if isinstance(source, dict):
            title = source.get("title", f"Source {i}")
            url = source.get("url", "")

            print(f"{i}. {title}")
            if url:
                print(f"   URL: {url}")
        else:
            print(f"{i}. {source}")

    print(f"\nTools used: {', '.join(output['tools_used'])}")

    # Test providing feedback
    print("\nProviding feedback...")

    feedback = "This research was excellent and very comprehensive!"

    learning_results = await assistant.update_from_feedback(
        query=query,
        response=output,
        feedback=feedback
    )

    print("\n--- Learning Results ---")
    print(f"State: {learning_results['state']}")
    print(f"Tools used: {', '.join(learning_results['tools_used'])}")
    print(f"Reward: {learning_results['reward']}")

    print("\nReward components:")
    for component, value in learning_results["reward_components"].items():
        print(f"- {component}: {value}")

    # Clean up
    if os.path.exists(db_path):
        os.remove(db_path)

async def main():
    """Run the tests."""
    # Load environment variables
    load_dotenv()

    # Test the Enhanced Research Assistant
    await test_enhanced_research_assistant()

    # Test the RL-Enhanced Research Assistant
    await test_rl_enhanced_research_assistant()

if __name__ == "__main__":
    asyncio.run(main())
