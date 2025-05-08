"""
Example script demonstrating the multi-agent learning system.
"""

import asyncio
import os
import sys
import time
from typing import Dict, List

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from src.agents.learning_capabilities import FeedbackCollector, LearningAgent
from src.agents.multi_agent_learning import (
    CollaborativeLearningSystem,
    KnowledgeTransferAgent,
    MultiAgentLearningSystem
)
from src.memory.collaborative_knowledge import CollaborativeKnowledgeBase
from src.memory.memory_persistence import MemoryDatabase
from src.utils.agent_metrics import AgentPerformanceTracker, MultiAgentPerformanceAnalyzer

load_dotenv()

model = ChatAnthropic(model="claude-3-5-sonnet-20240620")


async def run_example():
    """Run the multi-agent learning example."""
    print("Initializing multi-agent learning system...")
    
    # Initialize memory database
    db = MemoryDatabase("example_multi_agent_memory.db")
    
    # Initialize collaborative knowledge base
    knowledge_base = CollaborativeKnowledgeBase(db)
    
    # Initialize performance tracking
    performance_tracker = AgentPerformanceTracker(db)
    performance_analyzer = MultiAgentPerformanceAnalyzer(db, performance_tracker)
    
    # Initialize feedback collector
    feedback_collector = FeedbackCollector(model, db)
    
    # Initialize knowledge transfer agent
    knowledge_transfer_agent = KnowledgeTransferAgent(model, db)
    
    # Create learning agents
    learning_agents = {
        "search_agent": LearningAgent("search_agent", model, db, feedback_collector),
        "scraping_agent": LearningAgent("scraping_agent", model, db, feedback_collector),
        "analysis_agent": LearningAgent("analysis_agent", model, db, feedback_collector)
    }
    
    # Initialize collaborative learning system
    collaborative_learning = CollaborativeLearningSystem(
        model, db, learning_agents, knowledge_transfer_agent
    )
    
    # Initialize multi-agent learning system
    multi_agent_learning = MultiAgentLearningSystem(
        model, db, learning_agents, feedback_collector
    )
    
    print("Multi-agent learning system initialized.")
    
    # Simulate agent executions with different success rates
    print("\nSimulating agent executions...")
    
    # Simulate search agent executions
    for i in range(10):
        success = i % 3 != 0  # 7/10 success rate
        performance_tracker.record_agent_execution(
            "search_agent", success, 1.5 + (0.2 * i % 3)
        )
    
    # Simulate scraping agent executions
    for i in range(10):
        success = i % 4 != 0  # 3/4 success rate
        performance_tracker.record_agent_execution(
            "scraping_agent", success, 2.0 + (0.3 * i % 4)
        )
    
    # Simulate analysis agent executions
    for i in range(10):
        success = i % 2 == 0  # 1/2 success rate
        performance_tracker.record_agent_execution(
            "analysis_agent", success, 3.0 + (0.5 * i % 2)
        )
    
    # Record collaborative metrics
    performance_tracker.record_collaborative_metric(
        "success_rate", 0.8, ["search_agent", "scraping_agent"]
    )
    performance_tracker.record_collaborative_metric(
        "execution_time", 4.5, ["search_agent", "scraping_agent"]
    )
    
    performance_tracker.record_collaborative_metric(
        "success_rate", 0.7, ["search_agent", "analysis_agent"]
    )
    performance_tracker.record_collaborative_metric(
        "execution_time", 5.2, ["search_agent", "analysis_agent"]
    )
    
    performance_tracker.record_collaborative_metric(
        "success_rate", 0.9, ["scraping_agent", "analysis_agent"]
    )
    performance_tracker.record_collaborative_metric(
        "execution_time", 6.1, ["scraping_agent", "analysis_agent"]
    )
    
    # Simulate knowledge extraction and sharing
    print("\nSimulating knowledge extraction and sharing...")
    
    # Extract knowledge from search agent
    search_knowledge = await knowledge_transfer_agent.extract_knowledge(
        "The search for 'multi-agent systems' returned 15 relevant results. "
        "The most relevant sources were academic papers from the last 5 years. "
        "Key topics included collaborative learning, knowledge sharing, and performance optimization.",
        "Search for information about multi-agent systems"
    )
    
    # Store knowledge in the database
    search_knowledge_id = knowledge_base.store_knowledge(search_knowledge, "search_agent")
    
    # Extract knowledge from scraping agent
    scraping_knowledge = await knowledge_transfer_agent.extract_knowledge(
        "The scraping of the research papers revealed that successful multi-agent systems "
        "typically use a centralized coordinator for task allocation. The most effective "
        "systems implement knowledge sharing mechanisms and performance tracking.",
        "Scrape research papers about multi-agent systems"
    )
    
    # Store knowledge in the database
    scraping_knowledge_id = knowledge_base.store_knowledge(scraping_knowledge, "scraping_agent")
    
    # Share knowledge between agents
    print("\nSharing knowledge between agents...")
    
    # Share search knowledge with analysis agent
    transfer_result = await collaborative_learning.share_knowledge(
        "search_agent", "analysis_agent", search_knowledge
    )
    
    # Record the knowledge transfer
    knowledge_base.record_knowledge_transfer(
        "search_agent", "analysis_agent", search_knowledge_id, True
    )
    
    # Assign knowledge to agents
    knowledge_base.assign_knowledge_to_agent("search_agent", search_knowledge_id, 0.9)
    knowledge_base.assign_knowledge_to_agent("analysis_agent", search_knowledge_id, 0.6)
    
    # Share scraping knowledge with search agent
    transfer_result = await collaborative_learning.share_knowledge(
        "scraping_agent", "search_agent", scraping_knowledge
    )
    
    # Record the knowledge transfer
    knowledge_base.record_knowledge_transfer(
        "scraping_agent", "search_agent", scraping_knowledge_id, True
    )
    
    # Assign knowledge to agents
    knowledge_base.assign_knowledge_to_agent("scraping_agent", scraping_knowledge_id, 0.8)
    knowledge_base.assign_knowledge_to_agent("search_agent", scraping_knowledge_id, 0.5)
    
    # Analyze agent performance
    print("\nAnalyzing agent performance...")
    
    # Get individual agent performance
    search_performance = performance_tracker.get_agent_performance_summary("search_agent")
    scraping_performance = performance_tracker.get_agent_performance_summary("scraping_agent")
    analysis_performance = performance_tracker.get_agent_performance_summary("analysis_agent")
    
    print(f"Search Agent Success Rate: {search_performance['success_rate']:.2f}")
    print(f"Scraping Agent Success Rate: {scraping_performance['success_rate']:.2f}")
    print(f"Analysis Agent Success Rate: {analysis_performance['success_rate']:.2f}")
    
    # Compare agents
    comparison = performance_tracker.compare_agents(
        ["search_agent", "scraping_agent", "analysis_agent"]
    )
    
    print("\nRelative Performance:")
    for agent, score in comparison["relative_performance"].items():
        print(f"{agent}: {score:.2f}")
    
    # Analyze agent synergy
    print("\nAnalyzing agent synergy...")
    
    # Analyze synergy between search and scraping agents
    synergy = performance_analyzer.analyze_agent_synergy(
        ["search_agent", "scraping_agent"]
    )
    
    print("Search + Scraping Synergy:")
    print(f"Success Rate Synergy: {synergy['synergy_metrics']['success_rate_synergy']:.2f}")
    print(f"Execution Time Synergy: {synergy['synergy_metrics']['execution_time_synergy']:.2f}")
    
    # Identify optimal agent combinations
    print("\nIdentifying optimal agent combinations...")
    
    optimal_combinations = performance_analyzer.identify_optimal_agent_combinations(
        ["search_agent", "scraping_agent", "analysis_agent"]
    )
    
    print("Top Agent Combinations:")
    for i, combo in enumerate(optimal_combinations[:2]):
        print(f"{i+1}. {' + '.join(combo['agents'])}: {combo['synergy_score']:.2f}")
    
    # Execute a learning cycle
    print("\nExecuting a learning cycle...")
    
    learning_results = await multi_agent_learning.execute_learning_cycle()
    
    print("Learning cycle completed.")
    
    # Simulate collaborative problem-solving
    print("\nSimulating collaborative problem-solving...")
    
    agent_results = {
        "search_agent": {
            "success": True,
            "response": "Found 10 relevant research papers on multi-agent learning systems. "
                       "The most cited papers focus on knowledge sharing mechanisms and "
                       "collaborative problem-solving approaches."
        },
        "scraping_agent": {
            "success": True,
            "response": "Extracted key information from the research papers. The most effective "
                       "multi-agent systems use a combination of centralized coordination and "
                       "decentralized execution. Knowledge transfer is typically implemented "
                       "using a shared knowledge base."
        },
        "analysis_agent": {
            "success": True,
            "response": "Analysis of the extracted information reveals that successful multi-agent "
                       "learning systems typically implement three key components: knowledge "
                       "extraction, knowledge transfer, and collaborative problem-solving. "
                       "Performance tracking is also critical for identifying optimal agent "
                       "combinations."
        }
    }
    
    collaborative_solution = await multi_agent_learning.process_request_collaboratively(
        "Research and analyze multi-agent learning systems",
        agent_results
    )
    
    print("\nCollaborative Solution:")
    print(collaborative_solution["collaborative_solution"][:300] + "...")
    
    # Get agent knowledge statistics
    print("\nAgent Knowledge Statistics:")
    
    search_stats = knowledge_base.get_agent_knowledge_stats("search_agent")
    scraping_stats = knowledge_base.get_agent_knowledge_stats("scraping_agent")
    analysis_stats = knowledge_base.get_agent_knowledge_stats("analysis_agent")
    
    print(f"Search Agent Knowledge: {search_stats['total_knowledge']} items, "
          f"Avg Proficiency: {search_stats['average_proficiency']:.2f}")
    print(f"Scraping Agent Knowledge: {scraping_stats['total_knowledge']} items, "
          f"Avg Proficiency: {scraping_stats['average_proficiency']:.2f}")
    print(f"Analysis Agent Knowledge: {analysis_stats['total_knowledge']} items, "
          f"Avg Proficiency: {analysis_stats['average_proficiency']:.2f}")
    
    print("\nMulti-agent learning example completed.")


if __name__ == "__main__":
    asyncio.run(run_example())