"""
Multi-agent learning module for DataMCPServerAgent.
This module provides mechanisms for agents to learn from each other and collaborate.
"""

import json
from typing import Any, Dict

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from src.agents.learning_capabilities import FeedbackCollector, LearningAgent
from src.memory.memory_persistence import MemoryDatabase


class KnowledgeTransferAgent:
    """Agent responsible for transferring knowledge between specialized agents."""

    def __init__(self, model: ChatAnthropic, db: MemoryDatabase):
        """Initialize the knowledge transfer agent.

        Args:
            model: Language model to use
            db: Memory database for persistence
        """
        self.model = model
        self.db = db

        # Create the knowledge extraction prompt
        self.extraction_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""You are a knowledge extraction agent responsible for identifying valuable knowledge from agent interactions.
Your job is to analyze agent responses and extract reusable knowledge that could benefit other agents.

For each response, you should:
1. Identify key insights and knowledge
2. Extract patterns in successful problem-solving
3. Recognize effective strategies
4. Formalize the knowledge in a structured format
5. Assess the confidence and domain applicability

Respond with a JSON object containing:
- "knowledge_items": Array of knowledge items
- "confidence": Confidence score for each item (0-100)
- "domain": Domain or context where this knowledge applies
- "applicability": Array of agent types that could benefit from this knowledge
- "prerequisites": Any prerequisites for applying this knowledge
"""
                ),
                HumanMessage(
                    content="""
Agent response:
{response}

Task context:
{context}

Extract valuable knowledge from this interaction.
"""
                ),
            ]
        )

        # Create the knowledge integration prompt
        self.integration_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""You are a knowledge integration agent responsible for adapting knowledge for use by other agents.
Your job is to take knowledge extracted from one agent and adapt it for use by another agent with different capabilities.

For each knowledge item, you should:
1. Analyze the knowledge and its original context
2. Consider the target agent's capabilities and limitations
3. Adapt the knowledge to fit the target agent's context
4. Provide clear instructions for applying the knowledge
5. Identify potential challenges in knowledge transfer

Respond with a JSON object containing:
- "adapted_knowledge": The adapted knowledge
- "application_instructions": Instructions for applying the knowledge
- "potential_challenges": Potential challenges in applying the knowledge
- "expected_benefits": Expected benefits from applying the knowledge
"""
                ),
                HumanMessage(
                    content="""
Original knowledge:
{knowledge}

Source agent: {source_agent}
Target agent: {target_agent}

Adapt this knowledge for the target agent.
"""
                ),
            ]
        )

    async def extract_knowledge(self, response: str, context: str) -> Dict[str, Any]:
        """Extract valuable knowledge from an agent's response.

        Args:
            response: Agent's response
            context: Context of the task

        Returns:
            Extracted knowledge
        """
        # Prepare the input for the extraction prompt
        input_values = {"response": response, "context": context}

        # Get the extracted knowledge from the model
        messages = self.extraction_prompt.format_messages(**input_values)
        response = await self.model.ainvoke(messages)

        try:
            # Parse the response as JSON
            knowledge = json.loads(response.content)

            # Store the knowledge in the database
            self.db.store_knowledge(knowledge)

            return knowledge
        except json.JSONDecodeError:
            # If the response is not valid JSON, extract it manually
            return {
                "knowledge_items": [response.content],
                "confidence": 50,
                "domain": "general",
                "applicability": ["all"],
                "prerequisites": [],
            }

    async def adapt_knowledge(
        self, knowledge: Dict[str, Any], source_agent: str, target_agent: str
    ) -> Dict[str, Any]:
        """Adapt knowledge from one agent for use by another.

        Args:
            knowledge: Knowledge to adapt
            source_agent: Source agent type
            target_agent: Target agent type

        Returns:
            Adapted knowledge
        """
        # Prepare the input for the integration prompt
        input_values = {
            "knowledge": json.dumps(knowledge, indent=2),
            "source_agent": source_agent,
            "target_agent": target_agent,
        }

        # Get the adapted knowledge from the model
        messages = self.integration_prompt.format_messages(**input_values)
        response = await self.model.ainvoke(messages)

        try:
            # Parse the response as JSON
            adapted_knowledge = json.loads(response.content)
            return adapted_knowledge
        except json.JSONDecodeError:
            # If the response is not valid JSON, extract it manually
            return {
                "adapted_knowledge": response.content,
                "application_instructions": "Apply this knowledge as appropriate.",
                "potential_challenges": ["Format conversion issues"],
                "expected_benefits": ["Improved performance"],
            }


class CollaborativeLearningSystem:
    """System for collaborative learning between multiple agents."""

    def __init__(
        self,
        model: ChatAnthropic,
        db: MemoryDatabase,
        learning_agents: Dict[str, LearningAgent],
        knowledge_transfer_agent: KnowledgeTransferAgent,
    ):
        """Initialize the collaborative learning system.

        Args:
            model: Language model to use
            db: Memory database for persistence
            learning_agents: Dictionary of learning agents by name
            knowledge_transfer_agent: Knowledge transfer agent
        """
        self.model = model
        self.db = db
        self.learning_agents = learning_agents
        self.knowledge_transfer_agent = knowledge_transfer_agent

        # Create the collaboration strategy prompt
        self.strategy_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""You are a collaboration strategist responsible for developing strategies for agent collaboration.
Your job is to analyze agent performance and develop strategies for effective collaboration.

For each analysis, you should:
1. Identify strengths and weaknesses of each agent
2. Recognize complementary capabilities
3. Develop strategies for effective collaboration
4. Identify opportunities for knowledge sharing
5. Create a plan for collaborative problem-solving

Respond with a JSON object containing:
- "agent_profiles": Object with strengths and weaknesses of each agent
- "complementary_pairs": Array of agent pairs with complementary capabilities
- "collaboration_strategies": Array of strategies for collaboration
- "knowledge_sharing_opportunities": Array of opportunities for knowledge sharing
- "collaborative_problem_solving_plan": Plan for collaborative problem-solving
"""
                ),
                HumanMessage(
                    content="""
Agent performance:
{agent_performance}

Develop collaboration strategies for these agents.
"""
                ),
            ]
        )

    async def develop_collaboration_strategy(
        self, agent_performance: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Develop strategies for agent collaboration.

        Args:
            agent_performance: Performance metrics for each agent

        Returns:
            Collaboration strategies
        """
        # Format agent performance for the prompt
        formatted_performance = json.dumps(agent_performance, indent=2)

        # Prepare the input for the strategy prompt
        input_values = {"agent_performance": formatted_performance}

        # Get the collaboration strategies from the model
        messages = self.strategy_prompt.format_messages(**input_values)
        response = await self.model.ainvoke(messages)

        try:
            # Parse the response as JSON
            strategies = json.loads(response.content)
            return strategies
        except json.JSONDecodeError:
            # If the response is not valid JSON, extract it manually
            return {
                "collaboration_strategies": [response.content],
                "agent_profiles": {},
                "complementary_pairs": [],
                "knowledge_sharing_opportunities": [],
                "collaborative_problem_solving_plan": "Implement collaborative problem-solving.",
            }

    async def share_knowledge(
        self, source_agent: str, target_agent: str, knowledge: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Share knowledge from one agent to another.

        Args:
            source_agent: Source agent name
            target_agent: Target agent name
            knowledge: Knowledge to share

        Returns:
            Result of knowledge sharing
        """
        # Adapt the knowledge for the target agent
        adapted_knowledge = await self.knowledge_transfer_agent.adapt_knowledge(
            knowledge, source_agent, target_agent
        )

        # Apply the knowledge to the target agent
        if target_agent in self.learning_agents:
            learning_agent = self.learning_agents[target_agent]
            await learning_agent.incorporate_knowledge(adapted_knowledge)

        return {
            "source_agent": source_agent,
            "target_agent": target_agent,
            "original_knowledge": knowledge,
            "adapted_knowledge": adapted_knowledge,
            "status": "success" if target_agent in self.learning_agents else "failed",
        }

    async def collaborative_problem_solving(
        self, request: str, agent_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Solve a problem collaboratively using multiple agents.

        Args:
            request: User request
            agent_results: Results from individual agents

        Returns:
            Collaborative solution
        """
        # Create the collaborative problem-solving prompt
        prompt = f"""
You are a collaborative problem-solving coordinator. Your task is to synthesize results from multiple agents to solve the following problem:

User request: {request}

Agent results:
{json.dumps(agent_results, indent=2)}

Based on these results, provide a comprehensive solution that leverages the strengths of each agent.
Identify any conflicts or inconsistencies and resolve them.
Explain how the collaborative approach improved the solution compared to individual agents.
"""

        # Get the collaborative solution from the model
        messages = [
            {"role": "system", "content": "You are a collaborative problem-solving coordinator."},
            {"role": "user", "content": prompt},
        ]

        response = await self.model.ainvoke(messages)

        # Extract knowledge from the collaborative solution
        knowledge = await self.knowledge_transfer_agent.extract_knowledge(
            response.content, f"Collaborative solution for: {request}"
        )

        return {"collaborative_solution": response.content, "extracted_knowledge": knowledge}


class MultiAgentLearningSystem:
    """System for multi-agent learning and collaboration."""

    def __init__(
        self,
        model: ChatAnthropic,
        db: MemoryDatabase,
        learning_agents: Dict[str, LearningAgent],
        feedback_collector: FeedbackCollector,
    ):
        """Initialize the multi-agent learning system.

        Args:
            model: Language model to use
            db: Memory database for persistence
            learning_agents: Dictionary of learning agents by name
            feedback_collector: Feedback collector
        """
        self.model = model
        self.db = db
        self.learning_agents = learning_agents
        self.feedback_collector = feedback_collector

        # Initialize knowledge transfer agent
        self.knowledge_transfer_agent = KnowledgeTransferAgent(model, db)

        # Initialize collaborative learning system
        self.collaborative_learning = CollaborativeLearningSystem(
            model, db, learning_agents, self.knowledge_transfer_agent
        )

        # Create the performance analysis prompt
        self.analysis_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""You are a performance analysis agent responsible for analyzing agent performance.
Your job is to analyze performance metrics for multiple agents and identify patterns and opportunities for improvement.

For each analysis, you should:
1. Identify high-performing and low-performing agents
2. Recognize patterns in performance across different tasks
3. Identify opportunities for knowledge transfer
4. Develop strategies for performance improvement
5. Create a plan for multi-agent learning

Respond with a JSON object containing:
- "performance_patterns": Object with patterns in agent performance
- "knowledge_transfer_opportunities": Array of opportunities for knowledge transfer
- "improvement_strategies": Array of strategies for performance improvement
- "multi_agent_learning_plan": Plan for multi-agent learning
"""
                ),
                HumanMessage(
                    content="""
Performance metrics:
{performance_metrics}

Analyze agent performance and identify opportunities for improvement.
"""
                ),
            ]
        )

    async def analyze_performance(self, performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze agent performance and identify opportunities for improvement.

        Args:
            performance_metrics: Performance metrics for each agent

        Returns:
            Performance analysis
        """
        # Format performance metrics for the prompt
        formatted_metrics = json.dumps(performance_metrics, indent=2)

        # Prepare the input for the analysis prompt
        input_values = {"performance_metrics": formatted_metrics}

        # Get the performance analysis from the model
        messages = self.analysis_prompt.format_messages(**input_values)
        response = await self.model.ainvoke(messages)

        try:
            # Parse the response as JSON
            analysis = json.loads(response.content)
            return analysis
        except json.JSONDecodeError:
            # If the response is not valid JSON, extract it manually
            return {
                "performance_patterns": {},
                "knowledge_transfer_opportunities": [response.content],
                "improvement_strategies": [],
                "multi_agent_learning_plan": "Implement multi-agent learning.",
            }

    async def execute_learning_cycle(self) -> Dict[str, Any]:
        """Execute a complete multi-agent learning cycle.

        Returns:
            Results of the learning cycle
        """
        # Get performance metrics for all agents
        performance_metrics = {}
        for agent_name in self.learning_agents:
            agent_metrics = self.db.get_agent_performance(agent_name)
            performance_metrics[agent_name] = agent_metrics

        # Analyze performance
        performance_analysis = await self.analyze_performance(performance_metrics)

        # Develop collaboration strategy
        collaboration_strategy = await self.collaborative_learning.develop_collaboration_strategy(
            performance_metrics
        )

        # Execute knowledge transfers
        knowledge_transfers = []
        for opportunity in performance_analysis.get("knowledge_transfer_opportunities", []):
            if (
                isinstance(opportunity, dict)
                and "source" in opportunity
                and "target" in opportunity
            ):
                source_agent = opportunity["source"]
                target_agent = opportunity["target"]

                # Get knowledge from the source agent
                knowledge = self.db.get_agent_knowledge(source_agent)

                # Share knowledge with the target agent
                transfer_result = await self.collaborative_learning.share_knowledge(
                    source_agent, target_agent, knowledge
                )

                knowledge_transfers.append(transfer_result)

        # Learn from feedback for each agent
        learning_results = {}
        for agent_name, learning_agent in self.learning_agents.items():
            agent_results = await learning_agent.learn_from_feedback()
            learning_results[agent_name] = agent_results

        return {
            "performance_analysis": performance_analysis,
            "collaboration_strategy": collaboration_strategy,
            "knowledge_transfers": knowledge_transfers,
            "learning_results": learning_results,
        }

    async def process_request_collaboratively(
        self, request: str, agent_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a request collaboratively using multiple agents.

        Args:
            request: User request
            agent_results: Results from individual agents

        Returns:
            Collaborative response
        """
        # Solve the problem collaboratively
        collaborative_solution = await self.collaborative_learning.collaborative_problem_solving(
            request, agent_results
        )

        # Extract knowledge from each agent's result
        for agent_name, result in agent_results.items():
            if "response" in result:
                knowledge = await self.knowledge_transfer_agent.extract_knowledge(
                    result["response"], f"Agent {agent_name}'s response to: {request}"
                )

                # Store the knowledge in the database
                self.db.store_agent_knowledge(agent_name, knowledge)

        return {
            "collaborative_solution": collaborative_solution["collaborative_solution"],
            "extracted_knowledge": collaborative_solution["extracted_knowledge"],
        }


# Factory function to create multi-agent learning system
def create_multi_agent_learning_system(
    model: ChatAnthropic,
    db: MemoryDatabase,
    learning_agents: Dict[str, LearningAgent],
    feedback_collector: FeedbackCollector,
) -> MultiAgentLearningSystem:
    """Create a multi-agent learning system.

    Args:
        model: Language model to use
        db: Memory database for persistence
        learning_agents: Dictionary of learning agents by name
        feedback_collector: Feedback collector

    Returns:
        Multi-agent learning system
    """
    return MultiAgentLearningSystem(model, db, learning_agents, feedback_collector)
