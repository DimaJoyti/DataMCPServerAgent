"""
SEO Agent for DataMCPServerAgent.

This module implements a specialized SEO agent that can analyze websites,
provide SEO recommendations, and optimize content for better search engine rankings.
"""

from typing import Dict, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import BaseTool

from src.agents.agent_architecture import AgentMemory, SpecializedSubAgent
from src.memory.memory_persistence import MemoryDatabase
from src.tools.seo_advanced_tools import competitor_analysis_tool, rank_tracking_tool
from src.tools.seo_bulk_tools import bulk_analysis_tool
from src.tools.seo_ml_tools import ml_content_optimizer_tool, ml_ranking_prediction_tool
from src.tools.seo_scheduled_reporting import scheduled_reporting_tool
from src.tools.seo_tools import (
    backlink_analyzer_tool,
    content_optimizer_tool,
    keyword_research_tool,
    metadata_generator_tool,
    seo_analyzer_tool,
)
from src.utils.error_handlers import format_error_for_user


class SEOAgent:
    """Specialized agent for SEO tasks."""

    def __init__(
        self,
        model: ChatAnthropic,
        memory: AgentMemory,
        db: Optional[MemoryDatabase] = None,
        additional_tools: Optional[List[BaseTool]] = None,
    ):
        """Initialize the SEO agent.

        Args:
            model: Language model to use
            memory: Agent memory
            db: Optional memory database for persistence
            additional_tools: Optional additional tools to use
        """
        self.model = model
        self.memory = memory
        self.db = db

        # Set up SEO tools
        self.tools = [
            seo_analyzer_tool,
            keyword_research_tool,
            content_optimizer_tool,
            metadata_generator_tool,
            backlink_analyzer_tool,
            competitor_analysis_tool,
            rank_tracking_tool,
            bulk_analysis_tool,
            ml_content_optimizer_tool,
            ml_ranking_prediction_tool,
            scheduled_reporting_tool,
            seo_visualization_tool,
        ]

        # Add additional tools if provided
        if additional_tools:
            self.tools.extend(additional_tools)

        # Create specialized sub-agents for different SEO tasks
        self.sub_agents = self._create_specialized_sub_agents()

        # Create the main SEO agent prompt
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""You are an expert SEO agent specialized in search engine optimization.
Your goal is to help users improve their website's visibility in search engines and drive more organic traffic.

You can perform the following tasks:
1. Analyze websites for SEO factors
2. Research keywords and provide recommendations
3. Optimize content for better search engine rankings
4. Generate SEO-friendly metadata
5. Analyze backlink profiles
6. Analyze competitors and compare websites
7. Track keyword rankings over time
8. Perform bulk analysis of multiple pages or entire websites
9. Use machine learning to optimize content
10. Predict search rankings with machine learning models
11. Schedule regular SEO reports
12. Generate visualizations of SEO metrics

When responding to requests:
- Break down complex SEO tasks into manageable steps
- Provide clear explanations of SEO concepts
- Support your recommendations with data
- Prioritize recommendations based on impact and effort
- Focus on white-hat SEO techniques that follow search engine guidelines

Always consider the latest SEO best practices, including:
- Mobile-first indexing
- Page speed optimization
- User experience signals
- E-A-T (Expertise, Authoritativeness, Trustworthiness)
- Core Web Vitals
- Content quality and relevance
- Competitive analysis and differentiation
"""
                ),
                MessagesPlaceholder(variable_name="history"),
                HumanMessage(content="{request}"),
            ]
        )

    def _create_specialized_sub_agents(self) -> Dict[str, SpecializedSubAgent]:
        """Create specialized sub-agents for different SEO tasks.

        Returns:
            Dictionary of sub-agents by name
        """
        sub_agents = {}

        # Website Analysis Agent
        website_analysis_prompt = """You are a specialized SEO analysis agent that focuses on analyzing websites for SEO factors.
Your primary goal is to identify SEO issues and provide actionable recommendations for improvement.

When analyzing websites:
- Check for technical SEO issues
- Analyze content quality and relevance
- Evaluate meta tags and structured data
- Assess mobile-friendliness and page speed
- Identify crawlability and indexation issues
- Analyze multiple pages or entire websites for comprehensive insights

Provide clear, prioritized recommendations based on impact and implementation effort.
"""
        website_analysis_tools = [seo_analyzer_tool, backlink_analyzer_tool, bulk_analysis_tool]
        sub_agents["website_analysis"] = SpecializedSubAgent(
            "Website Analysis Agent", self.model, website_analysis_tools, website_analysis_prompt
        )

        # Keyword Research Agent
        keyword_research_prompt = """You are a specialized keyword research agent that focuses on finding the best keywords for SEO.
Your primary goal is to identify valuable keywords that balance search volume, competition, and relevance.

When researching keywords:
- Focus on user intent behind searches
- Identify long-tail keyword opportunities
- Group keywords by topic and funnel stage
- Analyze keyword difficulty and competition
- Consider commercial value and conversion potential
- Track keyword rankings over time to measure progress

Provide strategic keyword recommendations that align with the user's goals and content strategy.
"""
        keyword_research_tools = [keyword_research_tool, rank_tracking_tool]
        sub_agents["keyword_research"] = SpecializedSubAgent(
            "Keyword Research Agent", self.model, keyword_research_tools, keyword_research_prompt
        )

        # Content Optimization Agent
        content_optimization_prompt = """You are a specialized content optimization agent that focuses on improving content for SEO.
Your primary goal is to help users create and optimize content that ranks well in search engines.

When optimizing content:
- Ensure proper keyword usage and placement
- Improve readability and user engagement
- Enhance content structure with appropriate headings
- Recommend content additions for comprehensiveness
- Suggest internal and external linking opportunities
- Compare content with competitors to identify gaps and opportunities
- Use machine learning to analyze content and provide data-driven recommendations
- Leverage ML models to predict the impact of content changes on SEO

Focus on creating high-quality, valuable content that satisfies user intent while following SEO best practices.
"""
        content_optimization_tools = [
            content_optimizer_tool,
            metadata_generator_tool,
            ml_content_optimizer_tool,
        ]
        sub_agents["content_optimization"] = SpecializedSubAgent(
            "Content Optimization Agent",
            self.model,
            content_optimization_tools,
            content_optimization_prompt,
        )

        # Competitive Analysis Agent
        competitive_analysis_prompt = """You are a specialized competitive analysis agent that focuses on analyzing competitors for SEO.
Your primary goal is to help users understand their competitive landscape and identify opportunities for improvement.

When analyzing competitors:
- Identify top competitors based on keyword overlap
- Compare website metrics and performance
- Analyze content strategies and gaps
- Evaluate backlink profiles and opportunities
- Track competitor rankings and changes over time
- Use machine learning to predict ranking potential
- Leverage ML models to identify key ranking factors

Provide strategic recommendations to help users outperform their competitors in search results.
"""
        competitive_analysis_tools = [
            competitor_analysis_tool,
            rank_tracking_tool,
            ml_ranking_prediction_tool,
        ]
        sub_agents["competitive_analysis"] = SpecializedSubAgent(
            "Competitive Analysis Agent",
            self.model,
            competitive_analysis_tools,
            competitive_analysis_prompt,
        )

        return sub_agents

    async def process_request(self, request: str) -> str:
        """Process an SEO-related request.

        Args:
            request: User request

        Returns:
            Response to the user
        """
        # Add the user request to memory
        self.memory.add_message({"role": "user", "content": request})

        # Get recent conversation history
        history = self.memory.get_recent_messages(5)

        # Determine which sub-agent to use based on the request
        sub_agent_name = await self._select_sub_agent(request)
        sub_agent = self.sub_agents.get(sub_agent_name)

        # If no specific sub-agent is selected, process with the main agent
        if not sub_agent:
            return await self._process_with_main_agent(request, history)

        # Execute the selected sub-agent
        result = await sub_agent.execute(request, self.memory)

        # Format the response
        if result["success"]:
            response = result["response"]
        else:
            response = f"I encountered an error while processing your request: {result['error']}\n\nLet me try a different approach."
            # Fall back to the main agent
            response = await self._process_with_main_agent(request, history)

        # Add the response to memory
        self.memory.add_message({"role": "assistant", "content": response})

        # Save to database if available
        if self.db:
            self.db.save_conversation_message({"role": "user", "content": request}, "seo_agent")
            self.db.save_conversation_message(
                {"role": "assistant", "content": response}, "seo_agent"
            )

        return response

    async def _select_sub_agent(self, request: str) -> str:
        """Select the most appropriate sub-agent for a request.

        Args:
            request: User request

        Returns:
            Name of the selected sub-agent
        """
        # Create a prompt to select the sub-agent
        prompt = f"""Based on the following user request, select the most appropriate specialized SEO sub-agent to handle it:

User request: {request}

Available sub-agents:
1. website_analysis - For analyzing websites, identifying SEO issues, and providing recommendations
2. keyword_research - For researching keywords, analyzing search volume, and identifying opportunities
3. content_optimization - For optimizing content, improving readability, and enhancing keyword usage
4. competitive_analysis - For analyzing competitors, comparing websites, and identifying competitive advantages

Select the most appropriate sub-agent by returning only the sub-agent name (e.g., "website_analysis").
If the request doesn't clearly match any specific sub-agent, return "main".
"""

        # Get the sub-agent selection from the model
        messages = [
            {
                "role": "system",
                "content": "You are a task router that selects the most appropriate specialized agent for a request.",
            },
            {"role": "user", "content": prompt},
        ]

        response = await self.model.ainvoke(messages)

        # Extract the sub-agent name from the response
        sub_agent_name = response.content.strip().lower()

        # Clean up the response to get just the sub-agent name
        sub_agent_name = sub_agent_name.replace('"', "").replace("'", "")
        if (
            "website" in sub_agent_name
            or "analysis" in sub_agent_name
            and "competitive" not in sub_agent_name
        ):
            return "website_analysis"
        elif "keyword" in sub_agent_name or "research" in sub_agent_name:
            return "keyword_research"
        elif "content" in sub_agent_name or "optimization" in sub_agent_name:
            return "content_optimization"
        elif (
            "competitive" in sub_agent_name
            or "competitor" in sub_agent_name
            or "competition" in sub_agent_name
        ):
            return "competitive_analysis"
        else:
            return "main"

    async def _process_with_main_agent(self, request: str, history: List[Dict[str, str]]) -> str:
        """Process a request with the main SEO agent.

        Args:
            request: User request
            history: Conversation history

        Returns:
            Response to the user
        """
        try:
            # Prepare the input for the prompt
            input_values = {"request": request, "history": history}

            # Get the response from the model
            messages = self.prompt.format_messages(**input_values)
            response = await self.model.ainvoke(messages)

            return response.content

        except Exception as e:
            error_message = format_error_for_user(e)
            return f"I encountered an error while processing your request: {error_message}\n\nPlease try rephrasing your request or breaking it down into smaller steps."
