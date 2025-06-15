"""
Research reports agent for DataMCPServerAgent.
This module provides a specialized agent for generating comprehensive research reports.
"""

import time
from typing import Any, Dict, List

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool

from src.memory.memory_persistence import MemoryDatabase
from src.utils.error_handlers import format_error_for_user


class ResearchReportsAgent:
    """Agent for generating comprehensive research reports."""

    def __init__(
        self,
        model: ChatAnthropic,
        tools: List[BaseTool],
        memory_db: MemoryDatabase,
        report_templates: Dict[str, Any] = None,
    ):
        """Initialize the research reports agent.

        Args:
            model: Language model to use
            tools: List of available tools
            memory_db: Memory database for persistence
            report_templates: Dictionary of report templates
        """
        self.model = model
        self.tools = tools
        self.memory_db = memory_db
        self.report_templates = report_templates or self._get_default_templates()

        # Initialize components
        from src.agents.research_reports.data_analyzer import DataAnalyzer
        from src.agents.research_reports.data_collector import DataCollector
        from src.agents.research_reports.report_formatter import ReportFormatter
        from src.agents.research_reports.report_generator import ReportGenerator

        self.data_collector = DataCollector(model, tools, memory_db)
        self.data_analyzer = DataAnalyzer(model, memory_db)
        self.report_generator = ReportGenerator(model, memory_db, self.report_templates)
        self.report_formatter = ReportFormatter(model, memory_db)

    def _get_default_templates(self) -> Dict[str, Any]:
        """Get default report templates.

        Returns:
            Dictionary of default report templates
        """
        return {
            "standard": {
                "sections": [
                    "Introduction",
                    "Background",
                    "Methodology",
                    "Findings",
                    "Analysis",
                    "Conclusion",
                    "Recommendations",
                ],
                "description": "Standard research report template with introduction, findings, and recommendations.",
            },
            "academic": {
                "sections": [
                    "Abstract",
                    "Introduction",
                    "Literature Review",
                    "Methodology",
                    "Results",
                    "Discussion",
                    "Conclusion",
                    "References",
                ],
                "description": "Academic research report template following scholarly conventions.",
            },
            "business": {
                "sections": [
                    "Executive Summary",
                    "Introduction",
                    "Market Analysis",
                    "Competitive Landscape",
                    "Key Findings",
                    "Strategic Implications",
                    "Recommendations",
                    "Action Plan",
                ],
                "description": "Business-oriented research report template focused on market analysis and strategic recommendations.",
            },
        }

    async def generate_research_report(
        self,
        topic: str,
        depth: str = "medium",
        template: str = "standard",
        format_type: str = "markdown",
        audience: str = "general",
    ) -> Dict[str, Any]:
        """Generate a comprehensive research report on a topic.

        Args:
            topic: Research topic
            depth: Research depth ("shallow", "medium", "deep")
            template: Report template to use
            format_type: Output format type
            audience: Target audience

        Returns:
            Generated report and metadata
        """
        try:
            # Step 1: Collect data
            print(f"Collecting data on '{topic}'...")
            collected_data = await self.data_collector.collect_data(topic, depth)

            # Step 2: Analyze data
            print("Analyzing collected data...")
            analyzed_data = await self.data_analyzer.analyze_data(collected_data)

            # Step 3: Generate report
            print(f"Generating {template} report for {audience} audience...")
            report = await self.report_generator.generate_report(
                topic, analyzed_data, template, audience
            )

            # Step 4: Format report
            print(f"Formatting report as {format_type}...")
            formatted_report = await self.report_formatter.format_report(report, format_type)

            return {
                "topic": topic,
                "report": report,
                "formatted_report": formatted_report,
                "metadata": {
                    "depth": depth,
                    "template": template,
                    "format_type": format_type,
                    "audience": audience,
                    "timestamp": time.time(),
                },
            }
        except Exception as e:
            error_message = format_error_for_user(e)
            print(f"Error generating research report: {error_message}")
            return {
                "error": error_message,
                "topic": topic,
                "metadata": {
                    "depth": depth,
                    "template": template,
                    "format_type": format_type,
                    "audience": audience,
                    "timestamp": time.time(),
                },
            }

    async def process_request(self, request: str) -> str:
        """Process a user request.

        Args:
            request: User request

        Returns:
            Response to the user
        """
        try:
            # Check if this is a research request
            if request.lower().startswith("research "):
                # Extract the topic
                topic = request[len("research ") :].strip()

                # Generate research report
                result = await self.generate_research_report(topic)

                if "error" in result:
                    return f"Error generating research report: {result['error']}"

                return f"Research report on '{topic}' has been generated and saved as {result.get('formatted_report', {}).get('filename', 'report.md')}."
            else:
                # Use the model to generate a response
                from langchain_core.messages import HumanMessage, SystemMessage

                messages = [
                    SystemMessage(
                        content="You are a research assistant that helps users find information and generate comprehensive research reports."
                    ),
                    HumanMessage(content=request),
                ]

                response = await self.model.ainvoke(messages)
                return response.content
        except Exception as e:
            error_message = format_error_for_user(e)
            return f"Error processing request: {error_message}"
