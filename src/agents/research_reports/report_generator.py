"""
Report generator component for the research reports agent.
This module provides functionality for generating research reports.
"""

import re
import time
from typing import Any, Dict, List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from src.memory.memory_persistence import MemoryDatabase
from src.utils.error_handlers import format_error_for_user

class ReportGenerator:
    """Component for generating research reports."""

    def __init__(
        self,
        model: ChatAnthropic,
        memory_db: MemoryDatabase,
        templates: Dict[str, Any]
    ):
        """Initialize the report generator.

        Args:
            model: Language model to use
            memory_db: Memory database for persistence
            templates: Report templates
        """
        self.model = model
        self.memory_db = memory_db
        self.templates = templates

    async def generate_report(
        self,
        topic: str,
        analyzed_data: Dict[str, Any],
        template_name: str = "standard",
        audience: str = "general"
    ) -> Dict[str, Any]:
        """Generate a research report.

        Args:
            topic: Research topic
            analyzed_data: Analyzed data
            template_name: Name of the template to use
            audience: Target audience

        Returns:
            Generated report
        """
        try:
            # Get the template
            template = self.templates.get(template_name, self.templates["standard"])

            # Generate report structure
            print("Generating report structure...")
            structure = await self._generate_structure(topic, template, audience)

            # Generate content for each section
            print("Generating section content...")
            sections = {}
            for section_name in structure:
                section_content = await self._generate_section_content(
                    section_name,
                    structure[section_name],
                    analyzed_data,
                    audience
                )
                sections[section_name] = section_content

            # Generate executive summary
            print("Generating executive summary...")
            executive_summary = await self._generate_executive_summary(
                topic,
                sections,
                audience
            )

            # Generate bibliography
            print("Generating bibliography...")
            bibliography = await self._generate_bibliography(analyzed_data)

            # Create report
            report = {
                "topic": topic,
                "executive_summary": executive_summary,
                "sections": sections,
                "bibliography": bibliography,
                "metadata": {
                    "generated_at": time.time(),
                    "template": template_name,
                    "audience": audience
                }
            }

            # Store report in memory
            self.memory_db.save_entity(
                "research_reports",
                f"report_{int(time.time())}",
                report
            )

            return report
        except Exception as e:
            error_message = format_error_for_user(e)
            print(f"Error generating report: {error_message}")
            return {"error": error_message}

    async def _generate_structure(
        self,
        topic: str,
        template: Dict[str, Any],
        audience: str
    ) -> Dict[str, str]:
        """Generate report structure based on template.

        Args:
            topic: Research topic
            template: Report template
            audience: Target audience

        Returns:
            Dictionary of section names and prompts
        """
        structure = {}

        # Get sections from template
        sections = template.get("sections", [])

        # Create a prompt for each section
        for section in sections:
            # Create a prompt based on the section name
            if section == "Introduction":
                prompt = f"Write an introduction for a research report on {topic}. The audience is {audience}."
            elif section == "Background":
                prompt = f"Write a background section for a research report on {topic}, providing context and historical information. The audience is {audience}."
            elif section == "Literature Review" or section == "Literature":
                prompt = f"Write a literature review section for a research report on {topic}, summarizing existing research. The audience is {audience}."
            elif section == "Methodology":
                prompt = f"Write a methodology section for a research report on {topic}, explaining how the research was conducted. The audience is {audience}."
            elif section == "Results" or section == "Findings":
                prompt = f"Write a results/findings section for a research report on {topic}, presenting the main research findings. The audience is {audience}."
            elif section == "Analysis" or section == "Discussion":
                prompt = f"Write an analysis/discussion section for a research report on {topic}, interpreting the findings. The audience is {audience}."
            elif section == "Conclusion":
                prompt = f"Write a conclusion section for a research report on {topic}, summarizing the main points and implications. The audience is {audience}."
            elif section == "Recommendations":
                prompt = f"Write a recommendations section for a research report on {topic}, suggesting actions based on the findings. The audience is {audience}."
            elif section == "Abstract":
                prompt = f"Write an abstract for a research report on {topic}. The audience is {audience}."
            elif section == "Executive Summary":
                prompt = f"Write an executive summary for a research report on {topic}. The audience is {audience}."
            elif section == "Market Analysis":
                prompt = f"Write a market analysis section for a research report on {topic}, analyzing market trends and dynamics. The audience is {audience}."
            elif section == "Competitive Landscape":
                prompt = f"Write a competitive landscape section for a research report on {topic}, analyzing competitors and market positioning. The audience is {audience}."
            elif section == "Strategic Implications":
                prompt = f"Write a strategic implications section for a research report on {topic}, discussing the strategic impact of the findings. The audience is {audience}."
            elif section == "Action Plan":
                prompt = f"Write an action plan section for a research report on {topic}, outlining specific actions to take. The audience is {audience}."
            else:
                prompt = f"Write a {section} section for a research report on {topic}. The audience is {audience}."

            structure[section] = prompt

        return structure

    async def _generate_section_content(
        self,
        section_name: str,
        section_prompt: str,
        analyzed_data: Dict[str, Any],
        audience: str
    ) -> str:
        """Generate content for a report section.

        Args:
            section_name: Name of the section
            section_prompt: Prompt for the section
            analyzed_data: Analyzed data
            audience: Target audience

        Returns:
            Section content
        """
        # Prepare input for the model
        synthesis = analyzed_data.get("synthesis", "")

        # Get relevant themes for this section
        themes = analyzed_data.get("themes", [])
        theme_text = "\n".join([f"Theme: {theme['name']}\nDescription: {theme['description']}" for theme in themes])

        # Get relevant key points for this section
        key_points = analyzed_data.get("key_points", {})
        points_text = ""
        for source, points in key_points.items():
            points_text += f"\nSource: {source}\n"
            for i, point in enumerate(points, 1):
                points_text += f"{i}. {point}\n"

        # Create a prompt for the model
        messages = [
            SystemMessage(content=f"You are an expert at writing research reports for a {audience} audience. Write a {section_name} section based on the provided information."),
            HumanMessage(content=f"{section_prompt}\n\nUse the following information to write this section:\n\nSYNTHESIS:\n{synthesis}\n\nTHEMES:\n{theme_text}\n\nKEY POINTS:{points_text}")
        ]

        response = await self.model.ainvoke(messages)
        return response.content

    async def _generate_executive_summary(
        self,
        topic: str,
        sections: Dict[str, str],
        audience: str
    ) -> str:
        """Generate an executive summary for the report.

        Args:
            topic: Research topic
            sections: Report sections
            audience: Target audience

        Returns:
            Executive summary
        """
        # Prepare input for the model
        sections_text = ""
        for section_name, section_content in sections.items():
            # Add the first paragraph of each section
            paragraphs = section_content.split("\n\n")
            first_paragraph = paragraphs[0] if paragraphs else section_content[:500]
            sections_text += f"\n\n{section_name}:\n{first_paragraph}"

        messages = [
            SystemMessage(content=f"You are an expert at writing executive summaries for research reports for a {audience} audience. Write a concise executive summary that captures the key points of the report."),
            HumanMessage(content=f"Write an executive summary for a research report on {topic}. The audience is {audience}.\n\nUse the following section excerpts to create the summary:{sections_text}")
        ]

        response = await self.model.ainvoke(messages)
        return response.content

    async def _generate_bibliography(self, analyzed_data: Dict[str, Any]) -> List[str]:
        """Generate a bibliography for the report.

        Args:
            analyzed_data: Analyzed data

        Returns:
            List of bibliography entries
        """
        # Extract sources from the analyzed data
        key_points = analyzed_data.get("key_points", {})
        sources = list(key_points.keys())

        # If no sources, return empty list
        if not sources:
            return []

        # Use the model to generate bibliography entries
        messages = [
            SystemMessage(content="You are an expert at creating bibliographies for research reports. Create bibliography entries for the provided sources."),
            HumanMessage(content="Create bibliography entries for the following sources:\n\n" + "\n".join([f"- {source}" for source in sources]))
        ]

        response = await self.model.ainvoke(messages)
        bibliography_text = response.content

        # Parse bibliography entries
        entries = []
        for line in bibliography_text.split("\n"):
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("Bibliography"):
                # Remove numbering if present
                entry = re.sub(r"^\d+\.\s+", "", line)
                entries.append(entry)

        return entries
