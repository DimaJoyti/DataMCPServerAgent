"""
Data analyzer component for the research reports agent.
This module provides functionality for analyzing and synthesizing collected data.
"""

import re
import time
from typing import Any, Dict, List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from src.memory.memory_persistence import MemoryDatabase
from src.utils.error_handlers import format_error_for_user

class DataAnalyzer:
    """Component for analyzing and synthesizing collected data."""

    def __init__(
        self,
        model: ChatAnthropic,
        memory_db: MemoryDatabase
    ):
        """Initialize the data analyzer.

        Args:
            model: Language model to use
            memory_db: Memory database for persistence
        """
        self.model = model
        self.memory_db = memory_db

    async def analyze_data(self, collected_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze collected data.

        Args:
            collected_data: Data collected from various sources

        Returns:
            Analyzed data
        """
        try:
            # Extract key points from each source
            print("Extracting key points from sources...")
            key_points = await self._extract_key_points(collected_data)

            # Identify common themes
            print("Identifying common themes...")
            themes = await self._identify_themes(key_points)

            # Synthesize information
            print("Synthesizing information...")
            synthesis = await self._synthesize_information(key_points, themes)

            # Identify gaps and contradictions
            print("Identifying gaps and contradictions...")
            gaps = await self._identify_gaps(synthesis)
            contradictions = await self._identify_contradictions(synthesis)

            # Create analysis result
            analysis_result = {
                "key_points": key_points,
                "themes": themes,
                "synthesis": synthesis,
                "gaps": gaps,
                "contradictions": contradictions
            }

            # Store analysis result in memory
            self.memory_db.save_entity(
                "research_data",
                f"analysis_{int(time.time())}",
                analysis_result
            )

            return analysis_result
        except Exception as e:
            error_message = format_error_for_user(e)
            print(f"Error analyzing data: {error_message}")
            return {"error": error_message}

    async def _extract_key_points(self, collected_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract key points from each source.

        Args:
            collected_data: Data collected from various sources

        Returns:
            Dictionary of key points by source
        """
        key_points = {}

        for source_name, source_data in collected_data.items():
            # Skip error entries
            if source_name.endswith("_error") or source_name == "error":
                continue

            # Use the model to extract key points

            messages = [
                SystemMessage(content="You are an expert at extracting key points from text. Extract the 5-10 most important points from the provided text."),
                HumanMessage(content=f"Extract key points from this text from {source_name}:\n\n{source_data}")
            ]

            response = await self.model.ainvoke(messages)
            points = self._parse_key_points(response.content)

            key_points[source_name] = points

        return key_points

    def _parse_key_points(self, text: str) -> List[str]:
        """Parse key points from model response.

        Args:
            text: Model response text

        Returns:
            List of key points
        """
        # Simple parsing logic - extract lines that start with numbers or bullet points
        lines = text.split("\n")
        points = []

        for line in lines:
            line = line.strip()
            if re.match(r"^\d+\.\s+", line) or line.startswith("- ") or line.startswith("• "):
                # Remove the prefix
                point = re.sub(r"^\d+\.\s+|^-\s+|^•\s+", "", line)
                points.append(point)

        # If no points were found, try to extract sentences
        if not points and text:
            sentences = re.split(r'(?<=[.!?])\s+', text)
            points = [s.strip() for s in sentences if len(s.strip()) > 20][:10]

        return points

    async def _identify_themes(self, key_points: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        """Identify common themes across sources.

        Args:
            key_points: Key points by source

        Returns:
            List of themes
        """
        # Flatten all key points
        all_points = []
        for source, points in key_points.items():
            all_points.extend(points)

        # If no points, return empty list
        if not all_points:
            return []

        # Use the model to identify themes
        messages = [
            SystemMessage(content="You are an expert at identifying themes and patterns in information. Identify 3-5 main themes from the provided key points."),
            HumanMessage(content="Identify the main themes from these key points:\n\n" + "\n".join([f"- {point}" for point in all_points]))
        ]

        response = await self.model.ainvoke(messages)
        themes_text = response.content

        # Parse themes
        themes = []
        current_theme = None
        current_description = []

        for line in themes_text.split("\n"):
            line = line.strip()
            if not line:
                continue

            # Check if this is a new theme
            theme_match = re.match(r"^(\d+\.\s+|#+\s+|[A-Z][A-Za-z\s]+:)", line)
            if theme_match:
                # Save the previous theme if it exists
                if current_theme:
                    themes.append({
                        "name": current_theme,
                        "description": "\n".join(current_description)
                    })

                # Start a new theme
                current_theme = re.sub(r"^(\d+\.\s+|#+\s+|:|-)","", line).strip()
                current_description = []
            elif current_theme:
                current_description.append(line)

        # Add the last theme
        if current_theme:
            themes.append({
                "name": current_theme,
                "description": "\n".join(current_description)
            })

        return themes

    async def _synthesize_information(self, key_points: Dict[str, List[str]], themes: List[Dict[str, Any]]) -> str:
        """Synthesize information from key points and themes.

        Args:
            key_points: Key points by source
            themes: Identified themes

        Returns:
            Synthesized information
        """
        # Prepare input for the model
        theme_text = "\n".join([f"Theme: {theme['name']}\nDescription: {theme['description']}" for theme in themes])

        points_text = ""
        for source, points in key_points.items():
            points_text += f"\nSource: {source}\n"
            for i, point in enumerate(points, 1):
                points_text += f"{i}. {point}\n"

        messages = [
            SystemMessage(content="You are an expert at synthesizing information from multiple sources. Create a coherent synthesis of the provided key points and themes."),
            HumanMessage(content=f"Synthesize the following information into a coherent narrative:\n\nTHEMES:\n{theme_text}\n\nKEY POINTS:{points_text}")
        ]

        response = await self.model.ainvoke(messages)
        return response.content

    async def _identify_gaps(self, synthesis: str) -> List[str]:
        """Identify gaps in the research.

        Args:
            synthesis: Synthesized information

        Returns:
            List of identified gaps
        """
        messages = [
            SystemMessage(content="You are an expert at identifying gaps in research. Identify 3-5 areas where more information is needed based on the provided synthesis."),
            HumanMessage(content=f"Identify gaps in the following research synthesis:\n\n{synthesis}")
        ]

        response = await self.model.ainvoke(messages)
        gaps_text = response.content

        # Parse gaps
        return self._parse_key_points(gaps_text)

    async def _identify_contradictions(self, synthesis: str) -> List[str]:
        """Identify contradictions in the research.

        Args:
            synthesis: Synthesized information

        Returns:
            List of identified contradictions
        """
        messages = [
            SystemMessage(content="You are an expert at identifying contradictions in research. Identify any contradictory information or perspectives in the provided synthesis."),
            HumanMessage(content=f"Identify contradictions in the following research synthesis:\n\n{synthesis}")
        ]

        response = await self.model.ainvoke(messages)
        contradictions_text = response.content

        # Parse contradictions
        return self._parse_key_points(contradictions_text)
