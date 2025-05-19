"""
Data collector component for the research reports agent.
This module provides functionality for collecting data from various sources.
"""

import time
from typing import Any, Dict, List, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool

from src.memory.memory_persistence import MemoryDatabase
from src.utils.error_handlers import format_error_for_user

class DataCollector:
    """Component for collecting data from various sources."""
    
    def __init__(
        self,
        model: ChatAnthropic,
        tools: List[BaseTool],
        memory_db: MemoryDatabase
    ):
        """Initialize the data collector.
        
        Args:
            model: Language model to use
            tools: List of available tools
            memory_db: Memory database for persistence
        """
        self.model = model
        self.memory_db = memory_db
        self.tools = self._organize_tools(tools)
    
    def _organize_tools(self, tools: List[BaseTool]) -> Dict[str, BaseTool]:
        """Organize tools by name for easy access.
        
        Args:
            tools: List of tools
            
        Returns:
            Dictionary of tools by name
        """
        return {tool.name: tool for tool in tools}
    
    async def collect_data(self, topic: str, depth: str = "medium") -> Dict[str, Any]:
        """Collect data on a topic from various sources.
        
        Args:
            topic: Research topic
            depth: Research depth ("shallow", "medium", "deep")
            
        Returns:
            Collected data
        """
        try:
            # Determine which sources to use based on depth and topic
            sources = self._determine_sources(topic, depth)
            
            # Collect data from each source
            collected_data = {}
            for source_name in sources:
                if source_name in self.tools:
                    try:
                        print(f"Collecting data from {source_name}...")
                        source_data = await self.tools[source_name].arun(topic)
                        collected_data[source_name] = source_data
                    except Exception as e:
                        error_message = format_error_for_user(e)
                        print(f"Error collecting data from {source_name}: {error_message}")
                        collected_data[f"{source_name}_error"] = error_message
            
            # If we have enhanced_web_search but not brave_web_search, try that as fallback
            if "enhanced_web_search" not in collected_data and "brave_web_search" in self.tools:
                try:
                    print("Collecting data from brave_web_search...")
                    source_data = await self.tools["brave_web_search"].arun(topic)
                    collected_data["brave_web_search"] = source_data
                except Exception as e:
                    error_message = format_error_for_user(e)
                    print(f"Error collecting data from brave_web_search: {error_message}")
            
            # Store collected data in memory
            self.memory_db.save_entity(
                "research_data",
                f"collected_{int(time.time())}",
                collected_data
            )
            
            return collected_data
        except Exception as e:
            error_message = format_error_for_user(e)
            print(f"Error collecting data: {error_message}")
            return {"error": error_message}
    
    def _determine_sources(self, topic: str, depth: str) -> List[str]:
        """Determine which sources to use based on topic and depth.
        
        Args:
            topic: Research topic
            depth: Research depth
            
        Returns:
            List of source names
        """
        # Basic sources for all depths
        sources = ["enhanced_web_search", "wikipedia"]
        
        # Add academic sources for medium and deep research
        if depth in ["medium", "deep"]:
            sources.extend(["google_scholar", "arxiv"])
            
            # Add specialized sources based on topic
            if any(term in topic.lower() for term in ["health", "medical", "disease", "treatment", "drug"]):
                sources.append("pubmed")
            
            if any(term in topic.lower() for term in ["book", "literature", "novel", "author", "publication"]):
                sources.extend(["google_books", "open_library"])
        
        # Add news sources for deep research
        if depth == "deep":
            sources.append("news_search")
        
        return sources
