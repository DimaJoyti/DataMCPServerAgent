"""
Citation tools for the Research Assistant.

This module provides tools for formatting citations and generating bibliographies
in various citation styles.
"""

from typing import List

from langchain.tools import Tool

from src.models.research_models import CitationFormat, Source


class CitationFormatter:
    """Tool for formatting citations in various styles."""

    def format_citation(self, source_data: dict, format: str = "apa") -> str:
        """
        Format a source as a citation in the specified format.

        Args:
            source_data: Dictionary containing source information
            format: Citation format (apa, mla, chicago, harvard, ieee)

        Returns:
            Formatted citation string
        """
        # Convert the dictionary to a Source object
        source = Source(**source_data)

        # Format the citation
        citation_format = CitationFormat(format.lower())
        return source.format_citation(citation_format)

    def run(self, input_str: str) -> str:
        """
        Run the citation formatter.

        Args:
            input_str: JSON string containing source data and format

        Returns:
            Formatted citation
        """
        try:
            import json

            data = json.loads(input_str)
            source_data = data.get("source", {})
            format = data.get("format", "apa")

            return self.format_citation(source_data, format)
        except Exception as e:
            return f"Error formatting citation: {str(e)}"


class BibliographyGenerator:
    """Tool for generating bibliographies in various styles."""

    def generate_bibliography(self, sources_data: List[dict], format: str = "apa") -> str:
        """
        Generate a bibliography from a list of sources.

        Args:
            sources_data: List of dictionaries containing source information
            format: Citation format (apa, mla, chicago, harvard, ieee)

        Returns:
            Formatted bibliography string
        """
        # Convert the dictionaries to Source objects
        sources = [Source(**source_data) for source_data in sources_data]

        # Format the citations
        citation_format = CitationFormat(format.lower())
        citations = [source.format_citation(citation_format) for source in sources]

        # Sort the citations alphabetically
        citations.sort()

        # Join the citations with appropriate formatting
        if format.lower() == "apa":
            return "\n\n".join(citations)
        elif format.lower() == "mla":
            return "\n\n".join(citations)
        elif format.lower() == "chicago":
            return "\n\n".join(citations)
        elif format.lower() == "harvard":
            return "\n\n".join(citations)
        elif format.lower() == "ieee":
            return "\n\n".join([f"[{i+1}] {citation}" for i, citation in enumerate(citations)])
        else:
            return "\n\n".join(citations)

    def run(self, input_str: str) -> str:
        """
        Run the bibliography generator.

        Args:
            input_str: JSON string containing sources data and format

        Returns:
            Formatted bibliography
        """
        try:
            import json

            data = json.loads(input_str)
            sources_data = data.get("sources", [])
            format = data.get("format", "apa")

            return self.generate_bibliography(sources_data, format)
        except Exception as e:
            return f"Error generating bibliography: {str(e)}"


# Create tool instances
citation_formatter = CitationFormatter()
bibliography_generator = BibliographyGenerator()

# Create LangChain tools
format_citation_tool = Tool(
    name="format_citation",
    func=citation_formatter.run,
    description="Format a source as a citation in a specified style (APA, MLA, Chicago, Harvard, IEEE). Input should be a JSON string with 'source' and 'format' fields.",
)

generate_bibliography_tool = Tool(
    name="generate_bibliography",
    func=bibliography_generator.run,
    description="Generate a bibliography from a list of sources in a specified style (APA, MLA, Chicago, Harvard, IEEE). Input should be a JSON string with 'sources' and 'format' fields.",
)
