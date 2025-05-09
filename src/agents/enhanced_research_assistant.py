"""
Enhanced Research Assistant with advanced features.
This module implements an enhanced research assistant with memory persistence,
tool selection, learning capabilities, and reinforcement learning integration.
"""

import json
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from src.memory.research_memory_persistence import (
    ResearchMemoryDatabase,
    ResearchProject,
    ResearchResult,
    Source,
)
from src.tools.enhanced_tool_selection import (
    EnhancedToolSelector,
    ToolPerformanceTracker,
)

# Import real tools instead of mock tools
from src.tools.research_assistant_tools import (
    export_to_docx_tool,
    export_to_html_tool,
    export_to_markdown_tool,
    export_to_pdf_tool,
    export_to_presentation_tool,
    generate_chart_tool,
    generate_mind_map_tool,
    generate_network_diagram_tool,
    generate_timeline_tool,
    save_tool,
    search_tool,
    wiki_tool,
)

# Try to import academic tools, use mock tools if not available
try:
    from src.tools.academic_tools import (
        arxiv_tool,
        format_citation_tool,
        generate_bibliography_tool,
        google_books_tool,
        google_scholar_tool,
        open_library_tool,
        pubmed_tool,
    )
except ImportError:
    # Create mock tools for academic sources if real ones aren't available
    class MockTool:
        def __init__(self, name):
            self.name = name

        def run(self, query):
            return f"Mock {self.name} result for: {query}"

    google_scholar_tool = MockTool("Google Scholar")
    pubmed_tool = MockTool("PubMed")
    arxiv_tool = MockTool("arXiv")
    google_books_tool = MockTool("Google Books")
    open_library_tool = MockTool("Open Library")
    format_citation_tool = MockTool("Citation Formatter")
    generate_bibliography_tool = MockTool("Bibliography Generator")


class EnhancedResearchResponseModel(BaseModel):
    """Enhanced structured response format for research results with advanced features."""

    topic: str
    summary: str
    sources: List[Union[str, Dict]]
    tools_used: List[str]
    citation_format: Optional[str] = None
    bibliography: Optional[str] = None
    project_id: Optional[str] = None
    query_id: Optional[str] = None
    visualizations: List[Dict] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class EnhancedResearchAssistant:
    """Enhanced Research Assistant with advanced features."""

    def __init__(
        self,
        model: Optional[ChatAnthropic] = None,
        db_path: str = "research_memory.db",
        tools: Optional[List[BaseTool]] = None,
    ):
        """Initialize the enhanced research assistant.

        Args:
            model: Language model to use
            db_path: Path to the research memory database
            tools: List of tools to use
        """
        # Initialize model
        self.model = model or ChatAnthropic(model="claude-3-5-sonnet-20240620")

        # Initialize memory database
        self.memory_db = ResearchMemoryDatabase(db_path)

        # Initialize tools
        self.tools = tools or self._get_default_tools()

        # Initialize tool performance tracker
        self.tool_performance_tracker = ToolPerformanceTracker(self.memory_db)

        # Initialize tool selector
        self.tool_selector = EnhancedToolSelector(
            model=self.model,
            tools=self.tools,
            db=self.memory_db,
            performance_tracker=self.tool_performance_tracker,
        )

        # Create a default project
        self.default_project = self._get_or_create_default_project()

        # Initialize research prompt
        self.research_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content="""You are an advanced research assistant that helps users gather, organize, and visualize information on various topics.
Your task is to research the given topic using the available tools and provide a comprehensive response.

Follow these steps:
1. Analyze the user's query to understand the research topic
2. Select the most appropriate tools to gather information
3. Use the selected tools to gather information
4. Synthesize the information into a coherent summary
5. Identify and cite sources properly
6. Organize the information with appropriate tags
7. Generate visualizations if helpful

Your response should be structured as a JSON object with the following fields:
- "topic": The research topic
- "summary": A comprehensive summary of the research findings
- "sources": An array of source objects with title, url, authors, source_type, etc.
- "tools_used": An array of tool names used in the research
- "citation_format": The citation format used (apa, mla, chicago, harvard, ieee)
- "bibliography": A formatted bibliography of the sources
- "tags": An array of relevant tags for the research
- "visualizations": An array of visualization objects (if applicable)

Be thorough, accurate, and provide well-organized information.
"""
                ),
                HumanMessage(
                    content="""
Research Query: {query}
Project ID: {project_id}
Citation Format: {citation_format}

Please research this topic and provide a comprehensive response.
"""
                ),
            ]
        )

    def _get_default_tools(self) -> List[BaseTool]:
        """Get the default tools for the research assistant.

        Returns:
            List of default tools
        """
        return [
            search_tool,
            wiki_tool,
            google_scholar_tool,
            pubmed_tool,
            arxiv_tool,
            google_books_tool,
            open_library_tool,
            format_citation_tool,
            generate_bibliography_tool,
            save_tool,
            export_to_markdown_tool,
            export_to_html_tool,
            export_to_pdf_tool,
            export_to_docx_tool,
            export_to_presentation_tool,
            generate_chart_tool,
            generate_mind_map_tool,
            generate_timeline_tool,
            generate_network_diagram_tool,
        ]

    def _get_or_create_default_project(self) -> ResearchProject:
        """Get or create the default research project.

        Returns:
            Default research project
        """
        # Get all projects
        projects = self.memory_db.get_all_projects()

        # If there are projects, use the first one as default
        if projects:
            return projects[0]

        # Otherwise, create a new default project
        return self.memory_db.create_project(
            name="General Research",
            description="Default project for research queries",
            tags=["general", "research"],
        )

    def _execute_tool(
        self, tool_name: str, args: Dict[str, Any], project_id: str, query_id: str
    ) -> Any:
        """Execute a tool and track its performance.

        Args:
            tool_name: Name of the tool to execute
            args: Arguments to pass to the tool
            project_id: Project ID
            query_id: Query ID

        Returns:
            Tool result
        """
        # Find the tool
        tool = None
        for t in self.tools:
            if t.name == tool_name:
                tool = t
                break

        if not tool:
            raise ValueError(f"Tool {tool_name} not found")

        # Execute the tool and track performance
        start_time = time.time()
        success = True
        result = None

        try:
            # Convert args to the format expected by the tool
            if len(args) == 1 and next(iter(args.keys())) == "query":
                # If there's only a query parameter, pass it directly
                result = tool.run(args["query"])
            else:
                # Otherwise, pass the args as a dictionary
                result = tool.run(**args)
        except Exception as e:
            success = False
            result = str(e)
            print(f"Error executing tool {tool_name}: {e}")

        execution_time = time.time() - start_time

        # Save tool performance
        self.tool_performance_tracker.track_performance(
            tool_name, success, execution_time
        )

        # Save tool usage in research memory
        self.memory_db.save_tool_usage(
            project_id=project_id,
            query_id=query_id,
            tool_name=tool_name,
            args=args,
            result=result,
            execution_time=execution_time,
        )

        return result

    async def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke the research assistant.

        Args:
            inputs: Input parameters including query, project_id, and citation_format

        Returns:
            Research results
        """
        query = inputs.get("query", "").lower()
        project_id = inputs.get("project_id", self.default_project.id)
        citation_format = inputs.get("citation_format", "apa")

        # Get or create the project
        project = self.memory_db.get_project(project_id)
        if not project:
            project = self.default_project
            project_id = project.id

        # Add the query to the project
        research_query = self.memory_db.add_query(project_id, query)
        if not research_query:
            raise ValueError(f"Failed to add query to project {project_id}")

        query_id = research_query.id

        # Select tools for this query
        selected_tools = await self.tool_selector.select_tools(query)

        # Execute selected tools
        tool_results = {}
        tools_used = []

        for tool_info in selected_tools.get("selected_tools", []):
            tool_name = tool_info["name"]
            tool_args = tool_info.get("args", {"query": query})

            try:
                result = self._execute_tool(tool_name, tool_args, project_id, query_id)
                tool_results[tool_name] = result
                tools_used.append(tool_name)
            except Exception as e:
                print(f"Error executing tool {tool_name}: {e}")

        # Generate research response using the model
        input_values = {
            "query": query,
            "project_id": project_id,
            "citation_format": citation_format,
        }

        research_message = self.research_prompt.format_messages(**input_values)

        # Add tool results to the message
        tool_results_str = "Tool Results:\n"
        for tool_name, result in tool_results.items():
            tool_results_str += f"\n--- {tool_name} ---\n{result}\n"

        research_message.append(HumanMessage(content=tool_results_str))

        # Get response from model
        model_response = self.model.invoke(research_message)

        # Parse the response
        try:
            response_data = json.loads(model_response.content)
        except json.JSONDecodeError:
            # If the response is not valid JSON, extract it using a simple heuristic
            content = model_response.content
            json_start = content.find("{")
            json_end = content.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                try:
                    response_data = json.loads(json_str)
                except json.JSONDecodeError:
                    # If still not valid, create a basic response
                    response_data = {
                        "topic": query.title(),
                        "summary": content,
                        "sources": [],
                        "tools_used": tools_used,
                    }
            else:
                # Create a basic response
                response_data = {
                    "topic": query.title(),
                    "summary": content,
                    "sources": [],
                    "tools_used": tools_used,
                }

        # Ensure required fields are present
        if "topic" not in response_data:
            response_data["topic"] = query.title()

        if "summary" not in response_data:
            response_data["summary"] = "No summary provided."

        if "sources" not in response_data:
            response_data["sources"] = []

        if "tools_used" not in response_data:
            response_data["tools_used"] = tools_used

        # Add project and query IDs
        response_data["project_id"] = project_id
        response_data["query_id"] = query_id

        # Set citation format if provided
        if citation_format:
            response_data["citation_format"] = citation_format

        # Generate bibliography if sources are available and not already provided
        if (
            "sources" in response_data
            and response_data["sources"]
            and "bibliography" not in response_data
        ):
            try:
                # Convert string sources to Source objects if needed
                sources_data = []
                for source in response_data["sources"]:
                    if isinstance(source, dict):
                        sources_data.append(source)
                    else:
                        sources_data.append({"title": source, "source_type": "web"})

                # Generate bibliography
                bibliography_input = json.dumps(
                    {
                        "sources": sources_data,
                        "format": response_data.get("citation_format", "apa"),
                    }
                )

                bibliography = generate_bibliography_tool.run(bibliography_input)
                response_data["bibliography"] = bibliography
            except Exception as e:
                print(f"Bibliography generation error: {e}")

        # Create a research result and add it to the project
        try:
            # Convert sources to Source objects
            sources = []
            for source_data in response_data["sources"]:
                if isinstance(source_data, dict):
                    source = Source(**source_data)
                else:
                    source = Source(title=source_data)
                sources.append(source)

            # Create a unique ID for the result
            result_id = f"result_{uuid.uuid4().hex[:8]}"

            # Create the research result
            result = ResearchResult(
                id=result_id,
                topic=response_data["topic"],
                summary=response_data["summary"],
                sources=sources,
                tools_used=response_data["tools_used"],
                citation_format=response_data.get("citation_format"),
                bibliography=response_data.get("bibliography"),
                visualizations=response_data.get("visualizations", []),
                tags=response_data.get("tags", []),
            )

            # Add the result to the project
            self.memory_db.add_result(project_id, query_id, result)
        except Exception as e:
            print(f"Error adding result to project: {e}")

        # Return the response in the expected format
        return {"output": json.dumps(response_data)}

    def get_project(self, project_id: str) -> Optional[ResearchProject]:
        """Get a project by ID.

        Args:
            project_id: Project ID

        Returns:
            Project or None if not found
        """
        return self.memory_db.get_project(project_id)

    def get_all_projects(self) -> List[ResearchProject]:
        """Get all projects.

        Returns:
            List of projects
        """
        return self.memory_db.get_all_projects()

    def create_project(
        self, name: str, description: str = "", tags: List[str] = None
    ) -> ResearchProject:
        """Create a new project.

        Args:
            name: Project name
            description: Project description
            tags: Project tags

        Returns:
            Created project
        """
        return self.memory_db.create_project(name, description, tags or [])

    def search_projects(self, search_term: str) -> List[ResearchProject]:
        """Search for projects.

        Args:
            search_term: Search term

        Returns:
            List of matching projects
        """
        return self.memory_db.search_projects(search_term)

    def search_queries(
        self, search_term: str, project_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for queries.

        Args:
            search_term: Search term
            project_id: Optional project ID to filter by

        Returns:
            List of matching queries
        """
        return self.memory_db.search_queries(search_term, project_id)

    def search_results(
        self, search_term: str, project_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for results.

        Args:
            search_term: Search term
            project_id: Optional project ID to filter by

        Returns:
            List of matching results
        """
        return self.memory_db.search_results(search_term, project_id)
