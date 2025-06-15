import json
from datetime import datetime
from typing import Dict, List, Optional, Union

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.tools.research_assistant_tools import save_tool, search_tool, wiki_tool


# Import mock tools for testing
class MockTool:
    def __init__(self, name):
        self.name = name

    def run(self, query):
        return f"Mock {self.name} result for: {query}"


# Create mock tools for academic sources
google_scholar_tool = MockTool("Google Scholar")
pubmed_tool = MockTool("PubMed")
arxiv_tool = MockTool("arXiv")
google_books_tool = MockTool("Google Books")
open_library_tool = MockTool("Open Library")

# Create mock tools for citation formatting
format_citation_tool = MockTool("Citation Formatter")
generate_bibliography_tool = MockTool("Bibliography Generator")

# Create mock tools for exporting
export_to_markdown_tool = MockTool("Markdown Exporter")
export_to_html_tool = MockTool("HTML Exporter")
export_to_pdf_tool = MockTool("PDF Exporter")
export_to_docx_tool = MockTool("DOCX Exporter")
export_to_presentation_tool = MockTool("Presentation Exporter")

# Create mock tools for visualization
generate_chart_tool = MockTool("Chart Generator")
generate_mind_map_tool = MockTool("Mind Map Generator")
generate_timeline_tool = MockTool("Timeline Generator")
generate_network_diagram_tool = MockTool("Network Diagram Generator")


# Create mock research project manager
class MockResearchProjectManager:
    def __init__(self):
        self.projects = {}
        self.next_id = 1

    def create_project(self, name, description="", tags=None):
        project_id = f"project_{self.next_id}"
        self.next_id += 1

        project = {
            "id": project_id,
            "name": name,
            "description": description,
            "tags": tags or [],
            "queries": [],
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
        }

        self.projects[project_id] = project
        return project

    def get_project(self, project_id):
        return self.projects.get(project_id)

    def get_all_projects(self):
        return list(self.projects.values())

    def add_query(self, project_id, query):
        project = self.get_project(project_id)
        if not project:
            return None

        query_id = f"query_{len(project['queries']) + 1}"
        query_obj = {
            "id": query_id,
            "query": query,
            "results": [],
            "created_at": datetime.now(),
        }

        project["queries"].append(query_obj)
        project["updated_at"] = datetime.now()

        return query_obj

    def add_result(self, project_id, query_id, result):
        project = self.get_project(project_id)
        if not project:
            return False

        for query in project["queries"]:
            if query["id"] == query_id:
                query["results"].append(result)
                project["updated_at"] = datetime.now()
                return True

        return False


# Create a mock research project manager instance
research_project_manager = MockResearchProjectManager()


# Create mock models
class Source:
    def __init__(self, title, **kwargs):
        self.title = title
        for key, value in kwargs.items():
            setattr(self, key, value)


class SourceType:
    WEB = "web"
    WIKIPEDIA = "wikipedia"
    ACADEMIC = "academic"
    BOOK = "book"
    JOURNAL = "journal"


class ResearchResult:
    def __init__(self, topic, summary, sources, tools_used, tags=None):
        self.topic = topic
        self.summary = summary
        self.sources = sources
        self.tools_used = tools_used
        self.tags = tags or []


load_dotenv()


class ResearchResponse(BaseModel):
    """
    Structured response format for research results.
    """

    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


# Define the enhanced research response model
class EnhancedResearchResponseModel(BaseModel):
    """
    Enhanced structured response format for research results with advanced features.
    """

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


# Use a mock agent for testing purposes
print("Using an enhanced mock research agent for testing purposes.")


# Create an enhanced mock agent that returns predefined responses with advanced features
class EnhancedMockResearchAgent:
    def __init__(self):
        # Create a default project for storing research
        self.default_project = research_project_manager.create_project(
            name="General Research",
            description="Default project for research queries",
            tags=["general", "research"],
        )

        # Initialize responses with enhanced features
        self.responses = {
            "machine learning": {
                "topic": "Machine Learning",
                "summary": "Machine Learning is a subset of artificial intelligence that enables computers to learn from data and improve their performance without being explicitly programmed. It involves algorithms that can analyze data, identify patterns, and make predictions or decisions. Key approaches include supervised learning (using labeled data), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through trial and error with rewards).",
                "sources": [
                    {
                        "title": "Introduction to Machine Learning",
                        "authors": ["Andrew Ng"],
                        "url": "https://www.coursera.org/learn/machine-learning",
                        "source_type": "academic",
                        "publisher": "Coursera",
                    },
                    {
                        "title": "Machine Learning: A Probabilistic Perspective",
                        "authors": ["Kevin P. Murphy"],
                        "source_type": "book",
                        "publisher": "MIT Press",
                        "isbn": "978-0262018029",
                    },
                    {
                        "title": "Machine Learning",
                        "url": "https://en.wikipedia.org/wiki/Machine_learning",
                        "source_type": "wikipedia",
                    },
                ],
                "tools_used": ["search", "wikipedia", "google_scholar"],
                "citation_format": "apa",
                "tags": ["machine learning", "artificial intelligence", "data science"],
                "visualizations": [
                    {
                        "title": "Machine Learning Approaches",
                        "type": "chart",
                        "data": {
                            "labels": [
                                "Supervised Learning",
                                "Unsupervised Learning",
                                "Reinforcement Learning",
                            ],
                            "values": [60, 25, 15],
                        },
                    },
                    {
                        "title": "Machine Learning Concepts",
                        "type": "mind_map",
                        "data": {
                            "central_topic": "Machine Learning",
                            "branches": [
                                {
                                    "name": "Supervised Learning",
                                    "sub_branches": ["Classification", "Regression"],
                                },
                                {
                                    "name": "Unsupervised Learning",
                                    "sub_branches": [
                                        "Clustering",
                                        "Dimensionality Reduction",
                                    ],
                                },
                                {
                                    "name": "Reinforcement Learning",
                                    "sub_branches": ["Q-Learning", "Policy Gradients"],
                                },
                            ],
                        },
                    },
                ],
            },
            "python programming": {
                "topic": "Python Programming",
                "summary": "Python is a high-level, interpreted programming language known for its readability and simplicity. It supports multiple programming paradigms including procedural, object-oriented, and functional programming. Python has a comprehensive standard library and a vast ecosystem of third-party packages, making it suitable for various applications such as web development, data analysis, artificial intelligence, and scientific computing.",
                "sources": [
                    {
                        "title": "Python Documentation",
                        "url": "https://docs.python.org/",
                        "source_type": "web",
                    },
                    {
                        "title": "Fluent Python",
                        "authors": ["Luciano Ramalho"],
                        "source_type": "book",
                        "publisher": "O'Reilly Media",
                        "isbn": "978-1491946008",
                    },
                    {
                        "title": "Python (programming language)",
                        "url": "https://en.wikipedia.org/wiki/Python_(programming_language)",
                        "source_type": "wikipedia",
                    },
                ],
                "tools_used": ["search", "wikipedia", "google_books"],
                "citation_format": "mla",
                "tags": ["python", "programming", "software development"],
                "visualizations": [
                    {
                        "title": "Python Applications",
                        "type": "pie",
                        "data": {
                            "labels": [
                                "Web Development",
                                "Data Science",
                                "AI/ML",
                                "Automation",
                                "Other",
                            ],
                            "values": [30, 25, 20, 15, 10],
                        },
                    },
                    {
                        "title": "Python Timeline",
                        "type": "timeline",
                        "data": {
                            "events": [
                                {
                                    "date": "1991",
                                    "description": "Python 0.9.0 released",
                                },
                                {"date": "2000", "description": "Python 2.0 released"},
                                {"date": "2008", "description": "Python 3.0 released"},
                                {
                                    "date": "2020",
                                    "description": "Python 2 reaches end of life",
                                },
                                {"date": "2023", "description": "Python 3.11 released"},
                            ]
                        },
                    },
                ],
            },
            "default": {
                "topic": "Research Topic",
                "summary": "This is a mock summary for testing purposes. The research assistant is working correctly, but is using a mock agent for demonstration.",
                "sources": [
                    {
                        "title": "Mock Source 1",
                        "url": "https://example.com/mock1",
                        "source_type": "web",
                    },
                    {
                        "title": "Mock Source 2",
                        "url": "https://example.com/mock2",
                        "source_type": "web",
                    },
                ],
                "tools_used": ["search", "wikipedia"],
                "citation_format": "apa",
                "tags": ["research", "mock"],
                "visualizations": [],
            },
        }

    def invoke(self, inputs):
        query = inputs.get("query", "").lower()
        project_id = inputs.get("project_id", self.default_project.id)
        citation_format = inputs.get("citation_format", "apa")

        # Get or create the project
        project = research_project_manager.get_project(project_id)
        if not project:
            project = self.default_project

        # Add the query to the project
        research_query = project.add_query(query)

        # Use search tool to simulate real tool usage
        try:
            search_result = search_tool.run(query)
            print(f"\nSearch result: {search_result[:100]}...\n")
        except Exception as e:
            print(f"Search tool error: {e}")

        # Use wiki tool to simulate real tool usage
        try:
            wiki_result = wiki_tool.run(query)
            print(f"\nWikipedia result: {wiki_result[:100]}...\n")
        except Exception as e:
            print(f"Wikipedia tool error: {e}")

        # Use academic tools to simulate real tool usage
        try:
            if "machine learning" in query or "artificial intelligence" in query:
                google_scholar_result = google_scholar_tool.run(query)
                print(f"\nGoogle Scholar result: {google_scholar_result[:100]}...\n")

                arxiv_result = arxiv_tool.run(query)
                print(f"\narXiv result: {arxiv_result[:100]}...\n")
            elif "book" in query or "literature" in query:
                google_books_result = google_books_tool.run(query)
                print(f"\nGoogle Books result: {google_books_result[:100]}...\n")

                open_library_result = open_library_tool.run(query)
                print(f"\nOpen Library result: {open_library_result[:100]}...\n")
            elif "medical" in query or "health" in query:
                pubmed_result = pubmed_tool.run(query)
                print(f"\nPubMed result: {pubmed_result[:100]}...\n")
        except Exception as e:
            print(f"Academic tool error: {e}")

        # Get the appropriate response based on the query
        for key in self.responses:
            if key in query:
                response = self.responses[key].copy()
                break
        else:
            response = self.responses["default"].copy()
            response["topic"] = query.title()

        # Add project and query IDs
        response["project_id"] = project.id
        response["query_id"] = research_query.id

        # Set citation format if provided
        if citation_format:
            response["citation_format"] = citation_format

        # Generate bibliography if sources are available
        if "sources" in response and response["sources"]:
            try:
                # Convert string sources to Source objects if needed
                sources_data = []
                for source in response["sources"]:
                    if isinstance(source, dict):
                        sources_data.append(source)
                    else:
                        sources_data.append({"title": source, "source_type": "web"})

                # Generate bibliography
                bibliography_input = json.dumps(
                    {
                        "sources": sources_data,
                        "format": response.get("citation_format", "apa"),
                    }
                )

                bibliography = generate_bibliography_tool.run(bibliography_input)
                response["bibliography"] = bibliography
            except Exception as e:
                print(f"Bibliography generation error: {e}")

        # Create a research result and add it to the project
        try:
            # Convert sources to Source objects
            sources = []
            for source_data in response["sources"]:
                if isinstance(source_data, dict):
                    source = Source(**source_data)
                else:
                    source = Source(title=source_data, source_type=SourceType.WEB)
                sources.append(source)

            # Create the research result
            result = ResearchResult(
                topic=response["topic"],
                summary=response["summary"],
                sources=sources,
                tools_used=response["tools_used"],
                tags=response.get("tags", []),
            )

            # Add the result to the project
            research_project_manager.add_result(project.id, research_query.id, result)
        except Exception as e:
            print(f"Error adding result to project: {e}")

        # Return the response in the expected format
        return {"output": json.dumps(response)}


# Create the enhanced mock agent
agent_executor = EnhancedMockResearchAgent()


def run_research_assistant():
    """
    Run the enhanced research assistant with user input and handle the response.
    Supports multiple queries in a session with chat history, project management,
    citation formatting, and visualization.
    """
    chat_history = []
    current_project = None
    current_query = None
    last_response = None
    citation_format = "apa"

    print("=== Enhanced Research Assistant ===")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'save' to save the last research results to a file.")
    print("Type 'export <format>' to export results (formats: md, html, pdf, docx, pptx).")
    print("Type 'projects' to list all research projects.")
    print("Type 'project create <name>' to create a new project.")
    print("Type 'project select <id>' to select a project.")
    print("Type 'project info' to view current project details.")
    print("Type 'citation <format>' to set citation format (apa, mla, chicago, harvard, ieee).")
    print("Type 'visualize <type>' to create a visualization (chart, mind_map, timeline, network).")
    print("Type 'help' to see all available commands.")

    try:
        # Get or create the default project
        projects = research_project_manager.get_all_projects()
        if projects:
            current_project = projects[0]
            print(f"Using project: {current_project['name']} (ID: {current_project['id']})")
        else:
            current_project = research_project_manager.create_project(
                name="General Research",
                description="Default project for research queries",
                tags=["general", "research"],
            )
            print(
                f"Created default project: {current_project['name']} (ID: {current_project['id']})"
            )

        while True:
            command = input("\nWhat can I help you research? ")
            command = command.strip()

            if not command:
                print("Please provide a research topic or command.")
                continue

            # Handle commands
            if command.lower() in ["exit", "quit"]:
                print("Ending research session. Goodbye!")
                break

            elif command.lower() == "help":
                print("\n=== Available Commands ===")
                print("exit, quit - End the session")
                print("save - Save the last research results to a file")
                print("export <format> - Export results (formats: md, html, pdf, docx, pptx)")
                print("projects - List all research projects")
                print("project create <name> - Create a new project")
                print("project select <id> - Select a project")
                print("project info - View current project details")
                print("citation <format> - Set citation format (apa, mla, chicago, harvard, ieee)")
                print(
                    "visualize <type> - Create a visualization (chart, mind_map, timeline, network)"
                )
                print("Any other input will be treated as a research query")
                continue

            elif command.lower() == "save" and last_response:
                filename = input("Enter filename to save results (default: research_output.txt): ")
                if not filename.strip():
                    filename = "research_output.txt"

                # Create content with bibliography if available
                content = f"{last_response.topic}\n\n{last_response.summary}\n\n"

                # Add sources
                content += "Sources:\n"
                for i, source in enumerate(last_response.sources, 1):
                    if isinstance(source, dict):
                        content += f"{i}. {source.get('title', 'Unknown')} - {source.get('url', 'No URL')}\n"
                    else:
                        content += f"{i}. {source}\n"

                # Add bibliography if available
                if hasattr(last_response, "bibliography") and last_response.bibliography:
                    content += f"\nBibliography ({last_response.citation_format}):\n{last_response.bibliography}\n"

                # Save the content
                save_tool.run(content, filename)
                print(f"Research saved to {filename}")
                continue

            elif command.lower().startswith("export ") and last_response:
                parts = command.split()
                if len(parts) < 2:
                    print("Please specify an export format (md, html, pdf, docx, pptx).")
                    continue

                export_format = parts[1].lower()
                filename = input(
                    f"Enter filename to export results (default: research_output.{export_format}): "
                )
                if not filename.strip():
                    filename = f"research_output.{export_format}"

                # Prepare research data for export
                research_data = {
                    "topic": last_response.topic,
                    "summary": last_response.summary,
                    "sources": last_response.sources,
                    "tools_used": last_response.tools_used,
                }

                # Add bibliography if available
                if hasattr(last_response, "bibliography") and last_response.bibliography:
                    research_data["bibliography"] = last_response.bibliography
                    research_data["citation_format"] = last_response.citation_format

                # Add visualizations if available
                if hasattr(last_response, "visualizations") and last_response.visualizations:
                    research_data["visualizations"] = last_response.visualizations

                # Export the research data
                export_input = json.dumps({"research_data": research_data, "filename": filename})

                try:
                    if export_format == "md":
                        result = export_to_markdown_tool.run(export_input)
                    elif export_format == "html":
                        result = export_to_html_tool.run(export_input)
                    elif export_format == "pdf":
                        result = export_to_pdf_tool.run(export_input)
                    elif export_format == "docx":
                        result = export_to_docx_tool.run(export_input)
                    elif export_format in ["pptx", "presentation"]:
                        result = export_to_presentation_tool.run(export_input)
                    else:
                        print(f"Unsupported export format: {export_format}")
                        continue

                    print(result)
                except Exception as e:
                    print(f"Error exporting research: {e}")

                continue

            elif command.lower() == "projects":
                projects = research_project_manager.get_all_projects()
                if not projects:
                    print("No research projects found.")
                else:
                    print("\n=== Research Projects ===")
                    for i, project in enumerate(projects, 1):
                        print(f"{i}. {project.name} (ID: {project.id})")
                        print(f"   Description: {project.description}")
                        print(f"   Queries: {len(project.queries)}")
                        print(f"   Tags: {', '.join(project.tags)}")
                        print(f"   Created: {project.created_at}")
                        print(f"   Updated: {project.updated_at}")
                        print()
                continue

            elif command.lower().startswith("project create "):
                name = command[14:].strip()
                if not name:
                    print("Please provide a project name.")
                    continue

                description = input("Enter project description (optional): ")
                tags_input = input("Enter project tags (comma-separated, optional): ")
                tags = [tag.strip() for tag in tags_input.split(",")] if tags_input.strip() else []

                project = research_project_manager.create_project(
                    name=name, description=description, tags=tags
                )

                current_project = project
                print(f"Created project: {project['name']} (ID: {project['id']})")
                continue

            elif command.lower().startswith("project select "):
                project_id = command[15:].strip()
                if not project_id:
                    print("Please provide a project ID.")
                    continue

                project = research_project_manager.get_project(project_id)
                if not project:
                    print(f"Project with ID {project_id} not found.")
                    continue

                current_project = project
                print(f"Selected project: {project['name']} (ID: {project['id']})")
                continue

            elif command.lower() == "project info":
                if not current_project:
                    print("No project selected.")
                    continue

                print(f"\n=== Project: {current_project['name']} ===")
                print(f"ID: {current_project['id']}")
                print(f"Description: {current_project['description']}")
                print(f"Tags: {', '.join(current_project['tags'])}")
                print(f"Created: {current_project['created_at']}")
                print(f"Updated: {current_project['updated_at']}")
                print(f"Queries: {len(current_project['queries'])}")

                if current_project["queries"]:
                    print("\nRecent Queries:")
                    for i, query in enumerate(current_project["queries"][-5:], 1):
                        print(f"{i}. {query['query']} (ID: {query['id']})")
                        print(f"   Results: {len(query['results'])}")
                        print(f"   Created: {query['created_at']}")
                        print()

                continue

            elif command.lower().startswith("citation "):
                format_name = command[9:].strip().lower()
                if format_name in ["apa", "mla", "chicago", "harvard", "ieee"]:
                    citation_format = format_name
                    print(f"Citation format set to {citation_format}.")
                else:
                    print(f"Unsupported citation format: {format_name}")
                    print("Supported formats: apa, mla, chicago, harvard, ieee")
                continue

            elif command.lower().startswith("visualize ") and last_response:
                viz_type = command[10:].strip().lower()
                if viz_type not in ["chart", "mind_map", "timeline", "network"]:
                    print(f"Unsupported visualization type: {viz_type}")
                    print("Supported types: chart, mind_map, timeline, network")
                    continue

                # Create a simple visualization based on the last response
                try:
                    if viz_type == "chart":
                        # Create a simple bar chart of source types
                        source_types = {}
                        for source in last_response.sources:
                            if isinstance(source, dict):
                                source_type = source.get("source_type", "unknown")
                            else:
                                source_type = "unknown"

                            source_types[source_type] = source_types.get(source_type, 0) + 1

                        chart_data = {
                            "labels": list(source_types.keys()),
                            "values": list(source_types.values()),
                        }

                        chart_input = json.dumps(
                            {
                                "data": chart_data,
                                "type": "bar",
                                "title": f"Source Types for {last_response.topic}",
                            }
                        )

                        result = generate_chart_tool.run(chart_input)
                        print(result)

                    elif viz_type == "mind_map":
                        # Create a simple mind map of the research topic
                        mind_map_data = {
                            "central_topic": last_response.topic,
                            "branches": [
                                {
                                    "name": "Sources",
                                    "sub_branches": [
                                        (
                                            source.get("title", source)
                                            if isinstance(source, dict)
                                            else source
                                        )
                                        for source in last_response.sources[:3]
                                    ],
                                },
                                {
                                    "name": "Tools Used",
                                    "sub_branches": last_response.tools_used,
                                },
                            ],
                        }

                        mind_map_input = json.dumps(
                            {
                                "data": mind_map_data,
                                "title": f"Mind Map for {last_response.topic}",
                            }
                        )

                        result = generate_mind_map_tool.run(mind_map_input)
                        print(result)

                    elif viz_type == "timeline":
                        # Create a simple timeline of the research process
                        timeline_data = {
                            "events": [
                                {
                                    "date": "Step 1",
                                    "description": "Research query submitted",
                                },
                                {
                                    "date": "Step 2",
                                    "description": "Web search performed",
                                },
                                {
                                    "date": "Step 3",
                                    "description": "Wikipedia consulted",
                                },
                                {
                                    "date": "Step 4",
                                    "description": "Academic sources reviewed",
                                },
                                {
                                    "date": "Step 5",
                                    "description": "Results synthesized",
                                },
                            ]
                        }

                        timeline_input = json.dumps(
                            {
                                "data": timeline_data,
                                "title": f"Research Process for {last_response.topic}",
                            }
                        )

                        result = generate_timeline_tool.run(timeline_input)
                        print(result)

                    elif viz_type == "network":
                        # Create a simple network diagram of the research topic
                        network_data = {
                            "nodes": [
                                {"id": 0, "label": last_response.topic},
                                {"id": 1, "label": "Sources"},
                                {"id": 2, "label": "Tools"},
                            ],
                            "edges": [
                                {"source": 0, "target": 1, "label": "uses"},
                                {"source": 0, "target": 2, "label": "requires"},
                            ],
                        }

                        # Add source nodes
                        for i, source in enumerate(last_response.sources[:3], 3):
                            source_label = (
                                source.get("title", source) if isinstance(source, dict) else source
                            )
                            network_data["nodes"].append({"id": i, "label": source_label})
                            network_data["edges"].append(
                                {"source": 1, "target": i, "label": "includes"}
                            )

                        # Add tool nodes
                        for i, tool in enumerate(last_response.tools_used[:3], 6):
                            network_data["nodes"].append({"id": i, "label": tool})
                            network_data["edges"].append(
                                {"source": 2, "target": i, "label": "includes"}
                            )

                        network_input = json.dumps(
                            {
                                "data": network_data,
                                "title": f"Network Diagram for {last_response.topic}",
                            }
                        )

                        result = generate_network_diagram_tool.run(network_input)
                        print(result)

                except Exception as e:
                    print(f"Error generating visualization: {e}")

                continue

            # Handle research query
            print(f"Researching: {command}...")

            try:
                # Prepare the inputs for the agent
                inputs = {"query": command, "chat_history": chat_history}

                # Add project ID if available
                if current_project:
                    inputs["project_id"] = current_project.id

                # Add citation format
                inputs["citation_format"] = citation_format

                # Invoke the agent
                raw_response = agent_executor.invoke(inputs)

                # Handle the agent's response format
                output = raw_response.get("output")

                # Parse the JSON response
                response_data = json.loads(output)

                # Create an enhanced response object
                enhanced_response = EnhancedResearchResponseModel(
                    topic=response_data["topic"],
                    summary=response_data["summary"],
                    sources=response_data["sources"],
                    tools_used=response_data["tools_used"],
                    citation_format=response_data.get("citation_format"),
                    bibliography=response_data.get("bibliography"),
                    project_id=response_data.get("project_id"),
                    query_id=response_data.get("query_id"),
                    visualizations=response_data.get("visualizations", []),
                    tags=response_data.get("tags", []),
                )

                last_response = enhanced_response

                # Update current query ID
                if enhanced_response.query_id:
                    current_query = enhanced_response.query_id

                # Update chat history
                chat_history.append((command, enhanced_response.summary))

                # Print results
                print("\n--- Research Results ---")
                print(f"Topic: {enhanced_response.topic}")
                print(f"Summary: {enhanced_response.summary}")

                # Print sources
                print("\nSources:")
                for i, source in enumerate(enhanced_response.sources, 1):
                    if isinstance(source, dict):
                        title = source.get("title", f"Source {i}")
                        url = source.get("url", "")
                        authors = source.get("authors", [])

                        print(f"{i}. {title}")
                        if authors:
                            print(f"   Authors: {', '.join(authors)}")
                        if url:
                            print(f"   URL: {url}")
                    else:
                        print(f"{i}. {source}")

                # Print tools used
                print(f"\nTools used: {', '.join(enhanced_response.tools_used)}")

                # Print bibliography if available
                if enhanced_response.bibliography:
                    print(f"\nBibliography ({enhanced_response.citation_format}):")
                    print(enhanced_response.bibliography)

                # Print project and query information
                if enhanced_response.project_id:
                    project = research_project_manager.get_project(enhanced_response.project_id)
                    if project:
                        print(f"\nProject: {project['name']} (ID: {enhanced_response.project_id})")

                # Print tags if available
                if enhanced_response.tags:
                    print(f"\nTags: {', '.join(enhanced_response.tags)}")

            except Exception as e:
                print(f"Error processing response: {e}")
                import traceback

                traceback.print_exc()
                if "raw_response" in locals():
                    print(f"Raw Response: {raw_response}")
                else:
                    print("No response received from the agent.")

    except KeyboardInterrupt:
        print("\nResearch assistant terminated by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()

    print("\nThank you for using the Enhanced Research Assistant!")


if __name__ == "__main__":
    run_research_assistant()
