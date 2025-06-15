"""
Main entry point for the Enhanced Research Assistant.
This module provides a command-line interface for the Enhanced Research Assistant
with memory persistence, tool selection, learning capabilities, and reinforcement learning integration.
"""

import asyncio
import json
import os
import traceback

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from src.agents.enhanced_research_assistant import (
    EnhancedResearchAssistant,
    EnhancedResearchResponseModel,
)

load_dotenv()

# Initialize model
model = ChatAnthropic(model=os.getenv("MODEL_NAME", "claude-3-5-sonnet-20240620"))

# Initialize research assistant
research_assistant = EnhancedResearchAssistant(
    model=model,
    db_path=os.getenv("RESEARCH_DB_PATH", "research_memory.db"),
)


async def run_research_assistant():
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
    print("Type 'project create <n>' to create a new project.")
    print("Type 'project select <id>' to select a project.")
    print("Type 'project info' to view current project details.")
    print("Type 'citation <format>' to set citation format (apa, mla, chicago, harvard, ieee).")
    print("Type 'visualize <type>' to create a visualization (chart, mind_map, timeline, network).")
    print("Type 'search projects <term>' to search for projects.")
    print("Type 'search queries <term>' to search for queries.")
    print("Type 'search results <term>' to search for results.")
    print("Type 'help' to see all available commands.")

    try:
        # Get or create the default project
        projects = research_assistant.get_all_projects()
        if projects:
            current_project = projects[0]
            print(f"Using project: {current_project.name} (ID: {current_project.id})")
        else:
            current_project = research_assistant.create_project(
                name="General Research",
                description="Default project for research queries",
                tags=["general", "research"],
            )
            print(f"Created default project: {current_project.name} (ID: {current_project.id})")

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
                print("project create <n> - Create a new project")
                print("project select <id> - Select a project")
                print("project info - View current project details")
                print("citation <format> - Set citation format (apa, mla, chicago, harvard, ieee)")
                print(
                    "visualize <type> - Create a visualization (chart, mind_map, timeline, network)"
                )
                print("search projects <term> - Search for projects")
                print("search queries <term> - Search for queries")
                print("search results <term> - Search for results")
                print("Any other input will be treated as a research query")
                continue

            elif command.lower() == "projects":
                projects = research_assistant.get_all_projects()
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

                project = research_assistant.create_project(
                    name=name, description=description, tags=tags
                )

                current_project = project
                print(f"Created project: {project.name} (ID: {project.id})")
                continue

            elif command.lower().startswith("project select "):
                project_id = command[15:].strip()
                if not project_id:
                    print("Please provide a project ID.")
                    continue

                project = research_assistant.get_project(project_id)
                if not project:
                    print(f"Project with ID {project_id} not found.")
                    continue

                current_project = project
                print(f"Selected project: {project.name} (ID: {project.id})")
                continue

            elif command.lower() == "project info":
                if not current_project:
                    print("No project selected.")
                    continue

                print(f"\n=== Project: {current_project.name} ===")
                print(f"ID: {current_project.id}")
                print(f"Description: {current_project.description}")
                print(f"Tags: {', '.join(current_project.tags)}")
                print(f"Created: {current_project.created_at}")
                print(f"Updated: {current_project.updated_at}")
                print(f"Queries: {len(current_project.queries)}")

                if current_project.queries:
                    print("\nRecent Queries:")
                    for i, query in enumerate(current_project.queries[-5:], 1):
                        print(f"{i}. {query.query} (ID: {query.id})")
                        print(f"   Results: {len(query.results)}")
                        print(f"   Created: {query.created_at}")
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

            elif command.lower().startswith("search projects "):
                search_term = command[15:].strip()
                if not search_term:
                    print("Please provide a search term.")
                    continue

                projects = research_assistant.search_projects(search_term)
                if not projects:
                    print(f"No projects found matching '{search_term}'.")
                else:
                    print(f"\n=== Projects matching '{search_term}' ===")
                    for i, project in enumerate(projects, 1):
                        print(f"{i}. {project.name} (ID: {project.id})")
                        print(f"   Description: {project.description}")
                        print(f"   Tags: {', '.join(project.tags)}")
                        print()
                continue

            elif command.lower().startswith("search queries "):
                search_term = command[14:].strip()
                if not search_term:
                    print("Please provide a search term.")
                    continue

                project_id = current_project.id if current_project else None
                queries = research_assistant.search_queries(search_term, project_id)
                if not queries:
                    print(f"No queries found matching '{search_term}'.")
                else:
                    print(f"\n=== Queries matching '{search_term}' ===")
                    for i, query in enumerate(queries, 1):
                        print(f"{i}. {query['query']} (ID: {query['query_id']})")
                        print(f"   Project: {query['project_name']} (ID: {query['project_id']})")
                        print(f"   Created: {query['created_at']}")
                        print()
                continue

            elif command.lower().startswith("search results "):
                search_term = command[15:].strip()
                if not search_term:
                    print("Please provide a search term.")
                    continue

                project_id = current_project.id if current_project else None
                results = research_assistant.search_results(search_term, project_id)
                if not results:
                    print(f"No results found matching '{search_term}'.")
                else:
                    print(f"\n=== Results matching '{search_term}' ===")
                    for i, result in enumerate(results, 1):
                        print(f"{i}. {result['topic']} (ID: {result['result_id']})")
                        print(f"   Query: {result['query']} (ID: {result['query_id']})")
                        print(f"   Project: {result['project_name']} (ID: {result['project_id']})")
                        print(f"   Summary: {result['summary'][:100]}...")
                        print(f"   Tags: {', '.join(result['tags'])}")
                        print(f"   Created: {result['created_at']}")
                        print()
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
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(content)
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
                    from src.tools.research_assistant_tools import (
                        export_to_docx_tool,
                        export_to_html_tool,
                        export_to_markdown_tool,
                        export_to_pdf_tool,
                        export_to_presentation_tool,
                    )

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

            elif command.lower().startswith("visualize ") and last_response:
                viz_type = command[10:].strip().lower()
                if viz_type not in ["chart", "mind_map", "timeline", "network"]:
                    print(f"Unsupported visualization type: {viz_type}")
                    print("Supported types: chart, mind_map, timeline, network")
                    continue

                # Create a simple visualization based on the last response
                try:
                    from src.tools.research_assistant_tools import (
                        generate_chart_tool,
                        generate_mind_map_tool,
                        generate_network_diagram_tool,
                        generate_timeline_tool,
                    )

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
                inputs = {"query": command}

                # Add project ID if available
                if current_project:
                    inputs["project_id"] = current_project.id

                # Add citation format
                inputs["citation_format"] = citation_format

                # Invoke the agent
                raw_response = await research_assistant.invoke(inputs)

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
                    project = research_assistant.get_project(enhanced_response.project_id)
                    if project:
                        print(f"\nProject: {project.name} (ID: {enhanced_response.project_id})")

                # Print tags if available
                if enhanced_response.tags:
                    print(f"\nTags: {', '.join(enhanced_response.tags)}")

            except Exception as e:
                print(f"Error processing response: {e}")
                traceback.print_exc()
                if "raw_response" in locals():
                    print(f"Raw Response: {raw_response}")
                else:
                    print("No response received from the agent.")

    except KeyboardInterrupt:
        print("\nResearch assistant terminated by user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()

    print("\nThank you for using the Enhanced Research Assistant!")


if __name__ == "__main__":
    asyncio.run(run_research_assistant())
