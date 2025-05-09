"""
Main entry point for the RL-Enhanced Research Assistant.
This module provides a command-line interface for the RL-Enhanced Research Assistant
with memory persistence, tool selection, learning capabilities, and reinforcement learning integration.
"""

import asyncio
import json
import os
import traceback

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from src.agents.enhanced_research_assistant import EnhancedResearchResponseModel
from src.agents.research_rl_integration import RLEnhancedResearchAssistant

load_dotenv()

# Initialize model
model = ChatAnthropic(model=os.getenv("MODEL_NAME", "claude-3-5-sonnet-20240620"))

# Initialize RL-enhanced research assistant
research_assistant = RLEnhancedResearchAssistant(
    model=model,
    db_path=os.getenv("RESEARCH_DB_PATH", "research_memory.db"),
    learning_rate=float(os.getenv("RL_LEARNING_RATE", "0.1")),
    discount_factor=float(os.getenv("RL_DISCOUNT_FACTOR", "0.9")),
    exploration_rate=float(os.getenv("RL_EXPLORATION_RATE", "0.2")),
)


async def run_research_assistant():
    """
    Run the RL-enhanced research assistant with user input and handle the response.
    Supports multiple queries in a session with chat history, project management,
    citation formatting, visualization, and reinforcement learning.
    """
    chat_history = []
    current_project = None
    current_query = None
    last_response = None
    citation_format = "apa"

    print("=== RL-Enhanced Research Assistant ===")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'save' to save the last research results to a file.")
    print(
        "Type 'export <format>' to export results (formats: md, html, pdf, docx, pptx)."
    )
    print("Type 'projects' to list all research projects.")
    print("Type 'project create <n>' to create a new project.")
    print("Type 'project select <id>' to select a project.")
    print("Type 'project info' to view current project details.")
    print(
        "Type 'citation <format>' to set citation format (apa, mla, chicago, harvard, ieee)."
    )
    print(
        "Type 'visualize <type>' to create a visualization (chart, mind_map, timeline, network)."
    )
    print("Type 'search projects <term>' to search for projects.")
    print("Type 'search queries <term>' to search for queries.")
    print("Type 'search results <term>' to search for results.")
    print("Type 'feedback <rating>' to provide feedback on the last research result.")
    print("Type 'rl info' to view reinforcement learning information.")
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
            print(
                f"Created default project: {current_project.name} (ID: {current_project.id})"
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
                print(
                    "export <format> - Export results (formats: md, html, pdf, docx, pptx)"
                )
                print("projects - List all research projects")
                print("project create <n> - Create a new project")
                print("project select <id> - Select a project")
                print("project info - View current project details")
                print(
                    "citation <format> - Set citation format (apa, mla, chicago, harvard, ieee)"
                )
                print(
                    "visualize <type> - Create a visualization (chart, mind_map, timeline, network)"
                )
                print("search projects <term> - Search for projects")
                print("search queries <term> - Search for queries")
                print("search results <term> - Search for results")
                print(
                    "feedback <rating> - Provide feedback on the last research result"
                )
                print("rl info - View reinforcement learning information")
                print("Any other input will be treated as a research query")
                continue

            elif command.lower().startswith("feedback ") and last_response:
                feedback = command[9:].strip()
                if not feedback:
                    print("Please provide feedback text.")
                    continue

                print("Processing feedback...")
                try:
                    # Parse the last response
                    response_data = {
                        "topic": last_response.topic,
                        "summary": last_response.summary,
                        "sources": last_response.sources,
                        "tools_used": last_response.tools_used,
                        "citation_format": last_response.citation_format,
                        "bibliography": last_response.bibliography,
                        "project_id": last_response.project_id,
                        "query_id": last_response.query_id,
                        "visualizations": last_response.visualizations,
                        "tags": last_response.tags,
                    }

                    # Get the last query
                    last_query = (
                        chat_history[-1][0] if chat_history else "Unknown query"
                    )

                    # Update from feedback
                    learning_results = await research_assistant.update_from_feedback(
                        query=last_query,
                        response=response_data,
                        feedback=feedback,
                    )

                    print("\n=== Learning Results ===")
                    print(f"State: {learning_results['state']}")
                    print(f"Tools used: {', '.join(learning_results['tools_used'])}")
                    print(f"Reward: {learning_results['reward']}")
                    print("\nReward components:")
                    for component, value in learning_results[
                        "reward_components"
                    ].items():
                        print(f"- {component}: {value}")
                    print("\nFeedback analysis:")
                    print(learning_results["feedback"])

                except Exception as e:
                    print(f"Error processing feedback: {e}")
                    traceback.print_exc()

                continue

            elif command.lower() == "rl info":
                print("\n=== Reinforcement Learning Information ===")
                try:
                    # Get Q-table
                    q_table = research_assistant.rl_agent.q_table

                    print(f"Q-table size: {len(q_table)} states")
                    print("\nTop states:")
                    for i, (state, actions) in enumerate(q_table.items()):
                        if i >= 5:
                            break
                        print(f"- State: {state}")
                        print("  Actions:")
                        sorted_actions = sorted(
                            actions.items(), key=lambda x: x[1], reverse=True
                        )
                        for action, value in sorted_actions[:3]:
                            print(f"  - {action}: {value:.2f}")

                    # Get recent rewards
                    rewards = research_assistant.memory_db.get_agent_rewards(
                        "research_assistant", 5
                    )

                    print("\nRecent rewards:")
                    for reward_data in rewards:
                        print(f"- Reward: {reward_data['reward']:.2f}")
                        print("  Components:")
                        for component, value in reward_data["components"].items():
                            print(f"  - {component}: {value:.2f}")

                    # Get learning parameters
                    print("\nLearning parameters:")
                    print(
                        f"- Learning rate: {research_assistant.rl_agent.learning_rate}"
                    )
                    print(
                        f"- Discount factor: {research_assistant.rl_agent.discount_factor}"
                    )
                    print(
                        f"- Exploration rate: {research_assistant.rl_agent.exploration_rate}"
                    )

                except Exception as e:
                    print(f"Error retrieving RL information: {e}")
                    traceback.print_exc()

                continue

            # Handle other commands (projects, project create, project select, etc.)
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
                tags = (
                    [tag.strip() for tag in tags_input.split(",")]
                    if tags_input.strip()
                    else []
                )

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
                        print(
                            f"   Project: {query['project_name']} (ID: {query['project_id']})"
                        )
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
                        print(
                            f"   Project: {result['project_name']} (ID: {result['project_id']})"
                        )
                        print(f"   Summary: {result['summary'][:100]}...")
                        print(f"   Tags: {', '.join(result['tags'])}")
                        print(f"   Created: {result['created_at']}")
                        print()
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
                    project = research_assistant.get_project(
                        enhanced_response.project_id
                    )
                    if project:
                        print(
                            f"\nProject: {project.name} (ID: {enhanced_response.project_id})"
                        )

                # Print tags if available
                if enhanced_response.tags:
                    print(f"\nTags: {', '.join(enhanced_response.tags)}")

                # Prompt for feedback
                print(
                    "\nYou can provide feedback on this research by typing 'feedback <rating>'."
                )

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

    print("\nThank you for using the RL-Enhanced Research Assistant!")


if __name__ == "__main__":
    asyncio.run(run_research_assistant())
