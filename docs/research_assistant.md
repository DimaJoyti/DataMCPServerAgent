# Enhanced Research Assistant

The Enhanced Research Assistant is a powerful agent designed to help users gather, organize, and visualize information on various topics. It uses a combination of web search, Wikipedia, academic databases, and other sources to provide comprehensive research results with advanced features for citation, visualization, and project management.

## Features

### Core Features

- **Topic Research**: Research any topic using multiple sources including web search, Wikipedia, academic databases, and books
- **Structured Results**: Get structured research results with topic, summary, sources, and tools used
- **Interactive Sessions**: Maintain context across multiple queries in a session
- **Save Results**: Save research results to a text file for later reference
- **Chat History**: Maintain a history of your research queries and results

### Advanced Features

- **Academic Sources**: Access academic papers from Google Scholar, PubMed, and arXiv
- **Book Sources**: Search for books on Google Books and Open Library
- **Citation Formatting**: Format citations in various styles (APA, MLA, Chicago, Harvard, IEEE)
- **Bibliography Generation**: Generate bibliographies from research sources
- **Project Management**: Organize research into projects with multiple queries and results
- **Visualization Tools**: Create charts, mind maps, timelines, and network diagrams from research data
- **Export Options**: Export research results in various formats (Markdown, HTML, PDF, DOCX, Presentation)
- **Tagging System**: Categorize research results with tags for better organization

## Usage

### Running the Research Assistant

You can run the Research Assistant directly using the Python module:

```bash
python -m src.agents.research_assistant
```

### Interactive Commands

The Enhanced Research Assistant supports a wide range of interactive commands:

#### Basic Commands

- **Research Query**: Enter any topic to research

  ```bash
  What can I help you research? machine learning
  ```

- **Save Results**: Save the last research results to a file

  ```bash
  What can I help you research? save
  Enter filename to save results (default: research_output.txt): my_research.txt
  ```

- **Help**: Display available commands

  ```bash
  What can I help you research? help
  ```

- **Exit**: End the research session
  ```bash
  What can I help you research? exit
  ```
  or
  ```bash
  What can I help you research? quit
  ```

#### Project Management Commands

- **List Projects**: List all research projects

  ```bash
  What can I help you research? projects
  ```

- **Create Project**: Create a new research project

  ```bash
  What can I help you research? project create AI Research
  Enter project description (optional): Research on artificial intelligence and its applications
  Enter project tags (comma-separated, optional): ai, machine learning, deep learning
  ```

- **Select Project**: Select a project to work with

  ```bash
  What can I help you research? project select project_1
  ```

- **Project Info**: View details about the current project
  ```bash
  What can I help you research? project info
  ```

#### Citation Commands

- **Set Citation Format**: Set the citation format for research results
  ```bash
  What can I help you research? citation apa
  ```
  Supported formats: apa, mla, chicago, harvard, ieee

#### Export Commands

- **Export Results**: Export research results in different formats
  ```bash
  What can I help you research? export md
  Enter filename to export results (default: research_output.md): my_research.md
  ```
  Supported formats: md (Markdown), html (HTML), pdf (PDF), docx (Word), pptx (PowerPoint)

#### Visualization Commands

- **Create Visualization**: Create visualizations from research data
  ```bash
  What can I help you research? visualize chart
  ```
  Supported visualization types: chart, mind_map, timeline, network

### Example Session

Here's an example of an enhanced research session showcasing the new features:

```bash
=== Enhanced Research Assistant ===
Type 'exit' or 'quit' to end the session.
Type 'save' to save the last research results to a file.
Type 'export <format>' to export results (formats: md, html, pdf, docx, pptx).
Type 'projects' to list all research projects.
Type 'project create <name>' to create a new project.
Type 'project select <id>' to select a project.
Type 'project info' to view current project details.
Type 'citation <format>' to set citation format (apa, mla, chicago, harvard, ieee).
Type 'visualize <type>' to create a visualization (chart, mind_map, timeline, network).
Type 'help' to see all available commands.

Using project: General Research (ID: project_1)

What can I help you research? project create AI Research
Enter project description (optional): Research on artificial intelligence and machine learning
Enter project tags (comma-separated, optional): ai, machine learning, neural networks
Created project: AI Research (ID: project_2)

What can I help you research? citation ieee
Citation format set to ieee.

What can I help you research? machine learning

Researching: machine learning...

Search result: Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use...

Wikipedia result: Machine learning (ML) is a field of study in artificial intelligence concerned with the development and...

Google Scholar result: Mock Google Scholar result for: machine learning

arXiv result: Mock arXiv result for: machine learning

--- Research Results ---
Topic: Machine Learning
Summary: Machine Learning is a subset of artificial intelligence that enables computers to learn from data and improve their performance without being explicitly programmed. It involves algorithms that can analyze data, identify patterns, and make predictions or decisions. Key approaches include supervised learning (using labeled data), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through trial and error with rewards).

Sources:
1. Introduction to Machine Learning
   Authors: Andrew Ng
   URL: https://www.coursera.org/learn/machine-learning
2. Machine Learning: A Probabilistic Perspective
   Authors: Kevin P. Murphy
3. Machine Learning
   URL: https://en.wikipedia.org/wiki/Machine_learning

Tools used: search, wikipedia, google_scholar, arxiv

Bibliography (ieee):
[1] A. Ng, "Introduction to Machine Learning," Coursera. [Online]. Available: https://www.coursera.org/learn/machine-learning
[2] K. P. Murphy, Machine Learning: A Probabilistic Perspective. MIT Press, 2012.
[3] "Machine Learning," Wikipedia. [Online]. Available: https://en.wikipedia.org/wiki/Machine_learning

Project: AI Research (ID: project_2)

Tags: machine learning, artificial intelligence, data science

What can I help you research? visualize mind_map
Creating mind map visualization...

Mind Map for Machine Learning:
- Machine Learning (central topic)
  - Sources
    - Introduction to Machine Learning
    - Machine Learning: A Probabilistic Perspective
    - Machine Learning (Wikipedia)
  - Tools Used
    - search
    - wikipedia
    - google_scholar
    - arxiv

What can I help you research? export pdf
Enter filename to export results (default: research_output.pdf): ml_research.pdf
Research exported to ml_research.pdf successfully.

What can I help you research? project info

=== Project: AI Research ===
ID: project_2
Description: Research on artificial intelligence and machine learning
Tags: ai, machine learning, neural networks
Created: 2023-06-15 14:32:45
Updated: 2023-06-15 14:35:12
Queries: 1

Recent Queries:
1. machine learning (ID: query_1)
   Results: 1
   Created: 2023-06-15 14:33:21

What can I help you research? exit
Ending research session. Goodbye!

Thank you for using the Enhanced Research Assistant!
```

## Architecture

The Enhanced Research Assistant is built using the following components:

### Core Components

- **Enhanced Research Models**: Pydantic models that define the structure of research results, projects, queries, and sources
- **LLM Integration**: Integration with language models for generating research summaries and analyses
- **Tool Integration**: Integration with multiple tools for gathering information from various sources
- **Project Management**: System for organizing research into projects with multiple queries and results
- **Interactive Interface**: A command-line interface for interacting with the research assistant

### Research Models

- **ResearchResponse**: Basic model for research results
- **EnhancedResearchResponse**: Advanced model with support for structured sources, citations, and visualizations
- **Source**: Model for representing research sources with detailed metadata
- **ResearchProject**: Model for organizing research into projects
- **ResearchQuery**: Model for tracking individual research queries
- **ResearchResult**: Model for storing research results with metadata

### Tools

The Enhanced Research Assistant uses the following tools:

#### Information Gathering Tools

- **Search Tool**: Uses DuckDuckGo to search the web for information
- **Wikipedia Tool**: Queries Wikipedia for background information and definitions
- **Google Scholar Tool**: Searches Google Scholar for academic papers
- **PubMed Tool**: Searches PubMed for medical and biological research
- **arXiv Tool**: Searches arXiv for physics, mathematics, and computer science papers
- **Google Books Tool**: Searches Google Books for book content
- **Open Library Tool**: Searches Open Library for book information

#### Citation Tools

- **Citation Formatter**: Formats citations in various styles (APA, MLA, Chicago, Harvard, IEEE)
- **Bibliography Generator**: Generates bibliographies from research sources

#### Export Tools

- **Markdown Exporter**: Exports research results to Markdown format
- **HTML Exporter**: Exports research results to HTML format
- **PDF Exporter**: Exports research results to PDF format
- **DOCX Exporter**: Exports research results to Word format
- **Presentation Exporter**: Exports research results to PowerPoint format

#### Visualization Tools

- **Chart Generator**: Creates charts from research data
- **Mind Map Generator**: Creates mind maps from research topics and sources
- **Timeline Generator**: Creates timelines from research events
- **Network Diagram Generator**: Creates network diagrams showing relationships between research entities

## Customization

### Configuring the LLM

The Research Assistant uses Claude 3.5 Sonnet by default, but you can configure it to use other language models by modifying the `llm` variable in the `research_assistant.py` file:

```python
# Use OpenAI GPT-4
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4")

# Use Anthropic Claude 3 Opus
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-3-opus-20240229")
```

### Customizing the Prompt

You can customize the system prompt by modifying the `prompt` variable in the `research_assistant.py` file:

```python
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that specializes in [YOUR DOMAIN].

            Your task is to:
            1. Understand the user's research query
            2. Use the appropriate tools to gather relevant information
            3. Synthesize the information into a concise, informative summary
            4. Cite your sources properly
            5. Track which tools you used

            [ADDITIONAL INSTRUCTIONS]

            Wrap your output in the format specified below and provide no other text:
            {format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())
```

### Adding Custom Tools

You can add custom tools to any of the tool categories by creating new tool modules:

#### Adding a Custom Information Gathering Tool

```python
from langchain.tools import Tool

def custom_academic_tool(query: str) -> str:
    # Implement your custom academic search logic here
    return f"Custom academic results for: {query}"

custom_academic_tool = Tool(
    name="custom_academic_tool",
    func=custom_academic_tool,
    description="Searches a custom academic database for research papers",
)

# Add the custom tool to the tools list in research_assistant.py
tools = [
    search_tool,
    wiki_tool,
    google_scholar_tool,
    arxiv_tool,
    pubmed_tool,
    custom_academic_tool
]
```

#### Adding a Custom Citation Tool

```python
from langchain.tools import Tool

def custom_citation_format(source_data: str) -> str:
    # Parse the source data (JSON string)
    import json
    sources = json.loads(source_data)

    # Implement your custom citation formatting logic
    formatted_citations = []
    for source in sources["sources"]:
        # Format each source according to your custom citation style
        citation = f"Custom citation for: {source['title']}"
        formatted_citations.append(citation)

    return "\n".join(formatted_citations)

custom_citation_tool = Tool(
    name="custom_citation_format",
    func=custom_citation_format,
    description="Formats citations in a custom citation style",
)

# Add the custom tool to the citation tools
```

#### Adding a Custom Visualization Tool

```python
from langchain.tools import Tool

def custom_visualization_tool(data: str) -> str:
    # Parse the visualization data (JSON string)
    import json
    viz_data = json.loads(data)

    # Implement your custom visualization logic
    # This could generate an image, HTML, or other visualization format
    visualization = f"Custom visualization for: {viz_data['title']}"

    return visualization

custom_viz_tool = Tool(
    name="custom_visualization",
    func=custom_visualization_tool,
    description="Creates a custom visualization from research data",
)

# Add the custom tool to the visualization tools
```

## Integration with Other Systems

### Using the Enhanced Research Assistant in Your Code

You can integrate the Enhanced Research Assistant into your own code with various options:

#### Basic Usage

```python
from src.agents.research_assistant import agent_executor

# Run a basic research query
result = agent_executor.invoke({"query": "machine learning"})

# Process the result
output = result.get("output")
print(f"Research result: {output}")
```

#### Advanced Usage with Project Management

```python
from src.agents.research_assistant import agent_executor
from src.agents.research_project_manager import research_project_manager
import json

# Create a research project
project = research_project_manager.create_project(
    name="AI Research",
    description="Research on artificial intelligence and machine learning",
    tags=["ai", "machine learning", "neural networks"]
)

# Run a research query with project context and citation format
result = agent_executor.invoke({
    "query": "machine learning",
    "project_id": project["id"],
    "citation_format": "ieee"
})

# Process the result
output = result.get("output")
research_data = json.loads(output)

# Print the research results
print(f"Topic: {research_data['topic']}")
print(f"Summary: {research_data['summary']}")
print(f"Bibliography: {research_data['bibliography']}")

# Get all queries in the project
project = research_project_manager.get_project(project["id"])
for query in project["queries"]:
    print(f"Query: {query['query']} (ID: {query['id']})")
    print(f"Results: {len(query['results'])}")
```

### Using Enhanced Research Results in Other Agents

You can use the enhanced research results in other agents by parsing the structured output:

```python
import json
from src.agents.research_assistant import agent_executor, EnhancedResearchResponseModel

# Run a research query with advanced options
result = agent_executor.invoke({
    "query": "machine learning",
    "citation_format": "apa",
    "project_id": "project_1"
})

# Parse the result
output = result.get("output")
research_data = json.loads(output)

# Create an EnhancedResearchResponseModel object
enhanced_response = EnhancedResearchResponseModel(
    topic=research_data["topic"],
    summary=research_data["summary"],
    sources=research_data["sources"],
    tools_used=research_data["tools_used"],
    citation_format=research_data.get("citation_format"),
    bibliography=research_data.get("bibliography"),
    project_id=research_data.get("project_id"),
    query_id=research_data.get("query_id"),
    visualizations=research_data.get("visualizations", []),
    tags=research_data.get("tags", [])
)

# Use the enhanced research response in your agent
print(f"Topic: {enhanced_response.topic}")
print(f"Summary: {enhanced_response.summary}")

# Print sources with detailed information
print("Sources:")
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

# Print bibliography
if enhanced_response.bibliography:
    print(f"\nBibliography ({enhanced_response.citation_format}):")
    print(enhanced_response.bibliography)

# Use visualizations
if enhanced_response.visualizations:
    for viz in enhanced_response.visualizations:
        print(f"\nVisualization: {viz['title']} (Type: {viz['type']})")
        # Process visualization data as needed
```

### Integrating with External Systems

The Enhanced Research Assistant can be integrated with external systems like document management systems, knowledge bases, or collaboration platforms:

```python
import json
from src.agents.research_assistant import agent_executor
from src.tools.export_tools import export_to_html_tool, export_to_pdf_tool

# Run a research query
result = agent_executor.invoke({"query": "machine learning"})
output = result.get("output")
research_data = json.loads(output)

# Export to HTML for web publishing
html_export_input = json.dumps({
    "research_data": research_data,
    "filename": "ml_research.html"
})
html_result = export_to_html_tool.run(html_export_input)

# Export to PDF for document management systems
pdf_export_input = json.dumps({
    "research_data": research_data,
    "filename": "ml_research.pdf"
})
pdf_result = export_to_pdf_tool.run(pdf_export_input)

# Send to external API (example)
import requests
response = requests.post(
    "https://your-knowledge-base-api.com/documents",
    json={
        "title": research_data["topic"],
        "content": research_data["summary"],
        "sources": research_data["sources"],
        "tags": research_data.get("tags", []),
        "attachments": [
            {"name": "ml_research.pdf", "content_type": "application/pdf"}
        ]
    },
    files={
        "file": open("ml_research.pdf", "rb")
    }
)
```

## Troubleshooting

### Common Issues

#### API Key Issues

If you encounter API key errors:

```bash
Error: Invalid API key or API key not found
```

Make sure you have set the required API keys in your `.env` file or environment variables:

```bash
ANTHROPIC_API_KEY=your_anthropic_api_key
OPENAI_API_KEY=your_openai_api_key
```

#### Tool Execution Issues

If you encounter tool execution errors:

```bash
Error: Failed to execute tool
```

Check the tool's requirements and make sure all dependencies are installed:

```bash
pip install -r requirements.txt
```

#### Mock Mode for Testing

The Enhanced Research Assistant includes a comprehensive mock mode for testing without making API calls:

```python
# In research_assistant.py, the EnhancedMockResearchAgent is used by default
from src.agents.research_assistant import EnhancedMockResearchAgent
agent_executor = EnhancedMockResearchAgent()
```

This mock agent provides realistic responses for various research topics and simulates all the advanced features:

- Academic source integration (Google Scholar, PubMed, arXiv)
- Book source integration (Google Books, Open Library)
- Citation formatting in multiple styles
- Bibliography generation
- Project management
- Visualization tools
- Export options

#### Troubleshooting Project Management Issues

If you encounter issues with project management:

```bash
Error: Project not found or Cannot create project
```

Check that the research project manager is properly initialized:

```python
# Verify the research project manager is initialized
from src.agents.research_project_manager import research_project_manager
print(research_project_manager.get_all_projects())

# If empty or None, initialize it
if not research_project_manager.get_all_projects():
    research_project_manager.create_project(
        name="General Research",
        description="Default project for research queries",
        tags=["general", "research"]
    )
```

#### Troubleshooting Visualization Issues

If you encounter issues with visualizations:

```bash
Error: Cannot generate visualization or Visualization tool error
```

Check that the visualization tools are properly initialized and that the data format is correct:

```python
# Verify the visualization tools are initialized
from src.tools.visualization_tools import generate_chart_tool
print(generate_chart_tool.name)  # Should print "Chart Generator"

# Test with simple data
test_data = {
    "data": {
        "labels": ["A", "B", "C"],
        "values": [1, 2, 3]
    },
    "type": "bar",
    "title": "Test Chart"
}
import json
result = generate_chart_tool.run(json.dumps(test_data))
print(result)
```

### Getting Help

If you encounter issues not covered in this guide, you can:

1. Check the logs for error messages
2. Run the Research Assistant with verbose output
3. Check the GitHub repository for known issues and solutions
4. Contact the maintainers for support

## Future Enhancements

The Enhanced Research Assistant is continuously being improved. Future enhancements may include:

### Planned Enhancements

- **Real-time Collaboration**: Support for multiple users working on the same research project simultaneously
- **Advanced Search Filters**: More advanced filtering and sorting of research results by date, relevance, source type, etc.
- **Visual Research**: Support for visual research with images, diagrams, and interactive visualizations
- **Research Recommendations**: AI-powered recommendations for related research topics and sources
- **Semantic Search**: Improved search capabilities using semantic understanding of research content
- **Research Workflows**: Support for customizable research workflows and templates
- **Integration with Reference Managers**: Direct integration with reference managers like Zotero, Mendeley, and EndNote
- **Research Analytics**: Analytics and insights about research projects, sources, and trends
- **Offline Mode**: Support for offline research with synchronization when online
- **Mobile Support**: Mobile-friendly interface for research on the go

### Experimental Features

- **Research Assistant API**: RESTful API for integrating the Research Assistant with other applications
- **Research Assistant Plugin System**: Plugin system for extending the Research Assistant with custom functionality
- **Research Assistant Web Interface**: Web-based interface for the Research Assistant
- **Research Assistant Mobile App**: Mobile app for the Research Assistant
- **Research Assistant Desktop App**: Desktop app for the Research Assistant
- **Research Assistant Browser Extension**: Browser extension for the Research Assistant
- **Research Assistant Voice Interface**: Voice interface for the Research Assistant
- **Research Assistant Chat Interface**: Chat interface for the Research Assistant
- **Research Assistant Email Interface**: Email interface for the Research Assistant
- **Research Assistant SMS Interface**: SMS interface for the Research Assistant
