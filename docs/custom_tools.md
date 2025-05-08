# Custom Tools Documentation

This document provides comprehensive documentation for the custom tools available in DataMCPServerAgent.

## Table of Contents

- [Custom Tools Documentation](#custom-tools-documentation)
  - [Table of Contents](#table-of-contents)
  - [BrightDataToolkit](#brightdatatoolkit)
    - [Initialization](#initialization)
    - [Enhanced Web Search Tool](#enhanced-web-search-tool)
      - [Usage](#usage)
      - [Parameters](#parameters)
      - [Return Value](#return-value)
    - [Enhanced Web Scraper Tool](#enhanced-web-scraper-tool)
      - [Usage](#usage-1)
      - [Parameters](#parameters-1)
      - [Return Value](#return-value-1)
    - [Product Comparison Tool](#product-comparison-tool)
      - [Usage](#usage-2)
      - [Parameters](#parameters-2)
      - [Return Value](#return-value-2)
    - [Social Media Analyzer Tool](#social-media-analyzer-tool)
      - [Usage](#usage-3)
      - [Parameters](#parameters-3)
      - [Return Value](#return-value-3)
  - [EnhancedToolSelector](#enhancedtoolselector)
    - [Initialization](#initialization-1)
    - [Tool Selection Process](#tool-selection-process)
      - [Usage](#usage-4)
      - [Parameters](#parameters-4)
      - [Return Value](#return-value-4)
    - [Tool Performance Tracking](#tool-performance-tracking)
      - [Usage](#usage-5)
    - [Execution Feedback](#execution-feedback)
      - [Usage](#usage-6)
      - [Parameters](#parameters-5)
      - [Return Value](#return-value-5)

## BrightDataToolkit

The `BrightDataToolkit` class provides specialized tools that extend the basic Bright Data MCP tools with enhanced functionality.

### Initialization

```python
from mcp import ClientSession
from src.tools.bright_data_tools import BrightDataToolkit

# Initialize with an MCP client session
toolkit = BrightDataToolkit(session)

# Create custom tools
custom_tools = await toolkit.create_custom_tools()
```

### Enhanced Web Search Tool

The Enhanced Web Search tool provides better formatting and filtering for web search results.

#### Usage

```python
# Get the enhanced web search tool
enhanced_search_tool = custom_tools[0]  # Assuming it's the first tool

# Use the tool
results = await enhanced_search_tool.invoke({
    "query": "Python async programming",
    "count": 5
})

print(results)  # Formatted markdown results
```

#### Parameters

- `query` (string, required): The search query
- `count` (integer, optional, default=10): Number of results to return (1-20)

#### Return Value

Returns a formatted markdown string with search results, including titles, URLs, and descriptions.

### Enhanced Web Scraper Tool

The Enhanced Web Scraper tool provides better content extraction and filtering from web pages.

#### Usage

```python
# Get the enhanced web scraper tool
enhanced_scraper_tool = custom_tools[1]  # Assuming it's the second tool

# Use the tool to get all content
all_content = await enhanced_scraper_tool.invoke({
    "url": "https://example.com",
    "extract_type": "all"
})

# Use the tool to get only the main content
main_content = await enhanced_scraper_tool.invoke({
    "url": "https://example.com",
    "extract_type": "main_content"
})

# Use the tool to get only tables
tables = await enhanced_scraper_tool.invoke({
    "url": "https://example.com",
    "extract_type": "tables"
})

# Use the tool to get only links
links = await enhanced_scraper_tool.invoke({
    "url": "https://example.com",
    "extract_type": "links"
})
```

#### Parameters

- `url` (string, required): The URL to scrape
- `extract_type` (string, optional, default="all"): Type of content to extract
  - `all`: All content
  - `main_content`: Main content (excluding headers, footers, sidebars)
  - `tables`: Only tables
  - `links`: Only links

#### Return Value

Returns a markdown string with the extracted content based on the specified `extract_type`.

### Product Comparison Tool

The Product Comparison tool analyzes and compares multiple product pages.

#### Usage

```python
# Get the product comparison tool
product_comparison_tool = custom_tools[2]  # Assuming it's the third tool

# Use the tool to analyze a single product
single_product = await product_comparison_tool.invoke({
    "urls": ["https://www.amazon.com/dp/B08N5KWB9H"]
})

# Use the tool to compare multiple products
comparison = await product_comparison_tool.invoke({
    "urls": [
        "https://www.amazon.com/dp/B08N5KWB9H",
        "https://www.amazon.com/dp/B08N5M7S6K"
    ]
})
```

#### Parameters

- `urls` (array of strings, required): List of product URLs to analyze or compare

#### Return Value

For a single product, returns a formatted markdown string with product information, including title, price, rating, availability, features, and description.

For multiple products, returns a formatted markdown string with a comparison table and detailed comparison of each product.

### Social Media Analyzer Tool

The Social Media Analyzer tool analyzes social media content across platforms (Instagram, Facebook, Twitter/X).

#### Usage

```python
# Get the social media analyzer tool
social_media_analyzer_tool = custom_tools[3]  # Assuming it's the fourth tool

# Use the tool for basic analysis
basic_analysis = await social_media_analyzer_tool.invoke({
    "url": "https://www.instagram.com/p/ABC123/",
    "analysis_type": "basic"
})

# Use the tool for detailed analysis
detailed_analysis = await social_media_analyzer_tool.invoke({
    "url": "https://www.instagram.com/p/ABC123/",
    "analysis_type": "detailed"
})

# Use the tool for engagement analysis
engagement_analysis = await social_media_analyzer_tool.invoke({
    "url": "https://www.instagram.com/p/ABC123/",
    "analysis_type": "engagement"
})
```

#### Parameters

- `url` (string, required): URL of the social media post or profile
- `analysis_type` (string, optional, default="basic"): Type of analysis to perform
  - `basic`: Basic information and metrics
  - `detailed`: Detailed information including bio, website, hashtags
  - `engagement`: Engagement metrics and analysis

#### Return Value

Returns a formatted markdown string with the analysis results based on the specified `analysis_type`.

## EnhancedToolSelector

The `EnhancedToolSelector` class provides advanced tool selection capabilities with learning and performance tracking.

### Initialization

```python
from langchain_anthropic import ChatAnthropic
from src.memory.memory_persistence import MemoryDatabase
from src.tools.enhanced_tool_selection import EnhancedToolSelector, ToolPerformanceTracker

# Initialize dependencies
model = ChatAnthropic(model="claude-3-sonnet-20240229")
db = MemoryDatabase()
performance_tracker = ToolPerformanceTracker(db)

# Initialize the tool selector
tool_selector = EnhancedToolSelector(
    model=model,
    tools=tools,  # List of BaseTool instances
    db=db,
    performance_tracker=performance_tracker
)
```

### Tool Selection Process

The tool selection process analyzes the user's request and selects the most appropriate tools based on historical performance and task requirements.

#### Usage

```python
# Select tools for a request
selection = await tool_selector.select_tools(
    request="Find information about Python async programming",
    history=[]  # Optional conversation history
)

print(selection["selected_tools"])  # List of selected tool names
print(selection["reasoning"])       # Reasoning for the selection
print(selection["execution_order"]) # Suggested execution order
print(selection["fallback_tools"])  # Fallback tools if primary tools fail
```

#### Parameters

- `request` (string, required): User request
- `history` (array, optional): Conversation history

#### Return Value

Returns a dictionary with the following keys:

- `selected_tools`: Array of selected tool names
- `reasoning`: Detailed explanation of the selection
- `execution_order`: Suggested order to use the tools
- `fallback_tools`: Alternative tools to try if the primary tools fail

### Tool Performance Tracking

The `ToolPerformanceTracker` class tracks tool performance metrics, including success rate and execution time.

#### Usage

```python
# Start tracking execution time
performance_tracker.start_execution("tool_name")

try:
    # Execute the tool
    result = await tool.invoke(args)
    success = True
except Exception:
    success = False

# End tracking execution time and save metrics
execution_time = performance_tracker.end_execution("tool_name", success)

# Get performance metrics
metrics = performance_tracker.get_performance("tool_name")
print(metrics["success_rate"])      # Success rate (percentage)
print(metrics["total_uses"])        # Total number of uses
print(metrics["avg_execution_time"]) # Average execution time (seconds)
```

### Execution Feedback

The `EnhancedToolSelector` can provide feedback on tool executions to improve future selection.

#### Usage

```python
# Provide feedback on a tool execution
feedback = await tool_selector.provide_execution_feedback(
    request="Find information about Python async programming",
    tool_name="enhanced_web_search",
    tool_args={"query": "Python async programming"},
    tool_result="...",  # Tool result
    execution_time=1.5,  # Execution time in seconds
    success=True  # Whether the execution was successful
)

print(feedback["appropriate"])  # Whether the tool was appropriate for the task
print(feedback["issues"])       # Identified issues
print(feedback["suggestions"])  # Suggestions for improvement
print(feedback["confidence"])   # Confidence score (0-100)
print(feedback["learning_points"])  # Key learning points
```

#### Parameters

- `request` (string, required): Original user request
- `tool_name` (string, required): Name of the tool used
- `tool_args` (object, required): Arguments passed to the tool
- `tool_result` (any, required): Result returned by the tool
- `execution_time` (number, required): Time taken to execute the tool (seconds)
- `success` (boolean, required): Whether the execution was successful

#### Return Value

Returns a dictionary with the following keys:

- `appropriate`: Boolean indicating whether the tool was appropriate for the task
- `issues`: Array of identified issues
- `suggestions`: Array of suggestions for improvement
- `confidence`: Confidence score (0-100)
- `learning_points`: Key learning points from this execution
