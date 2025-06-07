"""
Test script for the advanced visualization tools.
This script tests the basic functionality of the visualization tools.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.tools.research_visualization_tools import (
    ChartData,
    MapData,
    NetworkData,
    TimelineData,
    VisualizationConfig,
    VisualizationGenerator,
    WordCloudData,
    generate_chart_tool,
    generate_map_tool,
    generate_network_diagram_tool,
    generate_timeline_tool,
    generate_wordcloud_tool,
)

def test_chart_visualization():
    """Test chart visualization."""
    print("Testing chart visualization...")

    # Create visualization generator
    generator = VisualizationGenerator()

    # Create chart data
    chart_data = ChartData(
        chart_type="bar",
        x_data=["A", "B", "C", "D", "E"],
        y_data=[10, 20, 15, 25, 30],
        x_label="Categories",
        y_label="Values"
    )

    # Create chart configuration
    chart_config = VisualizationConfig(
        title="Test Bar Chart",
        width=800,
        height=600,
        interactive=True
    )

    # Generate chart
    result = generator.generate_chart(chart_data, chart_config)

    print(f"Chart generated: {result['filepath']}")
    print(f"Chart URL: {result['url']}")

    # Test tool function
    tool_input = {
        "data": {
            "chart_type": "line",
            "x_data": [1, 2, 3, 4, 5],
            "y_data": [10, 20, 15, 25, 30],
            "x_label": "X Axis",
            "y_label": "Y Axis"
        },
        "config": {
            "title": "Test Line Chart",
            "width": 800,
            "height": 600,
            "interactive": True
        }
    }

    tool_result = generate_chart_tool(json.dumps(tool_input))
    tool_result_dict = json.loads(tool_result)

    print(f"Chart tool result: {tool_result_dict['filepath']}")
    print(f"Chart tool URL: {tool_result_dict['url']}")

def test_network_visualization():
    """Test network visualization."""
    print("\nTesting network visualization...")

    # Create visualization generator
    generator = VisualizationGenerator()

    # Create network data
    network_data = NetworkData(
        nodes=[
            {"id": 1, "label": "Node 1"},
            {"id": 2, "label": "Node 2"},
            {"id": 3, "label": "Node 3"},
            {"id": 4, "label": "Node 4"},
            {"id": 5, "label": "Node 5"}
        ],
        edges=[
            {"source": 1, "target": 2, "label": "Edge 1-2"},
            {"source": 1, "target": 3, "label": "Edge 1-3"},
            {"source": 2, "target": 4, "label": "Edge 2-4"},
            {"source": 3, "target": 5, "label": "Edge 3-5"},
            {"source": 4, "target": 5, "label": "Edge 4-5"}
        ],
        layout="force"
    )

    # Create network configuration
    network_config = VisualizationConfig(
        title="Test Network Diagram",
        width=800,
        height=600,
        interactive=True
    )

    # Generate network
    result = generator.generate_network(network_data, network_config)

    print(f"Network generated: {result['filepath']}")
    print(f"Network URL: {result['url']}")

    # Test tool function
    tool_input = {
        "data": {
            "nodes": [
                {"id": 1, "label": "Node 1"},
                {"id": 2, "label": "Node 2"},
                {"id": 3, "label": "Node 3"}
            ],
            "edges": [
                {"source": 1, "target": 2, "label": "Edge 1-2"},
                {"source": 2, "target": 3, "label": "Edge 2-3"},
                {"source": 3, "target": 1, "label": "Edge 3-1"}
            ],
            "layout": "circular"
        },
        "config": {
            "title": "Test Network Diagram (Tool)",
            "width": 800,
            "height": 600,
            "interactive": True
        }
    }

    tool_result = generate_network_diagram_tool(json.dumps(tool_input))
    tool_result_dict = json.loads(tool_result)

    print(f"Network tool result: {tool_result_dict['filepath']}")
    print(f"Network tool URL: {tool_result_dict['url']}")

def test_wordcloud_visualization():
    """Test word cloud visualization."""
    print("\nTesting word cloud visualization...")

    # Create visualization generator
    generator = VisualizationGenerator()

    # Create word cloud data
    wordcloud_data = WordCloudData(
        text="This is a test word cloud visualization. It should show the most frequent words in larger font sizes. "
             "The more times a word appears, the larger it will be in the word cloud. "
             "Word clouds are useful for visualizing text data and identifying the most important terms. "
             "They can be used for research summaries, content analysis, and more."
    )

    # Create word cloud configuration
    wordcloud_config = VisualizationConfig(
        title="Test Word Cloud",
        width=800,
        height=600
    )

    # Generate word cloud
    try:
        result = generator.generate_wordcloud(wordcloud_data, wordcloud_config)

        print(f"Word cloud generated: {result['filepath']}")
        print(f"Word cloud URL: {result['url']}")
    except ImportError:
        print("WordCloud library not available. Skipping word cloud test.")

def test_timeline_visualization():
    """Test timeline visualization."""
    print("\nTesting timeline visualization...")

    # Create visualization generator
    generator = VisualizationGenerator()

    # Create timeline data
    timeline_data = TimelineData(
        events=[
            {"date": "2020-01-01", "description": "Event 1", "category": "Category A"},
            {"date": "2020-02-15", "description": "Event 2", "category": "Category B"},
            {"date": "2020-03-10", "description": "Event 3", "category": "Category A"},
            {"date": "2020-04-20", "description": "Event 4", "category": "Category C"},
            {"date": "2020-05-05", "description": "Event 5", "category": "Category B"}
        ],
        date_field="date",
        description_field="description",
        category_field="category"
    )

    # Create timeline configuration
    timeline_config = VisualizationConfig(
        title="Test Timeline",
        width=800,
        height=600,
        interactive=True
    )

    # Generate timeline
    try:
        result = generator.generate_timeline(timeline_data, timeline_config)

        print(f"Timeline generated: {result['filepath']}")
        print(f"Timeline URL: {result['url']}")
    except ImportError:
        print("Plotly library not available. Skipping timeline test.")

def main():
    """Run the tests."""
    # Create a temporary directory for test visualizations
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")

        # Test chart visualization
        test_chart_visualization()

        # Test network visualization
        test_network_visualization()

        # Test word cloud visualization
        test_wordcloud_visualization()

        # Test timeline visualization
        test_timeline_visualization()

        print("\nAll tests completed.")

if __name__ == "__main__":
    main()
