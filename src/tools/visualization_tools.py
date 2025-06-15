"""
Visualization tools for the Research Assistant.

This module provides tools for creating visualizations of research data,
including charts, mind maps, timelines, and network diagrams.
"""

import json
from typing import Dict

from langchain.tools import Tool

from src.models.research_models import Visualization, VisualizationType


class ChartGenerator:
    """Tool for generating charts from research data."""

    def generate_chart(
        self, data: Dict, chart_type: str = "bar", title: str = "Chart"
    ) -> Visualization:
        """
        Generate a chart visualization.

        Args:
            data: Chart data
            chart_type: Type of chart (bar, line, pie, scatter)
            title: Chart title

        Returns:
            Visualization object
        """
        # Note: In a real implementation, this would use a charting library
        # like matplotlib, plotly, or bokeh
        # This is a simplified mock implementation

        # Create a visualization object
        visualization = Visualization(
            title=title,
            description=f"A {chart_type} chart of the data",
            type=VisualizationType.CHART,
            data={"chart_type": chart_type, "chart_data": data},
        )

        return visualization

    def render_chart(self, visualization: Visualization) -> str:
        """
        Render a chart visualization as ASCII art.

        Args:
            visualization: Visualization object

        Returns:
            ASCII art representation of the chart
        """
        # This is a simplified mock implementation
        chart_type = visualization.data.get("chart_type", "bar")
        chart_data = visualization.data.get("chart_data", {})

        if chart_type == "bar":
            return self._render_bar_chart(chart_data, visualization.title)
        elif chart_type == "pie":
            return self._render_pie_chart(chart_data, visualization.title)
        else:
            return f"Chart: {visualization.title}\n[Chart visualization would be displayed here]"

    def _render_bar_chart(self, data: Dict, title: str) -> str:
        """Render a bar chart as ASCII art."""
        result = f"Bar Chart: {title}\n\n"

        # Get the data
        labels = data.get("labels", [])
        values = data.get("values", [])

        # Find the maximum value for scaling
        max_value = max(values) if values else 0
        max_bar_length = 20

        # Render the bars
        for i, (label, value) in enumerate(zip(labels, values)):
            bar_length = int((value / max_value) * max_bar_length) if max_value > 0 else 0
            bar = "#" * bar_length
            result += f"{label.ljust(15)}: {bar} {value}\n"

        return result

    def _render_pie_chart(self, data: Dict, title: str) -> str:
        """Render a pie chart as ASCII art."""
        result = f"Pie Chart: {title}\n\n"

        # Get the data
        labels = data.get("labels", [])
        values = data.get("values", [])

        # Calculate percentages
        total = sum(values) if values else 0
        percentages = [(value / total) * 100 if total > 0 else 0 for value in values]

        # Render the pie slices
        for i, (label, value, percentage) in enumerate(zip(labels, values, percentages)):
            result += f"{label.ljust(15)}: {percentage:.1f}% ({value})\n"

        return result

    def run(self, input_str: str) -> str:
        """
        Run the chart generator.

        Args:
            input_str: JSON string containing chart data, type, and title

        Returns:
            ASCII art representation of the chart
        """
        try:
            data = json.loads(input_str)
            chart_data = data.get("data", {})
            chart_type = data.get("type", "bar")
            title = data.get("title", "Chart")

            visualization = self.generate_chart(chart_data, chart_type, title)
            return self.render_chart(visualization)
        except Exception as e:
            return f"Error generating chart: {str(e)}"


class MindMapGenerator:
    """Tool for generating mind maps from research data."""

    def generate_mind_map(self, data: Dict, title: str = "Mind Map") -> Visualization:
        """
        Generate a mind map visualization.

        Args:
            data: Mind map data
            title: Mind map title

        Returns:
            Visualization object
        """
        # Note: In a real implementation, this would use a mind map library
        # This is a simplified mock implementation

        # Create a visualization object
        visualization = Visualization(
            title=title,
            description="A mind map of the data",
            type=VisualizationType.MIND_MAP,
            data=data,
        )

        return visualization

    def render_mind_map(self, visualization: Visualization) -> str:
        """
        Render a mind map visualization as ASCII art.

        Args:
            visualization: Visualization object

        Returns:
            ASCII art representation of the mind map
        """
        # This is a simplified mock implementation
        mind_map_data = visualization.data

        result = f"Mind Map: {visualization.title}\n\n"

        # Get the central topic
        central_topic = mind_map_data.get("central_topic", "Central Topic")
        result += f"{central_topic}\n"

        # Get the main branches
        branches = mind_map_data.get("branches", [])

        # Render the branches
        for i, branch in enumerate(branches):
            branch_name = branch.get("name", f"Branch {i+1}")
            result += f"├── {branch_name}\n"

            # Render the sub-branches
            sub_branches = branch.get("sub_branches", [])
            for j, sub_branch in enumerate(sub_branches):
                is_last = j == len(sub_branches) - 1
                prefix = "└── " if is_last else "├── "
                result += f"│   {prefix}{sub_branch}\n"

        return result

    def run(self, input_str: str) -> str:
        """
        Run the mind map generator.

        Args:
            input_str: JSON string containing mind map data and title

        Returns:
            ASCII art representation of the mind map
        """
        try:
            data = json.loads(input_str)
            mind_map_data = data.get("data", {})
            title = data.get("title", "Mind Map")

            visualization = self.generate_mind_map(mind_map_data, title)
            return self.render_mind_map(visualization)
        except Exception as e:
            return f"Error generating mind map: {str(e)}"


class TimelineGenerator:
    """Tool for generating timelines from research data."""

    def generate_timeline(self, data: Dict, title: str = "Timeline") -> Visualization:
        """
        Generate a timeline visualization.

        Args:
            data: Timeline data
            title: Timeline title

        Returns:
            Visualization object
        """
        # Note: In a real implementation, this would use a timeline library
        # This is a simplified mock implementation

        # Create a visualization object
        visualization = Visualization(
            title=title,
            description="A timeline of events",
            type=VisualizationType.TIMELINE,
            data=data,
        )

        return visualization

    def render_timeline(self, visualization: Visualization) -> str:
        """
        Render a timeline visualization as ASCII art.

        Args:
            visualization: Visualization object

        Returns:
            ASCII art representation of the timeline
        """
        # This is a simplified mock implementation
        timeline_data = visualization.data

        result = f"Timeline: {visualization.title}\n\n"

        # Get the events
        events = timeline_data.get("events", [])

        # Sort events by date
        events.sort(key=lambda x: x.get("date", ""))

        # Render the timeline
        for i, event in enumerate(events):
            date = event.get("date", "")
            description = event.get("description", "")
            result += f"{date.ljust(15)}: {description}\n"

            # Add a connecting line except for the last event
            if i < len(events) - 1:
                result += "               |\n"

        return result

    def run(self, input_str: str) -> str:
        """
        Run the timeline generator.

        Args:
            input_str: JSON string containing timeline data and title

        Returns:
            ASCII art representation of the timeline
        """
        try:
            data = json.loads(input_str)
            timeline_data = data.get("data", {})
            title = data.get("title", "Timeline")

            visualization = self.generate_timeline(timeline_data, title)
            return self.render_timeline(visualization)
        except Exception as e:
            return f"Error generating timeline: {str(e)}"


class NetworkDiagramGenerator:
    """Tool for generating network diagrams from research data."""

    def generate_network_diagram(self, data: Dict, title: str = "Network Diagram") -> Visualization:
        """
        Generate a network diagram visualization.

        Args:
            data: Network diagram data
            title: Network diagram title

        Returns:
            Visualization object
        """
        # Note: In a real implementation, this would use a network diagram library
        # This is a simplified mock implementation

        # Create a visualization object
        visualization = Visualization(
            title=title,
            description="A network diagram of the data",
            type=VisualizationType.NETWORK,
            data=data,
        )

        return visualization

    def render_network_diagram(self, visualization: Visualization) -> str:
        """
        Render a network diagram visualization as ASCII art.

        Args:
            visualization: Visualization object

        Returns:
            ASCII art representation of the network diagram
        """
        # This is a simplified mock implementation
        network_data = visualization.data

        result = f"Network Diagram: {visualization.title}\n\n"

        # Get the nodes and edges
        nodes = network_data.get("nodes", [])
        edges = network_data.get("edges", [])

        # Render the nodes
        result += "Nodes:\n"
        for i, node in enumerate(nodes):
            node_id = node.get("id", i)
            node_label = node.get("label", f"Node {node_id}")
            result += f"- {node_label} (ID: {node_id})\n"

        # Render the edges
        result += "\nEdges:\n"
        for i, edge in enumerate(edges):
            source = edge.get("source", "")
            target = edge.get("target", "")
            label = edge.get("label", "")

            # Find the source and target node labels
            source_label = next(
                (node.get("label", f"Node {source}") for node in nodes if node.get("id") == source),
                f"Node {source}",
            )
            target_label = next(
                (node.get("label", f"Node {target}") for node in nodes if node.get("id") == target),
                f"Node {target}",
            )

            result += f"- {source_label} --[{label}]--> {target_label}\n"

        return result

    def run(self, input_str: str) -> str:
        """
        Run the network diagram generator.

        Args:
            input_str: JSON string containing network diagram data and title

        Returns:
            ASCII art representation of the network diagram
        """
        try:
            data = json.loads(input_str)
            network_data = data.get("data", {})
            title = data.get("title", "Network Diagram")

            visualization = self.generate_network_diagram(network_data, title)
            return self.render_network_diagram(visualization)
        except Exception as e:
            return f"Error generating network diagram: {str(e)}"


# Create tool instances
chart_generator = ChartGenerator()
mind_map_generator = MindMapGenerator()
timeline_generator = TimelineGenerator()
network_diagram_generator = NetworkDiagramGenerator()

# Create LangChain tools
generate_chart_tool = Tool(
    name="generate_chart",
    func=chart_generator.run,
    description="Generate a chart visualization from data. Input should be a JSON string with 'data', 'type', and 'title' fields.",
)

generate_mind_map_tool = Tool(
    name="generate_mind_map",
    func=mind_map_generator.run,
    description="Generate a mind map visualization from data. Input should be a JSON string with 'data' and 'title' fields.",
)

generate_timeline_tool = Tool(
    name="generate_timeline",
    func=timeline_generator.run,
    description="Generate a timeline visualization from data. Input should be a JSON string with 'data' and 'title' fields.",
)

generate_network_diagram_tool = Tool(
    name="generate_network_diagram",
    func=network_diagram_generator.run,
    description="Generate a network diagram visualization from data. Input should be a JSON string with 'data' and 'title' fields.",
)
