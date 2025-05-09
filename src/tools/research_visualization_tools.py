"""
Advanced visualization tools for the Research Assistant.
This module provides interactive and sophisticated visualization capabilities
for research data, including interactive charts, 3D visualizations, and more.
"""

import json
import os
import tempfile
from typing import Any, Dict, List, Optional, Union

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel

# Try to import advanced visualization libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Install with 'pip install plotly'")

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: NetworkX not available. Install with 'pip install networkx'")

try:
    from wordcloud import WordCloud

    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print("Warning: WordCloud not available. Install with 'pip install wordcloud'")


class VisualizationConfig(BaseModel):
    """Configuration for visualizations."""

    title: str
    subtitle: Optional[str] = None
    width: int = 800
    height: int = 600
    theme: str = "default"
    interactive: bool = True
    output_format: str = "html"
    color_scheme: Optional[List[str]] = None
    show_legend: bool = True
    font_family: str = "Arial"
    background_color: str = "#ffffff"
    grid: bool = True
    animation: bool = False

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


class ChartData(BaseModel):
    """Data for chart visualizations."""

    chart_type: str
    x_data: List[Any]
    y_data: Union[List[Any], List[List[Any]]]
    labels: Optional[List[str]] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    series_names: Optional[List[str]] = None
    annotations: Optional[List[Dict[str, Any]]] = None

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


class NetworkData(BaseModel):
    """Data for network visualizations."""

    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    node_size_field: Optional[str] = None
    node_color_field: Optional[str] = None
    edge_width_field: Optional[str] = None
    edge_color_field: Optional[str] = None
    layout: str = "force"

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


class MapData(BaseModel):
    """Data for map visualizations."""

    locations: List[Dict[str, Any]]
    location_mode: str = "ISO-3"
    color_field: Optional[str] = None
    size_field: Optional[str] = None
    hover_data: Optional[List[str]] = None
    map_style: str = "carto-positron"

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


class WordCloudData(BaseModel):
    """Data for word cloud visualizations."""

    text: str
    max_words: int = 200
    mask_image: Optional[str] = None

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


class TimelineData(BaseModel):
    """Data for timeline visualizations."""

    events: List[Dict[str, Any]]
    date_field: str = "date"
    description_field: str = "description"
    category_field: Optional[str] = None

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


class VisualizationGenerator:
    """Generator for advanced visualizations."""

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the visualization generator.

        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = output_dir or tempfile.mkdtemp()
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_chart(
        self, data: ChartData, config: VisualizationConfig
    ) -> Dict[str, Any]:
        """Generate a chart visualization.

        Args:
            data: Chart data
            config: Visualization configuration

        Returns:
            Visualization metadata
        """
        if not PLOTLY_AVAILABLE and config.interactive:
            print("Warning: Plotly not available. Falling back to Matplotlib.")
            config.interactive = False

        if config.interactive:
            return self._generate_interactive_chart(data, config)
        else:
            return self._generate_static_chart(data, config)

    def _generate_interactive_chart(
        self, data: ChartData, config: VisualizationConfig
    ) -> Dict[str, Any]:
        """Generate an interactive chart using Plotly.

        Args:
            data: Chart data
            config: Visualization configuration

        Returns:
            Visualization metadata
        """
        # Create figure
        fig = make_subplots(specs=[[{"secondary_y": False}]])

        # Set theme
        template = "plotly" if config.theme == "default" else config.theme

        # Add traces based on chart type
        if data.chart_type == "line":
            if isinstance(data.y_data[0], list):
                # Multiple series
                for i, y_series in enumerate(data.y_data):
                    name = (
                        data.series_names[i]
                        if data.series_names and i < len(data.series_names)
                        else f"Series {i + 1}"
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=data.x_data, y=y_series, mode="lines+markers", name=name
                        )
                    )
            else:
                # Single series
                fig.add_trace(
                    go.Scatter(
                        x=data.x_data,
                        y=data.y_data,
                        mode="lines+markers",
                        name=data.series_names[0] if data.series_names else "Series 1",
                    )
                )

        elif data.chart_type == "bar":
            if isinstance(data.y_data[0], list):
                # Multiple series
                for i, y_series in enumerate(data.y_data):
                    name = (
                        data.series_names[i]
                        if data.series_names and i < len(data.series_names)
                        else f"Series {i + 1}"
                    )
                    fig.add_trace(go.Bar(x=data.x_data, y=y_series, name=name))
            else:
                # Single series
                fig.add_trace(
                    go.Bar(
                        x=data.x_data,
                        y=data.y_data,
                        name=data.series_names[0] if data.series_names else "Series 1",
                    )
                )

        elif data.chart_type == "scatter":
            if isinstance(data.y_data[0], list):
                # Multiple series
                for i, y_series in enumerate(data.y_data):
                    name = (
                        data.series_names[i]
                        if data.series_names and i < len(data.series_names)
                        else f"Series {i + 1}"
                    )
                    fig.add_trace(
                        go.Scatter(x=data.x_data, y=y_series, mode="markers", name=name)
                    )
            else:
                # Single series
                fig.add_trace(
                    go.Scatter(
                        x=data.x_data,
                        y=data.y_data,
                        mode="markers",
                        name=data.series_names[0] if data.series_names else "Series 1",
                    )
                )

        elif data.chart_type == "pie":
            fig = go.Figure(
                data=[
                    go.Pie(
                        labels=data.labels or data.x_data, values=data.y_data, hole=0.3
                    )
                ]
            )

        elif data.chart_type == "area":
            if isinstance(data.y_data[0], list):
                # Multiple series
                for i, y_series in enumerate(data.y_data):
                    name = (
                        data.series_names[i]
                        if data.series_names and i < len(data.series_names)
                        else f"Series {i + 1}"
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=data.x_data,
                            y=y_series,
                            mode="lines",
                            fill="tonexty",
                            name=name,
                        )
                    )
            else:
                # Single series
                fig.add_trace(
                    go.Scatter(
                        x=data.x_data,
                        y=data.y_data,
                        mode="lines",
                        fill="tozeroy",
                        name=data.series_names[0] if data.series_names else "Series 1",
                    )
                )

        # Update layout
        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            template=template,
            showlegend=config.show_legend,
            font=dict(family=config.font_family),
            plot_bgcolor=config.background_color,
            paper_bgcolor=config.background_color,
            xaxis=dict(title=data.x_label, showgrid=config.grid),
            yaxis=dict(title=data.y_label, showgrid=config.grid),
        )

        # Add annotations if provided
        if data.annotations:
            annotations = []
            for annotation in data.annotations:
                annotations.append(
                    dict(
                        x=annotation.get("x"),
                        y=annotation.get("y"),
                        text=annotation.get("text", ""),
                        showarrow=annotation.get("show_arrow", True),
                        arrowhead=annotation.get("arrow_head", 1),
                        ax=annotation.get("arrow_x", 0),
                        ay=annotation.get("arrow_y", -40),
                    )
                )
            fig.update_layout(annotations=annotations)

        # Save the figure
        filename = f"{config.title.lower().replace(' ', '_')}.html"
        filepath = os.path.join(self.output_dir, filename)
        fig.write_html(filepath)

        return {
            "type": "interactive_chart",
            "chart_type": data.chart_type,
            "title": config.title,
            "filepath": filepath,
            "url": f"file://{filepath}",
        }

    def _generate_static_chart(
        self, data: ChartData, config: VisualizationConfig
    ) -> Dict[str, Any]:
        """Generate a static chart using Matplotlib.

        Args:
            data: Chart data
            config: Visualization configuration

        Returns:
            Visualization metadata
        """
        # Create figure
        fig, ax = plt.subplots(
            figsize=(config.width / 100, config.height / 100), dpi=100
        )

        # Set style
        plt.style.use("seaborn-v0_8" if config.theme == "default" else config.theme)

        # Add data based on chart type
        if data.chart_type == "line":
            if isinstance(data.y_data[0], list):
                # Multiple series
                for i, y_series in enumerate(data.y_data):
                    name = (
                        data.series_names[i]
                        if data.series_names and i < len(data.series_names)
                        else f"Series {i + 1}"
                    )
                    ax.plot(data.x_data, y_series, marker="o", label=name)
            else:
                # Single series
                ax.plot(
                    data.x_data,
                    data.y_data,
                    marker="o",
                    label=data.series_names[0] if data.series_names else "Series 1",
                )

        elif data.chart_type == "bar":
            if isinstance(data.y_data[0], list):
                # Multiple series
                x = np.arange(len(data.x_data))
                width = 0.8 / len(data.y_data)

                for i, y_series in enumerate(data.y_data):
                    name = (
                        data.series_names[i]
                        if data.series_names and i < len(data.series_names)
                        else f"Series {i + 1}"
                    )
                    ax.bar(x + i * width - 0.4 + width / 2, y_series, width, label=name)

                ax.set_xticks(x)
                ax.set_xticklabels(data.x_data)
            else:
                # Single series
                ax.bar(
                    data.x_data,
                    data.y_data,
                    label=data.series_names[0] if data.series_names else "Series 1",
                )

        elif data.chart_type == "scatter":
            if isinstance(data.y_data[0], list):
                # Multiple series
                for i, y_series in enumerate(data.y_data):
                    name = (
                        data.series_names[i]
                        if data.series_names and i < len(data.series_names)
                        else f"Series {i + 1}"
                    )
                    ax.scatter(data.x_data, y_series, label=name)
            else:
                # Single series
                ax.scatter(
                    data.x_data,
                    data.y_data,
                    label=data.series_names[0] if data.series_names else "Series 1",
                )

        elif data.chart_type == "pie":
            ax.pie(
                data.y_data,
                labels=data.labels or data.x_data,
                autopct="%1.1f%%",
                shadow=True,
                startangle=90,
            )
            ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle

        elif data.chart_type == "area":
            if isinstance(data.y_data[0], list):
                # Multiple series
                for i, y_series in enumerate(data.y_data):
                    name = (
                        data.series_names[i]
                        if data.series_names and i < len(data.series_names)
                        else f"Series {i + 1}"
                    )
                    ax.fill_between(data.x_data, y_series, alpha=0.3)
                    ax.plot(data.x_data, y_series, label=name)
            else:
                # Single series
                ax.fill_between(data.x_data, data.y_data, alpha=0.3)
                ax.plot(
                    data.x_data,
                    data.y_data,
                    label=data.series_names[0] if data.series_names else "Series 1",
                )

        # Set labels and title
        ax.set_title(config.title)
        if data.x_label:
            ax.set_xlabel(data.x_label)
        if data.y_label:
            ax.set_ylabel(data.y_label)

        # Show grid if configured
        ax.grid(config.grid)

        # Show legend if configured and if there are labels
        if config.show_legend and (
            (data.series_names and len(data.series_names) > 0)
            or (isinstance(data.y_data[0], list) and len(data.y_data) > 1)
        ):
            ax.legend()

        # Add annotations if provided
        if data.annotations:
            for annotation in data.annotations:
                ax.annotate(
                    annotation.get("text", ""),
                    xy=(annotation.get("x"), annotation.get("y")),
                    xytext=(
                        annotation.get("arrow_x", 0),
                        annotation.get("arrow_y", -40),
                    ),
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
                )

        # Save the figure
        filename = f"{config.title.lower().replace(' ', '_')}.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(
            filepath, dpi=100, bbox_inches="tight", facecolor=config.background_color
        )
        plt.close(fig)

        return {
            "type": "static_chart",
            "chart_type": data.chart_type,
            "title": config.title,
            "filepath": filepath,
            "url": f"file://{filepath}",
        }

    def generate_network(
        self, data: NetworkData, config: VisualizationConfig
    ) -> Dict[str, Any]:
        """Generate a network visualization.

        Args:
            data: Network data
            config: Visualization configuration

        Returns:
            Visualization metadata
        """
        if not NETWORKX_AVAILABLE and not PLOTLY_AVAILABLE:
            raise ImportError(
                "NetworkX and Plotly are required for network visualizations"
            )

        if config.interactive and PLOTLY_AVAILABLE:
            return self._generate_interactive_network(data, config)
        else:
            return self._generate_static_network(data, config)

    def _generate_interactive_network(
        self, data: NetworkData, config: VisualizationConfig
    ) -> Dict[str, Any]:
        """Generate an interactive network visualization using Plotly.

        Args:
            data: Network data
            config: Visualization configuration

        Returns:
            Visualization metadata
        """
        # Create a graph
        G = nx.Graph()

        # Add nodes
        for node in data.nodes:
            node_id = node.get("id")
            G.add_node(node_id, **{k: v for k, v in node.items() if k != "id"})

        # Add edges
        for edge in data.edges:
            source = edge.get("source")
            target = edge.get("target")
            G.add_edge(
                source,
                target,
                **{k: v for k, v in edge.items() if k not in ["source", "target"]},
            )

        # Get positions based on layout
        if data.layout == "force":
            pos = nx.spring_layout(G)
        elif data.layout == "circular":
            pos = nx.circular_layout(G)
        elif data.layout == "random":
            pos = nx.random_layout(G)
        elif data.layout == "shell":
            pos = nx.shell_layout(G)
        elif data.layout == "spectral":
            pos = nx.spectral_layout(G)
        else:
            pos = nx.spring_layout(G)

        # Create edge trace
        edge_x = []
        edge_y = []
        edge_text = []
        edge_width = []
        edge_color = []

        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

            # Add edge attributes as hover text
            edge_data = G.get_edge_data(edge[0], edge[1])
            edge_text.append(str(edge_data.get("label", "")))

            # Set edge width if specified
            if data.edge_width_field and data.edge_width_field in edge_data:
                edge_width.append(edge_data[data.edge_width_field])
            else:
                edge_width.append(1)

            # Set edge color if specified
            if data.edge_color_field and data.edge_color_field in edge_data:
                edge_color.append(edge_data[data.edge_color_field])
            else:
                edge_color.append("#888")

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color="#888"),
            hoverinfo="text",
            mode="lines",
            text=edge_text,
        )

        # Create node trace
        node_x = []
        node_y = []
        node_text = []
        node_size = []
        node_color = []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Add node attributes as hover text
            node_data = G.nodes[node]
            node_text.append(str(node_data.get("label", node)))

            # Set node size if specified
            if data.node_size_field and data.node_size_field in node_data:
                node_size.append(node_data[data.node_size_field] * 10)
            else:
                node_size.append(10)

            # Set node color if specified
            if data.node_color_field and data.node_color_field in node_data:
                node_color.append(node_data[data.node_color_field])
            else:
                node_color.append("#1f77b4")

        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers",
            hoverinfo="text",
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale="YlGnBu",
                size=node_size,
                color=node_color,
                line=dict(width=2),
            ),
        )

        # Create figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=config.title,
                titlefont=dict(size=16),
                showlegend=False,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text=config.subtitle or "",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.005,
                        y=-0.002,
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                width=config.width,
                height=config.height,
            ),
        )

        # Save the figure
        filename = f"{config.title.lower().replace(' ', '_')}_network.html"
        filepath = os.path.join(self.output_dir, filename)
        fig.write_html(filepath)

        return {
            "type": "interactive_network",
            "title": config.title,
            "filepath": filepath,
            "url": f"file://{filepath}",
        }

    def _generate_static_network(
        self, data: NetworkData, config: VisualizationConfig
    ) -> Dict[str, Any]:
        """Generate a static network visualization using NetworkX and Matplotlib.

        Args:
            data: Network data
            config: Visualization configuration

        Returns:
            Visualization metadata
        """
        if not NETWORKX_AVAILABLE:
            raise ImportError("NetworkX is required for static network visualizations")

        # Create a graph
        G = nx.Graph()

        # Add nodes
        for node in data.nodes:
            node_id = node.get("id")
            G.add_node(node_id, **{k: v for k, v in node.items() if k != "id"})

        # Add edges
        for edge in data.edges:
            source = edge.get("source")
            target = edge.get("target")
            G.add_edge(
                source,
                target,
                **{k: v for k, v in edge.items() if k not in ["source", "target"]},
            )

        # Get positions based on layout
        if data.layout == "force":
            pos = nx.spring_layout(G)
        elif data.layout == "circular":
            pos = nx.circular_layout(G)
        elif data.layout == "random":
            pos = nx.random_layout(G)
        elif data.layout == "shell":
            pos = nx.shell_layout(G)
        elif data.layout == "spectral":
            pos = nx.spectral_layout(G)
        else:
            pos = nx.spring_layout(G)

        # Create figure
        plt.figure(figsize=(config.width / 100, config.height / 100), dpi=100)

        # Get node sizes
        node_sizes = []
        for node in G.nodes():
            node_data = G.nodes[node]
            if data.node_size_field and data.node_size_field in node_data:
                node_sizes.append(node_data[data.node_size_field] * 100)
            else:
                node_sizes.append(300)

        # Get node colors
        node_colors = []
        for node in G.nodes():
            node_data = G.nodes[node]
            if data.node_color_field and data.node_color_field in node_data:
                node_colors.append(node_data[data.node_color_field])
            else:
                node_colors.append("#1f77b4")

        # Get edge widths
        edge_widths = []
        for edge in G.edges():
            edge_data = G.get_edge_data(edge[0], edge[1])
            if data.edge_width_field and data.edge_width_field in edge_data:
                edge_widths.append(edge_data[data.edge_width_field])
            else:
                edge_widths.append(1)

        # Get edge colors
        edge_colors = []
        for edge in G.edges():
            edge_data = G.get_edge_data(edge[0], edge[1])
            if data.edge_color_field and data.edge_color_field in edge_data:
                edge_colors.append(edge_data[data.edge_color_field])
            else:
                edge_colors.append("#888")

        # Get node labels
        node_labels = {}
        for node in G.nodes():
            node_data = G.nodes[node]
            node_labels[node] = node_data.get("label", node)

        # Draw the network
        nx.draw_networkx_nodes(
            G, pos, node_size=node_sizes, node_color=node_colors, alpha=0.8
        )
        nx.draw_networkx_edges(
            G, pos, width=edge_widths, edge_color=edge_colors, alpha=0.5
        )
        nx.draw_networkx_labels(
            G, pos, labels=node_labels, font_size=10, font_family=config.font_family
        )

        # Set title
        plt.title(config.title)
        plt.axis("off")

        # Save the figure
        filename = f"{config.title.lower().replace(' ', '_')}_network.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(
            filepath, dpi=100, bbox_inches="tight", facecolor=config.background_color
        )
        plt.close()

        return {
            "type": "static_network",
            "title": config.title,
            "filepath": filepath,
            "url": f"file://{filepath}",
        }

    def generate_wordcloud(
        self, data: WordCloudData, config: VisualizationConfig
    ) -> Dict[str, Any]:
        """Generate a word cloud visualization.

        Args:
            data: Word cloud data
            config: Visualization configuration

        Returns:
            Visualization metadata
        """
        if not WORDCLOUD_AVAILABLE:
            raise ImportError("WordCloud is required for word cloud visualizations")

        # Create figure
        plt.figure(figsize=(config.width / 100, config.height / 100), dpi=100)

        # Create word cloud
        wordcloud = WordCloud(
            max_words=data.max_words,
            background_color=config.background_color,
            width=config.width,
            height=config.height,
            colormap="viridis",
            font_path=None,
            min_font_size=10,
            max_font_size=None,
            relative_scaling=0.5,
            normalize_plurals=True,
            include_numbers=False,
            min_word_length=3,
        ).generate(data.text)

        # Display word cloud
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title(config.title)

        # Save the figure
        filename = f"{config.title.lower().replace(' ', '_')}_wordcloud.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(
            filepath, dpi=100, bbox_inches="tight", facecolor=config.background_color
        )
        plt.close()

        return {
            "type": "wordcloud",
            "title": config.title,
            "filepath": filepath,
            "url": f"file://{filepath}",
        }

    def generate_map(
        self, data: MapData, config: VisualizationConfig
    ) -> Dict[str, Any]:
        """Generate a map visualization.

        Args:
            data: Map data
            config: Visualization configuration

        Returns:
            Visualization metadata
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for map visualizations")

        # Convert locations to DataFrame
        df = pd.DataFrame(data.locations)

        # Create choropleth map
        if data.location_mode == "ISO-3":
            fig = px.choropleth(
                df,
                locations=df.columns[0],  # First column should contain country codes
                color=data.color_field if data.color_field else df.columns[1],
                hover_name=df.columns[0],
                hover_data=data.hover_data,
                color_continuous_scale=px.colors.sequential.Plasma,
                title=config.title,
            )
        else:
            # Create scatter map
            fig = px.scatter_geo(
                df,
                lat=df.columns[0],  # First column should contain latitude
                lon=df.columns[1],  # Second column should contain longitude
                color=data.color_field if data.color_field else df.columns[2],
                size=data.size_field if data.size_field else None,
                hover_name=df.columns[2] if len(df.columns) > 2 else None,
                hover_data=data.hover_data,
                title=config.title,
            )

        # Update layout
        fig.update_layout(
            width=config.width,
            height=config.height,
            geo=dict(
                showland=True,
                landcolor="rgb(217, 217, 217)",
                showocean=True,
                oceancolor="rgb(204, 229, 255)",
                showlakes=True,
                lakecolor="rgb(204, 229, 255)",
                showrivers=True,
                rivercolor="rgb(204, 229, 255)",
            ),
        )

        # Save the figure
        filename = f"{config.title.lower().replace(' ', '_')}_map.html"
        filepath = os.path.join(self.output_dir, filename)
        fig.write_html(filepath)

        return {
            "type": "map",
            "title": config.title,
            "filepath": filepath,
            "url": f"file://{filepath}",
        }

    def generate_timeline(
        self, data: TimelineData, config: VisualizationConfig
    ) -> Dict[str, Any]:
        """Generate a timeline visualization.

        Args:
            data: Timeline data
            config: Visualization configuration

        Returns:
            Visualization metadata
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for timeline visualizations")

        # Convert events to DataFrame
        df = pd.DataFrame(data.events)

        # Create figure
        fig = go.Figure()

        # Add events to timeline
        if data.category_field and data.category_field in df.columns:
            # Group events by category
            categories = df[data.category_field].unique()

            for i, category in enumerate(categories):
                category_df = df[df[data.category_field] == category]

                fig.add_trace(
                    go.Scatter(
                        x=category_df[data.date_field],
                        y=[i] * len(category_df),
                        mode="markers+text",
                        name=category,
                        text=category_df[data.description_field],
                        textposition="top center",
                        hoverinfo="text",
                        hovertext=category_df[data.description_field],
                        marker=dict(size=10, symbol="circle"),
                    )
                )

            # Set y-axis labels to categories
            fig.update_layout(
                yaxis=dict(
                    tickmode="array",
                    tickvals=list(range(len(categories))),
                    ticktext=categories,
                )
            )
        else:
            # Simple timeline without categories
            fig.add_trace(
                go.Scatter(
                    x=df[data.date_field],
                    y=[0] * len(df),
                    mode="markers+text",
                    text=df[data.description_field],
                    textposition="top center",
                    hoverinfo="text",
                    hovertext=df[data.description_field],
                    marker=dict(size=10, symbol="circle"),
                )
            )

            # Hide y-axis
            fig.update_layout(yaxis=dict(showticklabels=False, zeroline=False))

        # Update layout
        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            showlegend=config.show_legend,
            xaxis=dict(title="Time", showgrid=config.grid),
            plot_bgcolor=config.background_color,
            paper_bgcolor=config.background_color,
        )

        # Save the figure
        filename = f"{config.title.lower().replace(' ', '_')}_timeline.html"
        filepath = os.path.join(self.output_dir, filename)
        fig.write_html(filepath)

        return {
            "type": "timeline",
            "title": config.title,
            "filepath": filepath,
            "url": f"file://{filepath}",
        }

    def generate_visualization(
        self,
        visualization_type: str,
        data: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a visualization based on type.

        Args:
            visualization_type: Type of visualization to generate
            data: Visualization data
            config: Optional visualization configuration

        Returns:
            Visualization metadata
        """
        # Create configuration
        if config is None:
            config = {}

        visualization_config = VisualizationConfig(
            title=config.get("title", "Visualization"),
            subtitle=config.get("subtitle"),
            width=config.get("width", 800),
            height=config.get("height", 600),
            theme=config.get("theme", "default"),
            interactive=config.get("interactive", True),
            output_format=config.get("output_format", "html"),
            color_scheme=config.get("color_scheme"),
            show_legend=config.get("show_legend", True),
            font_family=config.get("font_family", "Arial"),
            background_color=config.get("background_color", "#ffffff"),
            grid=config.get("grid", True),
            animation=config.get("animation", False),
        )

        # Generate visualization based on type
        if visualization_type == "chart":
            chart_data = ChartData(
                chart_type=data.get("chart_type", "bar"),
                x_data=data.get("x_data", []),
                y_data=data.get("y_data", []),
                labels=data.get("labels"),
                x_label=data.get("x_label"),
                y_label=data.get("y_label"),
                series_names=data.get("series_names"),
                annotations=data.get("annotations"),
            )
            return self.generate_chart(chart_data, visualization_config)

        elif visualization_type == "network":
            network_data = NetworkData(
                nodes=data.get("nodes", []),
                edges=data.get("edges", []),
                node_size_field=data.get("node_size_field"),
                node_color_field=data.get("node_color_field"),
                edge_width_field=data.get("edge_width_field"),
                edge_color_field=data.get("edge_color_field"),
                layout=data.get("layout", "force"),
            )
            return self.generate_network(network_data, visualization_config)

        elif visualization_type == "wordcloud":
            wordcloud_data = WordCloudData(
                text=data.get("text", ""),
                max_words=data.get("max_words", 200),
                mask_image=data.get("mask_image"),
            )
            return self.generate_wordcloud(wordcloud_data, visualization_config)

        elif visualization_type == "map":
            map_data = MapData(
                locations=data.get("locations", []),
                location_mode=data.get("location_mode", "ISO-3"),
                color_field=data.get("color_field"),
                size_field=data.get("size_field"),
                hover_data=data.get("hover_data"),
                map_style=data.get("map_style", "carto-positron"),
            )
            return self.generate_map(map_data, visualization_config)

        elif visualization_type == "timeline":
            timeline_data = TimelineData(
                events=data.get("events", []),
                date_field=data.get("date_field", "date"),
                description_field=data.get("description_field", "description"),
                category_field=data.get("category_field"),
            )
            return self.generate_timeline(timeline_data, visualization_config)

        else:
            raise ValueError(f"Unsupported visualization type: {visualization_type}")


def generate_chart_tool(data_str: str) -> str:
    """Generate a chart visualization.

    Args:
        data_str: JSON string with chart data and configuration

    Returns:
        JSON string with visualization metadata
    """
    try:
        data = json.loads(data_str)

        # Create visualization generator
        generator = VisualizationGenerator()

        # Generate chart
        result = generator.generate_visualization(
            visualization_type="chart",
            data=data.get("data", {}),
            config=data.get("config", {}),
        )

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


def generate_network_diagram_tool(data_str: str) -> str:
    """Generate a network diagram visualization.

    Args:
        data_str: JSON string with network data and configuration

    Returns:
        JSON string with visualization metadata
    """
    try:
        data = json.loads(data_str)

        # Create visualization generator
        generator = VisualizationGenerator()

        # Generate network diagram
        result = generator.generate_visualization(
            visualization_type="network",
            data=data.get("data", {}),
            config=data.get("config", {}),
        )

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


def generate_wordcloud_tool(data_str: str) -> str:
    """Generate a word cloud visualization.

    Args:
        data_str: JSON string with word cloud data and configuration

    Returns:
        JSON string with visualization metadata
    """
    try:
        data = json.loads(data_str)

        # Create visualization generator
        generator = VisualizationGenerator()

        # Generate word cloud
        result = generator.generate_visualization(
            visualization_type="wordcloud",
            data=data.get("data", {}),
            config=data.get("config", {}),
        )

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


def generate_map_tool(data_str: str) -> str:
    """Generate a map visualization.

    Args:
        data_str: JSON string with map data and configuration

    Returns:
        JSON string with visualization metadata
    """
    try:
        data = json.loads(data_str)

        # Create visualization generator
        generator = VisualizationGenerator()

        # Generate map
        result = generator.generate_visualization(
            visualization_type="map",
            data=data.get("data", {}),
            config=data.get("config", {}),
        )

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


def generate_timeline_tool(data_str: str) -> str:
    """Generate a timeline visualization.

    Args:
        data_str: JSON string with timeline data and configuration

    Returns:
        JSON string with visualization metadata
    """
    try:
        data = json.loads(data_str)

        # Create visualization generator
        generator = VisualizationGenerator()

        # Generate timeline
        result = generator.generate_visualization(
            visualization_type="timeline",
            data=data.get("data", {}),
            config=data.get("config", {}),
        )

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    # Example usage
    generator = VisualizationGenerator()

    # Generate a chart
    chart_data = ChartData(
        chart_type="bar",
        x_data=["A", "B", "C", "D", "E"],
        y_data=[10, 20, 15, 25, 30],
        x_label="Categories",
        y_label="Values",
    )

    chart_config = VisualizationConfig(
        title="Example Bar Chart", width=800, height=600, interactive=True
    )

    chart_result = generator.generate_chart(chart_data, chart_config)
    print(f"Chart generated: {chart_result['filepath']}")

    # Generate a network diagram
    network_data = NetworkData(
        nodes=[
            {"id": 1, "label": "Node 1"},
            {"id": 2, "label": "Node 2"},
            {"id": 3, "label": "Node 3"},
            {"id": 4, "label": "Node 4"},
            {"id": 5, "label": "Node 5"},
        ],
        edges=[
            {"source": 1, "target": 2, "label": "Edge 1-2"},
            {"source": 1, "target": 3, "label": "Edge 1-3"},
            {"source": 2, "target": 4, "label": "Edge 2-4"},
            {"source": 3, "target": 5, "label": "Edge 3-5"},
            {"source": 4, "target": 5, "label": "Edge 4-5"},
        ],
        layout="force",
    )

    network_config = VisualizationConfig(
        title="Example Network Diagram", width=800, height=600, interactive=True
    )

    network_result = generator.generate_network(network_data, network_config)
    print(f"Network diagram generated: {network_result['filepath']}")
