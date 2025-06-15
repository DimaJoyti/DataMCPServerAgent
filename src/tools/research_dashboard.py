"""
Interactive dashboards for the Research Assistant.
This module provides capabilities for creating interactive dashboards
for visualizing and analyzing research data.
"""

import json
import os
import tempfile
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

# Try to import the libraries for the dashboards
try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output

    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    print("Warning: Dash is not available. Install it using 'pip install dash'")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly is not available. Install it using 'pip install plotly'")


class DashboardConfig(BaseModel):
    """Configuration for dashboards."""

    title: str
    subtitle: Optional[str] = None
    width: str = "100%"
    height: str = "800px"
    theme: str = "default"
    background_color: str = "#ffffff"
    font_family: str = "Arial"
    layout: str = "grid"  # grid, tabs, vertical, horizontal
    refresh_interval: Optional[int] = None  # in seconds

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True


class DashboardPanel(BaseModel):
    """Panel for the dashboard."""

    id: str
    title: str
    type: str  # chart, table, map, text, html, iframe
    data: Dict[str, Any]
    config: Dict[str, Any] = {}
    width: int = 1  # relative width (1-12 for grid layout)
    height: int = 1  # relative height (1-12 for grid layout)
    x: int = 0  # x position for grid layout
    y: int = 0  # y position for grid layout
    tab: Optional[str] = None  # for tab layout

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True


class Dashboard(BaseModel):
    """Dashboard for visualizing research data."""

    id: str
    title: str
    config: DashboardConfig
    panels: List[DashboardPanel]

    class Config:
        """Pydantic configuration."""

        arbitrary_types_allowed = True


class DashboardGenerator:
    """Generator for dashboards."""

    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the dashboard generator.

        Args:
            output_dir: Directory for saving dashboards
        """
        self.output_dir = output_dir or tempfile.mkdtemp()
        os.makedirs(self.output_dir, exist_ok=True)

        if not DASH_AVAILABLE or not PLOTLY_AVAILABLE:
            print("Warning: Dash or Plotly is not available. Dashboards will be limited.")

    def generate_dashboard(self, dashboard: Dashboard) -> Dict[str, Any]:
        """Generate a dashboard.

        Args:
            dashboard: Dashboard to generate

        Returns:
            Metadata of the dashboard
        """
        if not DASH_AVAILABLE:
            raise ImportError("Dash is required to generate dashboards")

        # Create Dash app
        app = dash.Dash(__name__, suppress_callback_exceptions=True)

        # Set the title
        app.title = dashboard.title

        # Create layout based on configuration
        if dashboard.config.layout == "tabs":
            app.layout = self._create_tabs_layout(dashboard)
        elif dashboard.config.layout == "vertical":
            app.layout = self._create_vertical_layout(dashboard)
        elif dashboard.config.layout == "horizontal":
            app.layout = self._create_horizontal_layout(dashboard)
        else:  # grid (default)
            app.layout = self._create_grid_layout(dashboard)

        # Add callbacks for interactivity
        self._add_callbacks(app, dashboard)

        # Save the dashboard
        filename = f"{dashboard.id.lower().replace(' ', '_')}_dashboard.html"
        filepath = os.path.join(self.output_dir, filename)

        # Run the server and save HTML
        app.run_server(debug=False, port=8050, mode="inline")

        return {
            "type": "dashboard",
            "id": dashboard.id,
            "title": dashboard.title,
            "filepath": filepath,
            "url": "http://localhost:8050",
        }

    def _create_grid_layout(self, dashboard: Dashboard) -> html.Div:
        """Create a grid layout for the dashboard.

        Args:
            dashboard: Dashboard to generate

        Returns:
            Dashboard layout
        """
        # Create a header
        header = html.Div(
            [
                html.H1(dashboard.title, style={"textAlign": "center"}),
                (
                    html.H3(dashboard.config.subtitle, style={"textAlign": "center"})
                    if dashboard.config.subtitle
                    else None
                ),
            ]
        )

        # Create a grid
        grid = html.Div(
            [
                html.Div(
                    [self._create_panel(panel) for panel in dashboard.panels],
                    className="grid-container",
                )
            ]
        )

        # Create styles for the grid
        grid_style = {
            "display": "grid",
            "gridTemplateColumns": "repeat(12, 1fr)",
            "gridGap": "10px",
            "padding": "10px",
        }

        # Add styles for each panel
        panel_styles = {}
        for panel in dashboard.panels:
            panel_styles[f"#{panel.id}"] = {
                "gridColumn": f"span {panel.width}",
                "gridRow": f"span {panel.height}",
            }

        # Create styles
        styles = html.Style(
            f"""
            .grid-container {{
                display: grid;
                grid-template-columns: repeat(12, 1fr);
                grid-gap: 10px;
                padding: 10px;
            }}

            {" ".join([f"#{panel.id} {{ grid-column: span {panel.width}; grid-row: span {panel.height}; }}" for panel in dashboard.panels])}
        """
        )

        # Create layout
        layout = html.Div(
            [styles, header, grid],
            style={
                "fontFamily": dashboard.config.font_family,
                "backgroundColor": dashboard.config.background_color,
                "padding": "20px",
            },
        )

        return layout

    def _create_tabs_layout(self, dashboard: Dashboard) -> html.Div:
        """Create a tabbed layout for the dashboard.

        Args:
            dashboard: Dashboard to generate

        Returns:
            Dashboard layout
        """
        # Create a header
        header = html.Div(
            [
                html.H1(dashboard.title, style={"textAlign": "center"}),
                (
                    html.H3(dashboard.config.subtitle, style={"textAlign": "center"})
                    if dashboard.config.subtitle
                    else None
                ),
            ]
        )

        # Get unique tabs
        tabs = list(set([panel.tab for panel in dashboard.panels if panel.tab]))

        # Create tabs
        tab_content = []
        for tab in tabs:
            tab_panels = [panel for panel in dashboard.panels if panel.tab == tab]
            tab_content.append(
                dcc.Tab(
                    label=tab,
                    children=html.Div(
                        [self._create_panel(panel) for panel in tab_panels],
                        style={"padding": "20px"},
                    ),
                )
            )

        # Add panels without tabs
        no_tab_panels = [panel for panel in dashboard.panels if not panel.tab]
        if no_tab_panels:
            tab_content.append(
                dcc.Tab(
                    label="General",
                    children=html.Div(
                        [self._create_panel(panel) for panel in no_tab_panels],
                        style={"padding": "20px"},
                    ),
                )
            )

        # Create tabs component
        tabs_component = dcc.Tabs(id="tabs", children=tab_content, style={"marginTop": "20px"})

        # Create layout
        layout = html.Div(
            [header, tabs_component],
            style={
                "fontFamily": dashboard.config.font_family,
                "backgroundColor": dashboard.config.background_color,
                "padding": "20px",
            },
        )

        return layout

    def _create_vertical_layout(self, dashboard: Dashboard) -> html.Div:
        """Create a vertical layout for the dashboard.

        Args:
            dashboard: Dashboard to generate

        Returns:
            Dashboard layout
        """
        # Create a header
        header = html.Div(
            [
                html.H1(dashboard.title, style={"textAlign": "center"}),
                (
                    html.H3(dashboard.config.subtitle, style={"textAlign": "center"})
                    if dashboard.config.subtitle
                    else None
                ),
            ]
        )

        # Create panels
        panels = html.Div(
            [self._create_panel(panel) for panel in dashboard.panels],
            style={"display": "flex", "flexDirection": "column", "gap": "20px"},
        )

        # Create layout
        layout = html.Div(
            [header, panels],
            style={
                "fontFamily": dashboard.config.font_family,
                "backgroundColor": dashboard.config.background_color,
                "padding": "20px",
            },
        )

        return layout

    def _create_horizontal_layout(self, dashboard: Dashboard) -> html.Div:
        """Create a horizontal layout for the dashboard.

        Args:
            dashboard: Dashboard to generate

        Returns:
            Dashboard layout
        """
        # Create a header
        header = html.Div(
            [
                html.H1(dashboard.title, style={"textAlign": "center"}),
                (
                    html.H3(dashboard.config.subtitle, style={"textAlign": "center"})
                    if dashboard.config.subtitle
                    else None
                ),
            ]
        )

        # Create panels
        panels = html.Div(
            [self._create_panel(panel) for panel in dashboard.panels],
            style={
                "display": "flex",
                "flexDirection": "row",
                "flexWrap": "wrap",
                "gap": "20px",
            },
        )

        # Create layout
        layout = html.Div(
            [header, panels],
            style={
                "fontFamily": dashboard.config.font_family,
                "backgroundColor": dashboard.config.background_color,
                "padding": "20px",
            },
        )

        return layout

    def _create_panel(self, panel: DashboardPanel) -> html.Div:
        """Create a panel for the dashboard.

        Args:
            panel: Panel to create

        Returns:
            Panel component
        """
        # Create panel content based on type
        if panel.type == "chart":
            content = self._create_chart_panel(panel)
        elif panel.type == "table":
            content = self._create_table_panel(panel)
        elif panel.type == "map":
            content = self._create_map_panel(panel)
        elif panel.type == "text":
            content = self._create_text_panel(panel)
        elif panel.type == "html":
            content = self._create_html_panel(panel)
        elif panel.type == "iframe":
            content = self._create_iframe_panel(panel)
        else:
            content = html.Div(f"Unsupported panel type: {panel.type}")

        # Create panel
        panel_div = html.Div(
            [
                html.H3(panel.title, style={"textAlign": "center", "marginBottom": "10px"}),
                content,
            ],
            id=panel.id,
            style={
                "border": "1px solid #ddd",
                "borderRadius": "5px",
                "padding": "10px",
                "boxShadow": "0 2px 4px rgba(0, 0, 0, 0.1)",
            },
        )

        return panel_div

    def _create_chart_panel(self, panel: DashboardPanel) -> html.Div:
        """Create a panel with a chart.

        Args:
            panel: Panel to create

        Returns:
            Panel component
        """
        if not PLOTLY_AVAILABLE:
            return html.Div("Plotly is not available. Cannot create chart.")

        # Get data and configuration
        chart_type = panel.data.get("chart_type", "bar")
        x_data = panel.data.get("x_data", [])
        y_data = panel.data.get("y_data", [])

        # Create chart based on type
        if chart_type == "bar":
            fig = go.Figure(data=[go.Bar(x=x_data, y=y_data)])
        elif chart_type == "line":
            fig = go.Figure(data=[go.Scatter(x=x_data, y=y_data, mode="lines+markers")])
        elif chart_type == "scatter":
            fig = go.Figure(data=[go.Scatter(x=x_data, y=y_data, mode="markers")])
        elif chart_type == "pie":
            fig = go.Figure(data=[go.Pie(labels=x_data, values=y_data)])
        elif chart_type == "area":
            fig = go.Figure(data=[go.Scatter(x=x_data, y=y_data, fill="tozeroy")])
        else:
            return html.Div(f"Unsupported chart type: {chart_type}")

        # Update layout
        fig.update_layout(
            title=panel.config.get("title"),
            xaxis_title=panel.config.get("x_label"),
            yaxis_title=panel.config.get("y_label"),
            margin=dict(l=40, r=40, t=40, b=40),
        )

        # Create chart component
        graph = dcc.Graph(
            id=f"{panel.id}-graph",
            figure=fig,
            style={"height": "100%", "width": "100%"},
        )

        return html.Div(graph, style={"height": "100%", "width": "100%"})

    def _create_table_panel(self, panel: DashboardPanel) -> html.Div:
        """Create a panel with a table.

        Args:
            panel: Panel to create

        Returns:
            Panel component
        """
        # Get data and configuration
        columns = panel.data.get("columns", [])
        data = panel.data.get("data", [])

        # Create table headers
        header = html.Tr([html.Th(col) for col in columns])

        # Create table rows
        rows = []
        for row in data:
            rows.append(html.Tr([html.Td(cell) for cell in row]))

        # Create table
        table = html.Table(
            [html.Thead(header), html.Tbody(rows)],
            style={"width": "100%", "borderCollapse": "collapse"},
        )

        # Create styles for the table
        styles = html.Style(
            """
            table {
                width: 100%;
                border-collapse: collapse;
            }

            th, td {
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }

            th {
                background-color: #f2f2f2;
            }

            tr:hover {
                background-color: #f5f5f5;
            }
        """
        )

        return html.Div([styles, table], style={"overflowX": "auto"})

    def _create_map_panel(self, panel: DashboardPanel) -> html.Div:
        """Create a panel with a map.

        Args:
            panel: Panel to create

        Returns:
            Panel component
        """
        if not PLOTLY_AVAILABLE:
            return html.Div("Plotly is not available. Cannot create map.")

        # Get data and configuration
        locations = panel.data.get("locations", [])
        location_mode = panel.data.get("location_mode", "ISO-3")
        color_field = panel.data.get("color_field")

        # Create map
        if location_mode == "ISO-3":
            # Create choropleth
            fig = px.choropleth(
                locations=locations,
                locationmode="ISO-3",
                color=color_field,
                color_continuous_scale="Viridis",
                title=panel.config.get("title"),
            )
        else:
            # Create scatter map
            fig = px.scatter_geo(
                lat=[loc.get("lat") for loc in locations],
                lon=[loc.get("lon") for loc in locations],
                text=[loc.get("name") for loc in locations],
                title=panel.config.get("title"),
            )

        # Update layout
        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
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

        # Create map component
        graph = dcc.Graph(
            id=f"{panel.id}-map", figure=fig, style={"height": "100%", "width": "100%"}
        )

        return html.Div(graph, style={"height": "100%", "width": "100%"})

    def _create_text_panel(self, panel: DashboardPanel) -> html.Div:
        """Create a panel with text.

        Args:
            panel: Panel to create

        Returns:
            Panel component
        """
        # Get data and configuration
        text = panel.data.get("text", "")

        # Create text component
        return html.Div(
            text,
            style={
                "whiteSpace": "pre-wrap",
                "overflowY": "auto",
                "height": "100%",
                "padding": "10px",
            },
        )

    def _create_html_panel(self, panel: DashboardPanel) -> html.Div:
        """Create a panel with HTML.

        Args:
            panel: Panel to create

        Returns:
            Panel component
        """
        # Get data and configuration
        html_content = panel.data.get("html", "")

        # Create HTML component
        return html.Iframe(
            srcDoc=html_content,
            style={"width": "100%", "height": "100%", "border": "none"},
        )

    def _create_iframe_panel(self, panel: DashboardPanel) -> html.Div:
        """Create a panel with iframe.

        Args:
            panel: Panel to create

        Returns:
            Panel component
        """
        # Get data and configuration
        url = panel.data.get("url", "")

        # Create iframe component
        return html.Iframe(src=url, style={"width": "100%", "height": "100%", "border": "none"})

    def _add_callbacks(self, app: dash.Dash, dashboard: Dashboard) -> None:
        """Add callbacks for interactivity.

        Args:
            app: Dash app
            dashboard: Dashboard to generate
        """
        # Add callbacks for data updates
        if dashboard.config.refresh_interval:

            @app.callback(
                Output("dashboard-container", "children"),
                Input("refresh-interval", "n_intervals"),
            )
            def update_dashboard(n):
                # Logic for updating data can be added here
                return self._create_dashboard_content(dashboard)


def generate_dashboard_tool(data_str: str) -> str:
    """Generate a dashboard.

    Args:
        data_str: JSON string with dashboard data and configuration

    Returns:
        JSON string with dashboard metadata
    """
    try:
        data = json.loads(data_str)

        # Create dashboard generator
        generator = DashboardGenerator()

        # Create dashboard
        dashboard = Dashboard(
            id=data.get("id", "dashboard"),
            title=data.get("title", "Dashboard"),
            config=DashboardConfig(**data.get("config", {})),
            panels=[DashboardPanel(**panel) for panel in data.get("panels", [])],
        )

        # Generate dashboard
        result = generator.generate_dashboard(dashboard)

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})


if __name__ == "__main__":
    # Example usage
    generator = DashboardGenerator()

    # Create dashboard
    dashboard = Dashboard(
        id="example-dashboard",
        title="Example Dashboard",
        config=DashboardConfig(
            title="Example Dashboard",
            subtitle="Demonstration of dashboard capabilities",
            layout="grid",
        ),
        panels=[
            DashboardPanel(
                id="chart-panel",
                title="Chart",
                type="chart",
                data={
                    "chart_type": "bar",
                    "x_data": ["A", "B", "C", "D", "E"],
                    "y_data": [10, 20, 15, 25, 30],
                },
                config={
                    "title": "Example Chart",
                    "x_label": "Categories",
                    "y_label": "Values",
                },
                width=6,
                height=4,
                x=0,
                y=0,
            ),
            DashboardPanel(
                id="table-panel",
                title="Table",
                type="table",
                data={
                    "columns": ["Name", "Value", "Description"],
                    "data": [
                        ["A", 10, "Description A"],
                        ["B", 20, "Description B"],
                        ["C", 15, "Description C"],
                        ["D", 25, "Description D"],
                        ["E", 30, "Description E"],
                    ],
                },
                width=6,
                height=4,
                x=6,
                y=0,
            ),
            DashboardPanel(
                id="text-panel",
                title="Text",
                type="text",
                data={
                    "text": "This is an example of a text panel. Any text can be placed here, including research results, conclusions, etc."
                },
                width=12,
                height=2,
                x=0,
                y=4,
            ),
        ],
    )

    # Generate dashboard
    result = generator.generate_dashboard(dashboard)
    print(f"Dashboard generated: {result['url']}")
