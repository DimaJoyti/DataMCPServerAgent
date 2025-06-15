"""
Test script for dashboards.
This script tests the main functionality of the dashboards.
"""

import json
import sys
from pathlib import Path

# Adding the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))


try:
    from src.tools.research_dashboard import (
        Dashboard,
        DashboardConfig,
        DashboardGenerator,
        DashboardPanel,
        generate_dashboard_tool,
    )

    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False
    print("Warning: Dash or Plotly not available. Skipping dashboard tests.")


def test_grid_dashboard():
    """Testing dashboard with grid layout."""
    if not DASHBOARD_AVAILABLE:
        print("Dash or Plotly not available. Skipping grid layout dashboard test.")
        return

    print("Testing dashboard with grid layout...")

    # Creating a dashboard generator
    generator = DashboardGenerator()

    # Creating a dashboard
    dashboard = Dashboard(
        id="test-grid-dashboard",
        title="Test Dashboard with Grid Layout",
        config=DashboardConfig(
            title="Test Dashboard with Grid Layout",
            subtitle="Grid layout demonstration",
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
                config={"title": "Sample Chart", "x_label": "Categories", "y_label": "Values"},
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
                    "text": "This is a sample text panel. You can place any text here, including research results, conclusions, etc."
                },
                width=12,
                height=2,
                x=0,
                y=4,
            ),
        ],
    )

    # Generating the dashboard
    try:
        result = generator.generate_dashboard(dashboard)
        print(f"Dashboard with grid layout generated: {result['url']}")
    except Exception as e:
        print(f"Error generating dashboard with grid layout: {str(e)}")


def test_tabs_dashboard():
    """Testing dashboard with tabs."""
    if not DASHBOARD_AVAILABLE:
        print("Dash or Plotly not available. Skipping tabs dashboard test.")
        return

    print("\nTesting dashboard with tabs...")

    # Creating a dashboard generator
    generator = DashboardGenerator()

    # Creating a dashboard
    dashboard = Dashboard(
        id="test-tabs-dashboard",
        title="Test Dashboard with Tabs",
        config=DashboardConfig(
            title="Test Dashboard with Tabs",
            subtitle="Tabs layout demonstration",
            layout="tabs",
        ),
        panels=[
            DashboardPanel(
                id="chart-panel",
                title="Chart",
                type="chart",
                data={
                    "chart_type": "line",
                    "x_data": [1, 2, 3, 4, 5],
                    "y_data": [10, 20, 15, 25, 30],
                },
                config={"title": "Sample Line Chart", "x_label": "X", "y_label": "Y"},
                tab="Charts",
            ),
            DashboardPanel(
                id="pie-chart-panel",
                title="Pie Chart",
                type="chart",
                data={
                    "chart_type": "pie",
                    "x_data": ["A", "B", "C", "D", "E"],
                    "y_data": [10, 20, 15, 25, 30],
                },
                config={"title": "Sample Pie Chart"},
                tab="Charts",
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
                tab="Data",
            ),
            DashboardPanel(
                id="text-panel",
                title="Text",
                type="text",
                data={
                    "text": "This is a sample text panel. You can place any text here, including research results, conclusions, etc."
                },
                tab="Information",
            ),
        ],
    )

    # Generating the dashboard
    try:
        result = generator.generate_dashboard(dashboard)
        print(f"Dashboard with tabs generated: {result['url']}")
    except Exception as e:
        print(f"Error generating dashboard with tabs: {str(e)}")


def test_dashboard_tool():
    """Testing the dashboard generation tool."""
    if not DASHBOARD_AVAILABLE:
        print("Dash or Plotly not available. Skipping dashboard generation tool test.")
        return

    print("\nTesting the dashboard generation tool...")

    # Creating data for the tool
    tool_input = {
        "id": "tool-dashboard",
        "title": "Dashboard created by the tool",
        "config": {
            "title": "Dashboard created by the tool",
            "subtitle": "Demonstration of the dashboard generation tool",
            "layout": "grid",
        },
        "panels": [
            {
                "id": "chart-panel",
                "title": "Chart",
                "type": "chart",
                "data": {
                    "chart_type": "bar",
                    "x_data": ["A", "B", "C", "D", "E"],
                    "y_data": [10, 20, 15, 25, 30],
                },
                "config": {
                    "title": "Sample Chart",
                    "x_label": "Categories",
                    "y_label": "Values",
                },
                "width": 12,
                "height": 6,
                "x": 0,
                "y": 0,
            }
        ],
    }

    # Calling the tool
    try:
        result = generate_dashboard_tool(json.dumps(tool_input))
        result_dict = json.loads(result)

        print(f"Result of the dashboard generation tool: {result_dict}")
    except Exception as e:
        print(f"Error calling the dashboard generation tool: {str(e)}")


def main():
    """Running the tests."""
    if not DASHBOARD_AVAILABLE:
        print("Dash or Plotly not available. Skipping all dashboard tests.")
        return

    # Testing the grid layout dashboard
    test_grid_dashboard()

    # Testing the tabs dashboard
    test_tabs_dashboard()

    # Testing the dashboard generation tool
    test_dashboard_tool()

    print("\nAll tests completed.")


if __name__ == "__main__":
    main()
