"""
Test script for 3D visualizations.
This script tests the basic functionality of 3D visualizations.
"""

import json
import sys
import tempfile
from pathlib import Path

# Adding the parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

from src.tools.research_3d_visualization import (
    Scatter3DData,
    Surface3DData,
    Visualization3DConfig,
    Visualization3DGenerator,
    Volume3DData,
    generate_scatter_3d_tool,
    generate_surface_3d_tool,
    generate_volume_3d_tool,
)


def test_surface_3d_visualization():
    """Testing 3D surface visualization."""
    print("Testing 3D surface visualization...")

    # Create a visualization generator
    generator = Visualization3DGenerator()

    # Create data for the surface
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    surface_data = Surface3DData(
        x_data=X.tolist(),
        y_data=Y.tolist(),
        z_data=Z.tolist(),
        x_label="X",
        y_label="Y",
        z_label="Z",
    )

    # Create a surface configuration
    surface_config = Visualization3DConfig(
        title="Test 3D Surface", width=800, height=600, interactive=True
    )

    # Generate the surface
    result = generator.generate_surface_3d(surface_data, surface_config)

    print(f"3D surface generated: {result['filepath']}")
    print(f"3D surface URL: {result['url']}")

    # Test the tool function
    tool_input = {
        "data": {
            "x_data": X.tolist(),
            "y_data": Y.tolist(),
            "z_data": Z.tolist(),
            "x_label": "X",
            "y_label": "Y",
            "z_label": "Z",
        },
        "config": {
            "title": "Test 3D Surface (Tool)",
            "width": 800,
            "height": 600,
            "interactive": True,
        },
    }

    tool_result = generate_surface_3d_tool(json.dumps(tool_input))
    tool_result_dict = json.loads(tool_result)

    print(f"3D surface tool result: {tool_result_dict['filepath']}")
    print(f"3D surface tool URL: {tool_result_dict['url']}")


def test_scatter_3d_visualization():
    """Testing 3D scatter visualization."""
    print("\nTesting 3D scatter visualization...")

    # Create a visualization generator
    generator = Visualization3DGenerator()

    # Create data for the scatter plot
    n = 100
    x = np.random.randn(n)
    y = np.random.randn(n)
    z = np.random.randn(n)
    colors = np.sqrt(x**2 + y**2 + z**2)
    sizes = np.random.rand(n) * 30 + 10

    scatter_data = Scatter3DData(
        x_data=x.tolist(),
        y_data=y.tolist(),
        z_data=z.tolist(),
        color_data=colors.tolist(),
        size_data=sizes.tolist(),
        x_label="X",
        y_label="Y",
        z_label="Z",
    )

    # Create a scatter plot configuration
    scatter_config = Visualization3DConfig(
        title="Test 3D Scatter Plot", width=800, height=600, interactive=True
    )

    # Generate the scatter plot
    result = generator.generate_scatter_3d(scatter_data, scatter_config)

    print(f"3D scatter plot generated: {result['filepath']}")
    print(f"3D scatter plot URL: {result['url']}")

    # Test the tool function
    tool_input = {
        "data": {
            "x_data": x.tolist(),
            "y_data": y.tolist(),
            "z_data": z.tolist(),
            "color_data": colors.tolist(),
            "size_data": sizes.tolist(),
            "x_label": "X",
            "y_label": "Y",
            "z_label": "Z",
        },
        "config": {
            "title": "Test 3D Scatter Plot (Tool)",
            "width": 800,
            "height": 600,
            "interactive": True,
        },
    }

    tool_result = generate_scatter_3d_tool(json.dumps(tool_input))
    tool_result_dict = json.loads(tool_result)

    print(f"3D scatter tool result: {tool_result_dict['filepath']}")
    print(f"3D scatter tool URL: {tool_result_dict['url']}")


def test_volume_3d_visualization():
    """Testing 3D volume visualization."""
    print("\nTesting 3D volume visualization...")

    # Create a visualization generator
    generator = Visualization3DGenerator()

    # Create data for the volume visualization
    n = 20
    x = np.linspace(-5, 5, n)
    y = np.linspace(-5, 5, n)
    z = np.linspace(-5, 5, n)

    X, Y, Z = np.meshgrid(x, y, z)
    volume_data = np.exp(-(X**2 + Y**2 + Z**2) / 10)

    # Create data for the volume visualization
    volume_data_obj = Volume3DData(
        volume_data=volume_data.tolist(),
        x_range=x.tolist(),
        y_range=y.tolist(),
        z_range=z.tolist(),
        x_label="X",
        y_label="Y",
        z_label="Z",
    )

    # Create a volume visualization configuration
    volume_config = Visualization3DConfig(
        title="Test 3D Volume Visualization", width=800, height=600, interactive=True
    )

    # Generate the volume visualization
    try:
        result = generator.generate_volume_3d(volume_data_obj, volume_config)

        print(f"3D volume visualization generated: {result['filepath']}")
        print(f"3D volume visualization URL: {result['url']}")

        # Test the tool function
        tool_input = {
            "data": {
                "volume_data": volume_data.tolist(),
                "x_range": x.tolist(),
                "y_range": y.tolist(),
                "z_range": z.tolist(),
                "x_label": "X",
                "y_label": "Y",
                "z_label": "Z",
            },
            "config": {
                "title": "Test 3D Volume Visualization (Tool)",
                "width": 800,
                "height": 600,
                "interactive": True,
            },
        }

        tool_result = generate_volume_3d_tool(json.dumps(tool_input))
        tool_result_dict = json.loads(tool_result)

        print(f"3D volume tool result: {tool_result_dict['filepath']}")
        print(f"3D volume tool URL: {tool_result_dict['url']}")
    except ImportError:
        print("Plotly not available. Skipping 3D volume visualization test.")


def main():
    """Run tests."""
    # Create a temporary directory for test visualizations
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")

        # Test 3D surface visualization
        test_surface_3d_visualization()

        # Test 3D scatter visualization
        test_scatter_3d_visualization()

        # Test 3D volume visualization
        test_volume_3d_visualization()

        print("\nAll tests completed.")


if __name__ == "__main__":
    main()
