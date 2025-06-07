"""
Тестовий скрипт для 3D-візуалізацій.
Цей скрипт тестує основну функціональність 3D-візуалізацій.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Додаємо батьківську директорію до шляху Python
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
    """Тестування 3D-поверхневої візуалізації."""
    print("Тестування 3D-поверхневої візуалізації...")

    # Створюємо генератор візуалізації
    generator = Visualization3DGenerator()

    # Створюємо дані для поверхні
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
        z_label="Z"
    )

    # Створюємо конфігурацію поверхні
    surface_config = Visualization3DConfig(
        title="Тестова 3D-поверхня",
        width=800,
        height=600,
        interactive=True
    )

    # Генеруємо поверхню
    result = generator.generate_surface_3d(surface_data, surface_config)

    print(f"3D-поверхня згенерована: {result['filepath']}")
    print(f"URL 3D-поверхні: {result['url']}")

    # Тестуємо функцію інструменту
    tool_input = {
        "data": {
            "x_data": X.tolist(),
            "y_data": Y.tolist(),
            "z_data": Z.tolist(),
            "x_label": "X",
            "y_label": "Y",
            "z_label": "Z"
        },
        "config": {
            "title": "Тестова 3D-поверхня (Інструмент)",
            "width": 800,
            "height": 600,
            "interactive": True
        }
    }

    tool_result = generate_surface_3d_tool(json.dumps(tool_input))
    tool_result_dict = json.loads(tool_result)

    print(f"Результат інструменту 3D-поверхні: {tool_result_dict['filepath']}")
    print(f"URL інструменту 3D-поверхні: {tool_result_dict['url']}")

def test_scatter_3d_visualization():
    """Тестування 3D-точкової візуалізації."""
    print("\nТестування 3D-точкової візуалізації...")

    # Створюємо генератор візуалізації
    generator = Visualization3DGenerator()

    # Створюємо дані для точкової діаграми
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
        z_label="Z"
    )

    # Створюємо конфігурацію точкової діаграми
    scatter_config = Visualization3DConfig(
        title="Тестова 3D-точкова діаграма",
        width=800,
        height=600,
        interactive=True
    )

    # Генеруємо точкову діаграму
    result = generator.generate_scatter_3d(scatter_data, scatter_config)

    print(f"3D-точкова діаграма згенерована: {result['filepath']}")
    print(f"URL 3D-точкової діаграми: {result['url']}")

    # Тестуємо функцію інструменту
    tool_input = {
        "data": {
            "x_data": x.tolist(),
            "y_data": y.tolist(),
            "z_data": z.tolist(),
            "color_data": colors.tolist(),
            "size_data": sizes.tolist(),
            "x_label": "X",
            "y_label": "Y",
            "z_label": "Z"
        },
        "config": {
            "title": "Тестова 3D-точкова діаграма (Інструмент)",
            "width": 800,
            "height": 600,
            "interactive": True
        }
    }

    tool_result = generate_scatter_3d_tool(json.dumps(tool_input))
    tool_result_dict = json.loads(tool_result)

    print(f"Результат інструменту 3D-точкової діаграми: {tool_result_dict['filepath']}")
    print(f"URL інструменту 3D-точкової діаграми: {tool_result_dict['url']}")

def test_volume_3d_visualization():
    """Тестування 3D-об'ємної візуалізації."""
    print("\nТестування 3D-об'ємної візуалізації...")

    # Створюємо генератор візуалізації
    generator = Visualization3DGenerator()

    # Створюємо дані для об'ємної візуалізації
    n = 20
    x = np.linspace(-5, 5, n)
    y = np.linspace(-5, 5, n)
    z = np.linspace(-5, 5, n)

    X, Y, Z = np.meshgrid(x, y, z)
    volume_data = np.exp(-(X**2 + Y**2 + Z**2) / 10)

    # Створюємо дані для об'ємної візуалізації
    volume_data_obj = Volume3DData(
        volume_data=volume_data.tolist(),
        x_range=x.tolist(),
        y_range=y.tolist(),
        z_range=z.tolist(),
        x_label="X",
        y_label="Y",
        z_label="Z"
    )

    # Створюємо конфігурацію об'ємної візуалізації
    volume_config = Visualization3DConfig(
        title="Тестова 3D-об'ємна візуалізація",
        width=800,
        height=600,
        interactive=True
    )

    # Генеруємо об'ємну візуалізацію
    try:
        result = generator.generate_volume_3d(volume_data_obj, volume_config)

        print(f"3D-об'ємна візуалізація згенерована: {result['filepath']}")
        print(f"URL 3D-об'ємної візуалізації: {result['url']}")

        # Тестуємо функцію інструменту
        tool_input = {
            "data": {
                "volume_data": volume_data.tolist(),
                "x_range": x.tolist(),
                "y_range": y.tolist(),
                "z_range": z.tolist(),
                "x_label": "X",
                "y_label": "Y",
                "z_label": "Z"
            },
            "config": {
                "title": "Тестова 3D-об'ємна візуалізація (Інструмент)",
                "width": 800,
                "height": 600,
                "interactive": True
            }
        }

        tool_result = generate_volume_3d_tool(json.dumps(tool_input))
        tool_result_dict = json.loads(tool_result)

        print(f"Результат інструменту 3D-об'ємної візуалізації: {tool_result_dict['filepath']}")
        print(f"URL інструменту 3D-об'ємної візуалізації: {tool_result_dict['url']}")
    except ImportError:
        print("Plotly не доступний. Пропускаємо тест 3D-об'ємної візуалізації.")

def main():
    """Запуск тестів."""
    # Створюємо тимчасову директорію для тестових візуалізацій
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Використовуємо тимчасову директорію: {temp_dir}")

        # Тестуємо 3D-поверхневу візуалізацію
        test_surface_3d_visualization()

        # Тестуємо 3D-точкову візуалізацію
        test_scatter_3d_visualization()

        # Тестуємо 3D-об'ємну візуалізацію
        test_volume_3d_visualization()

        print("\nВсі тести завершено.")

if __name__ == "__main__":
    main()
