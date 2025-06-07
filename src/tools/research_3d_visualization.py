"""
3D візуалізації для Дослідницького Асистента.
Цей модуль надає можливості для створення інтерактивних 3D-візуалізацій
дослідницьких даних, включаючи 3D-графіки, поверхні та об'ємні дані.
"""

import os
import tempfile
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel

# Спробуємо імпортувати бібліотеки для 3D-візуалізації
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print(
        "Увага: Plotly не доступний. Встановіть його за допомогою 'pip install plotly'"
    )

try:
    import matplotlib

    matplotlib.use("Agg")  # Використовуємо неінтерактивний бекенд
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    MATPLOTLIB_3D_AVAILABLE = True
except ImportError:
    MATPLOTLIB_3D_AVAILABLE = False
    print("Увага: Matplotlib 3D не доступний.")

class Visualization3DConfig(BaseModel):
    """Конфігурація для 3D-візуалізацій."""

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
    camera_position: Optional[Dict[str, float]] = None

    class Config:
        """Pydantic конфігурація."""

        arbitrary_types_allowed = True

class Surface3DData(BaseModel):
    """Дані для 3D-поверхневих візуалізацій."""

    x_data: List[List[float]]
    y_data: List[List[float]]
    z_data: List[List[float]]
    color_data: Optional[List[List[float]]] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    z_label: Optional[str] = None

    class Config:
        """Pydantic конфігурація."""

        arbitrary_types_allowed = True

class Scatter3DData(BaseModel):
    """Дані для 3D-точкових візуалізацій."""

    x_data: List[float]
    y_data: List[float]
    z_data: List[float]
    color_data: Optional[List[float]] = None
    size_data: Optional[List[float]] = None
    text_data: Optional[List[str]] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    z_label: Optional[str] = None

    class Config:
        """Pydantic конфігурація."""

        arbitrary_types_allowed = True

class Volume3DData(BaseModel):
    """Дані для 3D-об'ємних візуалізацій."""

    volume_data: List[List[List[float]]]
    x_range: Optional[List[float]] = None
    y_range: Optional[List[float]] = None
    z_range: Optional[List[float]] = None
    opacity_scale: Optional[List[float]] = None
    color_scale: Optional[List[List[Union[float, str]]]] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    z_label: Optional[str] = None

    class Config:
        """Pydantic конфігурація."""

        arbitrary_types_allowed = True

class Visualization3DGenerator:
    """Генератор для 3D-візуалізацій."""

    def __init__(self, output_dir: Optional[str] = None):
        """Ініціалізація генератора 3D-візуалізацій.

        Args:
            output_dir: Директорія для збереження візуалізацій
        """
        self.output_dir = output_dir or tempfile.mkdtemp()
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_surface_3d(
        self, data: Surface3DData, config: Visualization3DConfig
    ) -> Dict[str, Any]:
        """Генерація 3D-поверхневої візуалізації.

        Args:
            data: Дані для 3D-поверхні
            config: Конфігурація візуалізації

        Returns:
            Метадані візуалізації
        """
        if not PLOTLY_AVAILABLE and config.interactive:
            print("Увага: Plotly не доступний. Використовуємо Matplotlib.")
            config.interactive = False

        if config.interactive:
            return self._generate_interactive_surface_3d(data, config)
        else:
            return self._generate_static_surface_3d(data, config)

    def _generate_interactive_surface_3d(
        self, data: Surface3DData, config: Visualization3DConfig
    ) -> Dict[str, Any]:
        """Генерація інтерактивної 3D-поверхневої візуалізації за допомогою Plotly.

        Args:
            data: Дані для 3D-поверхні
            config: Конфігурація візуалізації

        Returns:
            Метадані візуалізації
        """
        # Створюємо фігуру
        fig = go.Figure()

        # Встановлюємо тему
        template = "plotly" if config.theme == "default" else config.theme

        # Додаємо поверхню
        fig.add_trace(
            go.Surface(
                z=data.z_data,
                x=data.x_data,
                y=data.y_data,
                colorscale="Viridis",
                colorbar=dict(title=data.z_label) if data.z_label else None,
                surfacecolor=data.color_data if data.color_data else None,
            )
        )

        # Оновлюємо макет
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25),
        )

        if config.camera_position:
            camera["eye"] = config.camera_position

        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            template=template,
            scene=dict(
                xaxis=dict(title=data.x_label) if data.x_label else None,
                yaxis=dict(title=data.y_label) if data.y_label else None,
                zaxis=dict(title=data.z_label) if data.z_label else None,
                camera=camera,
            ),
            font=dict(family=config.font_family),
            paper_bgcolor=config.background_color,
        )

        # Зберігаємо фігуру
        filename = f"{config.title.lower().replace(' ', '_')}_surface_3d.html"
        filepath = os.path.join(self.output_dir, filename)
        fig.write_html(filepath)

        return {
            "type": "interactive_surface_3d",
            "title": config.title,
            "filepath": filepath,
            "url": f"file://{filepath}",
        }

    def _generate_static_surface_3d(
        self, data: Surface3DData, config: Visualization3DConfig
    ) -> Dict[str, Any]:
        """Генерація статичної 3D-поверхневої візуалізації за допомогою Matplotlib.

        Args:
            data: Дані для 3D-поверхні
            config: Конфігурація візуалізації

        Returns:
            Метадані візуалізації
        """
        if not MATPLOTLIB_3D_AVAILABLE:
            raise ImportError(
                "Matplotlib 3D необхідний для статичних 3D-поверхневих візуалізацій"
            )

        # Створюємо фігуру
        fig = plt.figure(figsize=(config.width / 100, config.height / 100), dpi=100)
        ax = fig.add_subplot(111, projection="3d")

        # Додаємо поверхню
        x_data = np.array(data.x_data)
        y_data = np.array(data.y_data)
        z_data = np.array(data.z_data)

        if data.color_data:
            color_data = np.array(data.color_data)
            surf = ax.plot_surface(
                x_data,
                y_data,
                z_data,
                facecolors=plt.cm.viridis(color_data),
                alpha=0.8,
                cmap="viridis",
            )
        else:
            surf = ax.plot_surface(x_data, y_data, z_data, cmap="viridis", alpha=0.8)

        # Додаємо колірну шкалу
        if data.color_data:
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        # Встановлюємо мітки та заголовок
        ax.set_title(config.title)
        if data.x_label:
            ax.set_xlabel(data.x_label)
        if data.y_label:
            ax.set_ylabel(data.y_label)
        if data.z_label:
            ax.set_zlabel(data.z_label)

        # Зберігаємо фігуру
        filename = f"{config.title.lower().replace(' ', '_')}_surface_3d.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(
            filepath, dpi=100, bbox_inches="tight", facecolor=config.background_color
        )
        plt.close(fig)

        return {
            "type": "static_surface_3d",
            "title": config.title,
            "filepath": filepath,
            "url": f"file://{filepath}",
        }

    def generate_scatter_3d(
        self, data: Scatter3DData, config: Visualization3DConfig
    ) -> Dict[str, Any]:
        """Генерація 3D-точкової візуалізації.

        Args:
            data: Дані для 3D-точкової візуалізації
            config: Конфігурація візуалізації

        Returns:
            Метадані візуалізації
        """
        if not PLOTLY_AVAILABLE and config.interactive:
            print("Увага: Plotly не доступний. Використовуємо Matplotlib.")
            config.interactive = False

        if config.interactive:
            return self._generate_interactive_scatter_3d(data, config)
        else:
            return self._generate_static_scatter_3d(data, config)

    def _generate_interactive_scatter_3d(
        self, data: Scatter3DData, config: Visualization3DConfig
    ) -> Dict[str, Any]:
        """Генерація інтерактивної 3D-точкової візуалізації за допомогою Plotly.

        Args:
            data: Дані для 3D-точкової візуалізації
            config: Конфігурація візуалізації

        Returns:
            Метадані візуалізації
        """
        # Створюємо фігуру
        fig = go.Figure()

        # Встановлюємо тему
        template = "plotly" if config.theme == "default" else config.theme

        # Додаємо точки
        marker_dict = {}
        if data.color_data:
            marker_dict["color"] = data.color_data
            marker_dict["colorscale"] = "Viridis"
            marker_dict["colorbar"] = dict(title="Значення")

        if data.size_data:
            marker_dict["size"] = data.size_data
            marker_dict["sizeref"] = 0.1
            marker_dict["sizemode"] = "diameter"
        else:
            marker_dict["size"] = 5

        fig.add_trace(
            go.Scatter3d(
                x=data.x_data,
                y=data.y_data,
                z=data.z_data,
                mode="markers",
                marker=marker_dict,
                text=data.text_data,
                hoverinfo="text" if data.text_data else "x+y+z",
            )
        )

        # Оновлюємо макет
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25),
        )

        if config.camera_position:
            camera["eye"] = config.camera_position

        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            template=template,
            scene=dict(
                xaxis=dict(title=data.x_label) if data.x_label else None,
                yaxis=dict(title=data.y_label) if data.y_label else None,
                zaxis=dict(title=data.z_label) if data.z_label else None,
                camera=camera,
            ),
            font=dict(family=config.font_family),
            paper_bgcolor=config.background_color,
        )

        # Зберігаємо фігуру
        filename = f"{config.title.lower().replace(' ', '_')}_scatter_3d.html"
        filepath = os.path.join(self.output_dir, filename)
        fig.write_html(filepath)

        return {
            "type": "interactive_scatter_3d",
            "title": config.title,
            "filepath": filepath,
            "url": f"file://{filepath}",
        }

    def _generate_static_scatter_3d(
        self, data: Scatter3DData, config: Visualization3DConfig
    ) -> Dict[str, Any]:
        """Генерація статичної 3D-точкової візуалізації за допомогою Matplotlib.

        Args:
            data: Дані для 3D-точкової візуалізації
            config: Конфігурація візуалізації

        Returns:
            Метадані візуалізації
        """
        if not MATPLOTLIB_3D_AVAILABLE:
            raise ImportError(
                "Matplotlib 3D необхідний для статичних 3D-точкових візуалізацій"
            )

        # Створюємо фігуру
        fig = plt.figure(figsize=(config.width / 100, config.height / 100), dpi=100)
        ax = fig.add_subplot(111, projection="3d")

        # Додаємо точки
        scatter_kwargs = {}
        if data.color_data:
            scatter_kwargs["c"] = data.color_data
            scatter_kwargs["cmap"] = "viridis"

        if data.size_data:
            scatter_kwargs["s"] = np.array(data.size_data) * 10
        else:
            scatter_kwargs["s"] = 30

        scatter = ax.scatter(data.x_data, data.y_data, data.z_data, **scatter_kwargs)

        # Додаємо колірну шкалу
        if data.color_data:
            fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)

        # Встановлюємо мітки та заголовок
        ax.set_title(config.title)
        if data.x_label:
            ax.set_xlabel(data.x_label)
        if data.y_label:
            ax.set_ylabel(data.y_label)
        if data.z_label:
            ax.set_zlabel(data.z_label)

        # Зберігаємо фігуру
        filename = f"{config.title.lower().replace(' ', '_')}_scatter_3d.png"
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(
            filepath, dpi=100, bbox_inches="tight", facecolor=config.background_color
        )
        plt.close(fig)

        return {
            "type": "static_scatter_3d",
            "title": config.title,
            "filepath": filepath,
            "url": f"file://{filepath}",
        }

    def generate_volume_3d(
        self, data: Volume3DData, config: Visualization3DConfig
    ) -> Dict[str, Any]:
        """Генерація 3D-об'ємної візуалізації.

        Args:
            data: Дані для 3D-об'ємної візуалізації
            config: Конфігурація візуалізації

        Returns:
            Метадані візуалізації
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly необхідний для 3D-об'ємних візуалізацій")

        # Створюємо фігуру
        fig = go.Figure()

        # Встановлюємо тему
        template = "plotly" if config.theme == "default" else config.theme

        # Підготовка даних
        volume_data = np.array(data.volume_data)

        # Створюємо об'ємну візуалізацію
        fig.add_trace(
            go.Volume(
                x=data.x_range if data.x_range else np.arange(volume_data.shape[0]),
                y=data.y_range if data.y_range else np.arange(volume_data.shape[1]),
                z=data.z_range if data.z_range else np.arange(volume_data.shape[2]),
                value=volume_data.flatten(),
                opacity=0.1,
                opacityscale=data.opacity_scale if data.opacity_scale else "uniform",
                colorscale=data.color_scale if data.color_scale else "Viridis",
                surface_count=20,
                colorbar=dict(title="Значення"),
            )
        )

        # Оновлюємо макет
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25),
        )

        if config.camera_position:
            camera["eye"] = config.camera_position

        fig.update_layout(
            title=config.title,
            width=config.width,
            height=config.height,
            template=template,
            scene=dict(
                xaxis=dict(title=data.x_label) if data.x_label else None,
                yaxis=dict(title=data.y_label) if data.y_label else None,
                zaxis=dict(title=data.z_label) if data.z_label else None,
                camera=camera,
            ),
            font=dict(family=config.font_family),
            paper_bgcolor=config.background_color,
        )

        # Зберігаємо фігуру
        filename = f"{config.title.lower().replace(' ', '_')}_volume_3d.html"
        filepath = os.path.join(self.output_dir, filename)
        fig.write_html(filepath)

        return {
            "type": "volume_3d",
            "title": config.title,
            "filepath": filepath,
            "url": f"file://{filepath}",
        }

    def generate_visualization_3d(
        self,
        visualization_type: str,
        data: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Генерація 3D-візуалізації на основі типу.

        Args:
            visualization_type: Тип візуалізації для генерації
            data: Дані візуалізації
            config: Опціональна конфігурація візуалізації

        Returns:
            Метадані візуалізації
        """
        # Створюємо конфігурацію
        if config is None:
            config = {}

        visualization_config = Visualization3DConfig(
            title=config.get("title", "3D Візуалізація"),
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
            camera_position=config.get("camera_position"),
        )

        # Генеруємо візуалізацію на основі типу
        if visualization_type == "surface_3d":
            surface_data = Surface3DData(
                x_data=data.get("x_data", [[]]),
                y_data=data.get("y_data", [[]]),
                z_data=data.get("z_data", [[]]),
                color_data=data.get("color_data"),
                x_label=data.get("x_label"),
                y_label=data.get("y_label"),
                z_label=data.get("z_label"),
            )
            return self.generate_surface_3d(surface_data, visualization_config)

        elif visualization_type == "scatter_3d":
            scatter_data = Scatter3DData(
                x_data=data.get("x_data", []),
                y_data=data.get("y_data", []),
                z_data=data.get("z_data", []),
                color_data=data.get("color_data"),
                size_data=data.get("size_data"),
                text_data=data.get("text_data"),
                x_label=data.get("x_label"),
                y_label=data.get("y_label"),
                z_label=data.get("z_label"),
            )
            return self.generate_scatter_3d(scatter_data, visualization_config)

        elif visualization_type == "volume_3d":
            volume_data = Volume3DData(
                volume_data=data.get("volume_data", [[[]]]),
                x_range=data.get("x_range"),
                y_range=data.get("y_range"),
                z_range=data.get("z_range"),
                opacity_scale=data.get("opacity_scale"),
                color_scale=data.get("color_scale"),
                x_label=data.get("x_label"),
                y_label=data.get("y_label"),
                z_label=data.get("z_label"),
            )
            return self.generate_volume_3d(volume_data, visualization_config)

        else:
            raise ValueError(
                f"Непідтримуваний тип 3D-візуалізації: {visualization_type}"
            )

def generate_surface_3d_tool(data_str: str) -> str:
    """Генерація 3D-поверхневої візуалізації.

    Args:
        data_str: JSON-рядок з даними поверхні та конфігурацією

    Returns:
        JSON-рядок з метаданими візуалізації
    """
    try:
        data = json.loads(data_str)

        # Створюємо генератор візуалізації
        generator = Visualization3DGenerator()

        # Генеруємо поверхню
        result = generator.generate_visualization_3d(
            visualization_type="surface_3d",
            data=data.get("data", {}),
            config=data.get("config", {}),
        )

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})

def generate_scatter_3d_tool(data_str: str) -> str:
    """Генерація 3D-точкової візуалізації.

    Args:
        data_str: JSON-рядок з даними точок та конфігурацією

    Returns:
        JSON-рядок з метаданими візуалізації
    """
    try:
        data = json.loads(data_str)

        # Створюємо генератор візуалізації
        generator = Visualization3DGenerator()

        # Генеруємо точкову діаграму
        result = generator.generate_visualization_3d(
            visualization_type="scatter_3d",
            data=data.get("data", {}),
            config=data.get("config", {}),
        )

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})

def generate_volume_3d_tool(data_str: str) -> str:
    """Генерація 3D-об'ємної візуалізації.

    Args:
        data_str: JSON-рядок з об'ємними даними та конфігурацією

    Returns:
        JSON-рядок з метаданими візуалізації
    """
    try:
        data = json.loads(data_str)

        # Створюємо генератор візуалізації
        generator = Visualization3DGenerator()

        # Генеруємо об'ємну візуалізацію
        result = generator.generate_visualization_3d(
            visualization_type="volume_3d",
            data=data.get("data", {}),
            config=data.get("config", {}),
        )

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    # Приклад використання
    generator = Visualization3DGenerator()

    # Генеруємо 3D-поверхню
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

    surface_config = Visualization3DConfig(
        title="Приклад 3D-поверхні", width=800, height=600, interactive=True
    )

    surface_result = generator.generate_surface_3d(surface_data, surface_config)
    print(f"3D-поверхня згенерована: {surface_result['filepath']}")

    # Генеруємо 3D-точкову діаграму
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

    scatter_config = Visualization3DConfig(
        title="Приклад 3D-точкової діаграми", width=800, height=600, interactive=True
    )

    scatter_result = generator.generate_scatter_3d(scatter_data, scatter_config)
    print(f"3D-точкова діаграма згенерована: {scatter_result['filepath']}")
