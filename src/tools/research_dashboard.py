"""
Інтерактивні дашборди для Дослідницького Асистента.
Цей модуль надає можливості для створення інтерактивних дашбордів
для візуалізації та аналізу дослідницьких даних.
"""

import json
import os
import tempfile
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

# Спробуємо імпортувати бібліотеки для дашбордів
try:
    import dash
    from dash import dcc, html
    from dash.dependencies import Input, Output

    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    print("Увага: Dash не доступний. Встановіть його за допомогою 'pip install dash'")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print(
        "Увага: Plotly не доступний. Встановіть його за допомогою 'pip install plotly'"
    )

class DashboardConfig(BaseModel):
    """Конфігурація для дашбордів."""

    title: str
    subtitle: Optional[str] = None
    width: str = "100%"
    height: str = "800px"
    theme: str = "default"
    background_color: str = "#ffffff"
    font_family: str = "Arial"
    layout: str = "grid"  # grid, tabs, vertical, horizontal
    refresh_interval: Optional[int] = None  # в секундах

    class Config:
        """Pydantic конфігурація."""

        arbitrary_types_allowed = True

class DashboardPanel(BaseModel):
    """Панель для дашборду."""

    id: str
    title: str
    type: str  # chart, table, map, text, html, iframe
    data: Dict[str, Any]
    config: Dict[str, Any] = {}
    width: int = 1  # відносна ширина (1-12 для grid layout)
    height: int = 1  # відносна висота (1-12 для grid layout)
    x: int = 0  # позиція x для grid layout
    y: int = 0  # позиція y для grid layout
    tab: Optional[str] = None  # для tab layout

    class Config:
        """Pydantic конфігурація."""

        arbitrary_types_allowed = True

class Dashboard(BaseModel):
    """Дашборд для візуалізації дослідницьких даних."""

    id: str
    title: str
    config: DashboardConfig
    panels: List[DashboardPanel]

    class Config:
        """Pydantic конфігурація."""

        arbitrary_types_allowed = True

class DashboardGenerator:
    """Генератор для дашбордів."""

    def __init__(self, output_dir: Optional[str] = None):
        """Ініціалізація генератора дашбордів.

        Args:
            output_dir: Директорія для збереження дашбордів
        """
        self.output_dir = output_dir or tempfile.mkdtemp()
        os.makedirs(self.output_dir, exist_ok=True)

        if not DASH_AVAILABLE or not PLOTLY_AVAILABLE:
            print("Увага: Dash або Plotly не доступні. Дашборди будуть обмежені.")

    def generate_dashboard(self, dashboard: Dashboard) -> Dict[str, Any]:
        """Генерація дашборду.

        Args:
            dashboard: Дашборд для генерації

        Returns:
            Метадані дашборду
        """
        if not DASH_AVAILABLE:
            raise ImportError("Dash необхідний для генерації дашбордів")

        # Створюємо додаток Dash
        app = dash.Dash(__name__, suppress_callback_exceptions=True)

        # Встановлюємо заголовок
        app.title = dashboard.title

        # Створюємо макет на основі конфігурації
        if dashboard.config.layout == "tabs":
            app.layout = self._create_tabs_layout(dashboard)
        elif dashboard.config.layout == "vertical":
            app.layout = self._create_vertical_layout(dashboard)
        elif dashboard.config.layout == "horizontal":
            app.layout = self._create_horizontal_layout(dashboard)
        else:  # grid (за замовчуванням)
            app.layout = self._create_grid_layout(dashboard)

        # Додаємо колбеки для інтерактивності
        self._add_callbacks(app, dashboard)

        # Зберігаємо дашборд
        filename = f"{dashboard.id.lower().replace(' ', '_')}_dashboard.html"
        filepath = os.path.join(self.output_dir, filename)

        # Запускаємо сервер і зберігаємо HTML
        app.run_server(debug=False, port=8050, mode="inline")

        return {
            "type": "dashboard",
            "id": dashboard.id,
            "title": dashboard.title,
            "filepath": filepath,
            "url": "http://localhost:8050",
        }

    def _create_grid_layout(self, dashboard: Dashboard) -> html.Div:
        """Створення макету сітки для дашборду.

        Args:
            dashboard: Дашборд для генерації

        Returns:
            Макет дашборду
        """
        # Створюємо заголовок
        header = html.Div(
            [
                html.H1(dashboard.title, style={"textAlign": "center"}),
                html.H3(dashboard.config.subtitle, style={"textAlign": "center"})
                if dashboard.config.subtitle
                else None,
            ]
        )

        # Створюємо сітку
        grid = html.Div(
            [
                html.Div(
                    [self._create_panel(panel) for panel in dashboard.panels],
                    className="grid-container",
                )
            ]
        )

        # Створюємо стилі для сітки
        grid_style = {
            "display": "grid",
            "gridTemplateColumns": "repeat(12, 1fr)",
            "gridGap": "10px",
            "padding": "10px",
        }

        # Додаємо стилі для кожної панелі
        panel_styles = {}
        for panel in dashboard.panels:
            panel_styles[f"#{panel.id}"] = {
                "gridColumn": f"span {panel.width}",
                "gridRow": f"span {panel.height}",
            }

        # Створюємо стилі
        styles = html.Style(f"""
            .grid-container {{
                display: grid;
                grid-template-columns: repeat(12, 1fr);
                grid-gap: 10px;
                padding: 10px;
            }}

            {" ".join([f"#{panel.id} {{ grid-column: span {panel.width}; grid-row: span {panel.height}; }}" for panel in dashboard.panels])}
        """)

        # Створюємо макет
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
        """Створення макету з вкладками для дашборду.

        Args:
            dashboard: Дашборд для генерації

        Returns:
            Макет дашборду
        """
        # Створюємо заголовок
        header = html.Div(
            [
                html.H1(dashboard.title, style={"textAlign": "center"}),
                html.H3(dashboard.config.subtitle, style={"textAlign": "center"})
                if dashboard.config.subtitle
                else None,
            ]
        )

        # Отримуємо унікальні вкладки
        tabs = list(set([panel.tab for panel in dashboard.panels if panel.tab]))

        # Створюємо вкладки
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

        # Додаємо панелі без вкладок
        no_tab_panels = [panel for panel in dashboard.panels if not panel.tab]
        if no_tab_panels:
            tab_content.append(
                dcc.Tab(
                    label="Загальне",
                    children=html.Div(
                        [self._create_panel(panel) for panel in no_tab_panels],
                        style={"padding": "20px"},
                    ),
                )
            )

        # Створюємо компонент вкладок
        tabs_component = dcc.Tabs(
            id="tabs", children=tab_content, style={"marginTop": "20px"}
        )

        # Створюємо макет
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
        """Створення вертикального макету для дашборду.

        Args:
            dashboard: Дашборд для генерації

        Returns:
            Макет дашборду
        """
        # Створюємо заголовок
        header = html.Div(
            [
                html.H1(dashboard.title, style={"textAlign": "center"}),
                html.H3(dashboard.config.subtitle, style={"textAlign": "center"})
                if dashboard.config.subtitle
                else None,
            ]
        )

        # Створюємо панелі
        panels = html.Div(
            [self._create_panel(panel) for panel in dashboard.panels],
            style={"display": "flex", "flexDirection": "column", "gap": "20px"},
        )

        # Створюємо макет
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
        """Створення горизонтального макету для дашборду.

        Args:
            dashboard: Дашборд для генерації

        Returns:
            Макет дашборду
        """
        # Створюємо заголовок
        header = html.Div(
            [
                html.H1(dashboard.title, style={"textAlign": "center"}),
                html.H3(dashboard.config.subtitle, style={"textAlign": "center"})
                if dashboard.config.subtitle
                else None,
            ]
        )

        # Створюємо панелі
        panels = html.Div(
            [self._create_panel(panel) for panel in dashboard.panels],
            style={
                "display": "flex",
                "flexDirection": "row",
                "flexWrap": "wrap",
                "gap": "20px",
            },
        )

        # Створюємо макет
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
        """Створення панелі для дашборду.

        Args:
            panel: Панель для створення

        Returns:
            Компонент панелі
        """
        # Створюємо вміст панелі на основі типу
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
            content = html.Div(f"Непідтримуваний тип панелі: {panel.type}")

        # Створюємо панель
        panel_div = html.Div(
            [
                html.H3(
                    panel.title, style={"textAlign": "center", "marginBottom": "10px"}
                ),
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
        """Створення панелі з графіком.

        Args:
            panel: Панель для створення

        Returns:
            Компонент панелі
        """
        if not PLOTLY_AVAILABLE:
            return html.Div("Plotly не доступний. Неможливо створити графік.")

        # Отримуємо дані та конфігурацію
        chart_type = panel.data.get("chart_type", "bar")
        x_data = panel.data.get("x_data", [])
        y_data = panel.data.get("y_data", [])

        # Створюємо графік на основі типу
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
            return html.Div(f"Непідтримуваний тип графіка: {chart_type}")

        # Оновлюємо макет
        fig.update_layout(
            title=panel.config.get("title"),
            xaxis_title=panel.config.get("x_label"),
            yaxis_title=panel.config.get("y_label"),
            margin=dict(l=40, r=40, t=40, b=40),
        )

        # Створюємо компонент графіка
        graph = dcc.Graph(
            id=f"{panel.id}-graph",
            figure=fig,
            style={"height": "100%", "width": "100%"},
        )

        return html.Div(graph, style={"height": "100%", "width": "100%"})

    def _create_table_panel(self, panel: DashboardPanel) -> html.Div:
        """Створення панелі з таблицею.

        Args:
            panel: Панель для створення

        Returns:
            Компонент панелі
        """
        # Отримуємо дані та конфігурацію
        columns = panel.data.get("columns", [])
        data = panel.data.get("data", [])

        # Створюємо заголовки таблиці
        header = html.Tr([html.Th(col) for col in columns])

        # Створюємо рядки таблиці
        rows = []
        for row in data:
            rows.append(html.Tr([html.Td(cell) for cell in row]))

        # Створюємо таблицю
        table = html.Table(
            [html.Thead(header), html.Tbody(rows)],
            style={"width": "100%", "borderCollapse": "collapse"},
        )

        # Створюємо стилі для таблиці
        styles = html.Style("""
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
        """)

        return html.Div([styles, table], style={"overflowX": "auto"})

    def _create_map_panel(self, panel: DashboardPanel) -> html.Div:
        """Створення панелі з картою.

        Args:
            panel: Панель для створення

        Returns:
            Компонент панелі
        """
        if not PLOTLY_AVAILABLE:
            return html.Div("Plotly не доступний. Неможливо створити карту.")

        # Отримуємо дані та конфігурацію
        locations = panel.data.get("locations", [])
        location_mode = panel.data.get("location_mode", "ISO-3")
        color_field = panel.data.get("color_field")

        # Створюємо карту
        if location_mode == "ISO-3":
            # Створюємо хороплет
            fig = px.choropleth(
                locations=locations,
                locationmode="ISO-3",
                color=color_field,
                color_continuous_scale="Viridis",
                title=panel.config.get("title"),
            )
        else:
            # Створюємо точкову карту
            fig = px.scatter_geo(
                lat=[loc.get("lat") for loc in locations],
                lon=[loc.get("lon") for loc in locations],
                text=[loc.get("name") for loc in locations],
                title=panel.config.get("title"),
            )

        # Оновлюємо макет
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

        # Створюємо компонент карти
        graph = dcc.Graph(
            id=f"{panel.id}-map", figure=fig, style={"height": "100%", "width": "100%"}
        )

        return html.Div(graph, style={"height": "100%", "width": "100%"})

    def _create_text_panel(self, panel: DashboardPanel) -> html.Div:
        """Створення панелі з текстом.

        Args:
            panel: Панель для створення

        Returns:
            Компонент панелі
        """
        # Отримуємо дані та конфігурацію
        text = panel.data.get("text", "")

        # Створюємо текстовий компонент
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
        """Створення панелі з HTML.

        Args:
            panel: Панель для створення

        Returns:
            Компонент панелі
        """
        # Отримуємо дані та конфігурацію
        html_content = panel.data.get("html", "")

        # Створюємо HTML компонент
        return html.Iframe(
            srcDoc=html_content,
            style={"width": "100%", "height": "100%", "border": "none"},
        )

    def _create_iframe_panel(self, panel: DashboardPanel) -> html.Div:
        """Створення панелі з iframe.

        Args:
            panel: Панель для створення

        Returns:
            Компонент панелі
        """
        # Отримуємо дані та конфігурацію
        url = panel.data.get("url", "")

        # Створюємо iframe компонент
        return html.Iframe(
            src=url, style={"width": "100%", "height": "100%", "border": "none"}
        )

    def _add_callbacks(self, app: dash.Dash, dashboard: Dashboard) -> None:
        """Додавання колбеків для інтерактивності.

        Args:
            app: Додаток Dash
            dashboard: Дашборд для генерації
        """
        # Додаємо колбеки для оновлення даних
        if dashboard.config.refresh_interval:

            @app.callback(
                Output("dashboard-container", "children"),
                Input("refresh-interval", "n_intervals"),
            )
            def update_dashboard(n):
                # Тут можна додати логіку для оновлення даних
                return self._create_dashboard_content(dashboard)

def generate_dashboard_tool(data_str: str) -> str:
    """Генерація дашборду.

    Args:
        data_str: JSON-рядок з даними дашборду та конфігурацією

    Returns:
        JSON-рядок з метаданими дашборду
    """
    try:
        data = json.loads(data_str)

        # Створюємо генератор дашбордів
        generator = DashboardGenerator()

        # Створюємо дашборд
        dashboard = Dashboard(
            id=data.get("id", "dashboard"),
            title=data.get("title", "Дашборд"),
            config=DashboardConfig(**data.get("config", {})),
            panels=[DashboardPanel(**panel) for panel in data.get("panels", [])],
        )

        # Генеруємо дашборд
        result = generator.generate_dashboard(dashboard)

        return json.dumps(result)
    except Exception as e:
        return json.dumps({"error": str(e)})

if __name__ == "__main__":
    # Приклад використання
    generator = DashboardGenerator()

    # Створюємо дашборд
    dashboard = Dashboard(
        id="example-dashboard",
        title="Приклад дашборду",
        config=DashboardConfig(
            title="Приклад дашборду",
            subtitle="Демонстрація можливостей дашбордів",
            layout="grid",
        ),
        panels=[
            DashboardPanel(
                id="chart-panel",
                title="Графік",
                type="chart",
                data={
                    "chart_type": "bar",
                    "x_data": ["A", "B", "C", "D", "E"],
                    "y_data": [10, 20, 15, 25, 30],
                },
                config={
                    "title": "Приклад графіка",
                    "x_label": "Категорії",
                    "y_label": "Значення",
                },
                width=6,
                height=4,
                x=0,
                y=0,
            ),
            DashboardPanel(
                id="table-panel",
                title="Таблиця",
                type="table",
                data={
                    "columns": ["Назва", "Значення", "Опис"],
                    "data": [
                        ["A", 10, "Опис A"],
                        ["B", 20, "Опис B"],
                        ["C", 15, "Опис C"],
                        ["D", 25, "Опис D"],
                        ["E", 30, "Опис E"],
                    ],
                },
                width=6,
                height=4,
                x=6,
                y=0,
            ),
            DashboardPanel(
                id="text-panel",
                title="Текст",
                type="text",
                data={
                    "text": "Це приклад текстової панелі. Тут можна розмістити будь-який текст, включаючи результати дослідження, висновки, тощо."
                },
                width=12,
                height=2,
                x=0,
                y=4,
            ),
        ],
    )

    # Генеруємо дашборд
    result = generator.generate_dashboard(dashboard)
    print(f"Дашборд згенеровано: {result['url']}")
