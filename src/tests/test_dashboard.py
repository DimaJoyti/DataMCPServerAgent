"""
Тестовий скрипт для дашбордів.
Цей скрипт тестує основну функціональність дашбордів.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Додаємо батьківську директорію до шляху Python
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np

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
    print("Увага: Dash або Plotly не доступні. Пропускаємо тести дашбордів.")

def test_grid_dashboard():
    """Тестування дашборду з сітковим макетом."""
    if not DASHBOARD_AVAILABLE:
        print("Dash або Plotly не доступні. Пропускаємо тест дашборду з сітковим макетом.")
        return

    print("Тестування дашборду з сітковим макетом...")

    # Створюємо генератор дашбордів
    generator = DashboardGenerator()

    # Створюємо дашборд
    dashboard = Dashboard(
        id="test-grid-dashboard",
        title="Тестовий дашборд з сітковим макетом",
        config=DashboardConfig(
            title="Тестовий дашборд з сітковим макетом",
            subtitle="Демонстрація сіткового макету",
            layout="grid"
        ),
        panels=[
            DashboardPanel(
                id="chart-panel",
                title="Графік",
                type="chart",
                data={
                    "chart_type": "bar",
                    "x_data": ["A", "B", "C", "D", "E"],
                    "y_data": [10, 20, 15, 25, 30]
                },
                config={
                    "title": "Приклад графіка",
                    "x_label": "Категорії",
                    "y_label": "Значення"
                },
                width=6,
                height=4,
                x=0,
                y=0
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
                        ["E", 30, "Опис E"]
                    ]
                },
                width=6,
                height=4,
                x=6,
                y=0
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
                y=4
            )
        ]
    )

    # Генеруємо дашборд
    try:
        result = generator.generate_dashboard(dashboard)
        print(f"Дашборд з сітковим макетом згенеровано: {result['url']}")
    except Exception as e:
        print(f"Помилка при генерації дашборду з сітковим макетом: {str(e)}")

def test_tabs_dashboard():
    """Тестування дашборду з вкладками."""
    if not DASHBOARD_AVAILABLE:
        print("Dash або Plotly не доступні. Пропускаємо тест дашборду з вкладками.")
        return

    print("\nТестування дашборду з вкладками...")

    # Створюємо генератор дашбордів
    generator = DashboardGenerator()

    # Створюємо дашборд
    dashboard = Dashboard(
        id="test-tabs-dashboard",
        title="Тестовий дашборд з вкладками",
        config=DashboardConfig(
            title="Тестовий дашборд з вкладками",
            subtitle="Демонстрація макету з вкладками",
            layout="tabs"
        ),
        panels=[
            DashboardPanel(
                id="chart-panel",
                title="Графік",
                type="chart",
                data={
                    "chart_type": "line",
                    "x_data": [1, 2, 3, 4, 5],
                    "y_data": [10, 20, 15, 25, 30]
                },
                config={
                    "title": "Приклад лінійного графіка",
                    "x_label": "X",
                    "y_label": "Y"
                },
                tab="Графіки"
            ),
            DashboardPanel(
                id="pie-chart-panel",
                title="Кругова діаграма",
                type="chart",
                data={
                    "chart_type": "pie",
                    "x_data": ["A", "B", "C", "D", "E"],
                    "y_data": [10, 20, 15, 25, 30]
                },
                config={
                    "title": "Приклад кругової діаграми"
                },
                tab="Графіки"
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
                        ["E", 30, "Опис E"]
                    ]
                },
                tab="Дані"
            ),
            DashboardPanel(
                id="text-panel",
                title="Текст",
                type="text",
                data={
                    "text": "Це приклад текстової панелі. Тут можна розмістити будь-який текст, включаючи результати дослідження, висновки, тощо."
                },
                tab="Інформація"
            )
        ]
    )

    # Генеруємо дашборд
    try:
        result = generator.generate_dashboard(dashboard)
        print(f"Дашборд з вкладками згенеровано: {result['url']}")
    except Exception as e:
        print(f"Помилка при генерації дашборду з вкладками: {str(e)}")

def test_dashboard_tool():
    """Тестування інструменту для генерації дашбордів."""
    if not DASHBOARD_AVAILABLE:
        print("Dash або Plotly не доступні. Пропускаємо тест інструменту для генерації дашбордів.")
        return

    print("\nТестування інструменту для генерації дашбордів...")

    # Створюємо дані для інструменту
    tool_input = {
        "id": "tool-dashboard",
        "title": "Дашборд, створений інструментом",
        "config": {
            "title": "Дашборд, створений інструментом",
            "subtitle": "Демонстрація інструменту для генерації дашбордів",
            "layout": "grid"
        },
        "panels": [
            {
                "id": "chart-panel",
                "title": "Графік",
                "type": "chart",
                "data": {
                    "chart_type": "bar",
                    "x_data": ["A", "B", "C", "D", "E"],
                    "y_data": [10, 20, 15, 25, 30]
                },
                "config": {
                    "title": "Приклад графіка",
                    "x_label": "Категорії",
                    "y_label": "Значення"
                },
                "width": 12,
                "height": 6,
                "x": 0,
                "y": 0
            }
        ]
    }

    # Викликаємо інструмент
    try:
        result = generate_dashboard_tool(json.dumps(tool_input))
        result_dict = json.loads(result)

        print(f"Результат інструменту для генерації дашбордів: {result_dict}")
    except Exception as e:
        print(f"Помилка при виклику інструменту для генерації дашбордів: {str(e)}")

def main():
    """Запуск тестів."""
    if not DASHBOARD_AVAILABLE:
        print("Dash або Plotly не доступні. Пропускаємо всі тести дашбордів.")
        return

    # Тестуємо дашборд з сітковим макетом
    test_grid_dashboard()

    # Тестуємо дашборд з вкладками
    test_tabs_dashboard()

    # Тестуємо інструмент для генерації дашбордів
    test_dashboard_tool()

    print("\nВсі тести завершено.")

if __name__ == "__main__":
    main()
