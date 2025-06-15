#!/usr/bin/env python3
"""
Phase 1 Demo Script for DataMCPServerAgent

This script demonstrates all the completed features of Phase 1:
- Consolidated entry point
- Configuration system
- Logging system
- CLI interface
- Semantic agents infrastructure
- Code quality tools
"""

import subprocess
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

def run_command_demo(cmd: list, description: str, show_output: bool = True) -> bool:
    """Run a command and show the result."""
    console.print(f"\n🔧 {description}", style="bold blue")
    console.print(f"Command: {' '.join(cmd)}", style="dim")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            console.print("✅ SUCCESS", style="green")
            if show_output and result.stdout:
                # Show first few lines of output
                lines = result.stdout.strip().split('\n')[:10]
                for line in lines:
                    console.print(f"  {line}", style="dim")
                if len(result.stdout.strip().split('\n')) > 10:
                    console.print("  ... (output truncated)", style="dim")
        else:
            console.print("❌ FAILED", style="red")
            if result.stderr:
                console.print(f"Error: {result.stderr[:200]}...", style="red")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        console.print("⏰ TIMEOUT", style="yellow")
        return False
    except Exception as e:
        console.print(f"❌ ERROR: {e}", style="red")
        return False

def show_welcome():
    """Show welcome message."""
    welcome_text = """
🎉 PHASE 1 COMPLETED: Results Demonstration

This script demonstrates all Phase 1 achievements:
✅ Consolidated codebase
✅ Single entry point
✅ Improved code quality
✅ Configuration system
✅ Structured logging
✅ CLI interface
✅ Semantic agents (basic infrastructure)
    """

    panel = Panel(
        welcome_text.strip(),
        title="🚀 DataMCPServerAgent Phase 1 Demo",
        border_style="green",
        padding=(1, 2)
    )

    console.print(panel)

def demo_main_commands():
    """Demo main entry point commands."""
    console.print("\n" + "="*60, style="bold")
    console.print("📋 ДЕМОНСТРАЦІЯ ОСНОВНИХ КОМАНД", style="bold magenta")
    console.print("="*60, style="bold")

    commands = [
        (["python", "app/main_improved.py", "--help"], "Головна команда --help"),
        (["python", "app/main_improved.py", "status"], "Статус системи"),
        (["python", "app/main_improved.py", "cli", "--help"], "CLI інтерфейс"),
        (["python", "app/main_improved.py", "semantic-agents", "--help"], "Семантичні агенти"),
        (["python", "app/main_improved.py", "api", "--help"], "API сервер"),
    ]

    results = []
    for cmd, desc in commands:
        success = run_command_demo(cmd, desc, show_output=True)
        results.append((desc, success))
        time.sleep(1)

    return results

def demo_code_quality():
    """Demo code quality tools."""
    console.print("\n" + "="*60, style="bold")
    console.print("🔍 ДЕМОНСТРАЦІЯ ЯКОСТІ КОДУ", style="bold magenta")
    console.print("="*60, style="bold")

    commands = [
        (["python", "-m", "black", "--check", "app/core/config.py"], "Black форматування"),
        (["python", "-m", "isort", "--check-only", "app/core/config.py"], "isort сортування імпортів"),
        (["python", "-m", "ruff", "check", "app/core/config.py"], "Ruff лінтинг"),
        (["python", "-c", "import app.core.config; print('✅ Config module imports successfully')"], "Імпорт конфігурації"),
        (["python", "-c", "import app.core.logging; print('✅ Logging module imports successfully')"], "Імпорт логування"),
    ]

    results = []
    for cmd, desc in commands:
        success = run_command_demo(cmd, desc, show_output=False)
        results.append((desc, success))
        time.sleep(0.5)

    return results

def demo_project_structure():
    """Demo project structure."""
    console.print("\n" + "="*60, style="bold")
    console.print("🏗️ СТРУКТУРА ПРОЕКТУ", style="bold magenta")
    console.print("="*60, style="bold")

    # Show key files
    key_files = [
        "app/main_improved.py",
        "app/core/config.py",
        "app/core/logging.py",
        "pyproject.toml",
        ".pre-commit-config.yaml",
        "docs/PHASE_1_FINAL_SUMMARY.md"
    ]

    table = Table(title="Ключові Файли Фази 1")
    table.add_column("Файл", style="cyan")
    table.add_column("Статус", style="bold")
    table.add_column("Опис", style="dim")

    for file_path in key_files:
        if Path(file_path).exists():
            status = "✅ Існує"
            style = "green"
        else:
            status = "❌ Відсутній"
            style = "red"

        descriptions = {
            "app/main_improved.py": "Консолідована точка входу",
            "app/core/config.py": "Об'єднана конфігурація",
            "app/core/logging.py": "Система логування",
            "pyproject.toml": "Консолідовані залежності",
            ".pre-commit-config.yaml": "Автоматичні перевірки",
            "docs/PHASE_1_FINAL_SUMMARY.md": "Звіт про завершення"
        }

        table.add_row(
            file_path,
            status,
            descriptions.get(file_path, "")
        )

    console.print(table)

def show_summary(main_results, quality_results):
    """Show final summary."""
    console.print("\n" + "="*60, style="bold")
    console.print("📊 ПІДСУМОК ДЕМОНСТРАЦІЇ", style="bold magenta")
    console.print("="*60, style="bold")

    # Main commands summary
    table = Table(title="Результати Тестування")
    table.add_column("Категорія", style="cyan")
    table.add_column("Тест", style="white")
    table.add_column("Результат", style="bold")

    for desc, success in main_results:
        status = "✅ PASS" if success else "❌ FAIL"
        table.add_row("Основні команди", desc, status)

    for desc, success in quality_results:
        status = "✅ PASS" if success else "❌ FAIL"
        table.add_row("Якість коду", desc, status)

    console.print(table)

    # Calculate success rate
    total_tests = len(main_results) + len(quality_results)
    passed_tests = sum(1 for _, success in main_results + quality_results if success)
    success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

    # Final message
    if success_rate >= 80:
        style = "green"
        message = f"🎉 ФАЗА 1 УСПІШНО ЗАВЕРШЕНА! ({success_rate:.1f}% тестів пройдено)"
    elif success_rate >= 60:
        style = "yellow"
        message = f"⚠️ Фаза 1 частково завершена ({success_rate:.1f}% тестів пройдено)"
    else:
        style = "red"
        message = f"❌ Потрібні додаткові виправлення ({success_rate:.1f}% тестів пройдено)"

    console.print(f"\n{message}", style=f"bold {style}")

    # Next steps
    next_steps = """
🚀 НАСТУПНІ КРОКИ:

1. Переходимо до Фази 2: Розширення LLM-driven Pipelines
2. Розробка мультимодальних можливостей
3. Інтеграція з векторними сховищами
4. Покращення семантичних агентів
5. Cloudflare Workers інтеграція
    """

    panel = Panel(
        next_steps.strip(),
        title="🎯 Готовність до Фази 2",
        border_style="blue",
        padding=(1, 2)
    )

    console.print(panel)

def main():
    """Main demo function."""
    show_welcome()

    # Demo main commands
    main_results = demo_main_commands()

    # Demo code quality
    quality_results = demo_code_quality()

    # Demo project structure
    demo_project_structure()

    # Show summary
    show_summary(main_results, quality_results)

if __name__ == "__main__":
    main()
