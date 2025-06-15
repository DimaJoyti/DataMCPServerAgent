#!/usr/bin/env python3
"""
Quick Start Script for DataMCPServerAgent v2.0
Простий скрипт для швидкого запуску системи без складних залежностей.
"""

import asyncio
import sys
from pathlib import Path

# Add app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def print_banner():
    """Відображення банера додатку."""
    print("=" * 60)
    print("🤖 DataMCPServerAgent v2.0 - Quick Start")
    print("=" * 60)
    print("Advanced AI Agent System with MCP Integration")
    print("Enhanced architecture with Clean Code principles")
    print("=" * 60)

def check_dependencies():
    """Перевірка наявності основних залежностей."""
    print("🔍 Перевірка залежностей...")

    required_modules = [
        'pydantic',
        'fastapi',
        'uvicorn',
        'structlog',
        'typer',
        'rich'
    ]

    missing = []
    available = []

    for module in required_modules:
        try:
            __import__(module)
            available.append(module)
            print(f"  ✅ {module}")
        except ImportError:
            missing.append(module)
            print(f"  ❌ {module}")

    if missing:
        print(f"\n⚠️ Відсутні модулі: {', '.join(missing)}")
        print("📦 Встановіть їх командою:")
        print(f"pip install {' '.join(missing)}")
        return False

    print("✅ Всі залежності доступні!")
    return True

def test_basic_functionality():
    """Тестування базової функціональності."""
    print("\n🧪 Тестування базової функціональності...")

    try:
        # Test configuration
        print("  📋 Тестування конфігурації...")
        from app.core.config_improved import Settings
        settings = Settings()
        print(f"    ✅ Додаток: {settings.app_name} v{settings.app_version}")
        print(f"    ✅ Середовище: {settings.environment}")

        # Test logging
        print("  📝 Тестування логування...")
        from app.core.logging_improved import get_logger, setup_logging
        setup_logging(settings)
        logger = get_logger("quick_start")
        logger.info("Тестове повідомлення логування")
        print("    ✅ Логування працює")

        # Test exceptions
        print("  ⚠️ Тестування винятків...")
        from app.core.exceptions_improved import ValidationError
        try:
            raise ValidationError("Тестова помилка валідації", field="test_field")
        except ValidationError as e:
            print(f"    ✅ Винятки працюють: {e.error_code}")

        return True

    except Exception as e:
        print(f"    ❌ Помилка: {e}")
        return False

def test_domain_models():
    """Тестування доменних моделей."""
    print("\n🏗️ Тестування доменних моделей...")

    try:
        # Test Agent model
        print("  🤖 Тестування моделі Agent...")
        from app.domain.models.agent import Agent, AgentConfiguration, AgentType

        config = AgentConfiguration(
            max_concurrent_tasks=5,
            timeout_seconds=300
        )

        agent = Agent(
            name="test-agent",
            agent_type=AgentType.WORKER,
            description="Тестовий агент",
            configuration=config
        )

        print(f"    ✅ Agent створено: {agent.name} (ID: {agent.id[:8]})")

        # Test Task model
        print("  📋 Тестування моделі Task...")
        from app.domain.models.task import Task, TaskPriority, TaskType

        task = Task(
            name="Тестове завдання",
            task_type=TaskType.DATA_ANALYSIS,
            agent_id=agent.id,
            priority=TaskPriority.NORMAL,
            description="Тестове завдання для перевірки"
        )

        print(f"    ✅ Task створено: {task.name} (ID: {task.id[:8]})")

        return True

    except Exception as e:
        print(f"    ❌ Помилка в доменних моделях: {e}")
        return False

async def test_api_server():
    """Тестування API сервера."""
    print("\n🌐 Тестування API сервера...")

    try:
        from app.api.server_improved import create_api_server
        from app.core.config_improved import Settings

        settings = Settings(debug=True)
        app = create_api_server(settings)

        print("    ✅ FastAPI додаток створено")
        print(f"    ✅ Назва: {app.title}")
        print(f"    ✅ Версія: {app.version}")
        print(f"    ✅ Маршрутів: {len(app.routes)}")

        return True

    except Exception as e:
        print(f"    ❌ Помилка API сервера: {e}")
        return False

def show_next_steps():
    """Показати наступні кроки."""
    print("\n🚀 Наступні кроки:")
    print("=" * 40)
    print("1. 📦 Встановити всі залежності:")
    print("   pip install -r requirements.txt")
    print()
    print("2. 🌐 Запустити API сервер:")
    print("   python app/main_improved.py api --reload")
    print()
    print("3. 🖥️ Запустити CLI інтерфейс:")
    print("   python app/main_improved.py cli")
    print()
    print("4. 📊 Перевірити статус:")
    print("   python app/main_improved.py status")
    print()
    print("5. 📚 Переглянути документацію:")
    print("   docs/README_IMPROVED.md")
    print("   docs/architecture/SYSTEM_ARCHITECTURE_V2.md")
    print("=" * 40)

async def main():
    """Головна функція."""
    print_banner()

    # Перевірка залежностей
    if not check_dependencies():
        print("\n❌ Не всі залежності доступні. Встановіть їх та спробуйте знову.")
        show_next_steps()
        return 1

    # Тестування функціональності
    tests = [
        ("Базова функціональність", test_basic_functionality),
        ("Доменні моделі", test_domain_models),
        ("API сервер", test_api_server),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🧪 Запуск тесту: {test_name}")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            if result:
                passed += 1
                print(f"✅ Тест '{test_name}' пройдено")
            else:
                print(f"❌ Тест '{test_name}' не пройдено")
        except Exception as e:
            print(f"💥 Тест '{test_name}' завершився з помилкою: {e}")

    # Результати
    print("\n" + "=" * 60)
    print(f"📊 Результати тестування: {passed}/{total} тестів пройдено")

    if passed == total:
        print("🎉 Всі тести пройшли успішно!")
        print("✅ DataMCPServerAgent v2.0 готовий до роботи!")

        print("\n🌟 Система готова! Можете:")
        print("  • Запускати API сервер")
        print("  • Використовувати CLI інтерфейс")
        print("  • Створювати та керувати агентами")
        print("  • Виконувати завдання")

    else:
        print("⚠️ Деякі тести не пройшли.")
        print("🔧 Перевірте залежності та конфігурацію.")

    show_next_steps()

    return 0 if passed == total else 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n👋 Роботу перервано користувачем")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Критична помилка: {e}")
        sys.exit(1)
