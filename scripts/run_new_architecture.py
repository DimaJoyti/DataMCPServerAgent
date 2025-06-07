#!/usr/bin/env python3
"""
Демонстрація нової архітектури DataMCPServerAgent.
Показує як працювати з новою структурою коду.
"""

import asyncio
import sys
from pathlib import Path

# Add app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import settings
from app.core.logging import get_logger, set_correlation_id
from app.domain.models.agent import Agent, AgentType, AgentConfiguration, AgentCapability
from app.domain.models.task import Task, TaskType, TaskPriority
from app.domain.services.agent_service import AgentService
from app.infrastructure.repositories.base import InMemoryRepository


logger = get_logger(__name__)


async def demonstrate_new_architecture():
    """Демонстрація роботи нової архітектури."""
    
    # Set correlation ID for request tracing
    set_correlation_id("demo_001")
    
    logger.info("🚀 Демонстрація нової архітектури DataMCPServerAgent")
    
    # 1. Створення агента з новою доменною моделлю
    logger.info("📦 Створення агента...")
    
    # Конфігурація агента
    config = AgentConfiguration(
        max_concurrent_tasks=5,
        timeout_seconds=300,
        memory_limit_mb=512,
        cpu_limit_cores=1.0
    )
    
    # Можливості агента
    capabilities = [
        AgentCapability(
            name="data_processing",
            version="1.0.0",
            description="Process and analyze data",
            enabled=True
        ),
        AgentCapability(
            name="email_handling",
            version="1.0.0", 
            description="Send and receive emails",
            enabled=True
        )
    ]
    
    # Створення агента
    agent = Agent(
        name="demo-analytics-agent",
        agent_type=AgentType.ANALYTICS,
        description="Демонстраційний аналітичний агент",
        configuration=config,
        capabilities=capabilities
    )
    
    logger.info(f"✅ Агент створено: {agent.name} (ID: {agent.id})")
    
    # 2. Демонстрація доменних подій
    logger.info("📡 Доменні події:")
    events = agent.clear_domain_events()
    for event in events:
        logger.info(f"  - {event.event_type}: {event.data}")
    
    # 3. Створення завдання
    logger.info("📋 Створення завдання...")
    
    task = Task(
        name="Аналіз даних клієнтів",
        task_type=TaskType.DATA_ANALYSIS,
        agent_id=agent.id,
        priority=TaskPriority.HIGH,
        description="Проаналізувати дані клієнтів за останній місяць",
        input_data={
            "dataset": "customers_2024_01",
            "analysis_type": "behavior_patterns",
            "output_format": "json"
        }
    )
    
    logger.info(f"✅ Завдання створено: {task.name} (ID: {task.id})")
    
    # 4. Демонстрація бізнес-логіки
    logger.info("🔄 Виконання бізнес-логіки...")
    
    # Перевірка можливостей агента
    if agent.has_capability("data_processing"):
        logger.info("✅ Агент має можливість обробки даних")
        
        # Зміна статусу завдання
        task.change_status(task.status.__class__.RUNNING)
        logger.info(f"📊 Статус завдання змінено на: {task.status}")
        
        # Оновлення прогресу
        from app.domain.models.task import TaskProgress
        progress = TaskProgress(
            percentage=50.0,
            current_step="Обробка даних",
            total_steps=4,
            completed_steps=2
        )
        task.update_progress(progress)
        logger.info(f"📈 Прогрес завдання: {progress.percentage}%")
        
        # Завершення завдання
        result_data = {
            "patterns_found": 15,
            "customer_segments": ["high_value", "regular", "new"],
            "recommendations": [
                "Збільшити персоналізацію для high_value сегменту",
                "Покращити onboarding для нових клієнтів"
            ]
        }
        task.complete_successfully(result_data)
        logger.info("✅ Завдання успішно завершено")
    
    # 5. Демонстрація масштабування
    logger.info("📈 Демонстрація масштабування...")
    
    if agent.is_scalable():
        agent.scale_to(3)
        logger.info(f"🔄 Агент масштабовано до {agent.desired_instances} інстансів")
    
    # 6. Демонстрація репозиторію
    logger.info("💾 Демонстрація роботи з репозиторієм...")
    
    # Створення in-memory репозиторію для демонстрації
    agent_repo = InMemoryRepository()
    
    # Збереження агента
    saved_agent = await agent_repo.save(agent)
    logger.info(f"💾 Агент збережено в репозиторії")
    
    # Завантаження агента
    loaded_agent = await agent_repo.get_by_id(saved_agent.id)
    if loaded_agent:
        logger.info(f"📖 Агент завантажено з репозиторію: {loaded_agent.name}")
    
    # Пошук агентів за типом
    agents = await agent_repo.list(agent_type=AgentType.ANALYTICS)
    logger.info(f"🔍 Знайдено {len(agents)} аналітичних агентів")
    
    # 7. Демонстрація доменного сервісу
    logger.info("🔧 Демонстрація доменного сервісу...")
    
    # Створення сервісу
    agent_service = AgentService()
    agent_service.register_repository("agent", agent_repo)
    
    # Пошук здорових агентів
    healthy_agents = await agent_service.get_healthy_agents()
    logger.info(f"💚 Знайдено {len(healthy_agents)} здорових агентів")
    
    # 8. Демонстрація конфігурації
    logger.info("⚙️ Демонстрація конфігурації...")
    logger.info(f"🌍 Середовище: {settings.environment}")
    logger.info(f"🐛 Debug режим: {settings.debug}")
    logger.info(f"📊 Cloudflare увімкнено: {settings.enable_cloudflare}")
    logger.info(f"📧 Email увімкнено: {settings.enable_email}")
    logger.info(f"🎥 WebRTC увімкнено: {settings.enable_webrtc}")
    
    # 9. Демонстрація валідації
    logger.info("✅ Демонстрація валідації...")
    
    try:
        # Спроба створити агента з некоректними даними
        invalid_agent = Agent(
            name="",  # Порожнє ім'я - має викликати помилку
            agent_type=AgentType.WORKER
        )
    except Exception as e:
        logger.info(f"🚫 Валідація спрацювала: {type(e).__name__}")
    
    # 10. Фінальна статистика
    logger.info("📊 Фінальна статистика:")
    logger.info(f"  - Агентів створено: 1")
    logger.info(f"  - Завдань виконано: 1") 
    logger.info(f"  - Доменних подій: {len(events)}")
    logger.info(f"  - Успішність: 100%")
    
    logger.info("🎉 Демонстрація нової архітектури завершена успішно!")


async def main():
    """Головна функція."""
    try:
        await demonstrate_new_architecture()
    except Exception as e:
        logger.error(f"❌ Помилка під час демонстрації: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
