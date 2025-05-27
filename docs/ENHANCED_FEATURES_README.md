# 🚀 Enhanced DataMCPServerAgent Features

Цей документ описує розширені можливості DataMCPServerAgent, включаючи персистентний стан, довготривалі завдання, горизонтальне масштабування, Email API, WebRTC та self-hosting.

## 📋 Огляд нових можливостей

### 1. 📦 Персистентний стан з Cloudflare
- **Cloudflare KV** для швидкого доступу до стану агентів
- **Durable Objects** для консистентності стану
- **Версіонування стану** для відстеження змін
- **Автоматичне відновлення** після збоїв

### 2. ⏳ Довготривалі завдання та горизонтальне масштабування
- **Cloudflare Queues** для асинхронних завдань
- **Автоматичне масштабування** на основі навантаження
- **Моніторинг прогресу** завдань в реальному часі
- **Розподілене виконання** між інстансами

### 3. 📧 Email API для Human-in-the-Loop
- **Підтримка кількох провайдерів**: Cloudflare Email Workers, SendGrid, Mailgun, SMTP
- **Approval workflows** для критичних рішень
- **Шаблони email** з динамічними змінними
- **Автоматичне відстеження** статусу доставки

### 4. 🎥 WebRTC для голосу та відео
- **Cloudflare Calls API** інтеграція
- **Voice-to-text** в реальному часі
- **Text-to-speech** для відповідей агентів
- **Відеоконференції** з записом

### 5. 🏠 Self-hosting можливості
- **Docker containerization** з multi-stage builds
- **Kubernetes deployment** з повною конфігурацією
- **Docker Compose** для локальної розробки
- **Автоматична генерація** конфігураційних файлів

## 🛠️ Швидкий старт

### Встановлення залежностей

```bash
# Встановити основні залежності
pip install -r requirements.txt

# Для email інтеграції
pip install sendgrid mailgun

# Для WebRTC (опціонально)
pip install aiortc websockets

# Для self-hosting
pip install docker kubernetes pyyaml
```

### Базове використання

```python
import asyncio
from enhanced_agent_example import EnhancedAgentDemo

async def main():
    demo = EnhancedAgentDemo()
    await demo.run_complete_demo()

asyncio.run(main())
```

## 📦 Персистентний стан

### Збереження стану агента

```python
from cloudflare_mcp_integration import enhanced_cloudflare_integration, AgentType

# Збереження стану
state_data = {
    "conversation_history": [...],
    "user_preferences": {...},
    "context": {...}
}

await enhanced_cloudflare_integration.save_agent_state(
    agent_id="agent_001",
    agent_type=AgentType.ANALYTICS,
    state_data=state_data
)
```

### Завантаження стану

```python
# Завантаження стану
state = await enhanced_cloudflare_integration.load_agent_state("agent_001")
if state:
    print(f"Версія стану: {state.version}")
    print(f"Дані: {state.state_data}")
```

## ⏳ Довготривалі завдання

### Створення завдання

```python
# Створення довготривалого завдання
task_id = await enhanced_cloudflare_integration.create_long_running_task(
    agent_id="agent_001",
    task_type="data_analysis",
    metadata={"dataset_size": "10GB"}
)
```

### Відстеження прогресу

```python
# Оновлення прогресу
await enhanced_cloudflare_integration.update_task_progress(
    task_id, 
    progress=50.0, 
    status=TaskStatus.RUNNING
)

# Завершення завдання
await enhanced_cloudflare_integration.complete_task(
    task_id, 
    result={"processed_items": 1000}
)
```

## 📈 Горизонтальне масштабування

### Масштабування агентів

```python
# Масштабування до 5 інстансів
result = await enhanced_cloudflare_integration.scale_agent_horizontally(
    agent_id="agent_001",
    target_instances=5
)

# Отримання метрик навантаження
metrics = await enhanced_cloudflare_integration.get_agent_load_metrics("agent_001")
print(f"Рекомендація: {metrics['scaling_recommendation']}")
```

## 📧 Email інтеграція

### Налаштування провайдера

```python
from email_integration import email_integration, EmailProvider

# Налаштування SMTP
email_integration.provider_configs[EmailProvider.SMTP] = {
    "host": "smtp.gmail.com",
    "port": 587,
    "username": "your-email@gmail.com",
    "password": "your-app-password",
    "use_tls": True
}
```

### Створення approval request

```python
# Створення запиту на підтвердження
approval_id = await email_integration.create_approval_request(
    agent_id="agent_001",
    task_id="sensitive_operation",
    title="Доступ до конфіденційних даних",
    description="Агент потребує доступу до фінансових даних клієнта",
    data={"customer_id": "12345"},
    approver_email="admin@company.com",
    expires_in_hours=24
)
```

### Обробка відповіді

```python
# Обробка підтвердження
success = await email_integration.process_approval_response(
    approval_id,
    action="approve",
    approver_email="admin@company.com",
    reason="Схвалено для аналізу шахрайства"
)
```

## 🎥 WebRTC інтеграція

### Створення дзвінка

```python
from webrtc_integration import webrtc_integration, CallDirection

# Створення сесії дзвінка
call_id = await webrtc_integration.create_call_session(
    agent_id="agent_001",
    user_id="user_001",
    direction=CallDirection.INBOUND,
    enable_recording=True,
    enable_video=True
)

# Початок дзвінка
await webrtc_integration.start_call(call_id)
```

### Voice-to-Text

```python
# Обробка голосу в текст
result = await webrtc_integration.process_voice_to_text(
    call_id,
    audio_data=audio_bytes,
    participant_id="user_001"
)
print(f"Розпізнаний текст: {result.text}")
```

### Text-to-Speech

```python
# Відтворення тексту як мовлення
await webrtc_integration.play_speech_in_call(
    call_id,
    text="Дякую за дзвінок. Як я можу вам допомогти?",
    voice="uk-UA-PolinaNeural"
)
```

## 🏠 Self-hosting

### Генерація Docker файлів

```python
from self_hosting_config import self_hosting_manager

# Генерація файлів для розгортання
self_hosting_manager.save_deployment_files(
    environment="production",
    output_dir="./deployment"
)
```

### Docker Compose

```bash
# Запуск з Docker Compose
cd deployment
docker-compose up -d

# Перегляд логів
docker-compose logs -f agent-server
```

### Kubernetes

```bash
# Розгортання в Kubernetes
kubectl apply -f deployment/kubernetes/

# Перевірка статусу
kubectl get pods -n datamcpserveragent
```

## 🔧 Конфігурація

### Змінні середовища

```bash
# .env файл
ENVIRONMENT=production
CLOUDFLARE_API_KEY=your-api-key
POSTGRES_PASSWORD=secure-password
JWT_SECRET=your-jwt-secret
EMAIL_PROVIDER=sendgrid
SENDGRID_API_KEY=your-sendgrid-key
```

### Cloudflare налаштування

```javascript
// wrangler.toml для Cloudflare Workers
name = "datamcpserveragent"
main = "src/worker.js"
compatibility_date = "2024-01-01"

[env.production]
kv_namespaces = [
  { binding = "AGENT_STATE", id = "your-kv-namespace-id" }
]

[[env.production.durable_objects.bindings]]
name = "AGENT_OBJECTS"
class_name = "AgentDurableObject"
```

## 📊 Моніторинг та логування

### Prometheus метрики

```python
# Автоматичні метрики для всіх компонентів
- agent_requests_total
- agent_response_time_seconds
- agent_errors_total
- task_duration_seconds
- call_duration_seconds
```

### Grafana дашборди

Автоматично генеруються дашборди для:
- Продуктивність агентів
- Статус довготривалих завдань
- Email delivery метрики
- WebRTC якість дзвінків

## 🚀 Розгортання в продакшені

### 1. Підготовка

```bash
# Клонування репозиторію
git clone https://github.com/your-org/DataMCPServerAgent.git
cd DataMCPServerAgent

# Генерація конфігурацій
python -c "
from self_hosting_config import self_hosting_manager
self_hosting_manager.save_deployment_files('production', './prod-deployment')
"
```

### 2. Kubernetes розгортання

```bash
# Створення namespace
kubectl create namespace datamcpserveragent

# Застосування конфігурацій
kubectl apply -f prod-deployment/kubernetes/

# Перевірка статусу
kubectl get all -n datamcpserveragent
```

### 3. Налаштування DNS та SSL

```bash
# Налаштування Ingress з Let's Encrypt
kubectl apply -f https://github.com/jetstack/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Перевірка сертифікатів
kubectl get certificates -n datamcpserveragent
```

## 🔒 Безпека

### Рекомендації

1. **Використовуйте HTTPS** для всіх з'єднань
2. **Шифруйте секрети** в Kubernetes
3. **Обмежте доступ** до Cloudflare API ключів
4. **Регулярно оновлюйте** залежності
5. **Моніторьте** підозрілу активність

### Аудит безпеки

```bash
# Сканування вразливостей
docker run --rm -v $(pwd):/app clair-scanner:latest /app

# Перевірка Kubernetes конфігурацій
kubectl run kube-score --image=zegl/kube-score:latest -- score deployment/kubernetes/*.yaml
```

## 📚 Додаткові ресурси

- [Cloudflare Workers Documentation](https://developers.cloudflare.com/workers/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [WebRTC API Reference](https://developer.mozilla.org/en-US/docs/Web/API/WebRTC_API)

## 🤝 Підтримка

Для питань та підтримки:
- 📧 Email: support@yourdomain.com
- 💬 Discord: [Your Discord Server]
- 📖 Documentation: [Your Docs Site]
- 🐛 Issues: [GitHub Issues]
