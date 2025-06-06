# 🚀 Розширені функції пайплайну обробки документів

## 📋 Огляд нових функцій

Ми успішно реалізували чотири основні напрямки розвитку системи:

### ✅ 1. Інтеграція з векторними сховищами - реалізація конкретних бекендів

**Реалізовані векторні сховища:**
- **Memory Store** - швидке in-memory сховище для розробки та тестування
- **ChromaDB** - популярне векторне сховище з підтримкою гібридного пошуку
- **FAISS** - високопродуктивне сховище від Facebook AI для великих обсягів даних
- **Pinecone, Weaviate, Qdrant** - підготовлені інтерфейси (потребують додаткових залежностей)

**Ключові можливості:**
- Автоматичне управління колекціями
- Гібридний пошук (векторний + ключові слова)
- Фільтрація та сортування результатів
- Статистика та моніторинг
- Батчева обробка для продуктивності

### ✅ 2. Веб-інтерфейс - інтеграція з agent-ui

**Реалізований веб-інтерфейс:**
- **FastAPI REST API** з повною документацією
- **Асинхронна обробка** документів з відстеженням прогресу
- **Інтерактивний веб-інтерфейс** з Alpine.js та Tailwind CSS
- **Інтеграція з agent-ui** через стандартні API ендпоінти

**API ендпоінти:**
- `POST /documents/upload` - завантаження та обробка документів
- `GET /documents/{task_id}/status` - статус обробки
- `POST /search` - пошук у векторних сховищах
- `GET /stats` - статистика системи
- `GET /collections` - управління колекціями

### ✅ 3. Розширення форматів - додаткові типи документів

**Нові підтримувані формати:**
- **Excel файли** (.xlsx, .xls) - з підтримкою множинних аркушів
- **PowerPoint презентації** (.pptx) - з витяганням тексту та нотаток
- **CSV/TSV файли** - з автоматичним визначенням параметрів
- **Покращена підтримка** існуючих форматів

**Функції парсерів:**
- Автоматичне визначення кодування та роздільників
- Збереження структури документів
- Витягання метаданих
- Обробка таблиць та зображень

### ✅ 4. Оптимізація - асинхронна обробка та розподілені обчислення

**Асинхронні компоненти:**
- **AsyncDocumentProcessor** - паралельна обробка документів
- **AsyncBatchProcessor** - асинхронна векторизація
- **TaskQueue & TaskManager** - система черг завдань з пріоритетами
- **DistributedProcessor** - розподілена обробка

**Оптимізації продуктивності:**
- Паралельна обробка з контролем ресурсів
- Батчева векторизація з кешуванням
- Система повторних спроб
- Моніторинг та статистика

## 🛠️ Встановлення додаткових залежностей

### Базові залежності (вже встановлені)
```bash
uv pip install fastapi uvicorn pydantic
uv pip install sentence-transformers transformers torch
```

### Векторні сховища
```bash
# ChromaDB
uv pip install chromadb

# FAISS
uv pip install faiss-cpu
# або для GPU
uv pip install faiss-gpu

# Опціональні сховища
uv pip install pinecone-client weaviate-client qdrant-client
```

### Нові формати документів
```bash
# Excel файли
uv pip install pandas openpyxl xlrd

# PowerPoint презентації
uv pip install python-pptx

# Покращена обробка CSV
uv pip install chardet
```

### Веб-інтерфейс
```bash
# FastAPI та залежності
uv pip install fastapi uvicorn python-multipart aiofiles

# Опціонально для продакшену
uv pip install gunicorn
```

## 🚀 Запуск нових функцій

### 1. Демонстрація векторних сховищ
```bash
cd d:\AI\DataMCPServerAgent
python examples/vector_stores_example.py
```

**Що демонструється:**
- Створення різних типів векторних сховищ
- Вставка та пошук векторів
- Гібридний пошук
- Статистика та управління

### 2. Запуск веб-інтерфейсу
```bash
# Запуск API сервера
python src/web_interface/server.py

# Або з налаштуваннями
HOST=0.0.0.0 PORT=8000 python src/web_interface/server.py
```

**Доступні URL:**
- `http://localhost:8000` - API документація (Swagger)
- `http://localhost:8000/ui` - веб-інтерфейс
- `http://localhost:8000/health` - перевірка здоров'я
- `http://localhost:8000/stats` - статистика системи

### 3. Тестування нових форматів
```bash
python examples/advanced_features_example.py
```

**Що тестується:**
- Обробка Excel, PowerPoint, CSV файлів
- Порівняння з існуючими форматами
- Витягання метаданих
- Продуктивність обробки

### 4. Демонстрація асинхронної обробки
```bash
# Включено в advanced_features_example.py
python examples/advanced_features_example.py
```

**Що демонструється:**
- Паралельна обробка документів
- Система черг завдань
- Порівняння продуктивності
- Моніторинг прогресу

## 📊 Приклади використання

### Векторні сховища
```python
from src.data_pipeline.vector_stores.vector_store_manager import VectorStoreManager
from src.data_pipeline.vector_stores.schemas import VectorStoreConfig, VectorStoreType

# Створення менеджера
manager = VectorStoreManager()

# Створення ChromaDB сховища
config = VectorStoreConfig(
    store_type=VectorStoreType.CHROMA,
    collection_name="my_documents",
    embedding_dimension=384,
    persist_directory="data/chroma"
)

store = await manager.create_store("chroma_store", config)

# Пошук
from src.data_pipeline.vector_stores.schemas.search_models import SearchQuery, SearchType

query = SearchQuery(
    query_text="machine learning",
    search_type=SearchType.HYBRID,
    limit=10
)

results = await store.search_vectors(query)
```

### Веб API
```python
import httpx

# Завантаження документа
files = {"file": open("document.pdf", "rb")}
data = {
    "enable_vectorization": True,
    "store_vectors": True,
    "collection_name": "my_docs"
}

response = httpx.post("http://localhost:8000/documents/upload", files=files, data=data)
task_id = response.json()["task_id"]

# Перевірка статусу
status = httpx.get(f"http://localhost:8000/documents/{task_id}/status")

# Пошук
search_data = {
    "query_text": "artificial intelligence",
    "search_type": "hybrid",
    "limit": 5
}

results = httpx.post("http://localhost:8000/search", json=search_data)
```

### Асинхронна обробка
```python
from src.data_pipeline.async_processing import AsyncDocumentProcessor, TaskManager

# Асинхронна обробка документів
async_processor = AsyncDocumentProcessor(max_workers=4)

files = ["doc1.pdf", "doc2.docx", "doc3.xlsx"]
results = await async_processor.process_files_async(files)

# Система черг
task_manager = TaskManager(max_workers=3)
await task_manager.start()

task_id = await task_manager.submit_task(
    my_processing_function,
    "argument1", "argument2",
    priority=TaskPriority.HIGH
)
```

### Нові формати документів
```python
from src.data_pipeline.document_processing import DocumentProcessor

processor = DocumentProcessor()

# Excel файл
excel_result = processor.process_file("spreadsheet.xlsx")
print(f"Sheets: {excel_result.metadata.custom_metadata['total_sheets']}")

# PowerPoint презентація
ppt_result = processor.process_file("presentation.pptx")
print(f"Slides: {ppt_result.metadata.page_count}")

# CSV файл з автоматичним визначенням параметрів
csv_result = processor.process_file("data.csv")
print(f"Delimiter: {csv_result.metadata.custom_metadata['delimiter']}")
```

## 🔧 Налаштування продуктивності

### Векторні сховища
```python
# FAISS для великих обсягів
config = VectorStoreConfig(
    store_type=VectorStoreType.FAISS,
    index_type="hnsw",  # Для швидкого пошуку
    index_params={
        "M": 16,
        "ef_construction": 200,
        "ef_search": 50
    }
)

# ChromaDB з оптимізацією
config = VectorStoreConfig(
    store_type=VectorStoreType.CHROMA,
    batch_size=100,  # Більші батчі
    persist_directory="data/chroma"
)
```

### Асинхронна обробка
```python
# Оптимізація для CPU-інтенсивних задач
async_processor = AsyncDocumentProcessor(
    max_workers=8,
    use_process_pool=True,  # Використання процесів
    chunk_size=20
)

# Налаштування черг
task_manager = TaskManager(
    max_workers=6,
    queue_maxsize=1000,
    cleanup_interval=1800  # 30 хвилин
)
```

### Веб-інтерфейс
```bash
# Продакшен запуск з Gunicorn
gunicorn src.web_interface.server:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 300
```

## 📈 Моніторинг та діагностика

### Статистика системи
```python
# Статистика векторних сховищ
stats = await manager.get_stats_all()
print(f"Total vectors: {sum(s.get('total_vectors', 0) for s in stats.values())}")

# Статистика черг завдань
task_stats = task_manager.get_stats()
print(f"Success rate: {task_stats['success_rate']:.1f}%")

# Статистика кешу
cache_stats = batch_processor.get_cache_stats()
print(f"Cache hit rate: {cache_stats.get('hit_rate', 0):.1f}%")
```

### Логування
```python
import logging

# Детальне логування для діагностики
logging.getLogger('src.data_pipeline').setLevel(logging.DEBUG)
logging.getLogger('src.web_interface').setLevel(logging.INFO)
logging.getLogger('src.async_processing').setLevel(logging.INFO)
```

## 🔄 Інтеграція з agent-ui

### Налаштування MCP сервера
```json
{
  "mcpServers": {
    "document-pipeline": {
      "command": "python",
      "args": ["src/web_interface/server.py"],
      "env": {
        "HOST": "localhost",
        "PORT": "8001"
      }
    }
  }
}
```

### Використання в агентах
```typescript
// Завантаження документа через агент
const uploadResponse = await fetch('/documents/upload', {
  method: 'POST',
  body: formData
});

// Пошук документів
const searchResponse = await fetch('/search', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query_text: userQuery,
    search_type: 'hybrid',
    limit: 10
  })
});
```

## 🎯 Наступні кроки

1. **Розширення векторних сховищ**
   - Реалізація Pinecone, Weaviate, Qdrant бекендів
   - Підтримка гібридних індексів
   - Автоматичне масштабування

2. **Покращення веб-інтерфейсу**
   - Реалтайм оновлення статусу
   - Візуалізація результатів пошуку
   - Адміністративна панель

3. **Додаткові формати**
   - Аудіо та відео файли
   - Архіви та стиснені файли
   - Спеціалізовані формати (CAD, GIS)

4. **Розподілені обчислення**
   - Кластерна обробка
   - Інтеграція з Kubernetes
   - Автоматичне балансування навантаження

---

**Система готова до використання в продакшені!** 🎉

Всі компоненти протестовані та оптимізовані для високої продуктивності та надійності.
