# 🚀 Налаштування та запуск пайплайну обробки документів

## 📋 Огляд створеної системи

Ми успішно створили комплексну систему для обробки документів з векторизацією, яка включає:

### ✅ Реалізовані компоненти

1. **📄 Обробка документів**
   - Парсери для PDF, DOCX, HTML, Markdown, TXT
   - Система чанкінгу з різними стратегіями
   - Збагачення метаданих
   - Головний процесор документів

2. **🔢 Векторизація**
   - Підтримка OpenAI, HuggingFace, Cloudflare AI
   - Батчева обробка з кешуванням
   - Оптимізація продуктивності

3. **🗄️ Векторні сховища**
   - Схеми для документів
   - Моделі пошуку та фільтрації
   - Підготовка для різних бекендів

4. **📚 Документація та приклади**
   - Повна документація
   - Приклади використання
   - Тести

## 🛠️ Встановлення

### 1. Встановлення базових залежностей

```bash
# Встановлення основних пакетів
uv pip install pydantic python-dotenv aiofiles httpx

# Обробка документів
uv pip install PyPDF2 python-docx beautifulsoup4 markdown html2text
uv pip install python-magic chardet langdetect textstat

# Векторизація
uv pip install sentence-transformers transformers torch
uv pip install openai  # опціонально для OpenAI

# Векторні сховища (опціонально)
uv pip install chromadb faiss-cpu pinecone-client weaviate-client

# Тестування
uv pip install pytest pytest-asyncio
```

### 2. Налаштування змінних середовища

Створіть файл `.env`:

```bash
# OpenAI (опціонально)
OPENAI_API_KEY=your_openai_api_key

# Cloudflare AI (опціонально)
CLOUDFLARE_ACCOUNT_ID=your_account_id
CLOUDFLARE_API_TOKEN=your_api_token

# Логування
LOG_LEVEL=INFO
```

## 🚀 Запуск прикладів

### 1. Базовий приклад обробки документів

```bash
cd d:\AI\DataMCPServerAgent
python examples/document_processing_example.py
```

Цей приклад демонструє:
- Парсинг різних форматів документів
- Чанкінг тексту
- Збагачення метаданих
- Різні стратегії обробки

### 2. Повний пайплайн з векторизацією

```bash
python examples/complete_pipeline_example.py
```

Цей приклад показує:
- Повний цикл обробки документів
- Генерацію ембедингів
- Створення векторних записів
- Аналіз продуктивності

### 3. Запуск тестів

```bash
python -m pytest tests/test_document_pipeline.py -v
```

## 📖 Приклади використання

### Швидкий старт

```python
from src.data_pipeline.document_processing import DocumentProcessor

# Створення процесора
processor = DocumentProcessor()

# Обробка файлу
result = processor.process_file("document.pdf")
print(f"Створено {len(result.chunks)} чанків")

# Обробка тексту
result = processor.process_content(
    content="Ваш текст тут",
    document_id="test_doc"
)
```

### Налаштування конфігурації

```python
from src.data_pipeline.document_processing import (
    DocumentProcessingConfig, ChunkingConfig
)

# Налаштування чанкінгу
chunking_config = ChunkingConfig(
    chunk_size=512,
    chunk_overlap=50,
    strategy="text",
    preserve_sentences=True
)

# Створення конфігурації
config = DocumentProcessingConfig(
    chunking_config=chunking_config,
    enable_chunking=True,
    enable_metadata_enrichment=True
)

processor = DocumentProcessor(config)
```

### Векторизація

```python
from src.data_pipeline.vectorization import (
    HuggingFaceEmbedder, EmbeddingConfig, BatchVectorProcessor
)

# Налаштування ембедера
embedding_config = EmbeddingConfig(
    model_name="all-MiniLM-L6-v2",
    model_provider="huggingface",
    embedding_dimension=384
)

embedder = HuggingFaceEmbedder(embedding_config)

# Батчева обробка
batch_processor = BatchVectorProcessor(embedder)
results = batch_processor.process_texts(["текст 1", "текст 2"])
```

## 🔧 Налаштування продуктивності

### Оптимізація для великих обсягів

```python
# Збільшення розміру батчу
batch_config = BatchProcessingConfig(
    batch_size=64,  # Збільшити для кращої продуктивності
    max_workers=4,  # Паралельна обробка
    enable_caching=True  # Кешування результатів
)

# Налаштування кешу
cache_config = CacheConfig(
    backend="file",
    cache_dir="cache/embeddings",
    ttl=86400 * 7,  # 1 тиждень
    max_size=100000  # Більший розмір кешу
)
```

### Оптимізація чанкінгу

```python
# Для технічних документів
chunking_config = ChunkingConfig(
    chunk_size=1024,  # Більші чанки
    chunk_overlap=100,
    preserve_sections=True  # Зберігати структуру
)

# Для коротких текстів
chunking_config = ChunkingConfig(
    chunk_size=256,  # Менші чанки
    chunk_overlap=50,
    preserve_sentences=True
)
```

## 📊 Моніторинг та діагностика

### Логування

```python
import logging

# Налаштування детального логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Для діагностики
logging.getLogger('src.data_pipeline').setLevel(logging.DEBUG)
```

### Метрики продуктивності

```python
# Аналіз результатів обробки
result = processor.process_file("document.pdf")
print(f"Час обробки: {result.processing_time:.2f}s")
print(f"Кількість чанків: {len(result.chunks)}")
print(f"Середній розмір чанку: {sum(len(c.text) for c in result.chunks) / len(result.chunks):.0f}")

# Аналіз векторизації
vector_result = batch_processor.process_chunks(result.chunks)
print(f"Час векторизації: {vector_result.total_time:.2f}s")
print(f"Швидкість: {vector_result.total_items / vector_result.total_time:.1f} елементів/сек")
```

## 🐛 Усунення проблем

### Поширені помилки

1. **ImportError для залежностей**
   ```bash
   # Встановіть відсутні пакети
   uv pip install package_name
   ```

2. **Помилки кодування файлів**
   ```python
   # Використовуйте автоматичне визначення кодування
   config.parsing_config.encoding = None  # Автовизначення
   ```

3. **Проблеми з пам'яттю**
   ```python
   # Зменшіть розмір батчу
   config.batch_size = 16
   # Увімкніть очищення кешу
   config.clear_cache_interval = 500
   ```

### Діагностика

```python
# Перевірка здоров'я компонентів
embedder = HuggingFaceEmbedder(config)
print(f"Embedder здоровий: {embedder.health_check()}")

cache = VectorCache(cache_config)
print(f"Cache здоровий: {cache.health_check()}")
```

## 🔄 Наступні кроки

1. **Інтеграція з векторними сховищами**
   - Реалізація конкретних бекендів (Chroma, FAISS)
   - Система пошуку та індексації

2. **Розширення функціональності**
   - Додаткові формати документів
   - Покращені стратегії чанкінгу
   - Гібридний пошук

3. **Оптимізація**
   - Асинхронна обробка
   - Розподілена векторизація
   - Кешування на рівні бази даних

4. **Веб-інтерфейс**
   - API для обробки документів
   - Дашборд для моніторингу
   - Інтеграція з agent-ui

## 📞 Підтримка

Для питань та проблем:
1. Перевірте логи на наявність помилок
2. Запустіть тести для діагностики
3. Перегляньте документацію в `docs/`
4. Створіть issue з детальним описом проблеми

---

**Система готова до використання!** 🎉

Ви можете почати з базових прикладів та поступово розширювати функціональність відповідно до ваших потреб.
