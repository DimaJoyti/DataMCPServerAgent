# Document Processing Pipeline

Комплексна система для обробки документів, векторизації та збереження у векторних сховищах.

## 🎯 Огляд

Система забезпечує повний пайплайн обробки документів:

1. **Парсинг документів** - підтримка PDF, DOCX, HTML, Markdown, TXT
2. **Чанкінг тексту** - розбиття на логічні частини з різними стратегіями
3. **Векторизація** - генерація ембедингів з кешуванням та батчевою обробкою
4. **Векторні сховища** - збереження та пошук з підтримкою різних бекендів

## 📁 Структура

```
src/data_pipeline/
├── document_processing/          # Обробка документів
│   ├── parsers/                 # Парсери для різних форматів
│   ├── chunking/                # Стратегії чанкінгу
│   ├── metadata/                # Метадані та збагачення
│   └── document_processor.py    # Головний процесор
├── vectorization/               # Векторизація
│   ├── embeddings/              # Генерація ембедингів
│   ├── batch_processor.py       # Батчева обробка
│   └── vector_cache.py          # Кешування векторів
└── vector_stores/               # Векторні сховища
    ├── schemas/                 # Схеми даних
    ├── backends/                # Різні бекенди
    └── search/                  # Пошук та фільтрація
```

## 🚀 Швидкий старт

### Базове використання

```python
from src.data_pipeline.document_processing import DocumentProcessor
from src.data_pipeline.vectorization import HuggingFaceEmbedder, EmbeddingConfig

# Ініціалізація процесора документів
processor = DocumentProcessor()

# Обробка документа
result = processor.process_file("document.pdf")
print(f"Створено {len(result.chunks)} чанків")

# Ініціалізація ембедера
config = EmbeddingConfig(
    model_name="all-MiniLM-L6-v2",
    model_provider="huggingface",
    embedding_dimension=384
)
embedder = HuggingFaceEmbedder(config)

# Генерація ембедингів
embedding_result = embedder.embed_text("Приклад тексту")
print(f"Розмірність вектора: {len(embedding_result.embedding)}")
```

### Повний пайплайн

```python
from src.data_pipeline.document_processing import DocumentProcessor
from src.data_pipeline.vectorization import BatchVectorProcessor
from src.data_pipeline.vector_stores.schemas import DocumentVectorSchema

# Обробка документа
processor = DocumentProcessor()
doc_result = processor.process_file("document.pdf")

# Векторизація чанків
batch_processor = BatchVectorProcessor(embedder)
vector_result = batch_processor.process_chunks(doc_result.chunks)

# Створення векторних записів
schema = DocumentVectorSchema(store_config)
vector_records = []
for chunk, embedding in zip(doc_result.chunks, vector_result.results):
    if embedding:
        record = schema.create_record(
            chunk_metadata=chunk.metadata,
            document_metadata=doc_result.get_metadata(),
            vector=embedding.embedding,
            embedding_model=embedding.model_name
        )
        vector_records.append(record)
```

## 📖 Детальна документація

### Обробка документів

#### Підтримувані формати
- **PDF** - з використанням PyPDF2 або pdfplumber
- **DOCX** - з використанням python-docx
- **HTML** - з використанням BeautifulSoup4
- **Markdown** - з підтримкою front matter
- **TXT** - з автоматичним визначенням кодування

#### Стратегії чанкінгу
- **TextChunker** - базовий чанкінг за розміром
- **SemanticChunker** - семантичний чанкінг (в розробці)
- **AdaptiveChunker** - адаптивний чанкінг за характеристиками тексту

#### Конфігурація

```python
from src.data_pipeline.document_processing import (
    DocumentProcessingConfig, ParsingConfig, ChunkingConfig
)

parsing_config = ParsingConfig(
    extract_metadata=True,
    extract_tables=True,
    normalize_whitespace=True
)

chunking_config = ChunkingConfig(
    chunk_size=1000,
    chunk_overlap=200,
    strategy="text",
    preserve_sentences=True
)

config = DocumentProcessingConfig(
    parsing_config=parsing_config,
    chunking_config=chunking_config,
    enable_chunking=True,
    enable_metadata_enrichment=True
)
```

### Векторизація

#### Підтримувані провайдери
- **OpenAI** - text-embedding-3-small, text-embedding-3-large
- **HuggingFace** - sentence-transformers та transformers
- **Cloudflare AI** - BGE моделі

#### Батчева обробка

```python
from src.data_pipeline.vectorization import (
    BatchVectorProcessor, BatchProcessingConfig
)

batch_config = BatchProcessingConfig(
    batch_size=32,
    max_workers=4,
    enable_caching=True,
    show_progress=True
)

processor = BatchVectorProcessor(embedder, batch_config)
result = processor.process_texts(texts)
```

#### Кешування

```python
from src.data_pipeline.vectorization import VectorCache, CacheConfig

cache_config = CacheConfig(
    backend="file",  # або "memory", "redis"
    cache_dir="cache/vectors",
    ttl=86400,  # 24 години
    max_size=10000
)

cache = VectorCache(cache_config)
```

### Векторні сховища

#### Схеми даних

```python
from src.data_pipeline.vector_stores.schemas import (
    DocumentVectorSchema, VectorStoreConfig
)

store_config = VectorStoreConfig(
    store_type="chroma",
    collection_name="documents",
    embedding_dimension=384,
    distance_metric="cosine"
)

schema = DocumentVectorSchema(store_config)
```

#### Пошук та фільтрація

```python
from src.data_pipeline.vector_stores.schemas import (
    SearchQuery, SearchFilters, SearchType
)

# Створення фільтрів
filters = SearchFilters()
filters.add_text_filter("document_type", "pdf")
filters.add_range_filter("word_count", min_value=100, max_value=1000)

# Створення запиту
query = SearchQuery(
    query_text="машинне навчання",
    search_type=SearchType.HYBRID,
    limit=10,
    filters=filters
)
```

## ⚙️ Встановлення

### Базові залежності

```bash
pip install -r requirements.txt
```

### Додаткові залежності

```bash
# Для PDF обробки
pip install PyPDF2 pdfplumber

# Для DOCX обробки  
pip install python-docx

# Для HTML обробки
pip install beautifulsoup4 html2text

# Для векторизації
pip install sentence-transformers transformers torch
pip install openai  # для OpenAI ембедингів

# Для векторних сховищ
pip install chromadb faiss-cpu pinecone-client weaviate-client
```

## 🔧 Конфігурація

### Змінні середовища

```bash
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Cloudflare
CLOUDFLARE_ACCOUNT_ID=your_account_id
CLOUDFLARE_API_TOKEN=your_api_token

# Redis (для кешування)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_password
```

## 📊 Продуктивність

### Рекомендації
- Використовуйте батчеву обробку для великих обсягів
- Увімкніть кешування для повторних обробок
- Налаштуйте розмір чанків під вашу модель ембедингів
- Використовуйте адаптивний чанкінг для різнорідного контенту

### Бенчмарки
- Обробка PDF: ~10-50 сторінок/сек
- Чанкінг: ~1000-5000 чанків/сек  
- Векторизація (HuggingFace): ~100-500 текстів/сек
- Векторизація (OpenAI): ~50-200 текстів/сек (залежить від API лімітів)

## 🧪 Тестування

```bash
# Запуск прикладів
python examples/document_processing_example.py
python examples/complete_pipeline_example.py

# Запуск тестів
pytest tests/
```

## 🤝 Розширення

### Додавання нового парсера

```python
from src.data_pipeline.document_processing.parsers import BaseParser

class CustomParser(BaseParser):
    @property
    def supported_types(self):
        return [DocumentType.CUSTOM]
    
    @property  
    def supported_extensions(self):
        return ['custom']
    
    def _parse_file_impl(self, file_path):
        # Ваша логіка парсингу
        pass
```

### Додавання нового ембедера

```python
from src.data_pipeline.vectorization.embeddings import BaseEmbedder

class CustomEmbedder(BaseEmbedder):
    def embed_text(self, text):
        # Ваша логіка генерації ембедингів
        pass
    
    def embed_batch(self, texts):
        # Батчева обробка
        pass
```

## 📝 Ліцензія

MIT License - дивіться файл LICENSE для деталей.

## 🆘 Підтримка

Для питань та проблем створюйте issue в репозиторії або звертайтеся до команди розробки.
