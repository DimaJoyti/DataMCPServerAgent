"""
FastAPI application for document processing pipeline.
"""

import logging

# Add src to path for imports
import sys
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.data_pipeline.document_processing import (
    ChunkingConfig,
    DocumentProcessingConfig,
    DocumentProcessor,
    ParsingConfig,
)
from src.data_pipeline.vector_stores.schemas import (
    DistanceMetric,
    DocumentVectorSchema,
    VectorStoreConfig,
    VectorStoreType,
)
from src.data_pipeline.vector_stores.schemas.search_models import (
    SearchFilters,
    SearchQuery,
    SearchType,
)
from src.data_pipeline.vector_stores.vector_store_manager import VectorStoreManager
from src.data_pipeline.vectorization import (
    BatchProcessingConfig,
    BatchVectorProcessor,
    EmbeddingConfig,
    HuggingFaceEmbedder,
)

from .models import (
    BatchProcessingResponse,
    ChunkInfo,
    DocumentProcessingResponse,
    DocumentUploadRequest,
    PipelineStatus,
    ProcessingStatus,
    SearchResultItem,
    TaskInfo,
    VectorSearchRequest,
    VectorSearchResponse,
)


class DocumentProcessingAPI:
    """Document processing API service."""

    def __init__(self):
        """Initialize API service."""
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        self.document_processor = None
        self.embedder = None
        self.batch_processor = None
        self.vector_store_manager = VectorStoreManager()

        # Task tracking
        self.tasks: Dict[str, TaskInfo] = {}
        self.batch_tasks: Dict[str, BatchProcessingResponse] = {}

        # Statistics
        self.stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "total_vectors": 0,
            "start_time": datetime.now()
        }

    async def initialize(self):
        """Initialize API components."""
        try:
            # Initialize document processor
            parsing_config = ParsingConfig(
                extract_metadata=True,
                normalize_whitespace=True,
                preserve_formatting=False
            )

            chunking_config = ChunkingConfig(
                chunk_size=1000,
                chunk_overlap=200,
                strategy="text",
                preserve_sentences=True
            )

            processing_config = DocumentProcessingConfig(
                parsing_config=parsing_config,
                chunking_config=chunking_config,
                enable_chunking=True,
                enable_metadata_enrichment=True
            )

            self.document_processor = DocumentProcessor(processing_config)

            # Initialize embedder
            embedding_config = EmbeddingConfig(
                model_name="all-MiniLM-L6-v2",
                model_provider="huggingface",
                embedding_dimension=384,
                normalize_embeddings=True,
                batch_size=32
            )

            self.embedder = HuggingFaceEmbedder(
                config=embedding_config,
                use_sentence_transformers=True
            )

            # Initialize batch processor
            batch_config = BatchProcessingConfig(
                batch_size=32,
                max_workers=2,
                enable_caching=True,
                show_progress=False,  # Disable for API
                continue_on_error=True
            )

            self.batch_processor = BatchVectorProcessor(self.embedder, batch_config)

            # Initialize default vector store
            store_config = VectorStoreConfig(
                store_type=VectorStoreType.MEMORY,
                collection_name="default",
                embedding_dimension=384,
                distance_metric=DistanceMetric.COSINE
            )

            await self.vector_store_manager.create_store("default", store_config)

            self.logger.info("API components initialized successfully")

        except Exception as e:
            self.logger.error(f"Failed to initialize API: {e}")
            raise

    async def process_document(
        self,
        file_content: bytes,
        request: DocumentUploadRequest,
        task_id: str
    ) -> DocumentProcessingResponse:
        """Process a document."""
        try:
            # Update task status
            self.tasks[task_id].status = ProcessingStatus.PROCESSING
            self.tasks[task_id].current_step = "parsing_document"
            self.tasks[task_id].started_at = datetime.now()

            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix=Path(request.filename).suffix, delete=False) as temp_file:
                temp_file.write(file_content)
                temp_path = Path(temp_file.name)

            try:
                # Process document
                doc_result = self.document_processor.process_file(temp_path)

                # Update task progress
                self.tasks[task_id].progress = 30.0
                self.tasks[task_id].current_step = "creating_chunks"

                # Create chunk info
                chunks_info = []
                for chunk in doc_result.chunks:
                    chunk_info = ChunkInfo(
                        chunk_id=chunk.chunk_id,
                        chunk_index=chunk.chunk_index,
                        text=chunk.text,
                        character_count=len(chunk.text),
                        word_count=len(chunk.text.split()),
                        start_char=chunk.start_char,
                        end_char=chunk.end_char
                    )
                    chunks_info.append(chunk_info)

                # Vectorize if requested
                vectorization_time = 0.0
                if request.enable_vectorization and doc_result.chunks:
                    self.tasks[task_id].progress = 60.0
                    self.tasks[task_id].current_step = "generating_embeddings"

                    vector_result = await self.batch_processor.process_chunks_async(doc_result.chunks)
                    vectorization_time = vector_result.total_time

                    # Update chunk info with embedding data
                    for i, (chunk_info, embedding_result) in enumerate(zip(chunks_info, vector_result.results)):
                        if embedding_result:
                            chunk_info.has_embedding = True
                            chunk_info.embedding_model = embedding_result.model_name
                            chunk_info.embedding_dimension = embedding_result.embedding_dimension

                # Store in vector store if requested
                stored_in_vector_store = False
                vector_store_collection = None

                if request.store_vectors and request.enable_vectorization:
                    self.tasks[task_id].progress = 80.0
                    self.tasks[task_id].current_step = "storing_vectors"

                    # Get or create vector store
                    collection_name = request.collection_name or "default"
                    store = await self.vector_store_manager.get_store(collection_name)

                    if not store:
                        # Create new store
                        store_config = VectorStoreConfig(
                            store_type=VectorStoreType(request.vector_store_type),
                            collection_name=collection_name,
                            embedding_dimension=384,
                            distance_metric=DistanceMetric.COSINE
                        )
                        store = await self.vector_store_manager.create_store(collection_name, store_config)

                    # Create vector records
                    schema = DocumentVectorSchema(store.config)
                    vector_records = []

                    successful_embeddings = vector_result.get_successful_results()
                    for chunk, embedding_result in zip(doc_result.chunks, successful_embeddings):
                        if embedding_result:
                            record = schema.create_record(
                                chunk_metadata=chunk.metadata,
                                document_metadata=doc_result.get_metadata(),
                                vector=embedding_result.embedding,
                                embedding_model=embedding_result.model_name,
                                processing_time=embedding_result.processing_time
                            )
                            vector_records.append(record)

                    # Insert into vector store
                    if vector_records:
                        await store.insert_vectors(vector_records)
                        stored_in_vector_store = True
                        vector_store_collection = collection_name

                # Update task completion
                self.tasks[task_id].progress = 100.0
                self.tasks[task_id].status = ProcessingStatus.COMPLETED
                self.tasks[task_id].completed_at = datetime.now()
                self.tasks[task_id].current_step = "completed"

                # Update statistics
                self.stats["total_documents"] += 1
                self.stats["total_chunks"] += len(doc_result.chunks)
                if request.enable_vectorization:
                    self.stats["total_vectors"] += len([c for c in chunks_info if c.has_embedding])

                # Create response
                response = DocumentProcessingResponse(
                    task_id=task_id,
                    status=ProcessingStatus.COMPLETED,
                    document_id=doc_result.document_id,
                    filename=request.filename,
                    file_size=request.file_size,
                    text_length=len(doc_result.get_text()),
                    chunks=chunks_info,
                    document_metadata=doc_result.get_metadata().dict(),
                    processing_time=doc_result.processing_time,
                    vectorization_time=vectorization_time,
                    stored_in_vector_store=stored_in_vector_store,
                    vector_store_collection=vector_store_collection,
                    errors=doc_result.errors,
                    warnings=doc_result.warnings
                )

                # Store result in task
                self.tasks[task_id].result = response.dict()

                return response

            finally:
                # Clean up temporary file
                temp_path.unlink(missing_ok=True)

        except Exception as e:
            # Update task with error
            self.tasks[task_id].status = ProcessingStatus.FAILED
            self.tasks[task_id].error = str(e)
            self.tasks[task_id].completed_at = datetime.now()

            self.logger.error(f"Document processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Document Processing Pipeline API",
        description="API for document processing, vectorization, and search",
        version="1.0.0"
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize API service
    api_service = DocumentProcessingAPI()

    @app.on_event("startup")
    async def startup_event():
        """Initialize API on startup."""
        await api_service.initialize()

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on shutdown."""
        await api_service.vector_store_manager.close_all_stores()

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {
            "message": "Document Processing Pipeline API",
            "version": "1.0.0",
            "status": "running"
        }

    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        try:
            # Check component health
            health_status = await api_service.vector_store_manager.health_check_all()

            is_healthy = all(health_status.values()) if health_status else True

            return {
                "status": "healthy" if is_healthy else "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "components": health_status
            }
        except Exception as e:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            )

    # Document processing endpoints
    @app.post("/documents/upload", response_model=DocumentProcessingResponse)
    async def upload_document(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        extract_metadata: bool = Form(True),
        enable_chunking: bool = Form(True),
        enable_vectorization: bool = Form(True),
        chunk_size: int = Form(1000),
        chunk_overlap: int = Form(200),
        chunking_strategy: str = Form("text"),
        embedding_model: str = Form("all-MiniLM-L6-v2"),
        embedding_provider: str = Form("huggingface"),
        store_vectors: bool = Form(True),
        vector_store_type: str = Form("memory"),
        collection_name: Optional[str] = Form(None)
    ):
        """Upload and process a document."""
        try:
            # Validate file
            if not file.filename:
                raise HTTPException(status_code=400, detail="No file provided")

            # Read file content
            file_content = await file.read()

            # Create request object
            request = DocumentUploadRequest(
                filename=file.filename,
                content_type=file.content_type or "application/octet-stream",
                file_size=len(file_content),
                extract_metadata=extract_metadata,
                enable_chunking=enable_chunking,
                enable_vectorization=enable_vectorization,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                chunking_strategy=chunking_strategy,
                embedding_model=embedding_model,
                embedding_provider=embedding_provider,
                store_vectors=store_vectors,
                vector_store_type=vector_store_type,
                collection_name=collection_name
            )

            # Create task
            task_id = str(uuid.uuid4())
            task_info = TaskInfo(
                task_id=task_id,
                status=ProcessingStatus.PENDING,
                task_type="document_processing",
                created_at=datetime.now(),
                metadata={"filename": file.filename}
            )
            api_service.tasks[task_id] = task_info

            # Start background processing
            background_tasks.add_task(
                api_service.process_document,
                file_content,
                request,
                task_id
            )

            # Return initial response
            return DocumentProcessingResponse(
                task_id=task_id,
                status=ProcessingStatus.PENDING,
                document_id="",  # Will be set during processing
                filename=file.filename,
                file_size=len(file_content)
            )

        except Exception as e:
            api_service.logger.error(f"Upload failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/documents/{task_id}/status", response_model=TaskInfo)
    async def get_task_status(task_id: str):
        """Get processing task status."""
        if task_id not in api_service.tasks:
            raise HTTPException(status_code=404, detail="Task not found")

        return api_service.tasks[task_id]

    @app.get("/documents/{task_id}/result", response_model=DocumentProcessingResponse)
    async def get_task_result(task_id: str):
        """Get processing task result."""
        if task_id not in api_service.tasks:
            raise HTTPException(status_code=404, detail="Task not found")

        task = api_service.tasks[task_id]

        if task.status == ProcessingStatus.PENDING:
            raise HTTPException(status_code=202, detail="Task still pending")
        elif task.status == ProcessingStatus.PROCESSING:
            raise HTTPException(status_code=202, detail="Task still processing")
        elif task.status == ProcessingStatus.FAILED:
            raise HTTPException(status_code=500, detail=task.error or "Processing failed")
        elif task.result:
            return DocumentProcessingResponse(**task.result)
        else:
            raise HTTPException(status_code=404, detail="Result not available")

    # Search endpoints
    @app.post("/search", response_model=VectorSearchResponse)
    async def search_vectors(request: VectorSearchRequest):
        """Search vectors in the vector store."""
        try:
            # Get vector store
            collection_name = request.collection_name or "default"
            store = await api_service.vector_store_manager.get_store(collection_name)

            if not store:
                raise HTTPException(status_code=404, detail=f"Collection '{collection_name}' not found")

            # Create search query
            search_query = SearchQuery(
                query_text=request.query_text,
                query_vector=request.query_vector,
                search_type=SearchType(request.search_type),
                limit=request.limit,
                offset=request.offset,
                similarity_threshold=request.similarity_threshold,
                vector_weight=request.vector_weight,
                keyword_weight=request.keyword_weight,
                include_metadata=request.include_metadata,
                include_vectors=request.include_vectors
            )

            # Add filters if provided
            if any([request.document_ids, request.document_types, request.tags, request.date_range]):
                filters = SearchFilters()

                if request.document_ids:
                    filters.add_list_filter("document_id", request.document_ids)

                if request.document_types:
                    filters.add_list_filter("document_type", request.document_types)

                if request.tags:
                    filters.add_list_filter("tags", request.tags)

                if request.date_range:
                    if "from" in request.date_range:
                        filters.add_filter("created_at", "gte", request.date_range["from"])
                    if "to" in request.date_range:
                        filters.add_filter("created_at", "lte", request.date_range["to"])

                search_query.filters = filters

            # Perform search
            results = await store.search_vectors(search_query)

            # Convert to response format
            search_results = []
            document_counts = {}
            type_counts = {}

            for result in results.results:
                # Extract metadata
                metadata = result.metadata
                document_id = metadata.get("document_id", "unknown")
                document_type = metadata.get("document_type", "unknown")

                # Count by document and type
                document_counts[document_id] = document_counts.get(document_id, 0) + 1
                type_counts[document_type] = type_counts.get(document_type, 0) + 1

                search_result = SearchResultItem(
                    id=result.id,
                    document_id=document_id,
                    chunk_id=result.id,
                    text=result.text if request.include_text else "",
                    score=result.score,
                    rank=result.rank,
                    metadata=metadata if request.include_metadata else {},
                    vector=result.vector if request.include_vectors else None,
                    document_title=metadata.get("document_title"),
                    chunk_index=metadata.get("chunk_index")
                )
                search_results.append(search_result)

            return VectorSearchResponse(
                results=search_results,
                query_text=request.query_text,
                search_type=request.search_type,
                total_results=results.total_results,
                search_time=results.search_time,
                offset=request.offset,
                limit=request.limit,
                has_more=results.has_more,
                document_counts=document_counts,
                type_counts=type_counts
            )

        except Exception as e:
            api_service.logger.error(f"Search failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Statistics and monitoring endpoints
    @app.get("/stats", response_model=PipelineStatus)
    async def get_pipeline_stats():
        """Get pipeline statistics and status."""
        try:
            # Get vector store stats
            store_stats = await api_service.vector_store_manager.get_stats_all()

            # Calculate uptime
            uptime = (datetime.now() - api_service.stats["start_time"]).total_seconds()

            # Get cache stats
            cache_stats = api_service.batch_processor.get_cache_stats()

            return PipelineStatus(
                is_healthy=True,
                status="running",
                document_processor_status="healthy",
                vectorizer_status="healthy",
                vector_store_status="healthy",
                total_documents=api_service.stats["total_documents"],
                total_chunks=api_service.stats["total_chunks"],
                total_vectors=api_service.stats["total_vectors"],
                cache_hit_rate=cache_stats.get("hit_rate") if cache_stats else None,
                cache_size=cache_stats.get("size") if cache_stats else None,
                uptime=uptime
            )

        except Exception as e:
            api_service.logger.error(f"Failed to get stats: {e}")
            return PipelineStatus(
                is_healthy=False,
                status="error",
                document_processor_status="unknown",
                vectorizer_status="unknown",
                vector_store_status="unknown"
            )

    @app.get("/collections")
    async def list_collections():
        """List available vector store collections."""
        try:
            collections = api_service.vector_store_manager.list_stores()

            # Get stats for each collection
            collection_info = {}
            for name, store_type in collections.items():
                store = await api_service.vector_store_manager.get_store(name)
                if store:
                    stats = await store.get_stats()
                    collection_info[name] = {
                        "type": store_type,
                        "total_vectors": stats.total_vectors,
                        "index_type": stats.index_type,
                        "is_trained": stats.is_trained
                    }

            return {
                "collections": collection_info,
                "total_collections": len(collections)
            }

        except Exception as e:
            api_service.logger.error(f"Failed to list collections: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    return app, api_service
