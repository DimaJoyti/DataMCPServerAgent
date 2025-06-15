"""
Asynchronous processing components for document pipeline.

This module provides async versions of document processing components
for improved performance and scalability.
"""

from .async_batch_processor import AsyncBatchProcessor
from .async_document_processor import AsyncDocumentProcessor
from .distributed_processor import DistributedProcessor
from .task_queue import TaskManager, TaskQueue
from .worker_pool import WorkerPool

__version__ = "1.0.0"
__author__ = "DataMCPServerAgent Team"

__all__ = [
    "AsyncDocumentProcessor",
    "AsyncBatchProcessor",
    "DistributedProcessor",
    "TaskQueue",
    "TaskManager",
    "WorkerPool",
]
