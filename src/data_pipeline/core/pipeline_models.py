"""
Core data models for the data pipeline system.

This module defines the fundamental data structures used throughout the pipeline system.
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class PipelineStatus(str, Enum):
    """Pipeline execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"

class TaskStatus(str, Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRY = "retry"

class TaskType(str, Enum):
    """Types of pipeline tasks."""
    INGESTION = "ingestion"
    TRANSFORMATION = "transformation"
    VALIDATION = "validation"
    PROCESSING = "processing"
    EXPORT = "export"
    CUSTOM = "custom"

class DataSourceType(str, Enum):
    """Types of data sources."""
    DATABASE = "database"
    FILE = "file"
    API = "api"
    STREAM = "stream"
    QUEUE = "queue"
    OBJECT_STORAGE = "object_storage"

class TaskConfig(BaseModel):
    """Configuration for a pipeline task."""
    task_id: str = Field(..., description="Unique task identifier")
    task_type: TaskType = Field(..., description="Type of task")
    name: str = Field(..., description="Human-readable task name")
    description: Optional[str] = Field(None, description="Task description")

    # Task execution configuration
    command: Optional[str] = Field(None, description="Command to execute")
    function: Optional[str] = Field(None, description="Python function to call")
    module: Optional[str] = Field(None, description="Python module containing function")

    # Dependencies and scheduling
    depends_on: List[str] = Field(default_factory=list, description="Task dependencies")
    retry_count: int = Field(default=3, description="Number of retries on failure")
    retry_delay: int = Field(default=60, description="Delay between retries in seconds")
    timeout: Optional[int] = Field(None, description="Task timeout in seconds")

    # Resource requirements
    cpu_limit: Optional[float] = Field(None, description="CPU limit")
    memory_limit: Optional[str] = Field(None, description="Memory limit")

    # Task-specific parameters
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Task parameters")

    # Data source/destination configuration
    input_sources: List[Dict[str, Any]] = Field(default_factory=list, description="Input data sources")
    output_destinations: List[Dict[str, Any]] = Field(default_factory=list, description="Output destinations")

class PipelineConfig(BaseModel):
    """Configuration for a data pipeline."""
    pipeline_id: str = Field(..., description="Unique pipeline identifier")
    name: str = Field(..., description="Human-readable pipeline name")
    description: Optional[str] = Field(None, description="Pipeline description")
    version: str = Field(default="1.0.0", description="Pipeline version")

    # Pipeline scheduling
    schedule: Optional[str] = Field(None, description="Cron expression for scheduling")
    timezone: str = Field(default="UTC", description="Timezone for scheduling")

    # Pipeline configuration
    max_parallel_tasks: int = Field(default=10, description="Maximum parallel tasks")
    default_retry_count: int = Field(default=3, description="Default retry count for tasks")
    pipeline_timeout: Optional[int] = Field(None, description="Pipeline timeout in seconds")

    # Tasks configuration
    tasks: List[TaskConfig] = Field(..., description="Pipeline tasks")

    # Global parameters
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Global pipeline parameters")

    # Notification configuration
    notifications: Dict[str, Any] = Field(default_factory=dict, description="Notification settings")

    # Tags and metadata
    tags: List[str] = Field(default_factory=list, description="Pipeline tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class PipelineTask(BaseModel):
    """Runtime representation of a pipeline task."""
    task_id: str = Field(..., description="Unique task identifier")
    pipeline_id: str = Field(..., description="Parent pipeline identifier")
    run_id: str = Field(..., description="Pipeline run identifier")

    # Task configuration
    config: TaskConfig = Field(..., description="Task configuration")

    # Runtime state
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current task status")
    start_time: Optional[datetime] = Field(None, description="Task start time")
    end_time: Optional[datetime] = Field(None, description="Task end time")
    duration: Optional[float] = Field(None, description="Task duration in seconds")

    # Execution details
    attempt_count: int = Field(default=0, description="Number of execution attempts")
    worker_id: Optional[str] = Field(None, description="Worker executing the task")
    process_id: Optional[int] = Field(None, description="Process ID")

    # Results and logs
    result: Optional[Dict[str, Any]] = Field(None, description="Task execution result")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    logs: List[str] = Field(default_factory=list, description="Task execution logs")

    # Metrics
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Task execution metrics")

class PipelineRun(BaseModel):
    """Runtime representation of a pipeline execution."""
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique run identifier")
    pipeline_id: str = Field(..., description="Pipeline identifier")

    # Pipeline configuration snapshot
    config: PipelineConfig = Field(..., description="Pipeline configuration")

    # Runtime state
    status: PipelineStatus = Field(default=PipelineStatus.PENDING, description="Current pipeline status")
    start_time: Optional[datetime] = Field(None, description="Pipeline start time")
    end_time: Optional[datetime] = Field(None, description="Pipeline end time")
    duration: Optional[float] = Field(None, description="Pipeline duration in seconds")

    # Execution details
    triggered_by: Optional[str] = Field(None, description="What triggered this run")
    trigger_time: datetime = Field(default_factory=datetime.utcnow, description="When the run was triggered")

    # Tasks
    tasks: List[PipelineTask] = Field(default_factory=list, description="Pipeline tasks")

    # Results and metrics
    result: Optional[Dict[str, Any]] = Field(None, description="Pipeline execution result")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Pipeline execution metrics")

    # Runtime parameters
    runtime_parameters: Dict[str, Any] = Field(default_factory=dict, description="Runtime parameters")

class Pipeline(BaseModel):
    """Complete pipeline definition."""
    pipeline_id: str = Field(..., description="Unique pipeline identifier")
    config: PipelineConfig = Field(..., description="Pipeline configuration")

    # Pipeline metadata
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")
    created_by: Optional[str] = Field(None, description="Creator identifier")

    # Pipeline state
    is_active: bool = Field(default=True, description="Whether pipeline is active")
    last_run_id: Optional[str] = Field(None, description="Last execution run ID")
    last_run_status: Optional[PipelineStatus] = Field(None, description="Last execution status")
    last_run_time: Optional[datetime] = Field(None, description="Last execution time")

    # Statistics
    total_runs: int = Field(default=0, description="Total number of runs")
    successful_runs: int = Field(default=0, description="Number of successful runs")
    failed_runs: int = Field(default=0, description="Number of failed runs")

    # Next scheduled run
    next_run_time: Optional[datetime] = Field(None, description="Next scheduled run time")

class DataSource(BaseModel):
    """Data source configuration."""
    source_id: str = Field(..., description="Unique source identifier")
    name: str = Field(..., description="Human-readable source name")
    source_type: DataSourceType = Field(..., description="Type of data source")

    # Connection configuration
    connection_config: Dict[str, Any] = Field(..., description="Connection configuration")

    # Data configuration
    schema_config: Optional[Dict[str, Any]] = Field(None, description="Schema configuration")
    format_config: Optional[Dict[str, Any]] = Field(None, description="Data format configuration")

    # Access configuration
    credentials: Optional[Dict[str, Any]] = Field(None, description="Access credentials")

    # Metadata
    description: Optional[str] = Field(None, description="Source description")
    tags: List[str] = Field(default_factory=list, description="Source tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class DataDestination(BaseModel):
    """Data destination configuration."""
    destination_id: str = Field(..., description="Unique destination identifier")
    name: str = Field(..., description="Human-readable destination name")
    destination_type: DataSourceType = Field(..., description="Type of data destination")

    # Connection configuration
    connection_config: Dict[str, Any] = Field(..., description="Connection configuration")

    # Data configuration
    schema_config: Optional[Dict[str, Any]] = Field(None, description="Schema configuration")
    format_config: Optional[Dict[str, Any]] = Field(None, description="Data format configuration")

    # Write configuration
    write_mode: str = Field(default="append", description="Write mode (append, overwrite, upsert)")
    partition_config: Optional[Dict[str, Any]] = Field(None, description="Partitioning configuration")

    # Metadata
    description: Optional[str] = Field(None, description="Destination description")
    tags: List[str] = Field(default_factory=list, description="Destination tags")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class ValidationRule(BaseModel):
    """Data validation rule."""
    rule_id: str = Field(..., description="Unique rule identifier")
    name: str = Field(..., description="Human-readable rule name")
    rule_type: str = Field(..., description="Type of validation rule")

    # Rule configuration
    column: Optional[str] = Field(None, description="Column to validate")
    condition: str = Field(..., description="Validation condition")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Rule parameters")

    # Severity and actions
    severity: str = Field(default="error", description="Rule severity (error, warning, info)")
    action_on_failure: str = Field(default="fail", description="Action on validation failure")

    # Metadata
    description: Optional[str] = Field(None, description="Rule description")
    tags: List[str] = Field(default_factory=list, description="Rule tags")

class QualityMetrics(BaseModel):
    """Data quality metrics."""
    total_records: int = Field(..., description="Total number of records")
    valid_records: int = Field(..., description="Number of valid records")
    invalid_records: int = Field(..., description="Number of invalid records")

    # Quality scores
    completeness_score: float = Field(..., description="Completeness score (0-1)")
    validity_score: float = Field(..., description="Validity score (0-1)")
    consistency_score: float = Field(..., description="Consistency score (0-1)")
    accuracy_score: Optional[float] = Field(None, description="Accuracy score (0-1)")

    # Detailed metrics
    null_count: int = Field(default=0, description="Number of null values")
    duplicate_count: int = Field(default=0, description="Number of duplicate records")
    outlier_count: int = Field(default=0, description="Number of outliers")

    # Rule violations
    rule_violations: Dict[str, int] = Field(default_factory=dict, description="Rule violation counts")

    # Timestamps
    measured_at: datetime = Field(default_factory=datetime.utcnow, description="Measurement timestamp")
    data_timestamp: Optional[datetime] = Field(None, description="Data timestamp")
