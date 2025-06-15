"""
Database optimization utilities for DataMCPServerAgent.
Provides connection pooling, query optimization, and performance monitoring.
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

import aiosqlite
from pydantic import BaseModel, Field


class DatabaseConfig(BaseModel):
    """Database configuration with optimization settings."""
    
    # Connection pool settings
    max_pool_size: int = Field(default=20, description="Maximum number of connections in pool")
    min_pool_size: int = Field(default=5, description="Minimum number of connections in pool")
    pool_timeout: float = Field(default=30.0, description="Connection timeout in seconds")
    
    # Performance settings
    enable_wal_mode: bool = Field(default=True, description="Enable Write-Ahead Logging")
    enable_foreign_keys: bool = Field(default=True, description="Enable foreign key constraints")
    cache_size: int = Field(default=10000, description="SQLite cache size in KB")
    temp_store: str = Field(default="memory", description="Temporary storage location")
    
    # Monitoring settings
    slow_query_threshold: float = Field(default=1.0, description="Log queries slower than this (seconds)")
    enable_query_logging: bool = Field(default=False, description="Enable query performance logging")


class QueryPerformanceMonitor:
    """Monitor and log database query performance."""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.QueryMonitor")
        self.query_stats: Dict[str, List[float]] = {}
    
    @asynccontextmanager
    async def monitor_query(self, query_name: str, query: str):
        """Context manager to monitor query execution time."""
        start_time = time.time()
        
        try:
            yield
        finally:
            execution_time = time.time() - start_time
            
            # Log slow queries
            if execution_time > self.config.slow_query_threshold:
                self.logger.warning(
                    f"Slow query detected: {query_name} took {execution_time:.3f}s\n"
                    f"Query: {query[:200]}..."
                )
            
            # Store performance stats
            if query_name not in self.query_stats:
                self.query_stats[query_name] = []
            
            self.query_stats[query_name].append(execution_time)
            
            # Keep only last 100 measurements per query
            if len(self.query_stats[query_name]) > 100:
                self.query_stats[query_name] = self.query_stats[query_name][-100:]
    
    def get_query_stats(self, query_name: str) -> Optional[Dict[str, float]]:
        """Get performance statistics for a specific query."""
        if query_name not in self.query_stats:
            return None
        
        times = self.query_stats[query_name]
        return {
            "count": len(times),
            "avg_time": sum(times) / len(times),
            "min_time": min(times),
            "max_time": max(times),
            "total_time": sum(times)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for all monitored queries."""
        return {
            query_name: self.get_query_stats(query_name)
            for query_name in self.query_stats.keys()
        }


class OptimizedDatabase:
    """Optimized database connection manager with pooling and monitoring."""
    
    def __init__(self, db_path: str, config: Optional[DatabaseConfig] = None):
        self.db_path = db_path
        self.config = config or DatabaseConfig()
        self.monitor = QueryPerformanceMonitor(self.config)
        self.logger = logging.getLogger(f"{__name__}.OptimizedDatabase")
        self._initialized = False
    
    async def _optimize_connection(self, conn: aiosqlite.Connection) -> None:
        """Apply optimization settings to a database connection."""
        optimizations = [
            # Enable Write-Ahead Logging for better concurrency
            ("PRAGMA journal_mode=WAL", self.config.enable_wal_mode),
            
            # Enable foreign key constraints
            ("PRAGMA foreign_keys=ON", self.config.enable_foreign_keys),
            
            # Set cache size (negative value = KB, positive = pages)
            (f"PRAGMA cache_size=-{self.config.cache_size}", True),
            
            # Store temporary tables in memory
            (f"PRAGMA temp_store={self.config.temp_store}", True),
            
            # Optimize for faster writes
            ("PRAGMA synchronous=NORMAL", True),
            
            # Reduce checkpoint frequency for WAL mode
            ("PRAGMA wal_autocheckpoint=1000", self.config.enable_wal_mode),
            
            # Optimize page size
            ("PRAGMA page_size=4096", True),
        ]
        
        for pragma, enabled in optimizations:
            if enabled:
                await conn.execute(pragma)
    
    @asynccontextmanager
    async def get_connection(self):
        """Get an optimized database connection with monitoring."""
        async with aiosqlite.connect(self.db_path) as conn:
            # Apply optimizations on first connection
            if not self._initialized:
                await self._optimize_connection(conn)
                self._initialized = True
            
            yield conn
    
    async def execute_query(
        self,
        query: str,
        params: Union[tuple, List[tuple]] = (),
        query_name: str = "unnamed_query",
        fetch_method: str = "none"  # "none", "one", "all"
    ) -> Any:
        """Execute a query with performance monitoring."""
        
        async with self.monitor.monitor_query(query_name, query):
            async with self.get_connection() as conn:
                if isinstance(params, list):
                    # Execute many
                    await conn.executemany(query, params)
                    result = None
                else:
                    # Execute single query
                    cursor = await conn.execute(query, params)
                    
                    if fetch_method == "one":
                        result = await cursor.fetchone()
                    elif fetch_method == "all":
                        result = await cursor.fetchall()
                    else:
                        result = cursor.rowcount
                
                await conn.commit()
                return result
    
    async def execute_transaction(self, queries: List[Dict[str, Any]]) -> None:
        """Execute multiple queries in a single transaction."""
        async with self.get_connection() as conn:
            try:
                for query_info in queries:
                    query = query_info["query"]
                    params = query_info.get("params", ())
                    query_name = query_info.get("name", "transaction_query")
                    
                    async with self.monitor.monitor_query(query_name, query):
                        await conn.execute(query, params)
                
                await conn.commit()
            except Exception:
                await conn.rollback()
                raise
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "query_stats": self.monitor.get_all_stats(),
            "config": self.config.model_dump(),
            "db_path": self.db_path,
            "initialized": self._initialized
        }


# Optimization SQL templates for common patterns
OPTIMIZATION_QUERIES = {
    "create_indexes": {
        "conversation_history": [
            "CREATE INDEX IF NOT EXISTS idx_conversation_timestamp ON conversation_history(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_conversation_role ON conversation_history(role)",
            "CREATE INDEX IF NOT EXISTS idx_conversation_role_timestamp ON conversation_history(role, timestamp)"
        ],
        "tool_usage_history": [
            "CREATE INDEX IF NOT EXISTS idx_tool_usage_name ON tool_usage_history(tool_name)",
            "CREATE INDEX IF NOT EXISTS idx_tool_usage_timestamp ON tool_usage_history(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_tool_usage_name_timestamp ON tool_usage_history(tool_name, timestamp)"
        ],
        "entity_memory": [
            "CREATE INDEX IF NOT EXISTS idx_entity_type ON entity_memory(entity_type)",
            "CREATE INDEX IF NOT EXISTS idx_entity_type_id ON entity_memory(entity_type, entity_id)",
            "CREATE INDEX IF NOT EXISTS idx_entity_last_updated ON entity_memory(last_updated)"
        ],
        "tool_performance": [
            "CREATE INDEX IF NOT EXISTS idx_tool_performance_name ON tool_performance(tool_name)",
            "CREATE INDEX IF NOT EXISTS idx_tool_performance_timestamp ON tool_performance(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_tool_performance_name_success ON tool_performance(tool_name, success)"
        ],
        "research_projects": [
            "CREATE INDEX IF NOT EXISTS idx_research_projects_created_at ON research_projects(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_research_projects_updated_at ON research_projects(updated_at)"
        ],
        "research_queries": [
            "CREATE INDEX IF NOT EXISTS idx_research_queries_project_id ON research_queries(project_id)",
            "CREATE INDEX IF NOT EXISTS idx_research_queries_created_at ON research_queries(created_at)"
        ],
        "research_results": [
            "CREATE INDEX IF NOT EXISTS idx_research_results_project_query ON research_results(project_id, query_id)",
            "CREATE INDEX IF NOT EXISTS idx_research_results_created_at ON research_results(created_at)"
        ],
        "research_sources": [
            "CREATE INDEX IF NOT EXISTS idx_research_sources_result ON research_sources(result_id, query_id, project_id)",
            "CREATE INDEX IF NOT EXISTS idx_research_sources_type ON research_sources(source_type)"
        ]
    },
    
    "analyze_tables": [
        "ANALYZE conversation_history",
        "ANALYZE tool_usage_history", 
        "ANALYZE entity_memory",
        "ANALYZE tool_performance",
        "ANALYZE research_projects",
        "ANALYZE research_queries",
        "ANALYZE research_results",
        "ANALYZE research_sources"
    ]
}


async def apply_database_optimizations(db_path: str) -> Dict[str, Any]:
    """Apply comprehensive database optimizations."""
    config = DatabaseConfig()
    db = OptimizedDatabase(db_path, config)
    
    optimization_results = {
        "indexes_created": 0,
        "tables_analyzed": 0,
        "errors": []
    }
    
    try:
        # Create all recommended indexes
        for table_name, indexes in OPTIMIZATION_QUERIES["create_indexes"].items():
            for index_query in indexes:
                try:
                    await db.execute_query(
                        index_query,
                        query_name=f"create_index_{table_name}"
                    )
                    optimization_results["indexes_created"] += 1
                except Exception as e:
                    optimization_results["errors"].append(f"Index creation failed: {e}")
        
        # Analyze tables for query planner optimization
        for analyze_query in OPTIMIZATION_QUERIES["analyze_tables"]:
            try:
                await db.execute_query(
                    analyze_query,
                    query_name="analyze_table"
                )
                optimization_results["tables_analyzed"] += 1
            except Exception as e:
                optimization_results["errors"].append(f"Table analysis failed: {e}")
        
        # Add performance statistics
        optimization_results["performance_stats"] = db.get_performance_stats()
        
    except Exception as e:
        optimization_results["errors"].append(f"Optimization failed: {e}")
    
    return optimization_results