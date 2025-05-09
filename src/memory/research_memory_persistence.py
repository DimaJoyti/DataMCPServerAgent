"""
Research memory persistence module for DataMCPServerAgent.
This module provides database integration for persisting research data including
projects, queries, results, sources, and visualizations.
"""

import json
import sqlite3
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Source(BaseModel):
    """Source model for research results."""

    title: str
    url: Optional[str] = None
    authors: Optional[List[str]] = None
    source_type: str = "web"
    publisher: Optional[str] = None
    isbn: Optional[str] = None
    doi: Optional[str] = None
    year: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert source to dictionary."""
        return self.model_dump(exclude_none=True)


class ResearchResult(BaseModel):
    """Research result model."""

    id: str
    topic: str
    summary: str
    sources: List[Source]
    tools_used: List[str]
    citation_format: Optional[str] = None
    bibliography: Optional[str] = None
    visualizations: List[Dict[str, Any]] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert research result to dictionary."""
        result = self.model_dump(exclude_none=True)
        result["sources"] = [source.to_dict() for source in self.sources]
        result["created_at"] = result["created_at"].isoformat()
        return result


class ResearchQuery(BaseModel):
    """Research query model."""

    id: str
    query: str
    results: List[ResearchResult] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert research query to dictionary."""
        result = self.model_dump(exclude_none=True)
        result["results"] = [r.to_dict() for r in self.results]
        result["created_at"] = result["created_at"].isoformat()
        return result

    def add_result(self, result: ResearchResult) -> None:
        """Add a result to the query."""
        self.results.append(result)


class ResearchProject(BaseModel):
    """Research project model."""

    id: str
    name: str
    description: str = ""
    tags: List[str] = Field(default_factory=list)
    queries: List[ResearchQuery] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert research project to dictionary."""
        result = self.model_dump(exclude_none=True)
        result["queries"] = [q.to_dict() for q in self.queries]
        result["created_at"] = result["created_at"].isoformat()
        result["updated_at"] = result["updated_at"].isoformat()
        return result

    def add_query(self, query: str) -> ResearchQuery:
        """Add a query to the project."""
        query_id = f"query_{len(self.queries) + 1}"
        research_query = ResearchQuery(id=query_id, query=query)
        self.queries.append(research_query)
        self.updated_at = datetime.now()
        return research_query

    def add_result(self, query_id: str, result: ResearchResult) -> bool:
        """Add a result to a query in the project."""
        for query in self.queries:
            if query.id == query_id:
                query.add_result(result)
                self.updated_at = datetime.now()
                return True
        return False


class ResearchMemoryDatabase:
    """Database for persisting research memory."""

    def __init__(self, db_path: str = "research_memory.db"):
        """Initialize the research memory database.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self) -> None:
        """Initialize the database schema if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create research projects table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS research_projects (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            description TEXT,
            tags TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """)

        # Create research queries table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS research_queries (
            id TEXT NOT NULL,
            project_id TEXT NOT NULL,
            query TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (id, project_id),
            FOREIGN KEY (project_id) REFERENCES research_projects (id) ON DELETE CASCADE
        )
        """)

        # Create research results table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS research_results (
            id TEXT NOT NULL,
            query_id TEXT NOT NULL,
            project_id TEXT NOT NULL,
            topic TEXT NOT NULL,
            summary TEXT NOT NULL,
            tools_used TEXT NOT NULL,
            citation_format TEXT,
            bibliography TEXT,
            tags TEXT,
            created_at REAL NOT NULL,
            PRIMARY KEY (id, query_id, project_id),
            FOREIGN KEY (query_id, project_id) REFERENCES research_queries (id, project_id) ON DELETE CASCADE
        )
        """)

        # Create research sources table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS research_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            result_id TEXT NOT NULL,
            query_id TEXT NOT NULL,
            project_id TEXT NOT NULL,
            title TEXT NOT NULL,
            url TEXT,
            authors TEXT,
            source_type TEXT NOT NULL,
            publisher TEXT,
            isbn TEXT,
            doi TEXT,
            year INTEGER,
            FOREIGN KEY (result_id, query_id, project_id) REFERENCES research_results (id, query_id, project_id) ON DELETE CASCADE
        )
        """)

        # Create research visualizations table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS research_visualizations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            result_id TEXT NOT NULL,
            query_id TEXT NOT NULL,
            project_id TEXT NOT NULL,
            visualization_data TEXT NOT NULL,
            FOREIGN KEY (result_id, query_id, project_id) REFERENCES research_results (id, query_id, project_id) ON DELETE CASCADE
        )
        """)

        # Create research tool usage table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS research_tool_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            project_id TEXT NOT NULL,
            query_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            args TEXT NOT NULL,
            result TEXT NOT NULL,
            execution_time REAL NOT NULL,
            timestamp REAL NOT NULL,
            FOREIGN KEY (query_id, project_id) REFERENCES research_queries (id, project_id) ON DELETE CASCADE
        )
        """)

        conn.commit()
        conn.close()

    def create_project(
        self, name: str, description: str = "", tags: List[str] = None
    ) -> ResearchProject:
        """Create a new research project.

        Args:
            name: Project name
            description: Project description
            tags: Project tags

        Returns:
            Created project
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Generate project ID
        cursor.execute("SELECT COUNT(*) FROM research_projects")
        count = cursor.fetchone()[0]
        project_id = f"project_{count + 1}"

        # Create project
        now = time.time()
        cursor.execute(
            "INSERT INTO research_projects (id, name, description, tags, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (project_id, name, description, json.dumps(tags or []), now, now),
        )

        conn.commit()
        conn.close()

        return ResearchProject(
            id=project_id,
            name=name,
            description=description,
            tags=tags or [],
            created_at=datetime.fromtimestamp(now),
            updated_at=datetime.fromtimestamp(now),
        )

    def get_project(self, project_id: str) -> Optional[ResearchProject]:
        """Get a research project by ID.

        Args:
            project_id: Project ID

        Returns:
            Project or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get project
        cursor.execute(
            "SELECT id, name, description, tags, created_at, updated_at FROM research_projects WHERE id = ?",
            (project_id,),
        )

        row = cursor.fetchone()
        if not row:
            conn.close()
            return None

        project_id, name, description, tags_json, created_at, updated_at = row

        # Get queries for this project
        cursor.execute(
            "SELECT id, query, created_at FROM research_queries WHERE project_id = ?",
            (project_id,),
        )

        queries = []
        for query_row in cursor.fetchall():
            query_id, query_text, query_created_at = query_row

            # Get results for this query
            cursor.execute(
                """
                SELECT id, topic, summary, tools_used, citation_format, bibliography, tags, created_at
                FROM research_results
                WHERE project_id = ? AND query_id = ?
                """,
                (project_id, query_id),
            )

            results = []
            for result_row in cursor.fetchall():
                (
                    result_id,
                    topic,
                    summary,
                    tools_used_json,
                    citation_format,
                    bibliography,
                    tags_json,
                    result_created_at,
                ) = result_row

                # Get sources for this result
                cursor.execute(
                    """
                    SELECT title, url, authors, source_type, publisher, isbn, doi, year
                    FROM research_sources
                    WHERE project_id = ? AND query_id = ? AND result_id = ?
                    """,
                    (project_id, query_id, result_id),
                )

                sources = []
                for source_row in cursor.fetchall():
                    (
                        title,
                        url,
                        authors_json,
                        source_type,
                        publisher,
                        isbn,
                        doi,
                        year,
                    ) = source_row

                    source = Source(
                        title=title,
                        url=url,
                        authors=json.loads(authors_json) if authors_json else None,
                        source_type=source_type,
                        publisher=publisher,
                        isbn=isbn,
                        doi=doi,
                        year=year,
                    )
                    sources.append(source)

                # Get visualizations for this result
                cursor.execute(
                    """
                    SELECT visualization_data
                    FROM research_visualizations
                    WHERE project_id = ? AND query_id = ? AND result_id = ?
                    """,
                    (project_id, query_id, result_id),
                )

                visualizations = []
                for viz_row in cursor.fetchall():
                    visualization_data = json.loads(viz_row[0])
                    visualizations.append(visualization_data)

                # Create result
                result = ResearchResult(
                    id=result_id,
                    topic=topic,
                    summary=summary,
                    sources=sources,
                    tools_used=json.loads(tools_used_json),
                    citation_format=citation_format,
                    bibliography=bibliography,
                    visualizations=visualizations,
                    tags=json.loads(tags_json) if tags_json else [],
                    created_at=datetime.fromtimestamp(result_created_at),
                )
                results.append(result)

            # Create query
            research_query = ResearchQuery(
                id=query_id,
                query=query_text,
                results=results,
                created_at=datetime.fromtimestamp(query_created_at),
            )
            queries.append(research_query)

        conn.close()

        # Create project
        return ResearchProject(
            id=project_id,
            name=name,
            description=description,
            tags=json.loads(tags_json) if tags_json else [],
            queries=queries,
            created_at=datetime.fromtimestamp(created_at),
            updated_at=datetime.fromtimestamp(updated_at),
        )

    def get_all_projects(self) -> List[ResearchProject]:
        """Get all research projects.

        Returns:
            List of projects
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all projects
        cursor.execute(
            "SELECT id, name, description, tags, created_at, updated_at FROM research_projects ORDER BY created_at DESC"
        )

        projects = []
        for row in cursor.fetchall():
            project_id, name, description, tags_json, created_at, updated_at = row

            # Create project (without queries for efficiency)
            project = ResearchProject(
                id=project_id,
                name=name,
                description=description,
                tags=json.loads(tags_json) if tags_json else [],
                created_at=datetime.fromtimestamp(created_at),
                updated_at=datetime.fromtimestamp(updated_at),
            )
            projects.append(project)

        conn.close()
        return projects

    def add_query(self, project_id: str, query: str) -> Optional[ResearchQuery]:
        """Add a query to a project.

        Args:
            project_id: Project ID
            query: Query text

        Returns:
            Created query or None if project not found
        """
        # Check if project exists
        project = self.get_project(project_id)
        if not project:
            return None

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Generate query ID
        cursor.execute(
            "SELECT COUNT(*) FROM research_queries WHERE project_id = ?", (project_id,)
        )
        count = cursor.fetchone()[0]
        query_id = f"query_{count + 1}"

        # Create query
        now = time.time()
        cursor.execute(
            "INSERT INTO research_queries (id, project_id, query, created_at) VALUES (?, ?, ?, ?)",
            (query_id, project_id, query, now),
        )

        # Update project updated_at
        cursor.execute(
            "UPDATE research_projects SET updated_at = ? WHERE id = ?",
            (now, project_id),
        )

        conn.commit()
        conn.close()

        return ResearchQuery(
            id=query_id, query=query, created_at=datetime.fromtimestamp(now)
        )

    def add_result(
        self, project_id: str, query_id: str, result: ResearchResult
    ) -> bool:
        """Add a result to a query.

        Args:
            project_id: Project ID
            query_id: Query ID
            result: Research result

        Returns:
            True if successful, False otherwise
        """
        # Check if project and query exist
        project = self.get_project(project_id)
        if not project:
            return False

        query_exists = False
        for query in project.queries:
            if query.id == query_id:
                query_exists = True
                break

        if not query_exists:
            return False

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Insert result
            now = time.time()
            cursor.execute(
                """
                INSERT INTO research_results
                (id, query_id, project_id, topic, summary, tools_used, citation_format, bibliography, tags, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.id,
                    query_id,
                    project_id,
                    result.topic,
                    result.summary,
                    json.dumps(result.tools_used),
                    result.citation_format,
                    result.bibliography,
                    json.dumps(result.tags),
                    now,
                ),
            )

            # Insert sources
            for source in result.sources:
                cursor.execute(
                    """
                    INSERT INTO research_sources
                    (result_id, query_id, project_id, title, url, authors, source_type, publisher, isbn, doi, year)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        result.id,
                        query_id,
                        project_id,
                        source.title,
                        source.url,
                        json.dumps(source.authors) if source.authors else None,
                        source.source_type,
                        source.publisher,
                        source.isbn,
                        source.doi,
                        source.year,
                    ),
                )

            # Insert visualizations
            for visualization in result.visualizations:
                cursor.execute(
                    """
                    INSERT INTO research_visualizations
                    (result_id, query_id, project_id, visualization_data)
                    VALUES (?, ?, ?, ?)
                    """,
                    (result.id, query_id, project_id, json.dumps(visualization)),
                )

            # Update project updated_at
            cursor.execute(
                "UPDATE research_projects SET updated_at = ? WHERE id = ?",
                (now, project_id),
            )

            conn.commit()
            return True

        except Exception as e:
            print(f"Error adding result: {e}")
            conn.rollback()
            return False

        finally:
            conn.close()

    def save_tool_usage(
        self,
        project_id: str,
        query_id: str,
        tool_name: str,
        args: Dict[str, Any],
        result: Any,
        execution_time: float,
    ) -> bool:
        """Save tool usage for a research query.

        Args:
            project_id: Project ID
            query_id: Query ID
            tool_name: Name of the tool used
            args: Arguments passed to the tool
            result: Result returned by the tool
            execution_time: Time taken to execute the tool in seconds

        Returns:
            True if successful, False otherwise
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Insert tool usage
            cursor.execute(
                """
                INSERT INTO research_tool_usage
                (project_id, query_id, tool_name, args, result, execution_time, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    project_id,
                    query_id,
                    tool_name,
                    json.dumps(args),
                    json.dumps(str(result)),
                    execution_time,
                    time.time(),
                ),
            )

            conn.commit()
            return True

        except Exception as e:
            print(f"Error saving tool usage: {e}")
            conn.rollback()
            return False

        finally:
            conn.close()

    def get_tool_usage(
        self, project_id: str, query_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get tool usage for a project or query.

        Args:
            project_id: Project ID
            query_id: Optional query ID to filter by

        Returns:
            List of tool usage entries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if query_id:
            cursor.execute(
                """
                SELECT tool_name, args, result, execution_time, timestamp
                FROM research_tool_usage
                WHERE project_id = ? AND query_id = ?
                ORDER BY timestamp DESC
                """,
                (project_id, query_id),
            )
        else:
            cursor.execute(
                """
                SELECT query_id, tool_name, args, result, execution_time, timestamp
                FROM research_tool_usage
                WHERE project_id = ?
                ORDER BY timestamp DESC
                """,
                (project_id,),
            )

        rows = cursor.fetchall()
        conn.close()

        result = []
        if query_id:
            for tool_name, args, res, execution_time, timestamp in rows:
                result.append(
                    {
                        "tool_name": tool_name,
                        "args": json.loads(args),
                        "result": json.loads(res),
                        "execution_time": execution_time,
                        "timestamp": timestamp,
                    }
                )
        else:
            for q_id, tool_name, args, res, execution_time, timestamp in rows:
                result.append(
                    {
                        "query_id": q_id,
                        "tool_name": tool_name,
                        "args": json.loads(args),
                        "result": json.loads(res),
                        "execution_time": execution_time,
                        "timestamp": timestamp,
                    }
                )

        return result

    def search_projects(self, search_term: str) -> List[ResearchProject]:
        """Search for projects by name, description, or tags.

        Args:
            search_term: Search term

        Returns:
            List of matching projects
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Search for projects
        cursor.execute(
            """
            SELECT id, name, description, tags, created_at, updated_at
            FROM research_projects
            WHERE name LIKE ? OR description LIKE ?
            ORDER BY updated_at DESC
            """,
            (f"%{search_term}%", f"%{search_term}%"),
        )

        projects = []
        for row in cursor.fetchall():
            project_id, name, description, tags_json, created_at, updated_at = row

            # Check if search term is in tags
            tags = json.loads(tags_json) if tags_json else []
            if not any(search_term.lower() in tag.lower() for tag in tags):
                # Create project (without queries for efficiency)
                project = ResearchProject(
                    id=project_id,
                    name=name,
                    description=description,
                    tags=tags,
                    created_at=datetime.fromtimestamp(created_at),
                    updated_at=datetime.fromtimestamp(updated_at),
                )
                projects.append(project)

        conn.close()
        return projects

    def search_queries(
        self, search_term: str, project_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for queries by text.

        Args:
            search_term: Search term
            project_id: Optional project ID to filter by

        Returns:
            List of matching queries with project information
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if project_id:
            cursor.execute(
                """
                SELECT q.id, q.project_id, q.query, q.created_at, p.name
                FROM research_queries q
                JOIN research_projects p ON q.project_id = p.id
                WHERE q.project_id = ? AND q.query LIKE ?
                ORDER BY q.created_at DESC
                """,
                (project_id, f"%{search_term}%"),
            )
        else:
            cursor.execute(
                """
                SELECT q.id, q.project_id, q.query, q.created_at, p.name
                FROM research_queries q
                JOIN research_projects p ON q.project_id = p.id
                WHERE q.query LIKE ?
                ORDER BY q.created_at DESC
                """,
                (f"%{search_term}%",),
            )

        rows = cursor.fetchall()
        conn.close()

        result = []
        for query_id, proj_id, query, created_at, project_name in rows:
            result.append(
                {
                    "query_id": query_id,
                    "project_id": proj_id,
                    "project_name": project_name,
                    "query": query,
                    "created_at": datetime.fromtimestamp(created_at),
                }
            )

        return result

    def search_results(
        self, search_term: str, project_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for results by topic, summary, or tags.

        Args:
            search_term: Search term
            project_id: Optional project ID to filter by

        Returns:
            List of matching results with project and query information
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if project_id:
            cursor.execute(
                """
                SELECT r.id, r.query_id, r.project_id, r.topic, r.summary, r.tags, r.created_at,
                       q.query, p.name
                FROM research_results r
                JOIN research_queries q ON r.query_id = q.id AND r.project_id = q.project_id
                JOIN research_projects p ON r.project_id = p.id
                WHERE r.project_id = ? AND (r.topic LIKE ? OR r.summary LIKE ?)
                ORDER BY r.created_at DESC
                """,
                (project_id, f"%{search_term}%", f"%{search_term}%"),
            )
        else:
            cursor.execute(
                """
                SELECT r.id, r.query_id, r.project_id, r.topic, r.summary, r.tags, r.created_at,
                       q.query, p.name
                FROM research_results r
                JOIN research_queries q ON r.query_id = q.id AND r.project_id = q.project_id
                JOIN research_projects p ON r.project_id = p.id
                WHERE r.topic LIKE ? OR r.summary LIKE ?
                ORDER BY r.created_at DESC
                """,
                (f"%{search_term}%", f"%{search_term}%"),
            )

        rows = cursor.fetchall()

        result = []
        for (
            result_id,
            query_id,
            proj_id,
            topic,
            summary,
            tags_json,
            created_at,
            query,
            project_name,
        ) in rows:
            # Check if search term is in tags
            tags = json.loads(tags_json) if tags_json else []
            if (
                search_term.lower() in topic.lower()
                or search_term.lower() in summary.lower()
                or any(search_term.lower() in tag.lower() for tag in tags)
            ):
                result.append(
                    {
                        "result_id": result_id,
                        "query_id": query_id,
                        "project_id": proj_id,
                        "project_name": project_name,
                        "query": query,
                        "topic": topic,
                        "summary": summary,
                        "tags": tags,
                        "created_at": datetime.fromtimestamp(created_at),
                    }
                )

        conn.close()
        return result
