"""
Research models for the Research Assistant.

This module defines the data models used by the Research Assistant for managing
research projects, queries, results, and related entities.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field


class CitationFormat(str, Enum):
    """Supported citation formats."""
    APA = "apa"
    MLA = "mla"
    CHICAGO = "chicago"
    HARVARD = "harvard"
    IEEE = "ieee"


class SourceType(str, Enum):
    """Types of research sources."""
    WEB = "web"
    WIKIPEDIA = "wikipedia"
    ACADEMIC = "academic"
    BOOK = "book"
    JOURNAL = "journal"
    CONFERENCE = "conference"
    PREPRINT = "preprint"
    NEWS = "news"
    OTHER = "other"


class Source(BaseModel):
    """A research source with detailed information."""
    title: str
    url: Optional[str] = None
    authors: Optional[List[str]] = None
    publication_date: Optional[datetime] = None
    source_type: SourceType = SourceType.WEB
    publisher: Optional[str] = None
    journal: Optional[str] = None
    volume: Optional[str] = None
    issue: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    isbn: Optional[str] = None
    
    def format_citation(self, format: CitationFormat) -> str:
        """Format the source as a citation in the specified format."""
        if format == CitationFormat.APA:
            return self._format_apa()
        elif format == CitationFormat.MLA:
            return self._format_mla()
        elif format == CitationFormat.CHICAGO:
            return self._format_chicago()
        elif format == CitationFormat.HARVARD:
            return self._format_harvard()
        elif format == CitationFormat.IEEE:
            return self._format_ieee()
        else:
            return f"{self.title} - {self.url}"
    
    def _format_apa(self) -> str:
        """Format the source in APA style."""
        if not self.authors:
            author_text = "No author"
        elif len(self.authors) == 1:
            author_text = f"{self.authors[0]}"
        elif len(self.authors) == 2:
            author_text = f"{self.authors[0]} & {self.authors[1]}"
        else:
            author_text = f"{self.authors[0]} et al."
        
        year = self.publication_date.year if self.publication_date else "n.d."
        
        if self.source_type == SourceType.WEB:
            return f"{author_text}. ({year}). {self.title}. {self.publisher or 'Retrieved from'} {self.url}"
        elif self.source_type == SourceType.JOURNAL:
            return f"{author_text}. ({year}). {self.title}. {self.journal}, {self.volume}({self.issue}), {self.pages}."
        else:
            return f"{author_text}. ({year}). {self.title}."
    
    def _format_mla(self) -> str:
        """Format the source in MLA style."""
        if not self.authors:
            author_text = "No author"
        elif len(self.authors) == 1:
            author_text = f"{self.authors[0]}"
        elif len(self.authors) == 2:
            author_text = f"{self.authors[0]} and {self.authors[1]}"
        else:
            author_text = f"{self.authors[0]} et al."
        
        if self.source_type == SourceType.WEB:
            date = self.publication_date.strftime("%d %b. %Y") if self.publication_date else "n.d."
            return f"{author_text}. \"{self.title}.\" {self.publisher or ''}, {date}, {self.url}."
        elif self.source_type == SourceType.JOURNAL:
            date = self.publication_date.strftime("%Y") if self.publication_date else "n.d."
            return f"{author_text}. \"{self.title}.\" {self.journal}, vol. {self.volume}, no. {self.issue}, {date}, pp. {self.pages}."
        else:
            return f"{author_text}. \"{self.title}.\""
    
    def _format_chicago(self) -> str:
        """Format the source in Chicago style."""
        # Simplified implementation
        if not self.authors:
            author_text = "No author"
        elif len(self.authors) == 1:
            author_text = f"{self.authors[0]}"
        else:
            author_text = f"{self.authors[0]} et al."
        
        year = self.publication_date.year if self.publication_date else "n.d."
        
        return f"{author_text}. {self.title}. {year}."
    
    def _format_harvard(self) -> str:
        """Format the source in Harvard style."""
        # Simplified implementation
        if not self.authors:
            author_text = "No author"
        elif len(self.authors) == 1:
            author_text = f"{self.authors[0]}"
        else:
            author_text = f"{self.authors[0]} et al."
        
        year = self.publication_date.year if self.publication_date else "n.d."
        
        return f"{author_text} ({year}) {self.title}."
    
    def _format_ieee(self) -> str:
        """Format the source in IEEE style."""
        # Simplified implementation
        if not self.authors:
            author_text = "No author"
        elif len(self.authors) == 1:
            author_text = f"{self.authors[0]}"
        else:
            author_text = f"{self.authors[0]} et al."
        
        year = self.publication_date.year if self.publication_date else "n.d."
        
        return f"{author_text}, \"{self.title},\" {year}."


class ResearchResult(BaseModel):
    """A research result with detailed information."""
    id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S"))
    topic: str
    summary: str
    sources: List[Source]
    tools_used: List[str]
    created_at: datetime = Field(default_factory=datetime.now)
    tags: List[str] = []
    
    def format_bibliography(self, format: CitationFormat) -> str:
        """Format the sources as a bibliography in the specified format."""
        citations = [source.format_citation(format) for source in self.sources]
        return "\n\n".join(citations)


class ResearchQuery(BaseModel):
    """A research query with its results."""
    id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S"))
    query: str
    results: List[ResearchResult] = []
    created_at: datetime = Field(default_factory=datetime.now)
    tags: List[str] = []


class ResearchProject(BaseModel):
    """A research project containing multiple queries and results."""
    id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S"))
    name: str
    description: str = ""
    queries: List[ResearchQuery] = []
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    tags: List[str] = []
    
    def add_query(self, query: str) -> ResearchQuery:
        """Add a new query to the project."""
        research_query = ResearchQuery(query=query)
        self.queries.append(research_query)
        self.updated_at = datetime.now()
        return research_query
    
    def add_result(self, query_id: str, result: ResearchResult) -> None:
        """Add a result to a query in the project."""
        for query in self.queries:
            if query.id == query_id:
                query.results.append(result)
                self.updated_at = datetime.now()
                break


class User(BaseModel):
    """A user of the Research Assistant."""
    id: str
    name: str
    email: Optional[str] = None


class Permission(str, Enum):
    """Permission levels for shared research."""
    READ = "read"
    COMMENT = "comment"
    EDIT = "edit"
    ADMIN = "admin"


class SharedResearch(BaseModel):
    """A shared research project with permissions."""
    project_id: str
    user_id: str
    permission: Permission = Permission.READ
    shared_at: datetime = Field(default_factory=datetime.now)


class Comment(BaseModel):
    """A comment on a research result."""
    id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S"))
    user_id: str
    content: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class Annotation(BaseModel):
    """An annotation on a specific part of a research result."""
    id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S"))
    user_id: str
    content: str
    target_text: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None


class ResearchResultWithComments(ResearchResult):
    """A research result with comments and annotations."""
    comments: List[Comment] = []
    annotations: List[Annotation] = []


class ExportFormat(str, Enum):
    """Supported export formats."""
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"
    HTML = "html"
    PRESENTATION = "presentation"


class VisualizationType(str, Enum):
    """Types of visualizations."""
    CHART = "chart"
    MIND_MAP = "mind_map"
    TIMELINE = "timeline"
    NETWORK = "network"


class ChartType(str, Enum):
    """Types of charts."""
    BAR = "bar"
    LINE = "line"
    PIE = "pie"
    SCATTER = "scatter"


class Visualization(BaseModel):
    """A visualization of research data."""
    id: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M%S"))
    title: str
    description: str = ""
    type: VisualizationType
    data: Dict = {}
    created_at: datetime = Field(default_factory=datetime.now)
    
    def render(self) -> str:
        """Render the visualization as a string."""
        # This would be implemented by subclasses
        return f"Visualization: {self.title}"


class EnhancedResearchResponse(BaseModel):
    """Enhanced structured response format for research results."""
    topic: str
    summary: str
    sources: List[Union[str, Source]]
    tools_used: List[str]
    citation_format: Optional[CitationFormat] = None
    bibliography: Optional[str] = None
    project_id: Optional[str] = None
    query_id: Optional[str] = None
    visualizations: List[Visualization] = []
    tags: List[str] = []