"""
Search models and query definitions for vector stores.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

from .base_schema import VectorRecord, DistanceMetric

class SearchType(str, Enum):
    """Types of search operations."""
    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    SEMANTIC = "semantic"

class SortOrder(str, Enum):
    """Sort order options."""
    ASC = "asc"
    DESC = "desc"

class FilterOperator(str, Enum):
    """Filter operators."""
    EQ = "eq"          # Equal
    NE = "ne"          # Not equal
    GT = "gt"          # Greater than
    GTE = "gte"        # Greater than or equal
    LT = "lt"          # Less than
    LTE = "lte"        # Less than or equal
    IN = "in"          # In list
    NOT_IN = "not_in"  # Not in list
    CONTAINS = "contains"      # Contains substring
    NOT_CONTAINS = "not_contains"  # Does not contain substring
    STARTS_WITH = "starts_with"    # Starts with
    ENDS_WITH = "ends_with"        # Ends with
    REGEX = "regex"    # Regular expression
    EXISTS = "exists"  # Field exists
    NOT_EXISTS = "not_exists"  # Field does not exist

class SearchFilter(BaseModel):
    """Individual search filter."""

    field: str = Field(..., description="Field name to filter on")
    operator: FilterOperator = Field(..., description="Filter operator")
    value: Union[str, int, float, bool, List[Any]] = Field(..., description="Filter value")

    @validator('value')
    def validate_value(cls, v, values):
        """Validate filter value based on operator."""
        operator = values.get('operator')

        if operator in [FilterOperator.IN, FilterOperator.NOT_IN]:
            if not isinstance(v, list):
                raise ValueError(f"Value must be a list for operator {operator}")
        elif operator in [FilterOperator.EXISTS, FilterOperator.NOT_EXISTS]:
            # These operators don't need a value
            pass
        elif v is None:
            raise ValueError(f"Value cannot be None for operator {operator}")

        return v

class SearchFilters(BaseModel):
    """Collection of search filters."""

    filters: List[SearchFilter] = Field(default_factory=list, description="List of filters")
    operator: str = Field(default="AND", description="Logical operator between filters (AND/OR)")

    def add_filter(
        self,
        field: str,
        operator: FilterOperator,
        value: Union[str, int, float, bool, List[Any]]
    ) -> None:
        """Add a filter."""
        filter_obj = SearchFilter(field=field, operator=operator, value=value)
        self.filters.append(filter_obj)

    def add_text_filter(self, field: str, text: str, exact: bool = False) -> None:
        """Add text filter."""
        operator = FilterOperator.EQ if exact else FilterOperator.CONTAINS
        self.add_filter(field, operator, text)

    def add_range_filter(
        self,
        field: str,
        min_value: Optional[Union[int, float]] = None,
        max_value: Optional[Union[int, float]] = None
    ) -> None:
        """Add range filter."""
        if min_value is not None:
            self.add_filter(field, FilterOperator.GTE, min_value)
        if max_value is not None:
            self.add_filter(field, FilterOperator.LTE, max_value)

    def add_list_filter(self, field: str, values: List[Any], include: bool = True) -> None:
        """Add list filter."""
        operator = FilterOperator.IN if include else FilterOperator.NOT_IN
        self.add_filter(field, operator, values)

    def is_empty(self) -> bool:
        """Check if filters are empty."""
        return len(self.filters) == 0

class SortCriteria(BaseModel):
    """Sort criteria for search results."""

    field: str = Field(..., description="Field to sort by")
    order: SortOrder = Field(default=SortOrder.DESC, description="Sort order")

    @classmethod
    def by_score(cls, descending: bool = True) -> "SortCriteria":
        """Sort by relevance score."""
        return cls(field="_score", order=SortOrder.DESC if descending else SortOrder.ASC)

    @classmethod
    def by_date(cls, field: str = "created_at", descending: bool = True) -> "SortCriteria":
        """Sort by date field."""
        return cls(field=field, order=SortOrder.DESC if descending else SortOrder.ASC)

class SearchQuery(BaseModel):
    """Search query for vector stores."""

    # Query content
    query_text: Optional[str] = Field(None, description="Text query for semantic search")
    query_vector: Optional[List[float]] = Field(None, description="Vector query for similarity search")

    # Search configuration
    search_type: SearchType = Field(default=SearchType.VECTOR, description="Type of search")
    limit: int = Field(default=10, description="Maximum number of results")
    offset: int = Field(default=0, description="Offset for pagination")

    # Similarity configuration
    distance_metric: Optional[DistanceMetric] = Field(None, description="Distance metric override")
    similarity_threshold: Optional[float] = Field(None, description="Minimum similarity threshold")

    # Filtering and sorting
    filters: Optional[SearchFilters] = Field(None, description="Search filters")
    sort_by: List[SortCriteria] = Field(default_factory=list, description="Sort criteria")

    # Hybrid search configuration
    keyword_weight: float = Field(default=0.3, description="Weight for keyword search in hybrid mode")
    vector_weight: float = Field(default=0.7, description="Weight for vector search in hybrid mode")

    # Result configuration
    include_metadata: bool = Field(default=True, description="Include metadata in results")
    include_vectors: bool = Field(default=False, description="Include vectors in results")

    # Advanced options
    rerank: bool = Field(default=False, description="Apply reranking to results")
    explain: bool = Field(default=False, description="Include explanation of scoring")

    @validator('limit')
    def validate_limit(cls, v):
        """Validate limit."""
        if v <= 0:
            raise ValueError("Limit must be positive")
        if v > 1000:
            raise ValueError("Limit cannot exceed 1000")
        return v

    @validator('offset')
    def validate_offset(cls, v):
        """Validate offset."""
        if v < 0:
            raise ValueError("Offset cannot be negative")
        return v

    @validator('similarity_threshold')
    def validate_similarity_threshold(cls, v):
        """Validate similarity threshold."""
        if v is not None and (v < 0 or v > 1):
            raise ValueError("Similarity threshold must be between 0 and 1")
        return v

    @validator('keyword_weight', 'vector_weight')
    def validate_weights(cls, v):
        """Validate search weights."""
        if v < 0 or v > 1:
            raise ValueError("Weights must be between 0 and 1")
        return v

    def has_text_query(self) -> bool:
        """Check if query has text."""
        return self.query_text is not None and self.query_text.strip() != ""

    def has_vector_query(self) -> bool:
        """Check if query has vector."""
        return self.query_vector is not None and len(self.query_vector) > 0

    def is_valid(self) -> bool:
        """Check if query is valid."""
        if self.search_type == SearchType.VECTOR:
            return self.has_vector_query()
        elif self.search_type == SearchType.KEYWORD:
            return self.has_text_query()
        elif self.search_type == SearchType.HYBRID:
            return self.has_text_query() or self.has_vector_query()
        elif self.search_type == SearchType.SEMANTIC:
            return self.has_text_query()
        return False

class SearchResult(BaseModel):
    """Individual search result."""

    # Record information
    id: str = Field(..., description="Record ID")
    score: float = Field(..., description="Relevance score")

    # Content
    text: str = Field(..., description="Text content")
    vector: Optional[List[float]] = Field(None, description="Embedding vector")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Record metadata")

    # Ranking information
    rank: Optional[int] = Field(None, description="Result rank")
    distance: Optional[float] = Field(None, description="Vector distance")

    # Explanation (if requested)
    explanation: Optional[Dict[str, Any]] = Field(None, description="Scoring explanation")

    def get_metadata_field(self, field: str, default: Any = None) -> Any:
        """Get metadata field value."""
        return self.metadata.get(field, default)

class SearchResults(BaseModel):
    """Collection of search results."""

    # Results
    results: List[SearchResult] = Field(..., description="Search results")

    # Query information
    query: SearchQuery = Field(..., description="Original query")

    # Statistics
    total_results: int = Field(..., description="Total number of matching results")
    search_time: float = Field(..., description="Search time in seconds")

    # Pagination
    offset: int = Field(..., description="Result offset")
    limit: int = Field(..., description="Result limit")
    has_more: bool = Field(..., description="Whether more results are available")

    # Aggregations (if supported)
    aggregations: Optional[Dict[str, Any]] = Field(None, description="Search aggregations")

    def __len__(self) -> int:
        """Return number of results."""
        return len(self.results)

    def __iter__(self):
        """Iterate over results."""
        return iter(self.results)

    def __getitem__(self, index: int) -> SearchResult:
        """Get result by index."""
        return self.results[index]

    def get_texts(self) -> List[str]:
        """Get list of result texts."""
        return [result.text for result in self.results]

    def get_scores(self) -> List[float]:
        """Get list of result scores."""
        return [result.score for result in self.results]

    def get_ids(self) -> List[str]:
        """Get list of result IDs."""
        return [result.id for result in self.results]

    def filter_by_score(self, min_score: float) -> "SearchResults":
        """Filter results by minimum score."""
        filtered_results = [r for r in self.results if r.score >= min_score]

        return SearchResults(
            results=filtered_results,
            query=self.query,
            total_results=len(filtered_results),
            search_time=self.search_time,
            offset=self.offset,
            limit=self.limit,
            has_more=False,  # Filtering may change this
            aggregations=self.aggregations
        )
