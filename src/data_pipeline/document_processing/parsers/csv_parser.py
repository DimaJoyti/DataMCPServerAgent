"""
CSV file parser for comma-separated values files.
"""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

from ..metadata.models import DocumentMetadata, DocumentType
from .base_parser import BaseParser, ParsedDocument


class CSVParser(BaseParser):
    """Parser for CSV files."""

    def __init__(self):
        """Initialize CSV parser."""
        super().__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def can_parse(self, file_path: Path) -> bool:
        """Check if this parser can handle the file."""
        return file_path.suffix.lower() in [".csv", ".tsv"]

    def parse(self, file_path: Path, **kwargs) -> ParsedDocument:
        """
        Parse CSV file.

        Args:
            file_path: Path to CSV file
            **kwargs: Additional parsing options
                - delimiter: Field delimiter (default: auto-detect)
                - encoding: File encoding (default: auto-detect)
                - has_header: Whether first row is header (default: auto-detect)
                - max_rows: Maximum number of rows to parse (default: None)
                - format_as_table: Whether to format as table (default: True)
                - include_row_numbers: Whether to include row numbers (default: False)
                - sample_size: Number of rows to use for detection (default: 1000)

        Returns:
            ParsedDocument: Parsed document
        """
        try:
            # Parse options
            delimiter = kwargs.get("delimiter", None)
            encoding = kwargs.get("encoding", None)
            has_header = kwargs.get("has_header", None)
            max_rows = kwargs.get("max_rows", None)
            format_as_table = kwargs.get("format_as_table", True)
            include_row_numbers = kwargs.get("include_row_numbers", False)
            sample_size = kwargs.get("sample_size", 1000)

            # Auto-detect parameters if not provided
            if delimiter is None or encoding is None or has_header is None:
                detected = self._detect_csv_parameters(file_path, sample_size)
                delimiter = delimiter or detected.get("delimiter", ",")
                encoding = encoding or detected.get("encoding", "utf-8")
                has_header = (
                    has_header if has_header is not None else detected.get("has_header", True)
                )

            # Use pandas if available for better handling
            if HAS_PANDAS:
                content, metadata = self._parse_with_pandas(
                    file_path,
                    delimiter,
                    encoding,
                    has_header,
                    max_rows,
                    format_as_table,
                    include_row_numbers,
                )
            else:
                content, metadata = self._parse_with_csv(
                    file_path,
                    delimiter,
                    encoding,
                    has_header,
                    max_rows,
                    format_as_table,
                    include_row_numbers,
                )

            # Create document metadata
            doc_metadata = DocumentMetadata(
                title=file_path.stem,
                document_type=DocumentType.SPREADSHEET,
                file_path=str(file_path),
                file_size=file_path.stat().st_size,
                language="unknown",  # CSV doesn't have inherent language
                page_count=1,  # CSV is single "page"
                word_count=len(content.split()),
                character_count=len(content),
                custom_metadata={
                    "delimiter": delimiter,
                    "encoding": encoding,
                    "has_header": has_header,
                    "parser": "CSVParser",
                    **metadata,
                },
            )

            return ParsedDocument(
                text=content,
                metadata=doc_metadata,
                raw_content=None,  # Could store DataFrame if pandas is used
                processing_time=0.0,  # Will be set by caller
            )

        except Exception as e:
            self.logger.error(f"Failed to parse CSV file {file_path}: {e}")
            raise

    def _detect_csv_parameters(self, file_path: Path, sample_size: int = 1000) -> Dict[str, Any]:
        """Auto-detect CSV parameters."""
        detected = {"delimiter": ",", "encoding": "utf-8", "has_header": True}

        try:
            # Detect encoding
            detected["encoding"] = self._detect_encoding(file_path)

            # Read sample for delimiter and header detection
            with open(file_path, encoding=detected["encoding"], newline="") as f:
                sample_lines = []
                for i, line in enumerate(f):
                    if i >= sample_size:
                        break
                    sample_lines.append(line)

                if sample_lines:
                    sample_text = "".join(sample_lines)

                    # Detect delimiter
                    sniffer = csv.Sniffer()
                    try:
                        dialect = sniffer.sniff(sample_text, delimiters=",;\t|")
                        detected["delimiter"] = dialect.delimiter
                    except csv.Error:
                        # Fallback: count occurrences of common delimiters
                        delimiters = [",", ";", "\t", "|"]
                        delimiter_counts = {d: sample_text.count(d) for d in delimiters}
                        detected["delimiter"] = max(delimiter_counts, key=delimiter_counts.get)

                    # Detect header
                    try:
                        detected["has_header"] = sniffer.has_header(sample_text)
                    except csv.Error:
                        # Fallback: assume header if first row looks different from others
                        lines = sample_lines[:10]  # Check first 10 lines
                        if len(lines) >= 2:
                            first_row = lines[0].strip().split(detected["delimiter"])
                            second_row = lines[1].strip().split(detected["delimiter"])

                            # Simple heuristic: if first row has no numbers but second does
                            first_has_numbers = any(
                                self._is_number(cell.strip()) for cell in first_row
                            )
                            second_has_numbers = any(
                                self._is_number(cell.strip()) for cell in second_row
                            )

                            detected["has_header"] = not first_has_numbers and second_has_numbers

        except Exception as e:
            self.logger.warning(f"Failed to auto-detect CSV parameters: {e}")

        return detected

    def _detect_encoding(self, file_path: Path) -> str:
        """Detect file encoding."""
        try:
            import chardet

            with open(file_path, "rb") as f:
                raw_data = f.read(10000)  # Read first 10KB
                result = chardet.detect(raw_data)
                return result.get("encoding", "utf-8")
        except ImportError:
            # Fallback: try common encodings
            encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
            for encoding in encodings:
                try:
                    with open(file_path, encoding=encoding) as f:
                        f.read(1000)  # Try to read first 1KB
                    return encoding
                except UnicodeDecodeError:
                    continue

            return "utf-8"  # Final fallback

    def _is_number(self, value: str) -> bool:
        """Check if a string represents a number."""
        try:
            float(value)
            return True
        except ValueError:
            return False

    def _parse_with_pandas(
        self,
        file_path: Path,
        delimiter: str,
        encoding: str,
        has_header: bool,
        max_rows: Optional[int],
        format_as_table: bool,
        include_row_numbers: bool,
    ) -> tuple[str, Dict]:
        """Parse CSV using pandas."""
        try:
            # Read CSV with pandas
            df = pd.read_csv(
                file_path,
                delimiter=delimiter,
                encoding=encoding,
                header=0 if has_header else None,
                nrows=max_rows,
                dtype=str,  # Keep everything as string to preserve formatting
                na_filter=False,  # Don't convert to NaN
            )

            # Generate metadata
            metadata = {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns) if has_header else None,
                "data_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "memory_usage": df.memory_usage(deep=True).sum(),
                "has_missing_values": df.isnull().any().any(),
            }

            # Convert to text
            if format_as_table:
                content = self._dataframe_to_table(df, include_row_numbers, has_header)
            else:
                content = self._dataframe_to_structured_text(df, include_row_numbers, has_header)

            return content, metadata

        except Exception as e:
            self.logger.warning(f"Pandas parsing failed, falling back to csv module: {e}")
            return self._parse_with_csv(
                file_path,
                delimiter,
                encoding,
                has_header,
                max_rows,
                format_as_table,
                include_row_numbers,
            )

    def _parse_with_csv(
        self,
        file_path: Path,
        delimiter: str,
        encoding: str,
        has_header: bool,
        max_rows: Optional[int],
        format_as_table: bool,
        include_row_numbers: bool,
    ) -> tuple[str, Dict]:
        """Parse CSV using built-in csv module."""
        rows = []
        headers = None

        with open(file_path, encoding=encoding, newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)

            # Read header if present
            if has_header:
                try:
                    headers = next(reader)
                except StopIteration:
                    headers = None

            # Read data rows
            for i, row in enumerate(reader):
                if max_rows and i >= max_rows:
                    break
                rows.append(row)

        # Generate metadata
        metadata = {
            "rows": len(rows),
            "columns": len(rows[0]) if rows else 0,
            "column_names": headers,
            "has_missing_values": any("" in row for row in rows),
        }

        # Convert to text
        if format_as_table:
            content = self._rows_to_table(rows, headers, include_row_numbers)
        else:
            content = self._rows_to_structured_text(rows, headers, include_row_numbers)

        return content, metadata

    def _dataframe_to_table(self, df, include_row_numbers: bool, has_header: bool) -> str:
        """Convert DataFrame to table format."""
        lines = []

        # Add headers
        if has_header:
            header_row = []
            if include_row_numbers:
                header_row.append("Row")
            header_row.extend(df.columns)
            lines.append(" | ".join(header_row))
            lines.append("-" * len(lines[0]))

        # Add data rows
        for idx, row in df.iterrows():
            data_row = []
            if include_row_numbers:
                data_row.append(str(idx + 1))
            data_row.extend(str(value) for value in row)
            lines.append(" | ".join(data_row))

        return "\n".join(lines)

    def _dataframe_to_structured_text(self, df, include_row_numbers: bool, has_header: bool) -> str:
        """Convert DataFrame to structured text format."""
        lines = []

        for idx, row in df.iterrows():
            row_parts = []
            if include_row_numbers:
                row_parts.append(f"Row {idx + 1}")

            if has_header:
                for col, value in row.items():
                    if str(value).strip():  # Only include non-empty values
                        row_parts.append(f"{col}: {value}")
            else:
                for i, value in enumerate(row):
                    if str(value).strip():
                        row_parts.append(f"Column {i + 1}: {value}")

            if row_parts:
                lines.append(", ".join(row_parts))

        return "\n".join(lines)

    def _rows_to_table(
        self, rows: List[List[str]], headers: Optional[List[str]], include_row_numbers: bool
    ) -> str:
        """Convert rows to table format."""
        lines = []

        # Add headers
        if headers:
            header_row = []
            if include_row_numbers:
                header_row.append("Row")
            header_row.extend(headers)
            lines.append(" | ".join(header_row))
            lines.append("-" * len(lines[0]))

        # Add data rows
        for i, row in enumerate(rows):
            data_row = []
            if include_row_numbers:
                data_row.append(str(i + 1))
            data_row.extend(row)
            lines.append(" | ".join(data_row))

        return "\n".join(lines)

    def _rows_to_structured_text(
        self, rows: List[List[str]], headers: Optional[List[str]], include_row_numbers: bool
    ) -> str:
        """Convert rows to structured text format."""
        lines = []

        for i, row in enumerate(rows):
            row_parts = []
            if include_row_numbers:
                row_parts.append(f"Row {i + 1}")

            if headers and len(headers) == len(row):
                for header, value in zip(headers, row):
                    if value.strip():
                        row_parts.append(f"{header}: {value}")
            else:
                for j, value in enumerate(row):
                    if value.strip():
                        row_parts.append(f"Column {j + 1}: {value}")

            if row_parts:
                lines.append(", ".join(row_parts))

        return "\n".join(lines)

    def extract_metadata(self, file_path: Path) -> Dict:
        """Extract metadata from CSV file."""
        try:
            detected = self._detect_csv_parameters(file_path)

            # Get basic file stats
            with open(file_path, encoding=detected["encoding"]) as f:
                line_count = sum(1 for _ in f)

            # Get column count from first line
            with open(file_path, encoding=detected["encoding"]) as f:
                first_line = f.readline().strip()
                if first_line:
                    column_count = len(first_line.split(detected["delimiter"]))
                else:
                    column_count = 0

            return {
                "line_count": line_count,
                "estimated_rows": line_count - (1 if detected["has_header"] else 0),
                "estimated_columns": column_count,
                **detected,
            }

        except Exception as e:
            self.logger.warning(f"Failed to extract CSV metadata: {e}")
            return {}

    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions."""
        return [".csv", ".tsv"]

    def get_parser_info(self) -> Dict:
        """Get information about this parser."""
        return {
            "name": "CSVParser",
            "description": "Parser for CSV and TSV files",
            "supported_extensions": self.get_supported_extensions(),
            "features": [
                "Auto-detection of delimiter, encoding, and headers",
                "Table and structured text formatting",
                "Configurable row limits",
                "Row numbering support",
                "Pandas integration when available",
            ],
            "dependencies": {"pandas": HAS_PANDAS, "chardet": "optional (for encoding detection)"},
        }
