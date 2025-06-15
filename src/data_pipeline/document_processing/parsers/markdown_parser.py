"""
Markdown file parser implementation.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Union

try:
    import markdown
    from markdown.extensions import codehilite, tables, toc

    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False

from ..metadata.extractor import MetadataExtractor
from ..metadata.models import DocumentMetadata, DocumentType
from .base_parser import BaseParser, ParsedDocument


class MarkdownParser(BaseParser):
    """Parser for Markdown files."""

    def __init__(self, *args, **kwargs):
        """Initialize Markdown parser."""
        super().__init__(*args, **kwargs)

        if not HAS_MARKDOWN:
            self.logger.warning(
                "Markdown parsing library not available. " "Install with: pip install markdown"
            )

    @property
    def supported_types(self) -> List[DocumentType]:
        """Return list of supported document types."""
        return [DocumentType.MARKDOWN]

    @property
    def supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        return ["md", "markdown", "mdown", "mkd", "mkdn"]

    def _parse_file_impl(self, file_path: Path) -> ParsedDocument:
        """
        Parse a Markdown file.

        Args:
            file_path: Path to Markdown file

        Returns:
            ParsedDocument: Parsed document result
        """
        # Extract basic metadata
        extractor = MetadataExtractor()
        metadata = extractor.extract_from_file(file_path)
        metadata.document_type = DocumentType.MARKDOWN

        # Read file content
        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                markdown_content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, encoding="latin-1", errors="ignore") as f:
                markdown_content = f.read()

        return self._parse_markdown_content(markdown_content, metadata)

    def _parse_content_impl(
        self,
        content: Union[str, bytes],
        document_id: str,
        document_type: DocumentType,
        **metadata_kwargs,
    ) -> ParsedDocument:
        """
        Parse Markdown content directly.

        Args:
            content: Markdown content
            document_id: Document identifier
            document_type: Type of document
            **metadata_kwargs: Additional metadata fields

        Returns:
            ParsedDocument: Parsed document result
        """
        # Convert bytes to string if needed
        if isinstance(content, bytes):
            try:
                content = content.decode("utf-8")
            except UnicodeDecodeError:
                content = content.decode("utf-8", errors="ignore")

        # Create metadata
        extractor = MetadataExtractor()
        metadata = extractor.extract_from_content(
            content, document_id, document_type, **metadata_kwargs
        )
        metadata.document_type = DocumentType.MARKDOWN

        return self._parse_markdown_content(content, metadata)

    def _parse_markdown_content(
        self, markdown_content: str, metadata: DocumentMetadata
    ) -> ParsedDocument:
        """Parse Markdown content."""
        warnings = []
        errors = []
        tables = []
        images = []
        links = []

        try:
            # Extract front matter if present
            front_matter = self._extract_front_matter(markdown_content, metadata)
            if front_matter:
                # Remove front matter from content
                markdown_content = self._remove_front_matter(markdown_content)

            # Extract markdown-specific metadata
            self._extract_markdown_metadata(markdown_content, metadata)

            # Convert to HTML if markdown library is available
            html_content = ""
            if HAS_MARKDOWN:
                try:
                    md = markdown.Markdown(
                        extensions=["toc", "tables", "codehilite", "fenced_code"],
                        extension_configs={
                            "toc": {"permalink": True},
                            "codehilite": {"css_class": "highlight"},
                        },
                    )
                    html_content = md.convert(markdown_content)
                except Exception as e:
                    warnings.append(f"Failed to convert markdown to HTML: {e}")

            # Extract text content
            if self.config.preserve_formatting:
                # Keep markdown formatting
                text_content = markdown_content
            else:
                # Extract plain text
                text_content = self._markdown_to_text(markdown_content)

            # Extract tables if configured
            if self.config.extract_tables:
                tables = self._extract_markdown_tables(markdown_content)

            # Extract images if configured
            if self.config.extract_images:
                images = self._extract_markdown_images(markdown_content)

            # Extract links
            links = self._extract_markdown_links(markdown_content)

        except Exception as e:
            error_msg = f"Error parsing Markdown content: {str(e)}"
            errors.append(error_msg)
            if not self.config.ignore_errors:
                raise
            text_content = markdown_content

        # Normalize text if configured
        if self.config.normalize_whitespace and not self.config.preserve_formatting:
            text_content = self._normalize_text(text_content)

        # Update text statistics
        if text_content:
            metadata.character_count = len(text_content)
            metadata.word_count = len(text_content.split())

        # Store HTML content if generated
        raw_data = {}
        if html_content:
            raw_data["html"] = html_content
        if front_matter:
            raw_data["front_matter"] = front_matter

        # Create result
        result = ParsedDocument(
            text=text_content,
            metadata=metadata,
            tables=tables,
            images=images,
            links=links,
            warnings=warnings,
            errors=errors,
            raw_data=raw_data if raw_data else None,
            parsing_time=0.0,
            parser_name=self._parser_name,
            parser_version=self._parser_version,
        )

        return result

    def _extract_front_matter(self, content: str, metadata: DocumentMetadata) -> Dict[str, Any]:
        """Extract YAML front matter from markdown."""
        front_matter = {}

        # Check for YAML front matter
        if content.startswith("---\n"):
            try:
                import yaml

                end_marker = content.find("\n---\n", 4)
                if end_marker != -1:
                    yaml_content = content[4:end_marker]
                    front_matter = yaml.safe_load(yaml_content)

                    # Update metadata with front matter
                    if isinstance(front_matter, dict):
                        for key, value in front_matter.items():
                            if key.lower() == "title":
                                metadata.title = str(value)
                            elif key.lower() == "author":
                                metadata.author = str(value)
                            elif key.lower() == "description":
                                metadata.subject = str(value)
                            elif key.lower() in ["tags", "keywords"]:
                                if isinstance(value, list):
                                    metadata.keywords = [str(v) for v in value]
                                else:
                                    metadata.keywords = [str(value)]
                            elif key.lower() == "date":
                                metadata.add_custom_field("date", str(value))
                            else:
                                metadata.add_custom_field(key, value)
            except ImportError:
                self.logger.warning("PyYAML not available for front matter parsing")
            except Exception as e:
                self.logger.warning(f"Failed to parse front matter: {e}")

        return front_matter

    def _remove_front_matter(self, content: str) -> str:
        """Remove front matter from markdown content."""
        if content.startswith("---\n"):
            end_marker = content.find("\n---\n", 4)
            if end_marker != -1:
                return content[end_marker + 5 :]
        return content

    def _extract_markdown_metadata(self, content: str, metadata: DocumentMetadata) -> None:
        """Extract metadata from markdown structure."""
        # Extract title from first heading if not already set
        if not metadata.title:
            title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
            if title_match:
                metadata.title = title_match.group(1).strip()

        # Count different heading levels
        h1_count = len(re.findall(r"^#\s+", content, re.MULTILINE))
        h2_count = len(re.findall(r"^##\s+", content, re.MULTILINE))
        h3_count = len(re.findall(r"^###\s+", content, re.MULTILINE))
        h4_count = len(re.findall(r"^####\s+", content, re.MULTILINE))
        h5_count = len(re.findall(r"^#####\s+", content, re.MULTILINE))
        h6_count = len(re.findall(r"^######\s+", content, re.MULTILINE))

        metadata.add_custom_field("h1_count", h1_count)
        metadata.add_custom_field("h2_count", h2_count)
        metadata.add_custom_field("h3_count", h3_count)
        metadata.add_custom_field("h4_count", h4_count)
        metadata.add_custom_field("h5_count", h5_count)
        metadata.add_custom_field("h6_count", h6_count)
        metadata.add_custom_field(
            "total_headings", h1_count + h2_count + h3_count + h4_count + h5_count + h6_count
        )

        # Count code blocks
        code_blocks = len(re.findall(r"```[\s\S]*?```", content))
        inline_code = len(re.findall(r"`[^`\n]+`", content))

        metadata.add_custom_field("code_blocks", code_blocks)
        metadata.add_custom_field("inline_code", inline_code)

        # Count lists
        bullet_lists = len(re.findall(r"^\s*[-*+]\s+", content, re.MULTILINE))
        numbered_lists = len(re.findall(r"^\s*\d+\.\s+", content, re.MULTILINE))

        metadata.add_custom_field("bullet_lists", bullet_lists)
        metadata.add_custom_field("numbered_lists", numbered_lists)

    def _markdown_to_text(self, content: str) -> str:
        """Convert markdown to plain text."""
        # Remove code blocks
        content = re.sub(r"```[\s\S]*?```", "", content)

        # Remove inline code
        content = re.sub(r"`[^`\n]+`", "", content)

        # Remove headers markup
        content = re.sub(r"^#{1,6}\s+", "", content, flags=re.MULTILINE)

        # Remove emphasis markup
        content = re.sub(r"\*\*([^*]+)\*\*", r"\1", content)  # Bold
        content = re.sub(r"\*([^*]+)\*", r"\1", content)  # Italic
        content = re.sub(r"__([^_]+)__", r"\1", content)  # Bold
        content = re.sub(r"_([^_]+)_", r"\1", content)  # Italic

        # Remove links but keep text
        content = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", content)

        # Remove images
        content = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", content)

        # Remove horizontal rules
        content = re.sub(r"^---+$", "", content, flags=re.MULTILINE)

        # Remove list markers
        content = re.sub(r"^\s*[-*+]\s+", "", content, flags=re.MULTILINE)
        content = re.sub(r"^\s*\d+\.\s+", "", content, flags=re.MULTILINE)

        return content

    def _extract_markdown_tables(self, content: str) -> List[Dict[str, Any]]:
        """Extract tables from markdown."""
        tables = []

        # Find markdown tables
        table_pattern = r"(\|.+\|\n)+(\|[-\s|:]+\|\n)?(\|.+\|\n)+"
        table_matches = re.finditer(table_pattern, content)

        for table_idx, match in enumerate(table_matches):
            table_text = match.group(0)
            lines = [line.strip() for line in table_text.split("\n") if line.strip()]

            table_data = []
            for line in lines:
                if "|" in line and not re.match(r"\|[-\s|:]+\|", line):  # Skip separator line
                    cells = [
                        cell.strip() for cell in line.split("|")[1:-1]
                    ]  # Remove empty first/last
                    if cells:
                        table_data.append(cells)

            if table_data:
                tables.append(
                    {
                        "table_index": table_idx,
                        "data": table_data,
                        "rows": len(table_data),
                        "columns": len(table_data[0]) if table_data else 0,
                        "has_header": True,  # Markdown tables typically have headers
                    }
                )

        return tables

    def _extract_markdown_images(self, content: str) -> List[Dict[str, Any]]:
        """Extract images from markdown."""
        images = []

        # Find markdown images: ![alt](src "title")
        image_pattern = r'!\[([^\]]*)\]\(([^)]+?)(?:\s+"([^"]*)")?\)'
        image_matches = re.finditer(image_pattern, content)

        for img_idx, match in enumerate(image_matches):
            alt_text = match.group(1)
            src = match.group(2)
            title = match.group(3) or ""

            images.append({"image_index": img_idx, "src": src, "alt": alt_text, "title": title})

        return images

    def _extract_markdown_links(self, content: str) -> List[Dict[str, str]]:
        """Extract links from markdown."""
        links = []

        # Find markdown links: [text](url "title")
        link_pattern = r'\[([^\]]+)\]\(([^)]+?)(?:\s+"([^"]*)")?\)'
        link_matches = re.finditer(link_pattern, content)

        for match in link_matches:
            text = match.group(1)
            url = match.group(2)
            title = match.group(3) or ""

            links.append({"text": text, "url": url, "title": title, "type": "markdown"})

        return links
