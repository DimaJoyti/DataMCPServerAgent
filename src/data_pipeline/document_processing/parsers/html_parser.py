"""
HTML file parser implementation.
"""

import logging
import re
from pathlib import Path
from typing import List, Union, Dict, Any

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    import html2text
    HAS_HTML2TEXT = True
except ImportError:
    HAS_HTML2TEXT = False

from .base_parser import BaseParser, ParsedDocument
from ..metadata.models import DocumentMetadata, DocumentType
from ..metadata.extractor import MetadataExtractor


class HTMLParser(BaseParser):
    """Parser for HTML files."""
    
    def __init__(self, *args, **kwargs):
        """Initialize HTML parser."""
        super().__init__(*args, **kwargs)
        
        if not HAS_BS4:
            raise ImportError(
                "HTML parsing requires BeautifulSoup4. "
                "Install with: pip install beautifulsoup4"
            )
    
    @property
    def supported_types(self) -> List[DocumentType]:
        """Return list of supported document types."""
        return [DocumentType.HTML]
    
    @property
    def supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        return ['html', 'htm', 'xhtml']
    
    def _parse_file_impl(self, file_path: Path) -> ParsedDocument:
        """
        Parse an HTML file.
        
        Args:
            file_path: Path to HTML file
            
        Returns:
            ParsedDocument: Parsed document result
        """
        # Extract basic metadata
        extractor = MetadataExtractor()
        metadata = extractor.extract_from_file(file_path)
        metadata.document_type = DocumentType.HTML
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                html_content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
                html_content = f.read()
        
        return self._parse_html_content(html_content, metadata)
    
    def _parse_content_impl(
        self,
        content: Union[str, bytes],
        document_id: str,
        document_type: DocumentType,
        **metadata_kwargs
    ) -> ParsedDocument:
        """
        Parse HTML content directly.
        
        Args:
            content: HTML content
            document_id: Document identifier
            document_type: Type of document
            **metadata_kwargs: Additional metadata fields
            
        Returns:
            ParsedDocument: Parsed document result
        """
        # Convert bytes to string if needed
        if isinstance(content, bytes):
            try:
                content = content.decode('utf-8')
            except UnicodeDecodeError:
                content = content.decode('utf-8', errors='ignore')
        
        # Create metadata
        extractor = MetadataExtractor()
        metadata = extractor.extract_from_content(
            content,
            document_id,
            document_type,
            **metadata_kwargs
        )
        metadata.document_type = DocumentType.HTML
        
        return self._parse_html_content(content, metadata)
    
    def _parse_html_content(self, html_content: str, metadata: DocumentMetadata) -> ParsedDocument:
        """Parse HTML content using BeautifulSoup."""
        warnings = []
        errors = []
        tables = []
        images = []
        links = []
        
        try:
            # Parse HTML with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract HTML metadata
            self._extract_html_metadata(soup, metadata)
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract text content
            if self.config.preserve_formatting and HAS_HTML2TEXT:
                # Use html2text to preserve some formatting
                h = html2text.HTML2Text()
                h.ignore_links = False
                h.ignore_images = False
                text_content = h.handle(html_content)
            else:
                # Extract plain text
                text_content = soup.get_text()
            
            # Extract tables if configured
            if self.config.extract_tables:
                tables = self._extract_tables(soup)
            
            # Extract images if configured
            if self.config.extract_images:
                images = self._extract_images(soup)
            
            # Extract links
            links = self._extract_html_links(soup)
            
        except Exception as e:
            error_msg = f"Error parsing HTML content: {str(e)}"
            errors.append(error_msg)
            if not self.config.ignore_errors:
                raise
            text_content = ""
        
        # Normalize text if configured
        if self.config.normalize_whitespace:
            text_content = self._normalize_text(text_content)
        
        # Update text statistics
        if text_content:
            metadata.character_count = len(text_content)
            metadata.word_count = len(text_content.split())
        
        # Create result
        result = ParsedDocument(
            text=text_content,
            metadata=metadata,
            tables=tables,
            images=images,
            links=links,
            warnings=warnings,
            errors=errors,
            parsing_time=0.0,
            parser_name=self._parser_name,
            parser_version=self._parser_version
        )
        
        return result
    
    def _extract_html_metadata(self, soup: BeautifulSoup, metadata: DocumentMetadata) -> None:
        """Extract metadata from HTML head section."""
        try:
            # Extract title
            title_tag = soup.find('title')
            if title_tag and title_tag.string:
                metadata.title = title_tag.string.strip()
            
            # Extract meta tags
            meta_tags = soup.find_all('meta')
            for meta in meta_tags:
                name = meta.get('name', '').lower()
                content = meta.get('content', '')
                
                if name == 'author':
                    metadata.author = content
                elif name == 'description':
                    metadata.subject = content
                elif name == 'keywords':
                    keywords = [k.strip() for k in content.split(',') if k.strip()]
                    metadata.keywords = keywords
                elif name == 'language':
                    metadata.language = content
                elif name in ['generator', 'creator']:
                    metadata.add_custom_field(name, content)
                
                # Handle property-based meta tags (Open Graph, etc.)
                property_name = meta.get('property', '').lower()
                if property_name:
                    metadata.add_custom_field(f'meta_{property_name}', content)
            
            # Extract language from html tag
            html_tag = soup.find('html')
            if html_tag and html_tag.get('lang'):
                metadata.language = html_tag.get('lang')
            
        except Exception as e:
            self.logger.warning(f"Failed to extract HTML metadata: {e}")
    
    def _extract_tables(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract tables from HTML."""
        tables = []
        
        try:
            table_tags = soup.find_all('table')
            for table_idx, table in enumerate(table_tags):
                rows = table.find_all('tr')
                table_data = []
                
                for row in rows:
                    cells = row.find_all(['td', 'th'])
                    row_data = [cell.get_text(strip=True) for cell in cells]
                    if row_data:  # Only add non-empty rows
                        table_data.append(row_data)
                
                if table_data:
                    tables.append({
                        'table_index': table_idx,
                        'data': table_data,
                        'rows': len(table_data),
                        'columns': len(table_data[0]) if table_data else 0,
                        'has_header': bool(table.find('th'))
                    })
        
        except Exception as e:
            self.logger.warning(f"Failed to extract tables: {e}")
        
        return tables
    
    def _extract_images(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract images from HTML."""
        images = []
        
        try:
            img_tags = soup.find_all('img')
            for img_idx, img in enumerate(img_tags):
                image_info = {
                    'image_index': img_idx,
                    'src': img.get('src', ''),
                    'alt': img.get('alt', ''),
                    'title': img.get('title', ''),
                    'width': img.get('width'),
                    'height': img.get('height')
                }
                images.append(image_info)
        
        except Exception as e:
            self.logger.warning(f"Failed to extract images: {e}")
        
        return images
    
    def _extract_html_links(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract links from HTML."""
        links = []
        
        try:
            # Extract anchor tags
            a_tags = soup.find_all('a', href=True)
            for link in a_tags:
                href = link.get('href', '')
                text = link.get_text(strip=True)
                title = link.get('title', '')
                
                links.append({
                    'url': href,
                    'text': text,
                    'title': title,
                    'type': 'html_anchor'
                })
            
            # Extract link tags (stylesheets, etc.)
            link_tags = soup.find_all('link', href=True)
            for link in link_tags:
                href = link.get('href', '')
                rel = link.get('rel', [])
                link_type = link.get('type', '')
                
                links.append({
                    'url': href,
                    'text': f"{rel} - {link_type}",
                    'rel': rel,
                    'type': 'html_link'
                })
        
        except Exception as e:
            self.logger.warning(f"Failed to extract links: {e}")
        
        return links
