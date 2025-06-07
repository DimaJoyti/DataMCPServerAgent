"""
Documentation Health Checker

Monitor documentation quality, freshness, and completeness.
"""

import re
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import logging
import markdown
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetrics:
    """Metrics for a single document"""
    file_path: str
    word_count: int
    line_count: int
    last_modified: datetime
    age_days: int
    has_title: bool
    has_toc: bool
    heading_structure: List[str]
    internal_links: List[str]
    external_links: List[str]
    broken_links: List[str]
    missing_sections: List[str]
    readability_score: float


@dataclass
class DocumentationHealth:
    """Overall documentation health metrics"""
    timestamp: datetime
    total_documents: int
    outdated_documents: int
    documents_with_broken_links: int
    total_broken_links: int
    average_age_days: float
    coverage_score: float  # 0-100
    quality_score: float   # 0-100
    freshness_score: float # 0-100
    overall_score: float   # 0-100
    document_metrics: Dict[str, DocumentMetrics]
    recommendations: List[str]
    missing_documentation: List[str]


class DocumentationHealthChecker:
    """Check documentation health and quality"""
    
    def __init__(self, project_root: str, docs_directories: List[str]):
        self.project_root = Path(project_root)
        self.docs_directories = docs_directories
        self.required_sections = [
            "installation", "usage", "api", "contributing", 
            "examples", "configuration", "troubleshooting"
        ]
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'DataMCPServerAgent-DocChecker/1.0'
        })
    
    def find_documentation_files(self) -> List[Path]:
        """Find all documentation files"""
        doc_files = []
        
        for docs_dir in self.docs_directories:
            dir_path = self.project_root / docs_dir
            
            if dir_path.is_file() and docs_dir.endswith('.md'):
                # Single file like README.md
                doc_files.append(dir_path)
            elif dir_path.is_dir():
                # Directory with multiple files
                for pattern in ['*.md', '*.rst', '*.txt']:
                    doc_files.extend(dir_path.rglob(pattern))
        
        return doc_files
    
    def analyze_document(self, file_path: Path) -> DocumentMetrics:
        """Analyze a single document"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic metrics
            word_count = len(content.split())
            line_count = len(content.splitlines())
            last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
            age_days = (datetime.now() - last_modified).days
            
            # Structure analysis
            has_title = self._has_title(content)
            has_toc = self._has_table_of_contents(content)
            heading_structure = self._extract_headings(content)
            
            # Link analysis
            internal_links, external_links = self._extract_links(content, file_path)
            broken_links = self._check_broken_links(internal_links, external_links)
            
            # Content analysis
            missing_sections = self._check_missing_sections(content, heading_structure)
            readability_score = self._calculate_readability(content)
            
            return DocumentMetrics(
                file_path=str(file_path.relative_to(self.project_root)),
                word_count=word_count,
                line_count=line_count,
                last_modified=last_modified,
                age_days=age_days,
                has_title=has_title,
                has_toc=has_toc,
                heading_structure=heading_structure,
                internal_links=internal_links,
                external_links=external_links,
                broken_links=broken_links,
                missing_sections=missing_sections,
                readability_score=readability_score
            )
            
        except Exception as e:
            logger.error(f"Failed to analyze {file_path}: {e}")
            return DocumentMetrics(
                file_path=str(file_path.relative_to(self.project_root)),
                word_count=0,
                line_count=0,
                last_modified=datetime.now(),
                age_days=0,
                has_title=False,
                has_toc=False,
                heading_structure=[],
                internal_links=[],
                external_links=[],
                broken_links=[],
                missing_sections=[],
                readability_score=0.0
            )
    
    def _has_title(self, content: str) -> bool:
        """Check if document has a title"""
        lines = content.strip().split('\n')
        if not lines:
            return False
        
        first_line = lines[0].strip()
        # Check for markdown title (# Title) or underlined title
        return (first_line.startswith('#') or 
                (len(lines) > 1 and lines[1].strip() and 
                 all(c in '=-' for c in lines[1].strip())))
    
    def _has_table_of_contents(self, content: str) -> bool:
        """Check if document has a table of contents"""
        toc_indicators = [
            'table of contents', 'toc', 'contents',
            '- [', '* [', '1. ['  # Common TOC patterns
        ]
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in toc_indicators)
    
    def _extract_headings(self, content: str) -> List[str]:
        """Extract heading structure"""
        headings = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('#'):
                # Markdown heading
                level = len(line) - len(line.lstrip('#'))
                title = line.lstrip('#').strip()
                headings.append(f"{'  ' * (level-1)}• {title}")
            elif line and len(lines) > lines.index(line) + 1:
                # Check for underlined heading
                next_line = lines[lines.index(line) + 1].strip()
                if next_line and all(c in '=-' for c in next_line):
                    headings.append(f"• {line}")
        
        return headings
    
    def _extract_links(self, content: str, file_path: Path) -> tuple:
        """Extract internal and external links"""
        internal_links = []
        external_links = []
        
        # Markdown links: [text](url)
        markdown_links = re.findall(r'\[([^\]]*)\]\(([^)]+)\)', content)
        
        for text, url in markdown_links:
            if url.startswith(('http://', 'https://')):
                external_links.append(url)
            elif url.startswith(('#', 'mailto:')):
                # Skip anchors and mailto links
                continue
            else:
                # Internal link
                internal_links.append(url)
        
        # HTML links in markdown
        html_links = re.findall(r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>', content)
        for url in html_links:
            if url.startswith(('http://', 'https://')):
                external_links.append(url)
            elif not url.startswith(('#', 'mailto:')):
                internal_links.append(url)
        
        return internal_links, external_links
    
    def _check_broken_links(self, internal_links: List[str], external_links: List[str]) -> List[str]:
        """Check for broken links"""
        broken_links = []
        
        # Check internal links
        for link in internal_links:
            link_path = self.project_root / link
            if not link_path.exists():
                broken_links.append(f"Internal: {link}")
        
        # Check external links (sample only to avoid rate limiting)
        for link in external_links[:5]:  # Check only first 5 external links
            try:
                response = self.session.head(link, timeout=10, allow_redirects=True)
                if response.status_code >= 400:
                    broken_links.append(f"External: {link} ({response.status_code})")
            except Exception as e:
                broken_links.append(f"External: {link} (Error: {str(e)[:50]})")
        
        return broken_links
    
    def _check_missing_sections(self, content: str, headings: List[str]) -> List[str]:
        """Check for missing required sections"""
        content_lower = content.lower()
        headings_text = ' '.join(headings).lower()
        
        missing_sections = []
        for section in self.required_sections:
            if section not in content_lower and section not in headings_text:
                missing_sections.append(section.title())
        
        return missing_sections
    
    def _calculate_readability(self, content: str) -> float:
        """Calculate basic readability score"""
        # Simple readability metrics
        sentences = len(re.findall(r'[.!?]+', content))
        words = len(content.split())
        
        if sentences == 0 or words == 0:
            return 0.0
        
        avg_words_per_sentence = words / sentences
        
        # Simple scoring: ideal is 15-20 words per sentence
        if 15 <= avg_words_per_sentence <= 20:
            score = 100
        elif 10 <= avg_words_per_sentence <= 25:
            score = 80
        elif 5 <= avg_words_per_sentence <= 30:
            score = 60
        else:
            score = 40
        
        # Adjust for content length
        if words < 100:
            score *= 0.8  # Penalize very short documents
        elif words > 5000:
            score *= 0.9  # Slightly penalize very long documents
        
        return round(score, 2)
    
    def calculate_scores(self, document_metrics: Dict[str, DocumentMetrics]) -> tuple:
        """Calculate overall scores"""
        if not document_metrics:
            return 0.0, 0.0, 0.0, 0.0
        
        total_docs = len(document_metrics)
        
        # Coverage score: based on required sections presence
        docs_with_good_coverage = 0
        for metrics in document_metrics.values():
            if len(metrics.missing_sections) <= 2:  # Allow 2 missing sections
                docs_with_good_coverage += 1
        coverage_score = (docs_with_good_coverage / total_docs) * 100
        
        # Quality score: based on structure and links
        quality_scores = []
        for metrics in document_metrics.values():
            doc_quality = 0
            if metrics.has_title:
                doc_quality += 20
            if metrics.has_toc and len(metrics.heading_structure) > 3:
                doc_quality += 20
            if len(metrics.broken_links) == 0:
                doc_quality += 30
            if metrics.word_count >= 200:
                doc_quality += 20
            doc_quality += min(metrics.readability_score * 0.1, 10)
            quality_scores.append(doc_quality)
        
        quality_score = sum(quality_scores) / len(quality_scores)
        
        # Freshness score: based on document age
        freshness_scores = []
        for metrics in document_metrics.values():
            if metrics.age_days <= 30:
                freshness_scores.append(100)
            elif metrics.age_days <= 90:
                freshness_scores.append(80)
            elif metrics.age_days <= 180:
                freshness_scores.append(60)
            elif metrics.age_days <= 365:
                freshness_scores.append(40)
            else:
                freshness_scores.append(20)
        
        freshness_score = sum(freshness_scores) / len(freshness_scores)
        
        # Overall score: weighted average
        overall_score = (coverage_score * 0.4 + quality_score * 0.4 + freshness_score * 0.2)
        
        return coverage_score, quality_score, freshness_score, overall_score
    
    def generate_recommendations(self, document_metrics: Dict[str, DocumentMetrics], 
                               coverage_score: float, quality_score: float, 
                               freshness_score: float) -> List[str]:
        """Generate documentation recommendations"""
        recommendations = []
        
        # Coverage recommendations
        if coverage_score < 70:
            recommendations.append("📚 Add missing documentation sections (Installation, Usage, API, etc.)")
        
        # Quality recommendations
        if quality_score < 70:
            recommendations.append("✨ Improve documentation structure with proper headings and TOC")
        
        # Freshness recommendations
        if freshness_score < 70:
            outdated_docs = [path for path, metrics in document_metrics.items() 
                           if metrics.age_days > 90]
            if outdated_docs:
                recommendations.append(f"🔄 Update {len(outdated_docs)} outdated documents")
        
        # Broken links
        total_broken = sum(len(metrics.broken_links) for metrics in document_metrics.values())
        if total_broken > 0:
            recommendations.append(f"🔗 Fix {total_broken} broken links")
        
        # Missing titles
        docs_without_titles = [path for path, metrics in document_metrics.items() 
                             if not metrics.has_title]
        if docs_without_titles:
            recommendations.append(f"📝 Add titles to {len(docs_without_titles)} documents")
        
        # Short documents
        short_docs = [path for path, metrics in document_metrics.items() 
                     if metrics.word_count < 100]
        if short_docs:
            recommendations.append(f"📖 Expand {len(short_docs)} documents with more content")
        
        if not recommendations:
            recommendations.append("✅ Documentation is in excellent condition!")
        
        return recommendations
    
    def check_missing_documentation(self) -> List[str]:
        """Check for missing documentation files"""
        missing_docs = []
        
        expected_docs = [
            "README.md",
            "docs/installation.md",
            "docs/usage.md", 
            "docs/api.md",
            "docs/contributing.md",
            "docs/changelog.md",
            "docs/troubleshooting.md"
        ]
        
        for doc_path in expected_docs:
            full_path = self.project_root / doc_path
            if not full_path.exists():
                missing_docs.append(doc_path)
        
        return missing_docs
    
    def generate_health_report(self) -> DocumentationHealth:
        """Generate comprehensive documentation health report"""
        logger.info("Analyzing documentation health...")
        
        # Find and analyze all documents
        doc_files = self.find_documentation_files()
        document_metrics = {}
        
        for doc_file in doc_files:
            metrics = self.analyze_document(doc_file)
            document_metrics[metrics.file_path] = metrics
        
        # Calculate scores
        coverage_score, quality_score, freshness_score, overall_score = self.calculate_scores(document_metrics)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(
            document_metrics, coverage_score, quality_score, freshness_score
        )
        
        # Check for missing documentation
        missing_documentation = self.check_missing_documentation()
        
        # Calculate summary statistics
        total_documents = len(document_metrics)
        outdated_documents = len([m for m in document_metrics.values() if m.age_days > 90])
        documents_with_broken_links = len([m for m in document_metrics.values() if m.broken_links])
        total_broken_links = sum(len(m.broken_links) for m in document_metrics.values())
        average_age_days = sum(m.age_days for m in document_metrics.values()) / total_documents if total_documents > 0 else 0
        
        return DocumentationHealth(
            timestamp=datetime.now(),
            total_documents=total_documents,
            outdated_documents=outdated_documents,
            documents_with_broken_links=documents_with_broken_links,
            total_broken_links=total_broken_links,
            average_age_days=average_age_days,
            coverage_score=coverage_score,
            quality_score=quality_score,
            freshness_score=freshness_score,
            overall_score=overall_score,
            document_metrics=document_metrics,
            recommendations=recommendations,
            missing_documentation=missing_documentation
        )
    
    def save_report(self, health_report: DocumentationHealth, output_path: str) -> None:
        """Save documentation health report"""
        output_data = {
            "timestamp": health_report.timestamp.isoformat(),
            "summary": {
                "total_documents": health_report.total_documents,
                "outdated_documents": health_report.outdated_documents,
                "documents_with_broken_links": health_report.documents_with_broken_links,
                "total_broken_links": health_report.total_broken_links,
                "average_age_days": health_report.average_age_days
            },
            "scores": {
                "coverage_score": health_report.coverage_score,
                "quality_score": health_report.quality_score,
                "freshness_score": health_report.freshness_score,
                "overall_score": health_report.overall_score
            },
            "recommendations": health_report.recommendations,
            "missing_documentation": health_report.missing_documentation,
            "document_details": {}
        }
        
        for file_path, metrics in health_report.document_metrics.items():
            output_data["document_details"][file_path] = {
                "word_count": metrics.word_count,
                "line_count": metrics.line_count,
                "last_modified": metrics.last_modified.isoformat(),
                "age_days": metrics.age_days,
                "has_title": metrics.has_title,
                "has_toc": metrics.has_toc,
                "heading_count": len(metrics.heading_structure),
                "internal_links_count": len(metrics.internal_links),
                "external_links_count": len(metrics.external_links),
                "broken_links_count": len(metrics.broken_links),
                "missing_sections": metrics.missing_sections,
                "readability_score": metrics.readability_score
            }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            import json
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Documentation health report saved to {output_path}")


def monitor_documentation_health(project_root: str, docs_directories: List[str], 
                                output_path: str) -> DocumentationHealth:
    """Main function to monitor documentation health"""
    checker = DocumentationHealthChecker(project_root, docs_directories)
    health_report = checker.generate_health_report()
    checker.save_report(health_report, output_path)
    
    logger.info(f"Documentation analysis complete. Overall score: {health_report.overall_score:.1f}/100")
    return health_report


if __name__ == "__main__":
    # Example usage
    health_report = monitor_documentation_health(
        project_root=".",
        docs_directories=["docs", "README.md"],
        output_path="monitoring/data/documentation_health.json"
    )
    
    print(f"Documentation Health Score: {health_report.overall_score:.1f}/100")
    print(f"Coverage: {health_report.coverage_score:.1f}/100")
    print(f"Quality: {health_report.quality_score:.1f}/100")
    print(f"Freshness: {health_report.freshness_score:.1f}/100")
    print(f"Total Documents: {health_report.total_documents}")
    print(f"Broken Links: {health_report.total_broken_links}")
    print("\nRecommendations:")
    for rec in health_report.recommendations:
        print(f"  {rec}")
