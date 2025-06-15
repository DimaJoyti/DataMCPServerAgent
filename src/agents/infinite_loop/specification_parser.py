"""
Specification Parser

Analyzes specification files to extract content generation requirements,
format specifications, evolution patterns, and quality constraints.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Union

import markdown
import yaml
from bs4 import BeautifulSoup


class SpecificationParser:
    """
    Parses specification files to extract generation requirements.

    Supports multiple formats:
    - Markdown (.md)
    - YAML (.yaml, .yml)
    - JSON (.json)
    - Plain text (.txt)
    """

    def __init__(self):
        """Initialize the specification parser."""
        self.logger = logging.getLogger("specification_parser")

        # Pattern matchers for extracting information
        self.content_type_patterns = {
            "code": [r"code", r"programming", r"script", r"function", r"class"],
            "documentation": [r"docs?", r"documentation", r"guide", r"manual", r"readme"],
            "configuration": [r"config", r"settings", r"parameters", r"options"],
            "data": [r"data", r"dataset", r"csv", r"json", r"xml"],
            "text": [r"text", r"content", r"article", r"story", r"description"],
            "template": [r"template", r"boilerplate", r"scaffold", r"skeleton"],
            "test": [r"test", r"spec", r"validation", r"verification"],
            "api": [r"api", r"endpoint", r"service", r"interface"],
        }

        self.format_patterns = {
            "markdown": [r"\.md", r"markdown", r"md format"],
            "json": [r"\.json", r"json format", r"javascript object"],
            "yaml": [r"\.ya?ml", r"yaml format", r"yml format"],
            "python": [r"\.py", r"python", r"python code"],
            "javascript": [r"\.js", r"javascript", r"js code"],
            "html": [r"\.html?", r"html", r"web page"],
            "css": [r"\.css", r"css", r"stylesheet"],
            "xml": [r"\.xml", r"xml format"],
            "csv": [r"\.csv", r"csv format", r"comma separated"],
            "txt": [r"\.txt", r"plain text", r"text file"],
        }

        self.evolution_patterns = {
            "incremental": [r"incremental", r"gradual", r"step by step", r"progressive"],
            "branching": [r"branch", r"variant", r"alternative", r"fork"],
            "refinement": [r"refine", r"improve", r"enhance", r"optimize"],
            "expansion": [r"expand", r"extend", r"add", r"grow"],
            "transformation": [r"transform", r"convert", r"change", r"modify"],
            "combination": [r"combine", r"merge", r"integrate", r"synthesize"],
        }

    async def parse_specification(self, spec_file: Union[str, Path]) -> Dict[str, Any]:
        """
        Parse a specification file and extract generation requirements.

        Args:
            spec_file: Path to the specification file

        Returns:
            Parsed specification with extracted requirements
        """
        spec_path = Path(spec_file)

        if not spec_path.exists():
            raise FileNotFoundError(f"Specification file not found: {spec_file}")

        self.logger.info(f"Parsing specification file: {spec_path}")

        # Read file content
        content = spec_path.read_text(encoding="utf-8")

        # Parse based on file extension
        if spec_path.suffix.lower() in [".md", ".markdown"]:
            parsed_spec = await self._parse_markdown(content)
        elif spec_path.suffix.lower() in [".yaml", ".yml"]:
            parsed_spec = await self._parse_yaml(content)
        elif spec_path.suffix.lower() == ".json":
            parsed_spec = await self._parse_json(content)
        else:
            parsed_spec = await self._parse_text(content)

        # Add metadata
        parsed_spec["source_file"] = str(spec_path)
        parsed_spec["file_format"] = spec_path.suffix.lower()
        parsed_spec["content_length"] = len(content)

        # Extract high-level patterns
        parsed_spec.update(await self._extract_patterns(content))

        # Validate and normalize
        parsed_spec = await self._validate_and_normalize(parsed_spec)

        self.logger.info(
            f"Specification parsing complete: {parsed_spec.get('content_type', 'unknown')} content"
        )

        return parsed_spec

    async def _parse_markdown(self, content: str) -> Dict[str, Any]:
        """Parse markdown specification."""
        # Convert to HTML for easier parsing
        html = markdown.markdown(content)
        soup = BeautifulSoup(html, "html.parser")

        spec = {
            "format": "markdown",
            "content_type": "documentation",
            "sections": [],
            "headers": [],
            "requirements": [],
            "constraints": [],
        }

        # Extract headers
        for header in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
            spec["headers"].append(
                {
                    "level": int(header.name[1]),
                    "text": header.get_text().strip(),
                }
            )

        # Extract sections
        current_section = None
        for element in soup.find_all(["h1", "h2", "h3", "p", "ul", "ol"]):
            if element.name.startswith("h"):
                current_section = {
                    "title": element.get_text().strip(),
                    "level": int(element.name[1]),
                    "content": [],
                }
                spec["sections"].append(current_section)
            elif current_section:
                current_section["content"].append(
                    {
                        "type": element.name,
                        "text": element.get_text().strip(),
                    }
                )

        # Extract requirements and constraints
        for section in spec["sections"]:
            title_lower = section["title"].lower()
            if any(keyword in title_lower for keyword in ["requirement", "must", "should", "need"]):
                for content_item in section["content"]:
                    if content_item["text"]:
                        spec["requirements"].append(content_item["text"])
            elif any(keyword in title_lower for keyword in ["constraint", "limit", "restriction"]):
                for content_item in section["content"]:
                    if content_item["text"]:
                        spec["constraints"].append(content_item["text"])

        return spec

    async def _parse_yaml(self, content: str) -> Dict[str, Any]:
        """Parse YAML specification."""
        try:
            data = yaml.safe_load(content)

            spec = {
                "format": "yaml",
                "raw_data": data,
            }

            # Extract common fields
            if isinstance(data, dict):
                spec.update(
                    {
                        "content_type": data.get("content_type", "unknown"),
                        "format_requirements": data.get("format", {}),
                        "evolution_pattern": data.get("evolution_pattern", "incremental"),
                        "requirements": data.get("requirements", []),
                        "constraints": data.get("constraints", []),
                        "quality_requirements": data.get("quality", {}),
                        "innovation_areas": data.get("innovation_areas", []),
                        "naming_pattern": data.get("naming_pattern", ""),
                        "output_structure": data.get("output_structure", {}),
                    }
                )

            return spec

        except yaml.YAMLError as e:
            self.logger.error(f"Failed to parse YAML: {e}")
            return await self._parse_text(content)

    async def _parse_json(self, content: str) -> Dict[str, Any]:
        """Parse JSON specification."""
        try:
            data = json.loads(content)

            spec = {
                "format": "json",
                "raw_data": data,
            }

            # Extract common fields
            if isinstance(data, dict):
                spec.update(
                    {
                        "content_type": data.get("content_type", "unknown"),
                        "format_requirements": data.get("format", {}),
                        "evolution_pattern": data.get("evolution_pattern", "incremental"),
                        "requirements": data.get("requirements", []),
                        "constraints": data.get("constraints", []),
                        "quality_requirements": data.get("quality", {}),
                        "innovation_areas": data.get("innovation_areas", []),
                        "naming_pattern": data.get("naming_pattern", ""),
                        "output_structure": data.get("output_structure", {}),
                    }
                )

            return spec

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON: {e}")
            return await self._parse_text(content)

    async def _parse_text(self, content: str) -> Dict[str, Any]:
        """Parse plain text specification."""
        lines = content.split("\n")

        spec = {
            "format": "text",
            "content_type": "text",
            "lines": lines,
            "requirements": [],
            "constraints": [],
        }

        # Extract requirements and constraints from text
        for line in lines:
            line_lower = line.lower().strip()
            if any(keyword in line_lower for keyword in ["must", "required", "should", "need to"]):
                spec["requirements"].append(line.strip())
            elif any(
                keyword in line_lower for keyword in ["cannot", "must not", "forbidden", "limit"]
            ):
                spec["constraints"].append(line.strip())

        return spec

    async def _extract_patterns(self, content: str) -> Dict[str, Any]:
        """Extract high-level patterns from content."""
        content_lower = content.lower()

        patterns = {
            "content_type": "unknown",
            "format": "unknown",
            "evolution_pattern": "incremental",
        }

        # Detect content type
        for content_type, keywords in self.content_type_patterns.items():
            if any(re.search(pattern, content_lower) for pattern in keywords):
                patterns["content_type"] = content_type
                break

        # Detect format
        for format_type, keywords in self.format_patterns.items():
            if any(re.search(pattern, content_lower) for pattern in keywords):
                patterns["format"] = format_type
                break

        # Detect evolution pattern
        for evolution_type, keywords in self.evolution_patterns.items():
            if any(re.search(pattern, content_lower) for pattern in keywords):
                patterns["evolution_pattern"] = evolution_type
                break

        return patterns

    async def _validate_and_normalize(self, spec: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and normalize the parsed specification."""
        # Ensure required fields exist
        defaults = {
            "content_type": "unknown",
            "format": "unknown",
            "evolution_pattern": "incremental",
            "requirements": [],
            "constraints": [],
            "quality_requirements": {},
            "innovation_areas": [],
            "naming_pattern": "iteration_{number}",
            "output_structure": {},
        }

        for key, default_value in defaults.items():
            if key not in spec:
                spec[key] = default_value

        # Normalize lists
        for list_field in ["requirements", "constraints", "innovation_areas"]:
            if not isinstance(spec[list_field], list):
                spec[list_field] = []

        # Normalize dictionaries
        for dict_field in ["quality_requirements", "output_structure"]:
            if not isinstance(spec[dict_field], dict):
                spec[dict_field] = {}

        # Set default naming pattern if empty
        if not spec["naming_pattern"]:
            spec["naming_pattern"] = "iteration_{number}"

        return spec
