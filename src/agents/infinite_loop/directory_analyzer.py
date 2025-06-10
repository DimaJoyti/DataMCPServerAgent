"""
Directory Analyzer

Analyzes output directories to understand existing iterations, naming patterns,
content evolution, and identify opportunities for new iterations.
"""

import logging
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import hashlib


class DirectoryAnalyzer:
    """
    Analyzes output directories to understand the current state of iterations.
    
    Features:
    - File discovery and naming pattern analysis
    - Iteration number extraction and sequencing
    - Content evolution tracking
    - Gap identification and opportunity analysis
    - Metadata extraction and statistics
    """
    
    def __init__(self):
        """Initialize the directory analyzer."""
        self.logger = logging.getLogger("directory_analyzer")
        
        # Common naming patterns for iteration files
        self.iteration_patterns = [
            r"iteration[_-]?(\d+)",
            r"iter[_-]?(\d+)",
            r"v(\d+)",
            r"version[_-]?(\d+)",
            r"gen[_-]?(\d+)",
            r"generation[_-]?(\d+)",
            r"(\d+)[_-]?iteration",
            r"(\d+)[_-]?iter",
            r"(\d+)[_-]?v",
            r"(\d+)[_-]?version",
            r"(\d+)[_-]?gen",
            r"(\d+)[_-]?generation",
            r"(\d+)",  # Pure number
        ]
        
        # File extensions to consider
        self.supported_extensions = {
            ".md", ".txt", ".py", ".js", ".html", ".css", ".json", ".yaml", ".yml",
            ".xml", ".csv", ".sql", ".sh", ".bat", ".ps1", ".dockerfile", ".toml",
            ".ini", ".cfg", ".conf", ".log", ".rst", ".tex", ".r", ".rb", ".go",
            ".java", ".cpp", ".c", ".h", ".hpp", ".cs", ".php", ".swift", ".kt",
        }
    
    async def analyze_directory(self, output_dir: Union[str, Path]) -> Dict[str, Any]:
        """
        Analyze an output directory to understand the current iteration state.
        
        Args:
            output_dir: Path to the output directory
            
        Returns:
            Analysis results with file information, patterns, and opportunities
        """
        dir_path = Path(output_dir)
        
        # Create directory if it doesn't exist
        if not dir_path.exists():
            self.logger.info(f"Creating output directory: {dir_path}")
            dir_path.mkdir(parents=True, exist_ok=True)
            
            return {
                "directory_path": str(dir_path),
                "exists": False,
                "is_empty": True,
                "existing_files": [],
                "iteration_files": [],
                "highest_iteration": 0,
                "naming_patterns": [],
                "content_evolution": [],
                "gaps": [],
                "opportunities": ["First iteration - no existing content"],
                "statistics": self._empty_statistics(),
            }
        
        self.logger.info(f"Analyzing directory: {dir_path}")
        
        # Scan directory
        all_files = await self._scan_directory(dir_path)
        iteration_files = await self._identify_iteration_files(all_files)
        naming_patterns = await self._analyze_naming_patterns(iteration_files)
        content_evolution = await self._analyze_content_evolution(iteration_files)
        gaps = await self._identify_gaps(iteration_files)
        opportunities = await self._identify_opportunities(iteration_files, content_evolution)
        statistics = await self._calculate_statistics(all_files, iteration_files)
        
        highest_iteration = max(
            (file_info["iteration_number"] for file_info in iteration_files),
            default=0
        )
        
        analysis = {
            "directory_path": str(dir_path),
            "exists": True,
            "is_empty": len(all_files) == 0,
            "existing_files": all_files,
            "iteration_files": iteration_files,
            "highest_iteration": highest_iteration,
            "naming_patterns": naming_patterns,
            "content_evolution": content_evolution,
            "gaps": gaps,
            "opportunities": opportunities,
            "statistics": statistics,
        }
        
        self.logger.info(f"Directory analysis complete:")
        self.logger.info(f"- Total files: {len(all_files)}")
        self.logger.info(f"- Iteration files: {len(iteration_files)}")
        self.logger.info(f"- Highest iteration: {highest_iteration}")
        self.logger.info(f"- Naming patterns: {len(naming_patterns)}")
        
        return analysis
    
    async def _scan_directory(self, dir_path: Path) -> List[Dict[str, Any]]:
        """Scan directory and collect file information."""
        files = []
        
        try:
            for item in dir_path.rglob("*"):
                if item.is_file() and item.suffix.lower() in self.supported_extensions:
                    try:
                        stat = item.stat()
                        content_hash = await self._calculate_file_hash(item)
                        
                        file_info = {
                            "path": str(item),
                            "name": item.name,
                            "stem": item.stem,
                            "suffix": item.suffix,
                            "size": stat.st_size,
                            "modified": datetime.fromtimestamp(stat.st_mtime),
                            "created": datetime.fromtimestamp(stat.st_ctime),
                            "content_hash": content_hash,
                            "relative_path": str(item.relative_to(dir_path)),
                        }
                        
                        files.append(file_info)
                        
                    except (OSError, PermissionError) as e:
                        self.logger.warning(f"Could not access file {item}: {e}")
                        
        except (OSError, PermissionError) as e:
            self.logger.error(f"Could not scan directory {dir_path}: {e}")
        
        return files
    
    async def _identify_iteration_files(self, all_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify files that appear to be iterations."""
        iteration_files = []
        
        for file_info in all_files:
            iteration_number = await self._extract_iteration_number(file_info["name"])
            
            if iteration_number is not None:
                file_info["iteration_number"] = iteration_number
                file_info["is_iteration"] = True
                iteration_files.append(file_info)
        
        # Sort by iteration number
        iteration_files.sort(key=lambda x: x["iteration_number"])
        
        return iteration_files
    
    async def _extract_iteration_number(self, filename: str) -> Optional[int]:
        """Extract iteration number from filename using various patterns."""
        filename_lower = filename.lower()
        
        for pattern in self.iteration_patterns:
            match = re.search(pattern, filename_lower)
            if match:
                try:
                    return int(match.group(1))
                except (ValueError, IndexError):
                    continue
        
        return None
    
    async def _analyze_naming_patterns(self, iteration_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze naming patterns in iteration files."""
        patterns = []
        
        if not iteration_files:
            return patterns
        
        # Group by pattern
        pattern_groups = defaultdict(list)
        
        for file_info in iteration_files:
            # Extract pattern by replacing iteration number with placeholder
            name = file_info["name"]
            iteration_num = file_info["iteration_number"]
            
            # Try to find the pattern
            for pattern_regex in self.iteration_patterns:
                if re.search(pattern_regex, name.lower()):
                    # Replace the number with a placeholder
                    pattern_name = re.sub(
                        pattern_regex, 
                        lambda m: pattern_regex.replace(r"(\d+)", "{number}"),
                        name.lower()
                    )
                    pattern_groups[pattern_name].append(file_info)
                    break
        
        # Analyze each pattern group
        for pattern_name, files in pattern_groups.items():
            patterns.append({
                "pattern": pattern_name,
                "count": len(files),
                "iterations": [f["iteration_number"] for f in files],
                "example_files": [f["name"] for f in files[:3]],
            })
        
        return patterns
    
    async def _analyze_content_evolution(self, iteration_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze how content has evolved across iterations."""
        evolution = []
        
        if len(iteration_files) < 2:
            return evolution
        
        # Compare consecutive iterations
        for i in range(1, len(iteration_files)):
            prev_file = iteration_files[i-1]
            curr_file = iteration_files[i]
            
            # Calculate differences
            size_change = curr_file["size"] - prev_file["size"]
            size_change_percent = (size_change / prev_file["size"] * 100) if prev_file["size"] > 0 else 0
            
            # Check if content is similar (same hash = identical)
            is_identical = prev_file["content_hash"] == curr_file["content_hash"]
            
            evolution_step = {
                "from_iteration": prev_file["iteration_number"],
                "to_iteration": curr_file["iteration_number"],
                "size_change": size_change,
                "size_change_percent": round(size_change_percent, 2),
                "is_identical": is_identical,
                "time_gap": (curr_file["modified"] - prev_file["modified"]).total_seconds(),
            }
            
            evolution.append(evolution_step)
        
        return evolution
    
    async def _identify_gaps(self, iteration_files: List[Dict[str, Any]]) -> List[int]:
        """Identify missing iteration numbers (gaps in sequence)."""
        if not iteration_files:
            return []
        
        iteration_numbers = [f["iteration_number"] for f in iteration_files]
        min_iter = min(iteration_numbers)
        max_iter = max(iteration_numbers)
        
        expected_numbers = set(range(min_iter, max_iter + 1))
        actual_numbers = set(iteration_numbers)
        
        gaps = sorted(expected_numbers - actual_numbers)
        return gaps
    
    async def _identify_opportunities(
        self, 
        iteration_files: List[Dict[str, Any]], 
        content_evolution: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify opportunities for new iterations."""
        opportunities = []
        
        if not iteration_files:
            opportunities.append("First iteration - establish baseline content")
            return opportunities
        
        # Check for identical consecutive iterations
        identical_pairs = [
            evo for evo in content_evolution if evo["is_identical"]
        ]
        if identical_pairs:
            opportunities.append(f"Found {len(identical_pairs)} identical consecutive iterations - opportunity for differentiation")
        
        # Check for large size changes
        large_changes = [
            evo for evo in content_evolution if abs(evo["size_change_percent"]) > 50
        ]
        if large_changes:
            opportunities.append(f"Found {len(large_changes)} iterations with large size changes - opportunity for gradual evolution")
        
        # Check for gaps
        gaps = await self._identify_gaps(iteration_files)
        if gaps:
            opportunities.append(f"Found {len(gaps)} gaps in iteration sequence - opportunity to fill missing iterations")
        
        # Check for recent activity
        if iteration_files:
            latest_file = max(iteration_files, key=lambda x: x["modified"])
            time_since_latest = (datetime.now() - latest_file["modified"]).total_seconds()
            
            if time_since_latest > 86400:  # More than 1 day
                opportunities.append("No recent iterations - opportunity for fresh content")
            elif time_since_latest < 3600:  # Less than 1 hour
                opportunities.append("Recent iteration activity - opportunity for rapid iteration")
        
        # Default opportunity
        if not opportunities:
            opportunities.append("Continue iteration sequence with novel improvements")
        
        return opportunities
    
    async def _calculate_statistics(
        self, 
        all_files: List[Dict[str, Any]], 
        iteration_files: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate directory statistics."""
        if not all_files:
            return self._empty_statistics()
        
        total_size = sum(f["size"] for f in all_files)
        avg_size = total_size / len(all_files) if all_files else 0
        
        # File type distribution
        extensions = defaultdict(int)
        for file_info in all_files:
            extensions[file_info["suffix"]] += 1
        
        # Iteration statistics
        iteration_stats = {}
        if iteration_files:
            iteration_sizes = [f["size"] for f in iteration_files]
            iteration_stats = {
                "count": len(iteration_files),
                "avg_size": sum(iteration_sizes) / len(iteration_sizes),
                "min_size": min(iteration_sizes),
                "max_size": max(iteration_sizes),
                "size_variance": self._calculate_variance(iteration_sizes),
            }
        
        return {
            "total_files": len(all_files),
            "total_size": total_size,
            "average_size": avg_size,
            "file_types": dict(extensions),
            "iteration_statistics": iteration_stats,
        }
    
    def _empty_statistics(self) -> Dict[str, Any]:
        """Return empty statistics for empty directories."""
        return {
            "total_files": 0,
            "total_size": 0,
            "average_size": 0,
            "file_types": {},
            "iteration_statistics": {},
        }
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    async def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file content."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.sha256(content).hexdigest()[:16]  # First 16 chars
        except (OSError, PermissionError):
            return "unknown"
