"""
Code Quality Monitor

Automated code quality checking and metrics tracking.
"""

import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Code quality metrics"""
    timestamp: datetime
    tool: str
    status: str  # "success", "warning", "error"
    issues_count: int
    files_checked: int
    execution_time_seconds: float
    details: Dict[str, Any]


@dataclass
class QualityReport:
    """Comprehensive quality report"""
    timestamp: datetime
    overall_score: float  # 0-100
    total_issues: int
    critical_issues: int
    warnings: int
    tool_results: Dict[str, QualityMetrics]
    trends: Dict[str, Any]


class CodeQualityMonitor:
    """Monitor and track code quality metrics"""
    
    def __init__(self, project_root: str, directories: List[str]):
        self.project_root = Path(project_root)
        self.directories = directories
        self.tools_config = {
            "black": {
                "command": ["black", "--check", "--diff"],
                "weight": 20
            },
            "isort": {
                "command": ["isort", "--check-only", "--diff"],
                "weight": 15
            },
            "ruff": {
                "command": ["ruff", "check", "--output-format=json"],
                "weight": 30
            },
            "mypy": {
                "command": ["mypy", "--ignore-missing-imports"],
                "weight": 25
            },
            "bandit": {
                "command": ["bandit", "-r", "-f", "json"],
                "weight": 10
            }
        }
    
    def run_tool(self, tool_name: str, directories: List[str]) -> QualityMetrics:
        """Run a specific quality tool"""
        start_time = time.time()
        tool_config = self.tools_config.get(tool_name)
        
        if not tool_config:
            return QualityMetrics(
                timestamp=datetime.now(),
                tool=tool_name,
                status="error",
                issues_count=0,
                files_checked=0,
                execution_time_seconds=0,
                details={"error": f"Unknown tool: {tool_name}"}
            )
        
        command = tool_config["command"] + directories
        
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            execution_time = time.time() - start_time
            
            # Parse results based on tool
            issues_count, files_checked, details = self._parse_tool_output(
                tool_name, result.stdout, result.stderr, result.returncode
            )
            
            status = "success" if result.returncode == 0 else "warning"
            
            return QualityMetrics(
                timestamp=datetime.now(),
                tool=tool_name,
                status=status,
                issues_count=issues_count,
                files_checked=files_checked,
                execution_time_seconds=execution_time,
                details=details
            )
            
        except subprocess.TimeoutExpired:
            return QualityMetrics(
                timestamp=datetime.now(),
                tool=tool_name,
                status="error",
                issues_count=0,
                files_checked=0,
                execution_time_seconds=time.time() - start_time,
                details={"error": "Tool execution timeout"}
            )
        except Exception as e:
            return QualityMetrics(
                timestamp=datetime.now(),
                tool=tool_name,
                status="error",
                issues_count=0,
                files_checked=0,
                execution_time_seconds=time.time() - start_time,
                details={"error": str(e)}
            )
    
    def _parse_tool_output(self, tool_name: str, stdout: str, stderr: str, returncode: int) -> tuple:
        """Parse tool output to extract metrics"""
        issues_count = 0
        files_checked = 0
        details = {"stdout": stdout, "stderr": stderr, "returncode": returncode}
        
        try:
            if tool_name == "ruff" and stdout:
                # Ruff outputs JSON
                ruff_data = json.loads(stdout)
                issues_count = len(ruff_data)
                files_checked = len(set(item.get("filename", "") for item in ruff_data))
                details["issues"] = ruff_data
                
            elif tool_name == "bandit" and stdout:
                # Bandit outputs JSON
                bandit_data = json.loads(stdout)
                issues_count = len(bandit_data.get("results", []))
                files_checked = len(bandit_data.get("metrics", {}).get("_totals", {}).get("loc", 0))
                details["results"] = bandit_data
                
            elif tool_name in ["black", "isort"]:
                # Count files mentioned in diff output
                if stdout:
                    lines = stdout.split('\n')
                    files_mentioned = set()
                    for line in lines:
                        if line.startswith("would reformat") or line.startswith("ERROR"):
                            issues_count += 1
                        if "would reformat" in line or ".py" in line:
                            # Extract filename
                            for part in line.split():
                                if part.endswith(".py"):
                                    files_mentioned.add(part)
                    files_checked = len(files_mentioned)
                
            elif tool_name == "mypy":
                # Count error lines
                if stdout:
                    lines = stdout.split('\n')
                    for line in lines:
                        if ": error:" in line or ": warning:" in line:
                            issues_count += 1
                        if ".py:" in line:
                            files_checked += 1
                
        except Exception as e:
            details["parse_error"] = str(e)
        
        return issues_count, files_checked, details
    
    def run_all_tools(self) -> Dict[str, QualityMetrics]:
        """Run all quality tools"""
        results = {}
        
        for tool_name in self.tools_config.keys():
            logger.info(f"Running {tool_name}...")
            
            # Adjust directories for specific tools
            dirs = self.directories.copy()
            if tool_name == "mypy":
                # MyPy works better with specific directories
                dirs = ["app", "src"]
            
            results[tool_name] = self.run_tool(tool_name, dirs)
            logger.info(f"{tool_name} completed: {results[tool_name].status}")
        
        return results
    
    def calculate_overall_score(self, tool_results: Dict[str, QualityMetrics]) -> float:
        """Calculate overall quality score (0-100)"""
        total_weight = sum(config["weight"] for config in self.tools_config.values())
        weighted_score = 0
        
        for tool_name, metrics in tool_results.items():
            tool_weight = self.tools_config[tool_name]["weight"]
            
            if metrics.status == "error":
                tool_score = 0
            elif metrics.status == "success":
                tool_score = 100
            else:  # warning
                # Score based on issues count (fewer issues = higher score)
                if metrics.issues_count == 0:
                    tool_score = 100
                elif metrics.issues_count <= 5:
                    tool_score = 80
                elif metrics.issues_count <= 20:
                    tool_score = 60
                elif metrics.issues_count <= 50:
                    tool_score = 40
                else:
                    tool_score = 20
            
            weighted_score += (tool_score * tool_weight) / total_weight
        
        return round(weighted_score, 2)
    
    def generate_report(self, tool_results: Dict[str, QualityMetrics]) -> QualityReport:
        """Generate comprehensive quality report"""
        overall_score = self.calculate_overall_score(tool_results)
        total_issues = sum(metrics.issues_count for metrics in tool_results.values())
        
        # Count critical issues (errors and high-severity warnings)
        critical_issues = 0
        warnings = 0
        
        for metrics in tool_results.values():
            if metrics.status == "error":
                critical_issues += 1
            elif metrics.status == "warning":
                if metrics.tool in ["bandit", "mypy"]:  # Consider these more critical
                    critical_issues += metrics.issues_count
                else:
                    warnings += metrics.issues_count
        
        # Calculate trends (would need historical data)
        trends = {
            "score_trend": "stable",  # Would calculate from historical data
            "issues_trend": "stable",
            "last_improvement": None
        }
        
        return QualityReport(
            timestamp=datetime.now(),
            overall_score=overall_score,
            total_issues=total_issues,
            critical_issues=critical_issues,
            warnings=warnings,
            tool_results=tool_results,
            trends=trends
        )
    
    def save_report(self, report: QualityReport, output_path: str) -> None:
        """Save quality report to JSON file"""
        output_data = {
            "timestamp": report.timestamp.isoformat(),
            "overall_score": report.overall_score,
            "total_issues": report.total_issues,
            "critical_issues": report.critical_issues,
            "warnings": report.warnings,
            "trends": report.trends,
            "tool_results": {}
        }
        
        for tool_name, metrics in report.tool_results.items():
            output_data["tool_results"][tool_name] = {
                "timestamp": metrics.timestamp.isoformat(),
                "status": metrics.status,
                "issues_count": metrics.issues_count,
                "files_checked": metrics.files_checked,
                "execution_time_seconds": metrics.execution_time_seconds,
                "details": metrics.details
            }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Quality report saved to {output_path}")


def monitor_code_quality(project_root: str, directories: List[str], output_path: str) -> QualityReport:
    """Main function to monitor code quality"""
    monitor = CodeQualityMonitor(project_root, directories)
    
    logger.info("Starting code quality analysis...")
    tool_results = monitor.run_all_tools()
    
    logger.info("Generating quality report...")
    report = monitor.generate_report(tool_results)
    
    monitor.save_report(report, output_path)
    
    logger.info(f"Quality analysis complete. Overall score: {report.overall_score}/100")
    return report


if __name__ == "__main__":
    # Example usage
    report = monitor_code_quality(
        project_root=".",
        directories=["app", "src", "examples", "scripts", "tests"],
        output_path="monitoring/data/quality_report.json"
    )
    
    print(f"Overall Quality Score: {report.overall_score}/100")
    print(f"Total Issues: {report.total_issues}")
    print(f"Critical Issues: {report.critical_issues}")
    print(f"Warnings: {report.warnings}")
