"""
Testing Coverage Monitor

Track test coverage, performance metrics, and test health.
"""

import subprocess
import json
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class CoverageMetrics:
    """Test coverage metrics"""
    timestamp: datetime
    overall_coverage: float
    line_coverage: float
    branch_coverage: float
    function_coverage: float
    files_covered: int
    total_files: int
    lines_covered: int
    total_lines: int
    missing_lines: List[str]
    file_coverage: Dict[str, float]


@dataclass
class TestPerformanceMetrics:
    """Test performance metrics"""
    timestamp: datetime
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    total_duration_seconds: float
    average_test_duration_seconds: float
    slowest_tests: List[Dict[str, Any]]
    fastest_tests: List[Dict[str, Any]]


@dataclass
class TestHealthReport:
    """Comprehensive test health report"""
    timestamp: datetime
    coverage_metrics: CoverageMetrics
    performance_metrics: TestPerformanceMetrics
    health_score: float  # 0-100
    trends: Dict[str, Any]
    recommendations: List[str]
    test_failures: List[Dict[str, Any]]


class TestingMonitor:
    """Monitor test coverage and performance"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
    
    def run_coverage_analysis(self) -> CoverageMetrics:
        """Run test coverage analysis"""
        logger.info("Running test coverage analysis...")
        
        try:
            # Run pytest with coverage
            result = subprocess.run([
                "python", "-m", "pytest",
                "tests/",
                "--cov=src",
                "--cov=app", 
                "--cov-report=xml:coverage.xml",
                "--cov-report=json:coverage.json",
                "--cov-report=term-missing",
                "-v"
            ], 
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=600
            )
            
            # Parse coverage results
            coverage_data = self._parse_coverage_results()
            
            return coverage_data
            
        except subprocess.TimeoutExpired:
            logger.error("Coverage analysis timeout")
            return self._empty_coverage_metrics()
        except Exception as e:
            logger.error(f"Coverage analysis failed: {e}")
            return self._empty_coverage_metrics()
    
    def _parse_coverage_results(self) -> CoverageMetrics:
        """Parse coverage results from XML and JSON files"""
        coverage_xml_path = self.project_root / "coverage.xml"
        coverage_json_path = self.project_root / "coverage.json"
        
        # Default values
        overall_coverage = 0.0
        line_coverage = 0.0
        branch_coverage = 0.0
        function_coverage = 0.0
        files_covered = 0
        total_files = 0
        lines_covered = 0
        total_lines = 0
        missing_lines = []
        file_coverage = {}
        
        try:
            # Parse XML coverage report
            if coverage_xml_path.exists():
                tree = ET.parse(coverage_xml_path)
                root = tree.getroot()
                
                # Get overall coverage
                coverage_elem = root.find(".//coverage")
                if coverage_elem is not None:
                    line_coverage = float(coverage_elem.get("line-rate", 0)) * 100
                    branch_coverage = float(coverage_elem.get("branch-rate", 0)) * 100
                
                # Get file-level coverage
                for package in root.findall(".//package"):
                    for class_elem in package.findall(".//class"):
                        filename = class_elem.get("filename", "")
                        if filename:
                            file_line_rate = float(class_elem.get("line-rate", 0)) * 100
                            file_coverage[filename] = file_line_rate
                            
                            if file_line_rate > 0:
                                files_covered += 1
                            total_files += 1
            
            # Parse JSON coverage report for more detailed info
            if coverage_json_path.exists():
                with open(coverage_json_path, 'r') as f:
                    json_data = json.load(f)
                
                totals = json_data.get("totals", {})
                overall_coverage = totals.get("percent_covered", 0)
                lines_covered = totals.get("covered_lines", 0)
                total_lines = totals.get("num_statements", 0)
                
                # Get missing lines
                for filename, file_data in json_data.get("files", {}).items():
                    missing = file_data.get("missing_lines", [])
                    if missing:
                        missing_lines.extend([f"{filename}:{line}" for line in missing])
        
        except Exception as e:
            logger.error(f"Failed to parse coverage results: {e}")
        
        return CoverageMetrics(
            timestamp=datetime.now(),
            overall_coverage=overall_coverage,
            line_coverage=line_coverage,
            branch_coverage=branch_coverage,
            function_coverage=function_coverage,
            files_covered=files_covered,
            total_files=total_files,
            lines_covered=lines_covered,
            total_lines=total_lines,
            missing_lines=missing_lines[:50],  # Limit to first 50
            file_coverage=file_coverage
        )
    
    def run_performance_analysis(self) -> TestPerformanceMetrics:
        """Run test performance analysis"""
        logger.info("Running test performance analysis...")
        
        try:
            # Run pytest with timing and JSON output
            result = subprocess.run([
                "python", "-m", "pytest",
                "tests/",
                "--durations=10",
                "--json-report",
                "--json-report-file=test_report.json",
                "-v"
            ],
            cwd=self.project_root,
            capture_output=True,
            text=True,
            timeout=600
            )
            
            # Parse performance results
            performance_data = self._parse_performance_results(result.stdout)
            
            return performance_data
            
        except subprocess.TimeoutExpired:
            logger.error("Performance analysis timeout")
            return self._empty_performance_metrics()
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            return self._empty_performance_metrics()
    
    def _parse_performance_results(self, stdout: str) -> TestPerformanceMetrics:
        """Parse test performance results"""
        test_report_path = self.project_root / "test_report.json"
        
        # Default values
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        skipped_tests = 0
        total_duration = 0.0
        slowest_tests = []
        fastest_tests = []
        
        try:
            # Parse JSON test report if available
            if test_report_path.exists():
                with open(test_report_path, 'r') as f:
                    report_data = json.load(f)
                
                summary = report_data.get("summary", {})
                total_tests = summary.get("total", 0)
                passed_tests = summary.get("passed", 0)
                failed_tests = summary.get("failed", 0)
                skipped_tests = summary.get("skipped", 0)
                total_duration = summary.get("duration", 0.0)
                
                # Get test durations
                tests = report_data.get("tests", [])
                test_durations = []
                
                for test in tests:
                    duration = test.get("duration", 0)
                    test_durations.append({
                        "name": test.get("nodeid", "unknown"),
                        "duration": duration,
                        "outcome": test.get("outcome", "unknown")
                    })
                
                # Sort by duration
                test_durations.sort(key=lambda x: x["duration"], reverse=True)
                slowest_tests = test_durations[:5]
                fastest_tests = test_durations[-5:]
            
            else:
                # Parse from stdout if JSON report not available
                lines = stdout.split('\n')
                
                # Look for test summary
                for line in lines:
                    if "passed" in line and "failed" in line:
                        # Extract numbers using regex
                        numbers = re.findall(r'(\d+) (\w+)', line)
                        for count, status in numbers:
                            if status == "passed":
                                passed_tests = int(count)
                            elif status == "failed":
                                failed_tests = int(count)
                            elif status == "skipped":
                                skipped_tests = int(count)
                
                total_tests = passed_tests + failed_tests + skipped_tests
                
                # Look for duration info
                duration_match = re.search(r'(\d+\.?\d*) seconds', stdout)
                if duration_match:
                    total_duration = float(duration_match.group(1))
        
        except Exception as e:
            logger.error(f"Failed to parse performance results: {e}")
        
        average_duration = total_duration / total_tests if total_tests > 0 else 0
        
        return TestPerformanceMetrics(
            timestamp=datetime.now(),
            total_tests=total_tests,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
            total_duration_seconds=total_duration,
            average_test_duration_seconds=average_duration,
            slowest_tests=slowest_tests,
            fastest_tests=fastest_tests
        )
    
    def calculate_health_score(self, coverage: CoverageMetrics, performance: TestPerformanceMetrics) -> float:
        """Calculate overall test health score (0-100)"""
        # Coverage score (40% weight)
        coverage_score = min(coverage.overall_coverage, 100)
        
        # Test success rate (30% weight)
        if performance.total_tests > 0:
            success_rate = (performance.passed_tests / performance.total_tests) * 100
        else:
            success_rate = 0
        
        # Performance score (20% weight) - based on average test duration
        if performance.average_test_duration_seconds <= 1.0:
            performance_score = 100
        elif performance.average_test_duration_seconds <= 5.0:
            performance_score = 80
        elif performance.average_test_duration_seconds <= 10.0:
            performance_score = 60
        else:
            performance_score = 40
        
        # Coverage breadth (10% weight) - how many files are covered
        if coverage.total_files > 0:
            breadth_score = (coverage.files_covered / coverage.total_files) * 100
        else:
            breadth_score = 0
        
        # Weighted average
        health_score = (
            coverage_score * 0.4 +
            success_rate * 0.3 +
            performance_score * 0.2 +
            breadth_score * 0.1
        )
        
        return round(health_score, 2)
    
    def generate_recommendations(self, coverage: CoverageMetrics, performance: TestPerformanceMetrics) -> List[str]:
        """Generate testing recommendations"""
        recommendations = []
        
        # Coverage recommendations
        if coverage.overall_coverage < 70:
            recommendations.append(f"🎯 Increase test coverage from {coverage.overall_coverage:.1f}% to at least 70%")
        elif coverage.overall_coverage < 85:
            recommendations.append(f"📈 Good coverage at {coverage.overall_coverage:.1f}%, aim for 85%+ for excellent coverage")
        
        if coverage.files_covered < coverage.total_files * 0.8:
            uncovered_files = coverage.total_files - coverage.files_covered
            recommendations.append(f"📁 Add tests for {uncovered_files} uncovered files")
        
        # Performance recommendations
        if performance.failed_tests > 0:
            recommendations.append(f"🔧 Fix {performance.failed_tests} failing tests")
        
        if performance.average_test_duration_seconds > 5.0:
            recommendations.append(f"⚡ Optimize test performance (avg: {performance.average_test_duration_seconds:.2f}s)")
        
        if performance.total_tests < 50:
            recommendations.append("📝 Consider adding more comprehensive tests")
        
        # Missing lines recommendations
        if len(coverage.missing_lines) > 0:
            recommendations.append(f"🎯 Add tests for {len(coverage.missing_lines)} uncovered lines")
        
        if not recommendations:
            recommendations.append("✅ Test suite is in excellent condition!")
        
        return recommendations
    
    def generate_report(self) -> TestHealthReport:
        """Generate comprehensive test health report"""
        coverage_metrics = self.run_coverage_analysis()
        performance_metrics = self.run_performance_analysis()
        
        health_score = self.calculate_health_score(coverage_metrics, performance_metrics)
        recommendations = self.generate_recommendations(coverage_metrics, performance_metrics)
        
        # Extract test failures
        test_failures = []
        if performance_metrics.failed_tests > 0:
            # This would need more detailed parsing of test results
            test_failures.append({
                "count": performance_metrics.failed_tests,
                "details": "See test output for specific failures"
            })
        
        # Calculate trends (would need historical data)
        trends = {
            "coverage_trend": "stable",
            "performance_trend": "stable",
            "test_count_trend": "stable"
        }
        
        return TestHealthReport(
            timestamp=datetime.now(),
            coverage_metrics=coverage_metrics,
            performance_metrics=performance_metrics,
            health_score=health_score,
            trends=trends,
            recommendations=recommendations,
            test_failures=test_failures
        )
    
    def save_report(self, report: TestHealthReport, output_path: str) -> None:
        """Save test health report to JSON file"""
        output_data = {
            "timestamp": report.timestamp.isoformat(),
            "health_score": report.health_score,
            "trends": report.trends,
            "recommendations": report.recommendations,
            "test_failures": report.test_failures,
            "coverage_metrics": {
                "timestamp": report.coverage_metrics.timestamp.isoformat(),
                "overall_coverage": report.coverage_metrics.overall_coverage,
                "line_coverage": report.coverage_metrics.line_coverage,
                "branch_coverage": report.coverage_metrics.branch_coverage,
                "function_coverage": report.coverage_metrics.function_coverage,
                "files_covered": report.coverage_metrics.files_covered,
                "total_files": report.coverage_metrics.total_files,
                "lines_covered": report.coverage_metrics.lines_covered,
                "total_lines": report.coverage_metrics.total_lines,
                "missing_lines_count": len(report.coverage_metrics.missing_lines),
                "file_coverage": report.coverage_metrics.file_coverage
            },
            "performance_metrics": {
                "timestamp": report.performance_metrics.timestamp.isoformat(),
                "total_tests": report.performance_metrics.total_tests,
                "passed_tests": report.performance_metrics.passed_tests,
                "failed_tests": report.performance_metrics.failed_tests,
                "skipped_tests": report.performance_metrics.skipped_tests,
                "total_duration_seconds": report.performance_metrics.total_duration_seconds,
                "average_test_duration_seconds": report.performance_metrics.average_test_duration_seconds,
                "slowest_tests": report.performance_metrics.slowest_tests,
                "fastest_tests": report.performance_metrics.fastest_tests
            }
        }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Test health report saved to {output_path}")
    
    def _empty_coverage_metrics(self) -> CoverageMetrics:
        """Return empty coverage metrics"""
        return CoverageMetrics(
            timestamp=datetime.now(),
            overall_coverage=0.0,
            line_coverage=0.0,
            branch_coverage=0.0,
            function_coverage=0.0,
            files_covered=0,
            total_files=0,
            lines_covered=0,
            total_lines=0,
            missing_lines=[],
            file_coverage={}
        )
    
    def _empty_performance_metrics(self) -> TestPerformanceMetrics:
        """Return empty performance metrics"""
        return TestPerformanceMetrics(
            timestamp=datetime.now(),
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            skipped_tests=0,
            total_duration_seconds=0.0,
            average_test_duration_seconds=0.0,
            slowest_tests=[],
            fastest_tests=[]
        )


def monitor_testing(project_root: str, output_path: str) -> TestHealthReport:
    """Main function to monitor testing metrics"""
    monitor = TestingMonitor(project_root)
    
    logger.info("Starting test health analysis...")
    report = monitor.generate_report()
    
    monitor.save_report(report, output_path)
    
    logger.info(f"Test analysis complete. Health score: {report.health_score}/100")
    return report


if __name__ == "__main__":
    # Example usage
    report = monitor_testing(
        project_root=".",
        output_path="monitoring/data/test_health_report.json"
    )
    
    print(f"Test Health Score: {report.health_score}/100")
    print(f"Coverage: {report.coverage_metrics.overall_coverage:.1f}%")
    print(f"Tests: {report.performance_metrics.passed_tests}/{report.performance_metrics.total_tests} passed")
    print("\nRecommendations:")
    for rec in report.recommendations:
        print(f"  {rec}")
