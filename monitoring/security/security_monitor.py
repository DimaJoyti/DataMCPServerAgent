"""
Security Monitor

Comprehensive security scanning and vulnerability tracking.
"""

import subprocess
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SeverityLevel(Enum):
    """Security issue severity levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class SecurityIssue:
    """Represents a security issue"""
    tool: str
    severity: SeverityLevel
    title: str
    description: str
    file_path: Optional[str]
    line_number: Optional[int]
    cwe_id: Optional[str]
    confidence: Optional[str]
    recommendation: Optional[str]


@dataclass
class SecurityMetrics:
    """Security scan metrics"""
    timestamp: datetime
    tool: str
    status: str
    execution_time_seconds: float
    issues_by_severity: Dict[str, int]
    total_issues: int
    files_scanned: int
    issues: List[SecurityIssue]


@dataclass
class SecurityReport:
    """Comprehensive security report"""
    timestamp: datetime
    overall_risk_score: float  # 0-100 (higher = more risk)
    total_issues: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    tool_results: Dict[str, SecurityMetrics]
    trends: Dict[str, Any]
    recommendations: List[str]


class SecurityMonitor:
    """Monitor security vulnerabilities and issues"""
    
    def __init__(self, project_root: str, directories: List[str]):
        self.project_root = Path(project_root)
        self.directories = directories
        self.tools_config = {
            "bandit": {
                "command": ["bandit", "-r", "-f", "json"],
                "weight": 40
            },
            "safety": {
                "command": ["safety", "check", "--json"],
                "weight": 35
            },
            "semgrep": {
                "command": ["semgrep", "--config=auto", "--json"],
                "weight": 25
            }
        }
    
    def run_bandit(self, directories: List[str]) -> SecurityMetrics:
        """Run Bandit security scanner"""
        start_time = time.time()
        command = ["bandit", "-r", "-f", "json"] + directories
        
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            execution_time = time.time() - start_time
            issues = []
            issues_by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            
            if result.stdout:
                try:
                    bandit_data = json.loads(result.stdout)
                    
                    for issue_data in bandit_data.get("results", []):
                        severity_map = {
                            "HIGH": SeverityLevel.HIGH,
                            "MEDIUM": SeverityLevel.MEDIUM,
                            "LOW": SeverityLevel.LOW
                        }
                        
                        severity = severity_map.get(
                            issue_data.get("issue_severity", "LOW"),
                            SeverityLevel.LOW
                        )
                        
                        issue = SecurityIssue(
                            tool="bandit",
                            severity=severity,
                            title=issue_data.get("test_name", "Unknown"),
                            description=issue_data.get("issue_text", ""),
                            file_path=issue_data.get("filename"),
                            line_number=issue_data.get("line_number"),
                            cwe_id=issue_data.get("test_id"),
                            confidence=issue_data.get("issue_confidence"),
                            recommendation=issue_data.get("more_info")
                        )
                        
                        issues.append(issue)
                        issues_by_severity[severity.value] += 1
                    
                    files_scanned = len(bandit_data.get("metrics", {}).get("_totals", {}).get("loc", 0))
                    
                except json.JSONDecodeError:
                    logger.error("Failed to parse Bandit JSON output")
                    files_scanned = 0
            else:
                files_scanned = 0
            
            return SecurityMetrics(
                timestamp=datetime.now(),
                tool="bandit",
                status="success" if result.returncode == 0 else "warning",
                execution_time_seconds=execution_time,
                issues_by_severity=issues_by_severity,
                total_issues=len(issues),
                files_scanned=files_scanned,
                issues=issues
            )
            
        except Exception as e:
            logger.error(f"Bandit execution failed: {e}")
            return SecurityMetrics(
                timestamp=datetime.now(),
                tool="bandit",
                status="error",
                execution_time_seconds=time.time() - start_time,
                issues_by_severity={"critical": 0, "high": 0, "medium": 0, "low": 0},
                total_issues=0,
                files_scanned=0,
                issues=[]
            )
    
    def run_safety(self) -> SecurityMetrics:
        """Run Safety dependency checker"""
        start_time = time.time()
        command = ["safety", "check", "--json"]
        
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            execution_time = time.time() - start_time
            issues = []
            issues_by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            
            if result.stdout:
                try:
                    safety_data = json.loads(result.stdout)
                    
                    for vuln in safety_data:
                        # Safety doesn't provide severity, so we estimate based on vulnerability type
                        severity = SeverityLevel.HIGH  # Default to high for dependencies
                        
                        issue = SecurityIssue(
                            tool="safety",
                            severity=severity,
                            title=f"Vulnerable dependency: {vuln.get('package_name')}",
                            description=vuln.get("advisory", ""),
                            file_path="requirements.txt",  # Assumption
                            line_number=None,
                            cwe_id=vuln.get("cve"),
                            confidence="HIGH",
                            recommendation=f"Update to version {vuln.get('safe_versions', 'latest')}"
                        )
                        
                        issues.append(issue)
                        issues_by_severity[severity.value] += 1
                    
                except json.JSONDecodeError:
                    logger.error("Failed to parse Safety JSON output")
            
            return SecurityMetrics(
                timestamp=datetime.now(),
                tool="safety",
                status="success" if result.returncode == 0 else "warning",
                execution_time_seconds=execution_time,
                issues_by_severity=issues_by_severity,
                total_issues=len(issues),
                files_scanned=1,  # Checking requirements file
                issues=issues
            )
            
        except Exception as e:
            logger.error(f"Safety execution failed: {e}")
            return SecurityMetrics(
                timestamp=datetime.now(),
                tool="safety",
                status="error",
                execution_time_seconds=time.time() - start_time,
                issues_by_severity={"critical": 0, "high": 0, "medium": 0, "low": 0},
                total_issues=0,
                files_scanned=0,
                issues=[]
            )
    
    def run_semgrep(self, directories: List[str]) -> SecurityMetrics:
        """Run Semgrep security scanner"""
        start_time = time.time()
        command = ["semgrep", "--config=auto", "--json"] + directories
        
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600  # Semgrep can be slower
            )
            
            execution_time = time.time() - start_time
            issues = []
            issues_by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
            
            if result.stdout:
                try:
                    semgrep_data = json.loads(result.stdout)
                    
                    for finding in semgrep_data.get("results", []):
                        # Map Semgrep severity to our levels
                        severity_map = {
                            "ERROR": SeverityLevel.HIGH,
                            "WARNING": SeverityLevel.MEDIUM,
                            "INFO": SeverityLevel.LOW
                        }
                        
                        severity = severity_map.get(
                            finding.get("extra", {}).get("severity", "INFO"),
                            SeverityLevel.LOW
                        )
                        
                        issue = SecurityIssue(
                            tool="semgrep",
                            severity=severity,
                            title=finding.get("check_id", "Unknown"),
                            description=finding.get("extra", {}).get("message", ""),
                            file_path=finding.get("path"),
                            line_number=finding.get("start", {}).get("line"),
                            cwe_id=None,
                            confidence="HIGH",
                            recommendation=finding.get("extra", {}).get("fix", "Review and fix manually")
                        )
                        
                        issues.append(issue)
                        issues_by_severity[severity.value] += 1
                    
                    files_scanned = len(set(f.get("path") for f in semgrep_data.get("results", [])))
                    
                except json.JSONDecodeError:
                    logger.error("Failed to parse Semgrep JSON output")
                    files_scanned = 0
            else:
                files_scanned = 0
            
            return SecurityMetrics(
                timestamp=datetime.now(),
                tool="semgrep",
                status="success" if result.returncode == 0 else "warning",
                execution_time_seconds=execution_time,
                issues_by_severity=issues_by_severity,
                total_issues=len(issues),
                files_scanned=files_scanned,
                issues=issues
            )
            
        except Exception as e:
            logger.error(f"Semgrep execution failed: {e}")
            return SecurityMetrics(
                timestamp=datetime.now(),
                tool="semgrep",
                status="error",
                execution_time_seconds=time.time() - start_time,
                issues_by_severity={"critical": 0, "high": 0, "medium": 0, "low": 0},
                total_issues=0,
                files_scanned=0,
                issues=[]
            )
    
    def run_all_scans(self) -> Dict[str, SecurityMetrics]:
        """Run all security scans"""
        results = {}
        
        logger.info("Running Bandit...")
        results["bandit"] = self.run_bandit(self.directories)
        
        logger.info("Running Safety...")
        results["safety"] = self.run_safety()
        
        logger.info("Running Semgrep...")
        results["semgrep"] = self.run_semgrep(self.directories)
        
        return results
    
    def calculate_risk_score(self, tool_results: Dict[str, SecurityMetrics]) -> float:
        """Calculate overall risk score (0-100, higher = more risk)"""
        total_weight = sum(config["weight"] for config in self.tools_config.values())
        weighted_risk = 0
        
        for tool_name, metrics in tool_results.items():
            tool_weight = self.tools_config[tool_name]["weight"]
            
            if metrics.status == "error":
                tool_risk = 50  # Unknown risk
            else:
                # Calculate risk based on severity distribution
                severity_weights = {"critical": 100, "high": 75, "medium": 50, "low": 25}
                tool_risk = 0
                
                for severity, count in metrics.issues_by_severity.items():
                    tool_risk += count * severity_weights.get(severity, 0)
                
                # Normalize to 0-100 scale (cap at reasonable maximum)
                tool_risk = min(tool_risk, 100)
            
            weighted_risk += (tool_risk * tool_weight) / total_weight
        
        return round(weighted_risk, 2)
    
    def generate_recommendations(self, tool_results: Dict[str, SecurityMetrics]) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        total_critical = sum(m.issues_by_severity.get("critical", 0) for m in tool_results.values())
        total_high = sum(m.issues_by_severity.get("high", 0) for m in tool_results.values())
        
        if total_critical > 0:
            recommendations.append(f"🚨 URGENT: Fix {total_critical} critical security issues immediately")
        
        if total_high > 0:
            recommendations.append(f"⚠️ HIGH PRIORITY: Address {total_high} high-severity security issues")
        
        # Tool-specific recommendations
        for tool_name, metrics in tool_results.items():
            if metrics.status == "error":
                recommendations.append(f"🔧 Fix {tool_name} execution issues to ensure complete security coverage")
            elif metrics.total_issues > 0:
                recommendations.append(f"📋 Review {metrics.total_issues} issues found by {tool_name}")
        
        if not recommendations:
            recommendations.append("✅ No immediate security concerns detected")
        
        return recommendations
    
    def generate_report(self, tool_results: Dict[str, SecurityMetrics]) -> SecurityReport:
        """Generate comprehensive security report"""
        risk_score = self.calculate_risk_score(tool_results)
        
        total_issues = sum(m.total_issues for m in tool_results.values())
        critical_issues = sum(m.issues_by_severity.get("critical", 0) for m in tool_results.values())
        high_issues = sum(m.issues_by_severity.get("high", 0) for m in tool_results.values())
        medium_issues = sum(m.issues_by_severity.get("medium", 0) for m in tool_results.values())
        low_issues = sum(m.issues_by_severity.get("low", 0) for m in tool_results.values())
        
        recommendations = self.generate_recommendations(tool_results)
        
        # Calculate trends (would need historical data)
        trends = {
            "risk_trend": "stable",
            "issues_trend": "stable",
            "last_scan": datetime.now().isoformat()
        }
        
        return SecurityReport(
            timestamp=datetime.now(),
            overall_risk_score=risk_score,
            total_issues=total_issues,
            critical_issues=critical_issues,
            high_issues=high_issues,
            medium_issues=medium_issues,
            low_issues=low_issues,
            tool_results=tool_results,
            trends=trends,
            recommendations=recommendations
        )
    
    def save_report(self, report: SecurityReport, output_path: str) -> None:
        """Save security report to JSON file"""
        output_data = {
            "timestamp": report.timestamp.isoformat(),
            "overall_risk_score": report.overall_risk_score,
            "total_issues": report.total_issues,
            "critical_issues": report.critical_issues,
            "high_issues": report.high_issues,
            "medium_issues": report.medium_issues,
            "low_issues": report.low_issues,
            "trends": report.trends,
            "recommendations": report.recommendations,
            "tool_results": {}
        }
        
        for tool_name, metrics in report.tool_results.items():
            output_data["tool_results"][tool_name] = {
                "timestamp": metrics.timestamp.isoformat(),
                "status": metrics.status,
                "execution_time_seconds": metrics.execution_time_seconds,
                "issues_by_severity": metrics.issues_by_severity,
                "total_issues": metrics.total_issues,
                "files_scanned": metrics.files_scanned,
                "issues": [
                    {
                        "severity": issue.severity.value,
                        "title": issue.title,
                        "description": issue.description,
                        "file_path": issue.file_path,
                        "line_number": issue.line_number,
                        "cwe_id": issue.cwe_id,
                        "confidence": issue.confidence,
                        "recommendation": issue.recommendation
                    }
                    for issue in metrics.issues
                ]
            }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Security report saved to {output_path}")


def monitor_security(project_root: str, directories: List[str], output_path: str) -> SecurityReport:
    """Main function to monitor security"""
    monitor = SecurityMonitor(project_root, directories)
    
    logger.info("Starting security analysis...")
    tool_results = monitor.run_all_scans()
    
    logger.info("Generating security report...")
    report = monitor.generate_report(tool_results)
    
    monitor.save_report(report, output_path)
    
    logger.info(f"Security analysis complete. Risk score: {report.overall_risk_score}/100")
    return report


if __name__ == "__main__":
    # Example usage
    report = monitor_security(
        project_root=".",
        directories=["app", "src", "examples", "scripts"],
        output_path="monitoring/data/security_report.json"
    )
    
    print(f"Overall Risk Score: {report.overall_risk_score}/100")
    print(f"Total Issues: {report.total_issues}")
    print(f"Critical Issues: {report.critical_issues}")
    print(f"High Issues: {report.high_issues}")
    print("\nRecommendations:")
    for rec in report.recommendations:
        print(f"  {rec}")
