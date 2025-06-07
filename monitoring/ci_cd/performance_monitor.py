"""
CI/CD Performance Monitor

Tracks GitHub Actions workflow performance, success rates, and build times.
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class WorkflowRun:
    """Represents a workflow run"""
    id: int
    name: str
    status: str
    conclusion: Optional[str]
    created_at: datetime
    updated_at: datetime
    duration_seconds: Optional[int]
    queue_time_seconds: Optional[int]
    url: str
    commit_sha: str
    branch: str


@dataclass
class WorkflowMetrics:
    """Workflow performance metrics"""
    name: str
    total_runs: int
    success_rate: float
    average_duration_seconds: float
    average_queue_time_seconds: float
    recent_failures: List[WorkflowRun]
    trend_7_days: Dict[str, Any]
    trend_30_days: Dict[str, Any]


class CICDPerformanceMonitor:
    """Monitor CI/CD performance using GitHub API"""
    
    def __init__(self, github_token: str, owner: str, repo: str):
        self.github_token = github_token
        self.owner = owner
        self.repo = repo
        self.base_url = "https://api.github.com"
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            headers={
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def get_workflows(self) -> List[Dict[str, Any]]:
        """Get all workflows for the repository"""
        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/actions/workflows"
        
        async with self.session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("workflows", [])
            else:
                logger.error(f"Failed to fetch workflows: {response.status}")
                return []
    
    async def get_workflow_runs(self, workflow_id: int, per_page: int = 100) -> List[WorkflowRun]:
        """Get recent runs for a specific workflow"""
        url = f"{self.base_url}/repos/{self.owner}/{self.repo}/actions/workflows/{workflow_id}/runs"
        params = {"per_page": per_page, "status": "completed"}
        
        async with self.session.get(url, params=params) as response:
            if response.status == 200:
                data = await response.json()
                runs = []
                
                for run_data in data.get("workflow_runs", []):
                    # Calculate duration and queue time
                    created_at = datetime.fromisoformat(run_data["created_at"].replace("Z", "+00:00"))
                    updated_at = datetime.fromisoformat(run_data["updated_at"].replace("Z", "+00:00"))
                    
                    duration_seconds = None
                    queue_time_seconds = None
                    
                    if run_data.get("run_started_at"):
                        started_at = datetime.fromisoformat(run_data["run_started_at"].replace("Z", "+00:00"))
                        queue_time_seconds = int((started_at - created_at).total_seconds())
                        duration_seconds = int((updated_at - started_at).total_seconds())
                    
                    run = WorkflowRun(
                        id=run_data["id"],
                        name=run_data["name"],
                        status=run_data["status"],
                        conclusion=run_data.get("conclusion"),
                        created_at=created_at,
                        updated_at=updated_at,
                        duration_seconds=duration_seconds,
                        queue_time_seconds=queue_time_seconds,
                        url=run_data["html_url"],
                        commit_sha=run_data["head_sha"],
                        branch=run_data["head_branch"]
                    )
                    runs.append(run)
                
                return runs
            else:
                logger.error(f"Failed to fetch workflow runs: {response.status}")
                return []
    
    async def calculate_workflow_metrics(self, workflow_name: str, runs: List[WorkflowRun]) -> WorkflowMetrics:
        """Calculate performance metrics for a workflow"""
        if not runs:
            return WorkflowMetrics(
                name=workflow_name,
                total_runs=0,
                success_rate=0.0,
                average_duration_seconds=0.0,
                average_queue_time_seconds=0.0,
                recent_failures=[],
                trend_7_days={},
                trend_30_days={}
            )
        
        # Calculate success rate
        successful_runs = [r for r in runs if r.conclusion == "success"]
        success_rate = (len(successful_runs) / len(runs)) * 100
        
        # Calculate average durations
        durations = [r.duration_seconds for r in runs if r.duration_seconds is not None]
        queue_times = [r.queue_time_seconds for r in runs if r.queue_time_seconds is not None]
        
        avg_duration = sum(durations) / len(durations) if durations else 0
        avg_queue_time = sum(queue_times) / len(queue_times) if queue_times else 0
        
        # Find recent failures
        recent_failures = [r for r in runs[:10] if r.conclusion != "success"]
        
        # Calculate trends
        now = datetime.now()
        seven_days_ago = now - timedelta(days=7)
        thirty_days_ago = now - timedelta(days=30)
        
        runs_7_days = [r for r in runs if r.created_at >= seven_days_ago]
        runs_30_days = [r for r in runs if r.created_at >= thirty_days_ago]
        
        trend_7_days = {
            "total_runs": len(runs_7_days),
            "success_rate": (len([r for r in runs_7_days if r.conclusion == "success"]) / len(runs_7_days) * 100) if runs_7_days else 0,
            "avg_duration": sum([r.duration_seconds for r in runs_7_days if r.duration_seconds]) / len(runs_7_days) if runs_7_days else 0
        }
        
        trend_30_days = {
            "total_runs": len(runs_30_days),
            "success_rate": (len([r for r in runs_30_days if r.conclusion == "success"]) / len(runs_30_days) * 100) if runs_30_days else 0,
            "avg_duration": sum([r.duration_seconds for r in runs_30_days if r.duration_seconds]) / len(runs_30_days) if runs_30_days else 0
        }
        
        return WorkflowMetrics(
            name=workflow_name,
            total_runs=len(runs),
            success_rate=success_rate,
            average_duration_seconds=avg_duration,
            average_queue_time_seconds=avg_queue_time,
            recent_failures=recent_failures,
            trend_7_days=trend_7_days,
            trend_30_days=trend_30_days
        )
    
    async def get_all_metrics(self) -> Dict[str, WorkflowMetrics]:
        """Get performance metrics for all workflows"""
        workflows = await self.get_workflows()
        metrics = {}
        
        for workflow in workflows:
            workflow_name = workflow["name"]
            workflow_id = workflow["id"]
            
            logger.info(f"Analyzing workflow: {workflow_name}")
            runs = await self.get_workflow_runs(workflow_id)
            workflow_metrics = await self.calculate_workflow_metrics(workflow_name, runs)
            metrics[workflow_name] = workflow_metrics
        
        return metrics
    
    async def save_metrics(self, metrics: Dict[str, WorkflowMetrics], output_path: str) -> None:
        """Save metrics to JSON file"""
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {}
        }
        
        for name, metric in metrics.items():
            output_data["metrics"][name] = {
                "name": metric.name,
                "total_runs": metric.total_runs,
                "success_rate": metric.success_rate,
                "average_duration_seconds": metric.average_duration_seconds,
                "average_queue_time_seconds": metric.average_queue_time_seconds,
                "recent_failures_count": len(metric.recent_failures),
                "trend_7_days": metric.trend_7_days,
                "trend_30_days": metric.trend_30_days
            }
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"Metrics saved to {output_path}")


async def monitor_cicd_performance(github_token: str, owner: str, repo: str, output_path: str) -> Dict[str, WorkflowMetrics]:
    """Main function to monitor CI/CD performance"""
    async with CICDPerformanceMonitor(github_token, owner, repo) as monitor:
        metrics = await monitor.get_all_metrics()
        await monitor.save_metrics(metrics, output_path)
        return metrics


if __name__ == "__main__":
    import os
    
    # Example usage
    github_token = os.getenv("GITHUB_TOKEN")
    if not github_token:
        print("Please set GITHUB_TOKEN environment variable")
        exit(1)
    
    asyncio.run(monitor_cicd_performance(
        github_token=github_token,
        owner="DimaJoyti",
        repo="DataMCPServerAgent",
        output_path="monitoring/data/cicd_metrics.json"
    ))
