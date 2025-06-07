"""
Main Monitoring Dashboard

Web-based dashboard for visualizing all monitoring metrics.
"""

import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

try:
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    import uvicorn
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    print("FastAPI not available. Install with: pip install fastapi uvicorn jinja2")

logger = logging.getLogger(__name__)


class MonitoringDashboard:
    """Web dashboard for monitoring metrics"""
    
    def __init__(self, data_directory: str, host: str = "localhost", port: int = 8080):
        self.data_directory = Path(data_directory)
        self.host = host
        self.port = port
        self.app = None
        self.websocket_connections: List[WebSocket] = []
        
        if FASTAPI_AVAILABLE:
            self.setup_app()
    
    def setup_app(self):
        """Setup FastAPI application"""
        self.app = FastAPI(title="DataMCPServerAgent Monitoring Dashboard")
        
        # Setup templates and static files
        dashboard_dir = Path(__file__).parent
        templates_dir = dashboard_dir / "templates"
        static_dir = dashboard_dir / "static"
        
        # Create directories if they don't exist
        templates_dir.mkdir(exist_ok=True)
        static_dir.mkdir(exist_ok=True)
        
        # Create basic template if it doesn't exist
        self.create_dashboard_template(templates_dir)
        self.create_static_files(static_dir)
        
        self.templates = Jinja2Templates(directory=str(templates_dir))
        self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        
        # Setup routes
        self.setup_routes()
    
    def create_dashboard_template(self, templates_dir: Path):
        """Create basic dashboard HTML template"""
        template_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DataMCPServerAgent Monitoring Dashboard</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        .metric-card { margin-bottom: 20px; }
        .status-good { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-error { color: #dc3545; }
        .refresh-indicator {
            position: fixed;
            top: 10px;
            right: 10px;
            background: #007bff;
            color: white;
            padding: 5px 10px;
            border-radius: 5px;
            display: none;
        }
    </style>
</head>
<body>
    <div class="container-fluid">
        <div class="refresh-indicator" id="refreshIndicator">Updating...</div>
        
        <header class="row mb-4">
            <div class="col-12">
                <h1 class="text-center">DataMCPServerAgent Monitoring Dashboard</h1>
                <p class="text-center text-muted">Last updated: <span id="lastUpdate">{{ last_update }}</span></p>
            </div>
        </header>
        
        <!-- Summary Cards -->
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body text-center">
                        <h5 class="card-title">CI/CD Health</h5>
                        <h2 id="cicdHealth" class="status-good">{{ cicd_health }}%</h2>
                        <small class="text-muted">Success Rate</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body text-center">
                        <h5 class="card-title">Code Quality</h5>
                        <h2 id="codeQuality" class="status-good">{{ code_quality }}/100</h2>
                        <small class="text-muted">Quality Score</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body text-center">
                        <h5 class="card-title">Security Risk</h5>
                        <h2 id="securityRisk" class="status-good">{{ security_risk }}/100</h2>
                        <small class="text-muted">Risk Score</small>
                    </div>
                </div>
            </div>
            <div class="col-md-3">
                <div class="card metric-card">
                    <div class="card-body text-center">
                        <h5 class="card-title">Test Coverage</h5>
                        <h2 id="testCoverage" class="status-good">{{ test_coverage }}%</h2>
                        <small class="text-muted">Coverage</small>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Detailed Metrics -->
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>CI/CD Performance</h5>
                    </div>
                    <div class="card-body">
                        <canvas id="cicdChart"></canvas>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Security Issues</h5>
                    </div>
                    <div class="card-body">
                        <canvas id="securityChart"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Test Metrics</h5>
                    </div>
                    <div class="card-body">
                        <canvas id="testChart"></canvas>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Documentation Health</h5>
                    </div>
                    <div class="card-body">
                        <canvas id="docChart"></canvas>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Recent Activity -->
        <div class="row mt-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5>Recent Activity & Recommendations</h5>
                    </div>
                    <div class="card-body">
                        <div id="recommendations">
                            {% for rec in recommendations %}
                            <div class="alert alert-info">{{ rec }}</div>
                            {% endfor %}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // WebSocket connection for real-time updates
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        };
        
        function updateDashboard(data) {
            document.getElementById('refreshIndicator').style.display = 'block';
            
            // Update summary cards
            if (data.cicd_health !== undefined) {
                document.getElementById('cicdHealth').textContent = data.cicd_health + '%';
            }
            if (data.code_quality !== undefined) {
                document.getElementById('codeQuality').textContent = data.code_quality + '/100';
            }
            if (data.security_risk !== undefined) {
                document.getElementById('securityRisk').textContent = data.security_risk + '/100';
            }
            if (data.test_coverage !== undefined) {
                document.getElementById('testCoverage').textContent = data.test_coverage + '%';
            }
            
            document.getElementById('lastUpdate').textContent = new Date().toLocaleString();
            
            setTimeout(() => {
                document.getElementById('refreshIndicator').style.display = 'none';
            }, 1000);
        }
        
        // Initialize charts
        function initCharts() {
            // CI/CD Chart
            const cicdCtx = document.getElementById('cicdChart').getContext('2d');
            new Chart(cicdCtx, {
                type: 'line',
                data: {
                    labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                    datasets: [{
                        label: 'Success Rate %',
                        data: [95, 92, 98, 94, 96, 99, 97],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
            
            // Security Chart
            const securityCtx = document.getElementById('securityChart').getContext('2d');
            new Chart(securityCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Critical', 'High', 'Medium', 'Low'],
                    datasets: [{
                        data: [0, 2, 5, 8],
                        backgroundColor: ['#dc3545', '#fd7e14', '#ffc107', '#28a745']
                    }]
                },
                options: {
                    responsive: true
                }
            });
            
            // Test Chart
            const testCtx = document.getElementById('testChart').getContext('2d');
            new Chart(testCtx, {
                type: 'bar',
                data: {
                    labels: ['Coverage', 'Performance', 'Success Rate'],
                    datasets: [{
                        label: 'Score',
                        data: [85, 92, 98],
                        backgroundColor: ['#007bff', '#28a745', '#17a2b8']
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
            
            // Documentation Chart
            const docCtx = document.getElementById('docChart').getContext('2d');
            new Chart(docCtx, {
                type: 'radar',
                data: {
                    labels: ['Coverage', 'Quality', 'Freshness', 'Structure', 'Links'],
                    datasets: [{
                        label: 'Documentation Health',
                        data: [80, 75, 90, 85, 95],
                        borderColor: 'rgb(255, 99, 132)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)'
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        r: {
                            beginAtZero: true,
                            max: 100
                        }
                    }
                }
            });
        }
        
        // Initialize charts when page loads
        document.addEventListener('DOMContentLoaded', initCharts);
    </script>
</body>
</html>
        '''
        
        template_path = templates_dir / "dashboard.html"
        with open(template_path, 'w') as f:
            f.write(template_content.strip())
    
    def create_static_files(self, static_dir: Path):
        """Create basic static files"""
        # Create a simple CSS file
        css_content = '''
        .metric-card {
            transition: transform 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-2px);
        }
        '''
        
        css_path = static_dir / "style.css"
        with open(css_path, 'w') as f:
            f.write(css_content)
    
    def setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard(request: Request):
            # Load latest metrics
            metrics = self.load_latest_metrics()
            
            return self.templates.TemplateResponse("dashboard.html", {
                "request": request,
                "last_update": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "cicd_health": metrics.get("cicd_health", 0),
                "code_quality": metrics.get("code_quality", 0),
                "security_risk": metrics.get("security_risk", 0),
                "test_coverage": metrics.get("test_coverage", 0),
                "recommendations": metrics.get("recommendations", [])
            })
        
        @self.app.get("/api/metrics")
        async def get_metrics():
            """API endpoint for current metrics"""
            return JSONResponse(self.load_latest_metrics())
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                while True:
                    # Send periodic updates
                    await asyncio.sleep(30)  # Update every 30 seconds
                    metrics = self.load_latest_metrics()
                    await websocket.send_text(json.dumps(metrics))
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
    
    def load_latest_metrics(self) -> Dict[str, Any]:
        """Load latest metrics from data files"""
        metrics = {
            "cicd_health": 0,
            "code_quality": 0,
            "security_risk": 0,
            "test_coverage": 0,
            "recommendations": []
        }
        
        try:
            # Load CI/CD metrics
            cicd_file = self.data_directory / "cicd_metrics.json"
            if cicd_file.exists():
                with open(cicd_file, 'r') as f:
                    cicd_data = json.load(f)
                    # Calculate average success rate
                    success_rates = []
                    for workflow_metrics in cicd_data.get("metrics", {}).values():
                        success_rates.append(workflow_metrics.get("success_rate", 0))
                    if success_rates:
                        metrics["cicd_health"] = round(sum(success_rates) / len(success_rates), 1)
            
            # Load code quality metrics
            quality_file = self.data_directory / "quality_report.json"
            if quality_file.exists():
                with open(quality_file, 'r') as f:
                    quality_data = json.load(f)
                    metrics["code_quality"] = quality_data.get("overall_score", 0)
            
            # Load security metrics
            security_file = self.data_directory / "security_report.json"
            if security_file.exists():
                with open(security_file, 'r') as f:
                    security_data = json.load(f)
                    metrics["security_risk"] = security_data.get("overall_risk_score", 0)
            
            # Load test metrics
            test_file = self.data_directory / "test_health_report.json"
            if test_file.exists():
                with open(test_file, 'r') as f:
                    test_data = json.load(f)
                    coverage_metrics = test_data.get("coverage_metrics", {})
                    metrics["test_coverage"] = round(coverage_metrics.get("overall_coverage", 0), 1)
            
            # Collect recommendations
            all_recommendations = []
            for file_name in ["quality_report.json", "security_report.json", "test_health_report.json"]:
                file_path = self.data_directory / file_name
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        recommendations = data.get("recommendations", [])
                        all_recommendations.extend(recommendations[:2])  # Limit to 2 per file
            
            metrics["recommendations"] = all_recommendations[:6]  # Limit total to 6
            
        except Exception as e:
            logger.error(f"Failed to load metrics: {e}")
        
        return metrics
    
    async def broadcast_update(self, metrics: Dict[str, Any]):
        """Broadcast metrics update to all connected WebSocket clients"""
        if self.websocket_connections:
            message = json.dumps(metrics)
            for websocket in self.websocket_connections.copy():
                try:
                    await websocket.send_text(message)
                except:
                    self.websocket_connections.remove(websocket)
    
    def run(self):
        """Run the dashboard server"""
        if not FASTAPI_AVAILABLE:
            print("FastAPI not available. Please install with: pip install fastapi uvicorn jinja2")
            return
        
        logger.info(f"Starting monitoring dashboard at http://{self.host}:{self.port}")
        uvicorn.run(self.app, host=self.host, port=self.port)


def start_dashboard(data_directory: str, host: str = "localhost", port: int = 8080):
    """Start the monitoring dashboard"""
    dashboard = MonitoringDashboard(data_directory, host, port)
    dashboard.run()


if __name__ == "__main__":
    # Example usage
    start_dashboard(
        data_directory="monitoring/data",
        host="localhost",
        port=8080
    )
