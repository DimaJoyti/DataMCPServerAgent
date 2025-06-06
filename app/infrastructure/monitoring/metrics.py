"""
Monitoring and metrics setup.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

logger = get_logger(__name__)

# Simple in-memory metrics
_metrics = {
    "requests_total": 0,
    "requests_by_method": {},
    "requests_by_status": {},
    "agents_total": 0,
    "tasks_total": 0,
    "uptime_seconds": 0,
}


def increment_metric(name: str, labels: dict = None) -> None:
    """Increment a metric counter."""
    if name in _metrics:
        if isinstance(_metrics[name], dict) and labels:
            for key, value in labels.items():
                metric_key = f"{key}_{value}"
                _metrics[name][metric_key] = _metrics[name].get(metric_key, 0) + 1
        else:
            _metrics[name] += 1


def set_metric(name: str, value: float) -> None:
    """Set a metric value."""
    _metrics[name] = value


def get_metrics() -> dict:
    """Get all metrics."""
    return _metrics.copy()


def setup_monitoring(app: FastAPI) -> None:
    """Setup monitoring and metrics."""
    logger.info("Setting up monitoring...")

    @app.get("/metrics")
    async def metrics():
        """Simple metrics endpoint."""
        return JSONResponse(
            {"metrics": get_metrics(), "format": "json", "timestamp": "2024-01-01T00:00:00Z"}
        )

    logger.info("Monitoring setup complete")
