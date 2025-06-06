#!/usr/bin/env python3
"""
Quick start script for DataMCPServerAgent.
Provides multiple deployment options.
"""

import asyncio
import sys
import subprocess
import argparse
from pathlib import Path

# Add app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.logging import get_logger


logger = get_logger(__name__)


class StartupManager:
    """Manages different startup modes."""

    def __init__(self):
        self.project_root = Path(__file__).parent

    async def start_development(self) -> None:
        """Start in development mode."""
        logger.info("🚀 Starting DataMCPServerAgent in DEVELOPMENT mode")
        logger.info("=" * 60)

        # Check if dependencies are installed
        try:
            import uvicorn
            logger.info("✅ uvicorn imported successfully")
        except ImportError as e:
            logger.error(f"❌ uvicorn not available: {e}")
            logger.error("❌ Run: python install_basic.py")
            return

        try:
            from app.main import app
            logger.info("✅ app imported successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import app: {e}")
            return

        # Start with hot reload
        logger.info("🔥 Starting with hot reload...")
        logger.info("📍 API: http://localhost:8002")
        logger.info("📚 Docs: http://localhost:8002/docs")
        logger.info("📊 Metrics: http://localhost:8002/metrics")
        logger.info("🛑 Press Ctrl+C to stop")

        try:
            import uvicorn
            uvicorn.run(
                "app.main:app",
                host="0.0.0.0",
                port=8002,
                reload=True,
                log_level="info"
            )
        except KeyboardInterrupt:
            logger.info("\n👋 Development server stopped")

    async def start_production(self) -> None:
        """Start in production mode."""
        logger.info("🚀 Starting DataMCPServerAgent in PRODUCTION mode")
        logger.info("=" * 60)

        try:
            import uvicorn
            from app.main import app
        except ImportError:
            logger.error("❌ Dependencies not installed. Run: python deploy.py")
            return

        logger.info("⚡ Starting production server...")
        logger.info("📍 API: http://localhost:8002")
        logger.info("📚 Docs: http://localhost:8002/docs")
        logger.info("📊 Metrics: http://localhost:8002/metrics")

        try:
            import uvicorn
            uvicorn.run(
                "app.main:app",
                host="0.0.0.0",
                port=8002,
                workers=4,
                log_level="info"
            )
        except KeyboardInterrupt:
            logger.info("\n👋 Production server stopped")

    async def start_docker(self) -> None:
        """Start with Docker Compose."""
        logger.info("🐳 Starting DataMCPServerAgent with Docker Compose")
        logger.info("=" * 60)

        compose_file = self.project_root / "docker-compose.yml"
        if not compose_file.exists():
            logger.error("❌ docker-compose.yml not found")
            return

        try:
            logger.info("🔨 Building and starting containers...")
            subprocess.run([
                "docker-compose", "up", "--build", "-d"
            ], check=True)

            logger.info("✅ Containers started successfully!")
            logger.info("📍 API: http://localhost:8002")
            logger.info("📚 Docs: http://localhost:8002/docs")
            logger.info("📊 Metrics: http://localhost:8002/metrics")
            logger.info("📈 Grafana: http://localhost:3000 (admin/admin123)")
            logger.info("🔍 Prometheus: http://localhost:9091")
            logger.info("")
            logger.info("🛑 To stop: docker-compose down")
            logger.info("📋 To view logs: docker-compose logs -f")

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Docker Compose failed: {e}")
        except FileNotFoundError:
            logger.error("❌ Docker Compose not found. Please install Docker.")

    async def start_kubernetes(self) -> None:
        """Start with Kubernetes."""
        logger.info("☸️ Deploying DataMCPServerAgent to Kubernetes")
        logger.info("=" * 60)

        k8s_dir = self.project_root / "deployment" / "kubernetes"
        if not k8s_dir.exists():
            logger.error("❌ Kubernetes manifests not found")
            return

        try:
            # Apply namespace and configs
            logger.info("📦 Creating namespace and configs...")
            subprocess.run([
                "kubectl", "apply", "-f", str(k8s_dir / "namespace.yaml")
            ], check=True)

            # Apply deployment
            logger.info("🚀 Deploying application...")
            subprocess.run([
                "kubectl", "apply", "-f", str(k8s_dir / "deployment.yaml")
            ], check=True)

            logger.info("✅ Kubernetes deployment successful!")
            logger.info("📋 Check status: kubectl get pods -n datamcp")
            logger.info("📍 Port forward: kubectl port-forward -n datamcp svc/datamcp-agent-service 8002:8002")

        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Kubernetes deployment failed: {e}")
        except FileNotFoundError:
            logger.error("❌ kubectl not found. Please install Kubernetes CLI.")

    async def show_status(self) -> None:
        """Show application status."""
        logger.info("📊 DataMCPServerAgent Status")
        logger.info("=" * 60)

        # Check if running locally
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get("http://localhost:8002/health", timeout=5)
                if response.status_code == 200:
                    logger.info("✅ Local server: RUNNING")
                    logger.info("📍 API: http://localhost:8002")
                else:
                    logger.info("⚠️ Local server: UNHEALTHY")
        except:
            logger.info("❌ Local server: NOT RUNNING")

        # Check Docker containers
        try:
            result = subprocess.run([
                "docker-compose", "ps", "--services", "--filter", "status=running"
            ], capture_output=True, text=True)

            if result.returncode == 0 and result.stdout.strip():
                running_services = result.stdout.strip().split('\n')
                logger.info(f"🐳 Docker: {len(running_services)} services running")
                for service in running_services:
                    logger.info(f"  ✅ {service}")
            else:
                logger.info("🐳 Docker: No services running")
        except:
            logger.info("🐳 Docker: Not available")

        # Check Kubernetes
        try:
            result = subprocess.run([
                "kubectl", "get", "pods", "-n", "datamcp", "--no-headers"
            ], capture_output=True, text=True)

            if result.returncode == 0 and result.stdout.strip():
                pods = result.stdout.strip().split('\n')
                running_pods = [p for p in pods if 'Running' in p]
                logger.info(f"☸️ Kubernetes: {len(running_pods)}/{len(pods)} pods running")
            else:
                logger.info("☸️ Kubernetes: No pods found")
        except:
            logger.info("☸️ Kubernetes: Not available")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="DataMCPServerAgent Startup Manager")
    parser.add_argument(
        "mode",
        choices=["dev", "prod", "docker", "k8s", "status"],
        help="Startup mode"
    )

    args = parser.parse_args()
    startup_manager = StartupManager()

    try:
        if args.mode == "dev":
            await startup_manager.start_development()
        elif args.mode == "prod":
            await startup_manager.start_production()
        elif args.mode == "docker":
            await startup_manager.start_docker()
        elif args.mode == "k8s":
            await startup_manager.start_kubernetes()
        elif args.mode == "status":
            await startup_manager.show_status()
    except KeyboardInterrupt:
        logger.info("\n👋 Startup interrupted")
    except Exception as e:
        logger.error(f"💥 Startup failed: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
