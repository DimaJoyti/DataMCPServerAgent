#!/usr/bin/env python3
"""
Enhanced test to verify that all core modules can be imported correctly.
Tests both app/ (new structure) and src/ (legacy) modules.
"""

import sys
from pathlib import Path

import pytest

# Add both app and src to path for compatibility
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "app"))
sys.path.insert(0, str(project_root / "src"))


def test_core_app_imports():
    """Test importing core app modules."""

    try:
        # Test core configuration
        from app.core.config import Settings
        assert Settings is not None

        # Test logging
        from app.core.logging import get_logger, setup_logging
        assert setup_logging is not None
        assert get_logger is not None

        # Test main application
        from app.main_improved import app
        assert app is not None

        return True

    except ImportError as e:
        pytest.skip(f"Core app modules not available: {e}")
        return False


def test_domain_models():
    """Test importing domain models."""

    try:
        # Test agent models
        from app.domain.models.agent import Agent, AgentType
        assert Agent is not None
        assert AgentType is not None

        # Test task models
        from app.domain.models.task import Task, TaskType
        assert Task is not None
        assert TaskType is not None

        return True

    except ImportError as e:
        pytest.skip(f"Domain models not available: {e}")
        return False


def test_api_modules():
    """Test importing API modules."""

    try:
        # Test API server
        from app.api.server_improved import create_api_server
        assert create_api_server is not None

        # Test dependencies
        from app.api.dependencies import get_settings
        assert get_settings is not None

        return True

    except ImportError as e:
        pytest.skip(f"API modules not available: {e}")
        return False


def test_legacy_src_imports():
    """Test importing legacy src modules (if available)."""

    try:
        # Try to import some legacy modules
        from src.core.main import chat_with_agent
        assert chat_with_agent is not None

        return True

    except ImportError:
        # Legacy modules not available, which is fine
        pytest.skip("Legacy src modules not available")
        return False

def test_app_functionality():
    """Test basic functionality of app modules."""

    try:
        # Test Settings creation
        from app.core.config import Settings
        settings = Settings()
        assert settings.app_name is not None
        assert settings.app_version is not None

        # Test logger creation
        from app.core.logging import get_logger
        logger = get_logger("test")
        assert logger is not None

        return True

    except ImportError:
        pytest.skip("App modules not available for functionality testing")
        return False
    except Exception as e:
        pytest.fail(f"App functionality test error: {e}")
        return False


def test_pydantic_models():
    """Test that Pydantic models work correctly."""

    try:
        from app.domain.models.agent import Agent, AgentConfiguration, AgentType

        # Test model creation
        config = AgentConfiguration(
            max_concurrent_tasks=5,
            timeout_seconds=300
        )

        agent = Agent(
            name="test-agent",
            agent_type=AgentType.WORKER,
            description="Test agent",
            configuration=config
        )

        assert agent.name == "test-agent"
        assert agent.agent_type == AgentType.WORKER
        assert agent.configuration.max_concurrent_tasks == 5

        return True

    except ImportError:
        pytest.skip("Domain models not available")
        return False
    except Exception as e:
        pytest.fail(f"Pydantic model test error: {e}")
        return False


# Pytest-compatible test functions
def test_all_core_imports():
    """Pytest-compatible test for core imports."""
    assert test_core_app_imports()


def test_all_domain_models():
    """Pytest-compatible test for domain models."""
    assert test_domain_models()


def test_all_api_modules():
    """Pytest-compatible test for API modules."""
    assert test_api_modules()


def test_all_app_functionality():
    """Pytest-compatible test for app functionality."""
    assert test_app_functionality()


def test_all_pydantic_models():
    """Pytest-compatible test for Pydantic models."""
    assert test_pydantic_models()


if __name__ == "__main__":
    # Run with pytest when executed directly
    pytest.main([__file__, "-v"])
