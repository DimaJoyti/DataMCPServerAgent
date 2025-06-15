#!/usr/bin/env python3
"""
System Integration Test for DataMCPServerAgent.
This test verifies that the main system components can be imported and initialized.
"""

import sys
from pathlib import Path

# Add app directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_core_imports():
    """Test that core modules can be imported."""
    try:
        from app.core.config import get_settings
        from app.core.logging_improved import get_logger
        from app.core.simple_config import SimpleSettings
        print("✅ Core modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Core import failed: {e}")
        return False

def test_rl_imports():
    """Test that RL modules can be imported."""
    try:
        from app.rl.rl_integration import get_rl_manager
        from app.rl.rl_manager import RLManager
        print("✅ RL modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ RL import failed: {e}")
        return False

def test_api_imports():
    """Test that API modules can be imported."""
    try:
        from app.api.main import app
        print("✅ API modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ API import failed: {e}")
        return False

def test_phase6_imports():
    """Test that Phase 6 modules can be imported (with fallback)."""
    try:
        # These might fail if cloud dependencies aren't installed
        from app.rl.federated_learning import create_federated_coordinator
        print("✅ Federated learning imported successfully")
    except ImportError as e:
        print(f"⚠️ Federated learning import failed (expected): {e}")

    try:
        from app.scaling.auto_scaling import create_auto_scaler
        print("✅ Auto-scaling imported successfully")
    except ImportError as e:
        print(f"⚠️ Auto-scaling import failed (expected): {e}")

    try:
        from app.monitoring.real_time_monitoring import get_real_time_monitor
        print("✅ Real-time monitoring imported successfully")
    except ImportError as e:
        print(f"⚠️ Real-time monitoring import failed (expected): {e}")

    return True

def test_basic_functionality():
    """Test basic system functionality."""
    try:
        from app.core.config import get_settings
        settings = get_settings()
        print(f"✅ Settings loaded: {type(settings).__name__}")

        from app.core.logging_improved import get_logger
        logger = get_logger("test")
        logger.info("Test log message")
        print("✅ Logging system working")

        return True
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("🧪 DataMCPServerAgent System Integration Test")
    print("=" * 50)

    tests = [
        ("Core Imports", test_core_imports),
        ("RL Imports", test_rl_imports),
        ("API Imports", test_api_imports),
        ("Phase 6 Imports", test_phase6_imports),
        ("Basic Functionality", test_basic_functionality),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} passed")
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! System is ready.")
        return 0
    else:
        print("⚠️ Some tests failed. Check dependencies and configuration.")
        return 1

if __name__ == "__main__":
    exit(main())
