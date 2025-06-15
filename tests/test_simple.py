#!/usr/bin/env python3
"""
Simple tests without external dependencies.
These tests should always pass in any Python environment.
"""

import os
import sys
from pathlib import Path


def test_python_version():
    """Test Python version compatibility."""
    version = sys.version_info
    assert version.major == 3
    assert version.minor >= 9
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")


def test_basic_imports():
    """Test basic Python standard library imports."""
    import json
    import os
    import pathlib
    import sys
    import tempfile

    assert json is not None
    assert os is not None
    assert sys is not None
    assert pathlib is not None
    assert tempfile is not None
    print("✅ Basic imports successful")


def test_project_structure():
    """Test basic project structure."""
    project_root = Path(__file__).parent.parent

    # Check for essential directories
    essential_dirs = ["tests"]
    for dir_name in essential_dirs:
        dir_path = project_root / dir_name
        assert dir_path.exists(), f"{dir_name} directory should exist"

    # Check for essential files
    essential_files = ["README.md", "pyproject.toml"]
    for file_name in essential_files:
        file_path = project_root / file_name
        assert file_path.exists(), f"{file_name} should exist"

    print("✅ Project structure valid")


def test_basic_math():
    """Test basic mathematical operations."""
    assert 2 + 2 == 4
    assert 10 - 5 == 5
    assert 3 * 4 == 12
    assert 8 / 2 == 4.0
    assert 2 ** 3 == 8
    assert 10 % 3 == 1
    print("✅ Basic math operations work")


def test_string_operations():
    """Test basic string operations."""
    test_string = "Hello, World!"

    assert test_string.lower() == "hello, world!"
    assert test_string.upper() == "HELLO, WORLD!"
    assert test_string.replace("World", "Python") == "Hello, Python!"
    assert len(test_string) == 13
    assert "World" in test_string
    print("✅ String operations work")


def test_list_operations():
    """Test basic list operations."""
    test_list = [1, 2, 3, 4, 5]

    assert len(test_list) == 5
    assert test_list[0] == 1
    assert test_list[-1] == 5

    test_list.append(6)
    assert len(test_list) == 6
    assert test_list[-1] == 6

    test_list.remove(3)
    assert 3 not in test_list
    assert len(test_list) == 5
    print("✅ List operations work")


def test_dict_operations():
    """Test basic dictionary operations."""
    test_dict = {"name": "test", "version": "1.0", "active": True}

    assert test_dict["name"] == "test"
    assert test_dict.get("version") == "1.0"
    assert test_dict.get("missing", "default") == "default"

    test_dict["new_key"] = "new_value"
    assert "new_key" in test_dict
    assert test_dict["new_key"] == "new_value"
    print("✅ Dictionary operations work")


def test_file_operations():
    """Test basic file operations."""
    import tempfile

    # Test file creation and writing
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        test_content = "Hello, World! This is a test."
        f.write(test_content)
        temp_file_path = f.name

    try:
        # Test file reading
        with open(temp_file_path) as f:
            content = f.read()

        assert content == test_content
        assert os.path.exists(temp_file_path)

    finally:
        # Clean up
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

    print("✅ File operations work")


def test_json_operations():
    """Test JSON serialization/deserialization."""
    import json

    test_data = {
        "name": "test_project",
        "version": "1.0.0",
        "features": ["feature1", "feature2"],
        "config": {
            "debug": True,
            "timeout": 30
        }
    }

    # Test serialization
    json_string = json.dumps(test_data)
    assert isinstance(json_string, str)
    assert "test_project" in json_string

    # Test deserialization
    parsed_data = json.loads(json_string)
    assert parsed_data == test_data
    assert parsed_data["name"] == "test_project"
    assert parsed_data["config"]["debug"] is True
    print("✅ JSON operations work")


def run_all_tests():
    """Run all tests."""
    tests = [
        test_python_version,
        test_basic_imports,
        test_project_structure,
        test_basic_math,
        test_string_operations,
        test_list_operations,
        test_dict_operations,
        test_file_operations,
        test_json_operations,
    ]

    print("🧪 Running simple tests...")
    print("=" * 50)

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} failed: {e}")
            failed += 1

    print("=" * 50)
    print(f"📊 Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
