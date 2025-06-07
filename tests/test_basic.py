"""
Basic tests for DataMCPServerAgent CI/CD validation.
"""

import sys
import os
import pytest

def test_python_version():
    """Test that we're running on a supported Python version."""
    version = sys.version_info
    assert version.major == 3
    assert version.minor >= 9
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")

def test_basic_imports():
    """Test that basic Python modules can be imported."""
    try:
        import json
        import os
        import sys
        import pathlib
        print("✅ Basic Python modules imported successfully")
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import basic modules: {e}")

def test_project_structure():
    """Test that the project has the expected structure."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    expected_dirs = ['app', 'src', 'docs', 'tests']
    expected_files = ['README.md', 'pyproject.toml', 'requirements.txt']

    for directory in expected_dirs:
        dir_path = os.path.join(project_root, directory)
        assert os.path.exists(dir_path), f"Directory {directory} should exist"
        print(f"✅ Directory {directory} exists")

    for file in expected_files:
        file_path = os.path.join(project_root, file)
        assert os.path.exists(file_path), f"File {file} should exist"
        print(f"✅ File {file} exists")

def test_environment_variables():
    """Test environment variable handling."""
    # Test that we can set and get environment variables
    test_key = "TEST_CI_VAR"
    test_value = "test_value"

    os.environ[test_key] = test_value
    assert os.getenv(test_key) == test_value

    # Clean up
    del os.environ[test_key]
    print("✅ Environment variables work correctly")

def test_basic_math():
    """Test basic mathematical operations."""
    assert 2 + 2 == 4
    assert 10 - 5 == 5
    assert 3 * 4 == 12
    assert 8 / 2 == 4
    print("✅ Basic math operations work")

@pytest.mark.parametrize("input_value,expected", [
    ("hello", "hello"),
    (123, 123),
    ([1, 2, 3], [1, 2, 3]),
])
def test_parametrized_values(input_value, expected):
    """Test parametrized test functionality."""
    assert input_value == expected
    print(f"✅ Parametrized test passed for {input_value}")

def test_exception_handling():
    """Test exception handling."""
    with pytest.raises(ZeroDivisionError):
        result = 1 / 0

    print("✅ Exception handling works correctly")

def test_file_operations():
    """Test basic file operations."""
    import tempfile

    with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
        f.write("test content")
        temp_file = f.name

    try:
        # Read the file
        with open(temp_file, 'r') as f:
            content = f.read()

        assert content == "test content"
        print("✅ File operations work correctly")

    finally:
        # Clean up
        os.unlink(temp_file)

if __name__ == "__main__":
    # Run tests when script is executed directly
    pytest.main([__file__, "-v"])
