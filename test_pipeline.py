#!/usr/bin/env python3
"""
Script to test the Document Processing Pipeline.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command: str, description: str = ""):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"🧪 {description or command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            command.split(),
            check=True,
            text=True
        )
        print(f"✅ Success: {description or command}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {description or command}")
        print(f"Exit code: {e.returncode}")
        return False


def main():
    """Run all pipeline tests."""
    print("🧪 Document Processing Pipeline - Test Suite")
    print("=" * 60)
    
    # Add src to Python path
    src_path = Path(__file__).parent / "src"
    sys.path.insert(0, str(src_path))
    
    # Run different test categories
    test_commands = [
        ("python examples/document_processing_example.py", "Document Processing Example"),
        ("python examples/complete_pipeline_example.py", "Complete Pipeline Example"),
        ("python examples/vector_stores_example.py", "Vector Stores Example"),
        ("python examples/advanced_features_example.py", "Advanced Features Example")
    ]
    
    passed = 0
    failed = 0
    
    for command, description in test_commands:
        success = run_command(command, description)
        if success:
            passed += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 Test Summary")
    print(f"{'='*60}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(passed / (passed + failed)) * 100:.1f}%")
    
    if failed > 0:
        print("\n⚠️  Some tests failed. Please check the output above.")
        sys.exit(1)
    else:
        print("\n🎉 All tests passed!")


if __name__ == "__main__":
    main()
