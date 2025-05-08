#!/usr/bin/env python3
"""
Script to run all tests for DataMCPServerAgent.
"""

import os
import sys
import unittest
import argparse


def run_tests(test_pattern=None):
    """Run all tests or tests matching a pattern.
    
    Args:
        test_pattern: Optional pattern to match test files
    """
    # Add the project root to the Python path
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    # Discover and run tests
    if test_pattern:
        print(f"Running tests matching pattern: {test_pattern}")
        test_suite = unittest.defaultTestLoader.discover('tests', pattern=f'*{test_pattern}*.py')
    else:
        print("Running all tests")
        test_suite = unittest.defaultTestLoader.discover('tests')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return non-zero exit code if tests failed
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tests for DataMCPServerAgent')
    parser.add_argument('--pattern', type=str, help='Pattern to match test files')
    args = parser.parse_args()
    
    sys.exit(run_tests(args.pattern))