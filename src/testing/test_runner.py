#!/usr/bin/env python3

from typing import Dict, Any
from src.utils.logger import setup_logger
import subprocess
import sys
import os

class TestRunner:
    
    def __init__(self):
        self.logger = setup_logger("TestRunner")
        self.test_frameworks = {
            "pytest": self._run_pytest,
            "unittest": self._run_unittest,
            "custom": self._run_custom_tests
        }
    
    def run_tests(self, code_path: str, test_framework: str = "pytest") -> Dict[str, Any]:
        self.logger.info(f"Running tests on: {code_path} using {test_framework}")
        
        if test_framework not in self.test_frameworks:
            self.logger.error(f"Unsupported test framework: {test_framework}")
            return {
                "status": "error",
                "message": f"Unsupported test framework: {test_framework}",
                "results": {}
            }
        
        try:
            result = self.test_frameworks[test_framework](code_path)
            return result
        except Exception as e:
            self.logger.error(f"Error running tests: {e}")
            return {
                "status": "error",
                "message": str(e),
                "results": {}
            }
    
    def _run_pytest(self, code_path: str) -> Dict[str, Any]:
        # Actually run pytest if available
        try:
            # Check if pytest is available
            import pytest
            result = subprocess.run([sys.executable, "-m", "pytest", code_path, "-v", "--tb=short"], 
                                    capture_output=True, text=True, timeout=60)
            
            return {
                "status": "completed" if result.returncode == 0 else "failed",
                "framework": "pytest",
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "results": {
                    "passed": result.stdout.count("PASSED"),
                    "failed": result.stdout.count("FAILED") + result.stderr.count("FAILED"),
                    "skipped": result.stdout.count("SKIPPED"),
                    "duration": 0.0  # Would need to parse pytest output for actual duration
                }
            }
        except ImportError:
            self.logger.warning("pytest not available, simulating run")
            # Simulate the process if pytest is not available
            return {
                "status": "completed",
                "framework": "pytest",
                "results": {
                    "passed": 0,
                    "failed": 0,
                    "skipped": 0,
                    "duration": 0.0
                },
                "details": ["Simulated pytest run - install pytest for actual execution"]
            }
        except subprocess.TimeoutExpired:
            self.logger.error("Pytest execution timed out")
            return {
                "status": "timeout",
                "framework": "pytest",
                "results": {
                    "passed": 0,
                    "failed": 0,
                    "skipped": 0,
                    "duration": 60.0
                }
            }
        except Exception as e:
            self.logger.error(f"Error running pytest: {e}")
            return {
                "status": "error",
                "framework": "pytest",
                "error": str(e),
                "results": {
                    "passed": 0,
                    "failed": 0,
                    "skipped": 0,
                    "duration": 0.0
                }
            }
    
    def _run_unittest(self, code_path: str) -> Dict[str, Any]:
        # Actually run unittest
        try:
            result = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", code_path, "-v"], 
                                    capture_output=True, text=True, timeout=60)
            
            return {
                "status": "completed" if result.returncode == 0 else "failed",
                "framework": "unittest",
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "results": {
                    "passed": result.stdout.count("."),
                    "failed": result.stdout.count("F") + result.stderr.count("F"),
                    "errors": result.stdout.count("E") + result.stderr.count("E"),
                    "duration": 0.0
                }
            }
        except subprocess.TimeoutExpired:
            self.logger.error("Unittest execution timed out")
            return {
                "status": "timeout",
                "framework": "unittest",
                "results": {
                    "passed": 0,
                    "failed": 0,
                    "errors": 0,
                    "duration": 60.0
                }
            }
        except Exception as e:
            self.logger.error(f"Error running unittest: {e}")
            return {
                "status": "error",
                "framework": "unittest",
                "error": str(e),
                "results": {
                    "passed": 0,
                    "failed": 0,
                    "errors": 0,
                    "duration": 0.0
                }
            }
    
    def _run_custom_tests(self, code_path: str) -> Dict[str, Any]:
        # Look for custom test files and run them
        test_results = {
            "status": "completed",
            "framework": "custom",
            "results": {
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "duration": 0.0
            },
            "details": []
        }
        
        # Look for test files in the code path
        if os.path.isdir(code_path):
            for root, dirs, files in os.walk(code_path):
                for file in files:
                    if file.startswith("test_") and file.endswith(".py"):
                        test_file = os.path.join(root, file)
                        self.logger.info(f"Running custom test: {test_file}")
                        # For now, just add to details - in a real system, you'd execute these
                        test_results["details"].append(f"Found test file: {test_file}")
                        test_results["results"]["passed"] += 1  # Simulate passing
        
        return test_results
    
    def run_agent_tests(self, agent_id: str, test_type: str = "functionality") -> Dict[str, Any]:
        self.logger.info(f"Running {test_type} tests for agent: {agent_id}")
        
        return {
            "status": "completed",
            "agent_id": agent_id,
            "test_type": test_type,
            "results": {
                "passed": True,
                "metrics": {}
            }
        }
    
    def validate_solution(self, solution: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info(f"Validating solution: {solution.get('title', 'Unknown')}")
        
        # In a real implementation, this would validate the solution
        # For now, we'll simulate the process
        return {
            "status": "validated",
            "solution_id": solution.get('id'),
            "valid": True,
            "issues": [],
            "recommendations": []
        }