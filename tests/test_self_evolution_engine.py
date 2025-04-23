#!/usr/bin/env python
# Tests for Self Evolution Engine
import unittest
import sys
import os
import asyncio
from unittest.mock import MagicMock, patch

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evogenesis_core.modules.self_evolution_engine import SelfEvolutionEngine

class TestSelfEvolutionEngine(unittest.TestCase):
    """Test cases for EvoGenesis Self Evolution Engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Creating a mock kernel for the engine to use
        self.mock_kernel = MagicMock()
        self.mock_kernel.config = {"self_evolution": {"enabled": True}}
        self.mock_kernel.logger = MagicMock()
        
        # Initialize the engine with the mock kernel
        with patch('threading.Thread'):  # Prevent background threads from starting
            self.engine = SelfEvolutionEngine(self.mock_kernel)
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.engine = None
        self.mock_kernel = None
    
    def test_ab_test_creation(self):
        """Test that A/B tests can be created and tracked."""
        # Mock the functions that would start background threads
        with patch.object(self.engine, '_run_ab_test_thread'):
            test_id = self.engine.run_ab_test(
                feature="test_feature",
                version_a="current",
                version_b="experimental",
                duration_seconds=60,
                metrics=["latency", "error_rate"]
            )
            
            # Check that test was created
            self.assertIsNotNone(test_id)
            self.assertIn(test_id, self.engine.active_ab_tests)
            
            # Check test configuration
            test_config = self.engine.active_ab_tests[test_id]
            self.assertEqual(test_config["feature"], "test_feature")
            self.assertEqual(test_config["version_a"], "current")
            self.assertEqual(test_config["version_b"], "experimental")
            self.assertEqual(test_config["status"], "initializing")
    
    def test_get_ab_test_status(self):
        """Test retrieving A/B test status."""
        # Create a test entry
        test_id = "test-abtest-123"
        self.engine.active_ab_tests[test_id] = {
            "feature": "test_feature",
            "status": "running",
            "version_a": "current",
            "version_b": "experimental"
        }
        
        # Get status
        status = self.engine.get_ab_test_status(test_id)
        
        # Verify status data
        self.assertEqual(status["feature"], "test_feature")
        self.assertEqual(status["status"], "running")

if __name__ == '__main__':
    unittest.main()
