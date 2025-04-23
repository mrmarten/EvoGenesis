#!/usr/bin/env python
# Tests for EvoGenesis Kernel
import unittest
import sys
import os

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evogenesis_core.kernel import Kernel

class TestKernel(unittest.TestCase):
    """Test cases for EvoGenesis Kernel."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.kernel = Kernel(config_path="config/default.json")
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.kernel = None
    
    def test_kernel_initialization(self):
        """Test that kernel initializes correctly."""
        self.assertIsNotNone(self.kernel)
        self.assertIsNotNone(self.kernel.config)
    
    def test_kernel_modules_loaded(self):
        """Test that kernel modules are loaded."""
        # These assertions will need to be updated based on your actual module structure
        # This is a basic test to ensure the framework initializes core components
        self.assertIsNotNone(getattr(self.kernel, "tooling_system", None))
        self.assertIsNotNone(getattr(self.kernel, "llm_orchestrator", None))
        self.assertIsNotNone(getattr(self.kernel, "agent_manager", None))

if __name__ == '__main__':
    unittest.main()
