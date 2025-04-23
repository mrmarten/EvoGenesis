#!/usr/bin/env python
# Tests for Tooling System
import unittest
import sys
import os
import asyncio

# Add parent directory to path to import the module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evogenesis_core.modules.tooling_system import ToolingSystem, Tool, ToolStatus

class TestToolingSystem(unittest.TestCase):
    """Test cases for EvoGenesis Tooling System."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tooling_system = ToolingSystem()
    
    def tearDown(self):
        """Tear down test fixtures."""
        self.tooling_system = None
    
    def test_tool_registration(self):
        """Test that tools can be registered correctly."""
        # Create a simple test tool
        test_tool = Tool(
            id="test-tool-1",
            name="Test Tool",
            description="A test tool for unit testing",
            status=ToolStatus.ACTIVE
        )
        
        # Register the tool
        self.tooling_system.register_tool(test_tool)
        
        # Verify the tool is registered
        self.assertIn(test_tool.id, self.tooling_system.get_tools())
    
    def test_tool_execution_result(self):
        """Test that the execute_tool_safely method returns expected results."""
        # This would normally need to run in an async context
        async def run_test():
            test_tool = Tool(
                id="test-tool-2",
                name="Echo Tool",
                description="A tool that echoes input",
                status=ToolStatus.ACTIVE,
                function=lambda args: args["input"]
            )
            
            self.tooling_system.register_tool(test_tool)
            
            result = await self.tooling_system.execute_tool_safely(
                tool_id="test-tool-2",
                args={"input": "test value"}
            )
            
            return result
            
        # Run the async test synchronously
        result = asyncio.run(run_test())
        
        # Check that the result is valid
        self.assertTrue(hasattr(result, 'success'))

if __name__ == '__main__':
    unittest.main()
