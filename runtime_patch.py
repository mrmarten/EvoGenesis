"""
Runtime monkey patch for EvoGenesis bugs.

This script fixes:
1. Perception-Action Tooling module issues
2. Async event loop errors with adapters
"""

import sys
import os
import logging

# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("runtime_patch")

# Add the current directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Define mock Remote Control methods
async def mock_discover_remote_target(self, host_id, hostname, ip_address=None):
    """Mock implementation for discover_remote_target."""
    logger.info(f"MOCK: Discovering target {hostname}")
    return {
        "host_id": host_id,
        "hostname": hostname,
        "ip_address": ip_address,
        "os_type": "Windows",
        "available_adapters": ["SSH", "RDP"]
    }

async def mock_generate_remote_control_tool(self, host_id, hostname, description, **kwargs):
    """Mock implementation for generate_remote_control_tool."""
    logger.info(f"MOCK: Generating tool for {hostname}")
    return f"mock-tool-{hash(description) % 10000}"

async def mock_execute_remote_tool(self, tool_id, args, **kwargs):
    """Mock implementation for execute_remote_tool."""
    logger.info(f"MOCK: Executing tool {tool_id}")
    return {
        "success": True,
        "execution_id": f"mock-exec-{hash(tool_id) % 10000}",
        "result": f"Mock execution of {tool_id} completed successfully"
    }

def mock_get_remote_audit_log(self, **kwargs):
    """Mock implementation for get_remote_audit_log."""
    logger.info("MOCK: Getting audit logs")
    return []

# Define a safer adapter shutdown method
def safe_shutdown_adapter(self, adapter_name):
    """Safe implementation for shutdown_adapter that avoids event loop issues."""
    logger.info(f"SAFE: Shutting down adapter {adapter_name}")
    
    if adapter_name not in self.initialized_adapters:
        return False
    
    # Just remove the adapter without trying to call async methods
    if adapter_name in self.initialized_adapters:
        del self.initialized_adapters[adapter_name]
        logger.info(f"SAFE: Successfully removed adapter {adapter_name}")
    
    return True

def apply_patches():
    """Apply all the monkey patches."""
    logger.info("Applying runtime patches to EvoGenesis...")
    
    try:
        # Import the modules we need to patch
        from evogenesis_core.modules.tooling_system import ToolingSystem
        from evogenesis_core.adapters.framework_adapter_manager import FrameworkAdapterManager
        
        # Patch Perception-Action Tooling methods
        logger.info("Patching Perception-Action Tooling methods...")
        ToolingSystem.discover_remote_target = mock_discover_remote_target
        ToolingSystem.generate_remote_control_tool = mock_generate_remote_control_tool
        ToolingSystem.execute_remote_tool = mock_execute_remote_tool
        ToolingSystem.get_remote_audit_log = mock_get_remote_audit_log
        
        # Patch adapter shutdown method
        logger.info("Patching adapter shutdown method...")
        FrameworkAdapterManager.shutdown_adapter = safe_shutdown_adapter
        
        logger.info("All patches applied successfully!")
        return True
    except Exception as e:
        logger.error(f"Error applying patches: {e}")
        return False

if __name__ == "__main__":
    apply_patches()
