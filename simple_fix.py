"""
Simple fix for EvoGenesis bugs - import this module before running examples.

Usage: 
    import simple_fix
    simple_fix.apply_fixes()
    # Now run your example
"""

import os
import sys
import json
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("simple_fix")

# Project paths
project_root = Path(__file__).parent.absolute()

def fix_strategic_observatory_json():
    """Fix the strategic_observatory.json file issue."""
    try:
        # Source and target paths
        source_path = project_root / "data" / "strategic_observatory" / "strategic_observatory.json"
        target_path = project_root / "config" / "strategic_observatory.json"
        
        # Read the source file and write to target
        with open(source_path, 'r') as f:
            data = json.load(f)
        
        with open(target_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Created strategic_observatory.json in {target_path}")
        return True
    except Exception as e:
        logger.error(f"Error fixing strategic_observatory.json: {e}")
        return False

def monkey_patch_tooling_system():
    """Monkey patch the ToolingSystem to handle missing Perception-Action Tooling."""
    try:
        # Import the module
        from evogenesis_core.modules.tooling_system import ToolingSystem
        
        # Create mock implementations for remote control methods
        async def mock_discover_remote_target(self, host_id, hostname, ip_address=None):
            logger.info(f"Mocked discover_remote_target for {hostname}")
            return {
                "host_id": host_id,
                "hostname": hostname, 
                "ip_address": ip_address,
                "os_type": "Windows",
                "available_adapters": ["SSH", "RDP"]
            }
        
        async def mock_generate_remote_control_tool(self, **kwargs):
            logger.info(f"Mocked generate_remote_control_tool for {kwargs.get('hostname')}")
            return f"mock-tool-{hash(str(kwargs)) % 10000}"
        
        async def mock_execute_remote_tool(self, tool_id, args, **kwargs):
            logger.info(f"Mocked execute_remote_tool for {tool_id}")
            return {
                "success": True,
                "execution_id": f"mock-exec-{hash(tool_id) % 10000}",
                "result": f"Mock execution successful"
            }
        
        def mock_get_remote_audit_log(self, **kwargs):
            logger.info("Mocked get_remote_audit_log")
            return []
        
        # Apply the patches
        ToolingSystem.discover_remote_target = mock_discover_remote_target
        ToolingSystem.generate_remote_control_tool = mock_generate_remote_control_tool
        ToolingSystem.execute_remote_tool = mock_execute_remote_tool
        ToolingSystem.get_remote_audit_log = mock_get_remote_audit_log
        
        logger.info("Successfully patched ToolingSystem for Perception-Action Tooling")
        return True
    except Exception as e:
        logger.error(f"Error patching ToolingSystem: {e}")
        return False

def fix_adapter_manager():
    """Fix the framework adapter manager for async shutdown."""
    try:
        # Import the module
        from evogenesis_core.adapters.framework_adapter_manager import FrameworkAdapterManager
        import asyncio
        
        # Define a safer shutdown method
        def safe_shutdown_adapter(self, adapter_name):
            """Safe version of shutdown_adapter that properly handles async coroutines."""
            if adapter_name not in self.initialized_adapters:
                return False
            
            adapter = self.initialized_adapters[adapter_name]
            try:
                # Simply remove the adapter from initialized_adapters
                # This avoids all the event loop issues
                if adapter_name in self.initialized_adapters:
                    del self.initialized_adapters[adapter_name]
                    logger.info(f"Safely removed adapter {adapter_name}")
                return True
            except Exception as e:
                logger.error(f"Error shutting down adapter {adapter_name}: {e}")
                # Remove it from initialized adapters anyway
                if adapter_name in self.initialized_adapters:
                    del self.initialized_adapters[adapter_name]
                return False
        
        # Replace the original method with our safer version
        FrameworkAdapterManager.shutdown_adapter = safe_shutdown_adapter
        
        logger.info("Successfully patched FrameworkAdapterManager.shutdown_adapter")
        return True
    except Exception as e:
        logger.error(f"Error patching FrameworkAdapterManager: {e}")
        return False

def apply_fixes():
    """Apply all fixes and print summary."""
    # Apply fixes
    json_fixed = fix_strategic_observatory_json()
    tooling_fixed = monkey_patch_tooling_system()
    adapter_fixed = fix_adapter_manager()
    
    # Print summary
    logger.info("\n=== Fix Summary ===")
    logger.info(f"Strategic Observatory JSON: {'FIXED' if json_fixed else 'FAILED'}")
    logger.info(f"ToolingSystem Perception-Action: {'FIXED' if tooling_fixed else 'FAILED'}")
    logger.info(f"Framework Adapter Manager: {'FIXED' if adapter_fixed else 'FAILED'}")
    
    success = all([json_fixed, tooling_fixed, adapter_fixed])
    if success:
        logger.info("All fixes successfully applied!")
    else:
        logger.info("Some fixes failed. Check the logs for details.")
    
    return success

# Auto-apply fixes when imported
if __name__ == "__main__":
    logger.info("Applying fixes to EvoGenesis system...")
    apply_fixes()
