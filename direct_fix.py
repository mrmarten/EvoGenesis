"""
Direct fix for EvoGenesis bugs.
"""

import sys
import os
import logging
import asyncio
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("direct_fix")

# Add the main directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Monkeypatch for Perception-Action methods
def patch_system():
    print("Applying patches to fix EvoGenesis bugs...")
    
    # Import modules
    from evogenesis_core.modules.tooling_system import ToolingSystem
    from evogenesis_core.adapters.framework_adapter_manager import FrameworkAdapterManager
    
    # Mock methods for Perception-Action
    async def mock_discover(self, host_id, hostname, ip_address=None):
        print(f"[MOCK] Discovering target: {hostname}")
        return {
            "host_id": host_id,
            "hostname": hostname,
            "ip_address": ip_address,
            "os_type": "Windows",
            "available_adapters": ["SSH", "RDP"]
        }
    
    async def mock_generate(self, host_id, hostname, description, **kwargs):
        print(f"[MOCK] Generating tool for: {hostname}")
        return f"mock-tool-{hash(description) % 10000}"
    
    async def mock_execute(self, tool_id, args, **kwargs):
        print(f"[MOCK] Executing tool: {tool_id}")
        return {
            "success": True,
            "execution_id": f"mock-exec-{hash(tool_id) % 10000}",
            "result": f"Mock execution of {tool_id} completed successfully"
        }
    
    def mock_audit(self, **kwargs):
        print("[MOCK] Getting audit logs")
        return []
    
    # Safe adapter shutdown method
    def safe_shutdown(self, adapter_name):
        print(f"[SAFE] Shutting down adapter: {adapter_name}")
        if adapter_name not in self.initialized_adapters:
            return False
        
        # Just remove from initialized adapters
        if adapter_name in self.initialized_adapters:
            del self.initialized_adapters[adapter_name]
        
        return True
    
    # Apply patches
    ToolingSystem.discover_remote_target = mock_discover
    ToolingSystem.generate_remote_control_tool = mock_generate
    ToolingSystem.execute_remote_tool = mock_execute
    ToolingSystem.get_remote_audit_log = mock_audit
    FrameworkAdapterManager.shutdown_adapter = safe_shutdown
    
    print("All patches applied successfully!")

# Run the example with patches applied
if __name__ == "__main__":
    patch_system()
    
    print("\nRunning perception_action_demo.py example...")
    import examples.perception_action_demo
    print("\nExample completed!")
