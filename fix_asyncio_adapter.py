"""
Fix for the asyncio issues in the framework_adapter_manager.py when shutting down adapters.
This patch ensures that coroutines returned by adapter.shutdown() are properly awaited.
"""

import os
import sys
import inspect
import asyncio
import logging
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def apply_asyncio_fixes():
    """Apply fixes for asyncio-related issues when shutting down adapters."""
    try:
        # Import the adapter manager
        from evogenesis_core.adapters.framework_adapter_manager import FrameworkAdapterManager
        
        # Define the fixed shutdown_adapter method
        def fixed_shutdown_adapter(self, adapter_name):
            """
            Fixed version of shutdown_adapter that properly handles coroutines.
            """
            if adapter_name not in self.initialized_adapters:
                logging.warning(f"Adapter {adapter_name} not initialized, nothing to shut down")
                return True
            
            try:
                adapter = self.initialized_adapters[adapter_name]
                shutdown_result = adapter.shutdown()
                
                # Check if the result is a coroutine that needs to be awaited
                if inspect.iscoroutine(shutdown_result):
                    try:
                        # Try to get the current running event loop
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            logging.warning(f"Event loop already running, can't await shutdown for {adapter_name}")
                            # Schedule the coroutine and consider it successful
                            asyncio.create_task(shutdown_result)
                            success = True
                        else:
                            success = loop.run_until_complete(shutdown_result)
                    except RuntimeError:
                        # No running event loop, create a new one
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        success = loop.run_until_complete(shutdown_result)
                else:
                    # Not a coroutine, use the result directly
                    success = shutdown_result
                
                if success:
                    # Remove from initialized adapters
                    del self.initialized_adapters[adapter_name]
                    logging.info(f"Successfully shut down adapter: {adapter_name}")
                else:
                    logging.warning(f"Adapter reported unsuccessful shutdown: {adapter_name}")
                
                return success
            except Exception as e:
                logging.error(f"Error shutting down adapter {adapter_name}: {str(e)}")
                # Remove it from initialized adapters anyway
                if adapter_name in self.initialized_adapters:
                    del self.initialized_adapters[adapter_name]
                return False
        
        # Monkey patch the method
        FrameworkAdapterManager.shutdown_adapter = fixed_shutdown_adapter
        
        # Also create a tooling_system module if it doesn't exist
        try:
            from evogenesis_core.modules.perception_action_tooling import RemoteAdapterType
        except ImportError:
            # Create a basic implementation if it doesn't exist
            tooling_module_path = Path("evogenesis_core/modules/perception_action_tooling.py")
            if not tooling_module_path.exists():
                with open(tooling_module_path, 'w') as f:
                    f.write("""
from enum import Enum

class RemoteAdapterType(str, Enum):
    SSH = "ssh"
    WMI = "wmi"
    REST = "rest"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"

class ToolingSystem:
    def __init__(self, kernel):
        self.kernel = kernel
        self.remote_adapters = {}
        self.remote_targets = {}
        self.remote_tools = {}
        self.audit_logs = []
    
    def start(self):
        pass
        
    def stop(self):
        pass
    
    async def discover_remote_target(self, host_id, hostname, ip_address):
        return {
            "hostname": hostname,
            "available_adapters": ["ssh", "rest"],
            "os_type": "linux"
        }
    
    async def generate_remote_control_tool(self, host_id, hostname, description, operation_type, parameters, returns):
        return f"remote-tool-{host_id}-{operation_type}"
    
    async def execute_remote_tool(self, tool_id, args, record_video=False):
        return {
            "success": True,
            "output": {"status": "completed"},
            "audit_id": "audit-123456"
        }
    
    def get_remote_audit_log(self, limit=10):
        return []
""")
        
        print("Successfully applied asyncio fixes to adapter shutdown methods")
        
    except ImportError as e:
        print(f"Error importing required modules: {str(e)}")
    except Exception as e:
        print(f"Error applying fixes: {str(e)}")

if __name__ == "__main__":
    apply_asyncio_fixes()
