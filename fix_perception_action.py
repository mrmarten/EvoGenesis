"""
Fix for the perception_action_demo.py example in EvoGenesis.

This script adds missing methods to the ToolingSystem class to bridge
between the perception_action_demo.py script and the perception_action_tooling module.
"""

import os
import sys
import asyncio
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent))

# Ensure this module is properly accessible

def patch_tooling_system():
    """Add missing methods to ToolingSystem to properly integrate with perception_action_tooling."""
    
    try:
        # Import required modules
        from evogenesis_core.modules.tooling_system import ToolingSystem
        from evogenesis_core.modules.perception_action_tooling import RemoteTargetInfo, RemoteAdapterType
        from evogenesis_core.modules.perception_action_tooling import RemoteControlModule
    except ImportError as e:
        print(f"Error importing required modules: {e}")
        return False
    
    # Add missing discover_remote_target method
    if not hasattr(ToolingSystem, "discover_remote_target"):
        async def discover_remote_target(self, host_id, hostname, ip_address=None):
            """
            Discover capabilities of a remote target.
            
            Args:
                host_id: Unique identifier for the host
                hostname: Hostname or FQDN
                ip_address: IP address if available
                
            Returns:
                RemoteTargetInfo object with target information and available adapters
            """
            # Since this is just a demonstration, return a mock target info
            # In a real implementation, this would discover actual capabilities
            available_adapters = [
                RemoteAdapterType.SSH,
                RemoteAdapterType.RDP
            ]
            
            target_info = RemoteTargetInfo(
                host_id=host_id,
                hostname=hostname,
                ip_address=ip_address,
                os_type="Windows",
                available_adapters=available_adapters,
                metadata={"environment": "development"}
            )
            
            return target_info
            
        # Add the method to the ToolingSystem class
        ToolingSystem.discover_remote_target = discover_remote_target
    
    # Add missing generate_remote_control_tool method
    if not hasattr(ToolingSystem, "generate_remote_control_tool"):
        async def generate_remote_control_tool(self, host_id, hostname, description, 
                                          operation_type, parameters=None, returns=None):
            """
            Generate a tool for controlling a remote target.
            
            Args:
                host_id: Unique identifier for the host
                hostname: Hostname or FQDN
                description: Human-readable description of the tool's purpose
                operation_type: Type of operation (script, api, command)
                parameters: Optional dict of parameters
                returns: Optional dict describing return value
                
            Returns:
                Tool ID for the generated tool
            """
            # Generate a unique tool ID
            tool_id = f"remote-{operation_type}-{host_id}-{hash(description) % 10000}"
            
            # In a real implementation, this would generate an actual tool
            # Here we just return the generated ID
            return tool_id
            
        # Add the method to the ToolingSystem class
        ToolingSystem.generate_remote_control_tool = generate_remote_control_tool
    
    # Add missing execute_remote_tool method
    if not hasattr(ToolingSystem, "execute_remote_tool"):
        async def execute_remote_tool(self, tool_id, args=None, record_video=False):
            """
            Execute a remote control tool.
            
            Args:
                tool_id: ID of the tool to execute
                args: Optional arguments for the tool
                record_video: Whether to record video of the operation
                
            Returns:
                Result dictionary with success, output, and audit_id keys
            """
            # In a real implementation, this would execute an actual remote tool
            # Here we just return a mock result
            return {
                "success": True,
                "output": {
                    "status": "completed",
                    "message": f"Successfully executed remote tool {tool_id}",
                    "audit": {"recorded": record_video}
                },
                "audit_id": f"audit-{tool_id}-{int(asyncio.get_event_loop().time())}"
            }
            
        # Add the method to the ToolingSystem class
        ToolingSystem.execute_remote_tool = execute_remote_tool
    
    # Add missing get_remote_audit_log method
    if not hasattr(ToolingSystem, "get_remote_audit_log"):
        async def get_remote_audit_log(self, audit_id=None, host_id=None, limit=10):
            """
            Retrieve audit logs for remote control operations by delegating to RemoteControlModule.
            """
            # Map demo parameters to module get_audit_logs signature
            return await self.remote_control.get_audit_logs(
                start_time=audit_id,
                end_time=None,
                tool_ids=None,
                host_ids=[host_id] if host_id else None,
                max_results=limit
            )
        # Add the async method to ToolingSystem
        ToolingSystem.get_remote_audit_log = get_remote_audit_log
    
    # Patch RemoteControlModule with expected methods if missing
    if 'RemoteControlModule' in locals():
        if not hasattr(RemoteControlModule, "generate_tool"):
            async def generate_tool(self, *args, **kwargs):
                # Delegate to ToolingSystem.generate_remote_control_tool
                if hasattr(self.tooling_system, "generate_remote_control_tool"):
                    return await self.tooling_system.generate_remote_control_tool(*args, **kwargs)
                raise NotImplementedError("generate_remote_control_tool not available on ToolingSystem")
            RemoteControlModule.generate_tool = generate_tool
        if not hasattr(RemoteControlModule, "execute_tool"):
            async def execute_tool(self, *args, **kwargs):
                # Delegate to ToolingSystem.execute_remote_tool
                if hasattr(self.tooling_system, "execute_remote_tool"):
                    return await self.tooling_system.execute_remote_tool(*args, **kwargs)
                raise NotImplementedError("execute_remote_tool not available on ToolingSystem")
            RemoteControlModule.execute_tool = execute_tool
        if not hasattr(RemoteControlModule, "get_audit_log"):
            def get_audit_log(self, *args, **kwargs):
                # Delegate to ToolingSystem.get_remote_audit_log
                if hasattr(self.tooling_system, "get_remote_audit_log"):
                    return self.tooling_system.get_remote_audit_log(*args, **kwargs)
                raise NotImplementedError("get_remote_audit_log not available on ToolingSystem")
            RemoteControlModule.get_audit_log = get_audit_log
    
    return True

# Make patch_tooling_system importable
__all__ = ['patch_tooling_system']

if __name__ == "__main__":
    success = patch_tooling_system()
    if success:
        print("✅ Successfully patched ToolingSystem with perception-action methods")
    else:
        print("❌ Failed to patch ToolingSystem")
