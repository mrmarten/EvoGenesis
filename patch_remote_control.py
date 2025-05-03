"""
Fix for the perception_action_demo.py bugs.

This script adds missing methods to the RemoteControlModule class
in the perception_action_tooling.py module.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from evogenesis_core.modules.perception_action_tooling import RemoteControlModule


def patch_remote_control_module():
    """Add missing methods to RemoteControlModule."""
    
    # Add alias for generate_tool -> generate_remote_control_tool
    if not hasattr(RemoteControlModule, "generate_remote_control_tool"):
        RemoteControlModule.generate_remote_control_tool = RemoteControlModule.generate_tool
        print("✅ Added generate_remote_control_tool alias")
    
    # Add missing execute_remote_tool method
    if not hasattr(RemoteControlModule, "execute_remote_tool"):
        async def execute_remote_tool(self, tool_id, args=None, record_video=False):
            """
            Execute a remote control tool.
            
            Args:
                tool_id: ID of the tool to execute
                args: Arguments to pass to the tool
                record_video: Whether to record video of the execution
                
            Returns:
                Dictionary with execution results
            """
            # Execute the tool with the provided arguments
            result = await self.tooling_system.execute_tool(
                tool_id=tool_id,
                args=args or {}
            )
            
            # Create an audit record
            tool = self.tooling_system.get_tool(tool_id)
            hostname = tool.metadata.get("hostname", "unknown") if tool else "unknown"
            host_id = tool.metadata.get("host_id", "unknown") if tool else "unknown"
            
            # Generate an audit ID
            from datetime import datetime
            import uuid
            audit_id = str(uuid.uuid4())
            
            # Create an audit record
            audit_record = {
                "audit_id": audit_id,
                "tool_id": tool_id,
                "tool_name": tool.name if tool else "unknown",
                "hostname": hostname,
                "host_id": host_id,
                "timestamp_start": datetime.now().isoformat(),
                "timestamp_end": datetime.now().isoformat(),
                "args": args,
                "success": result.get("success", False),
                "result_summary": str(result.get("output", {}))[:200],
                "has_video": record_video
            }
            
            # Store the audit record
            if not hasattr(self, "audit_log"):
                self.audit_log = []
            self.audit_log.append(audit_record)
            
            # Return the result with the audit ID
            return {
                "success": result.get("success", False),
                "output": result.get("output", {}),
                "audit_id": audit_id
            }
        
        RemoteControlModule.execute_remote_tool = execute_remote_tool
        print("✅ Added execute_remote_tool method")
    
    # Add missing get_remote_audit_log method
    if not hasattr(RemoteControlModule, "get_remote_audit_log"):
        def get_remote_audit_log(self, limit=None):
            """
            Get the remote audit log entries.
            
            Args:
                limit: Maximum number of entries to return
                
            Returns:
                List of audit log entries
            """
            if not hasattr(self, "audit_log"):
                self.audit_log = []
                
            log_entries = self.audit_log
            
            if limit:
                log_entries = log_entries[-limit:]
                
            return log_entries
        
        RemoteControlModule.get_remote_audit_log = get_remote_audit_log
        print("✅ Added get_remote_audit_log method")


if __name__ == "__main__":
    patch_remote_control_module()
    print("All fixes have been applied to RemoteControlModule")
