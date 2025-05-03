"""
Fix for RemoteTargetInfo class in perception_action_tooling.py

This script updates the perception_action_demo.py file to properly 
access RemoteTargetInfo objects instead of treating them as dictionaries.
"""

import os
import sys
import re
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def fix_demo_script():
    """Fix the perception_action_demo.py script."""
    demo_path = os.path.join(os.path.dirname(__file__), 
                           "examples", "perception_action_demo.py")
    
    if not os.path.exists(demo_path):
        print(f"Error: Demo script not found at {demo_path}")
        return False
    
    # Read the current file contents
    with open(demo_path, 'r') as f:
        content = f.read()
    
    # Fix 1: Update target_info access to use object properties instead of dictionary access
    pattern1 = r"print\(f\"Discovered target: \{target_info\['hostname'\]\}\"\)"
    replacement1 = r"print(f\"Discovered target: {target_info.hostname}\")"
    content = re.sub(pattern1, replacement1, content)
    
    pattern2 = r"print\(f\"Available adapters: \{target_info\['available_adapters'\]\}\"\)"
    replacement2 = r"print(f\"Available adapters: {target_info.available_adapters}\")"
    content = re.sub(pattern2, replacement2, content)
    
    pattern3 = r"print\(f\"OS type: \{target_info\['os_type'\]\}\"\)"
    replacement3 = r"print(f\"OS type: {target_info.os_type}\")"
    content = re.sub(pattern3, replacement3, content)
    
    # Write the updated content back to the file
    with open(demo_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated {demo_path} to correctly access RemoteTargetInfo properties")
    return True


def fix_tooling_system():
    """Fix the tooling_system.py to handle RemoteTargetInfo objects."""
    tooling_system_path = os.path.join(os.path.dirname(__file__),
                                 "evogenesis_core", "modules", "tooling_system.py")
    
    if not os.path.exists(tooling_system_path):
        print(f"Error: Tooling system not found at {tooling_system_path}")
        return False
    
    # Read the current file contents
    with open(tooling_system_path, 'r') as f:
        content = f.read()
    
    # Find the RemoteDiscoveryService mock implementation 
    if "async def discover_remote_target" not in content:
        print("Adding missing discover_remote_target method to ToolingSystem")
        
        # Find a good insertion point - after the end of the class def
        insertion_point = content.find("    def get_tools_with_tag")
        if insertion_point > 0:
            # Define the new method
            new_method = """
    async def discover_remote_target(self, host_id, hostname, ip_address=None):
        """Discover capabilities of a remote target."""
        if not PERCEPTION_ACTION_AVAILABLE:
            raise NotImplementedError("Perception-Action Tooling module is not available")
        
        # Get the remote control module
        remote_control_module = self.kernel.get_module("remote_control")
        
        # Discover the target
        target_info = await remote_control_module.discover_target(
            host_id=host_id,
            hostname=hostname,
            ip_address=ip_address
        )
        
        return target_info
        
"""
            # Insert the new method
            content = content[:insertion_point] + new_method + content[insertion_point:]
    
    # Add the generate_remote_control_tool method if it doesn't exist
    if "async def generate_remote_control_tool" not in content:
        print("Adding missing generate_remote_control_tool method to ToolingSystem")
        
        # Find a good insertion point - after the discover_remote_target method
        insertion_point = content.find("    def get_tools_with_tag")
        if insertion_point > 0:
            # Define the new method
            new_method = """
    async def generate_remote_control_tool(self, host_id, hostname, description, 
                                        operation_type="general", parameters=None, returns=None, 
                                        ip_address=None):
        """Generate a tool for remote machine control."""
        if not PERCEPTION_ACTION_AVAILABLE:
            raise NotImplementedError("Perception-Action Tooling module is not available")
        
        # Get the remote control module
        remote_control_module = self.kernel.get_module("remote_control")
        
        # Generate the tool
        tool_id = await remote_control_module.generate_remote_control_tool(
            host_id=host_id,
            hostname=hostname,
            description=description,
            operation_type=operation_type,
            parameters=parameters,
            returns=returns,
            ip_address=ip_address
        )
        
        return tool_id
        
"""
            # Insert the new method
            content = content[:insertion_point] + new_method + content[insertion_point:]
    
    # Add the execute_remote_tool method if it doesn't exist
    if "async def execute_remote_tool" not in content:
        print("Adding missing execute_remote_tool method to ToolingSystem")
        
        # Find a good insertion point - after the generate_remote_control_tool method
        insertion_point = content.find("    def get_tools_with_tag")
        if insertion_point > 0:
            # Define the new method
            new_method = """
    async def execute_remote_tool(self, tool_id, args=None, record_video=False):
        """Execute a remote control tool."""
        if not PERCEPTION_ACTION_AVAILABLE:
            raise NotImplementedError("Perception-Action Tooling module is not available")
        
        # Get the remote control module
        remote_control_module = self.kernel.get_module("remote_control")
        
        # Execute the tool
        result = await remote_control_module.execute_remote_tool(
            tool_id=tool_id,
            args=args or {},
            record_video=record_video
        )
        
        return result
        
"""
            # Insert the new method
            content = content[:insertion_point] + new_method + content[insertion_point:]
    
    # Add the get_remote_audit_log method if it doesn't exist
    if "def get_remote_audit_log" not in content:
        print("Adding missing get_remote_audit_log method to ToolingSystem")
        
        # Find a good insertion point - after the execute_remote_tool method
        insertion_point = content.find("    def get_tools_with_tag")
        if insertion_point > 0:
            # Define the new method
            new_method = """
    def get_remote_audit_log(self, limit=5):
        """Get remote control audit logs."""
        if not PERCEPTION_ACTION_AVAILABLE:
            raise NotImplementedError("Perception-Action Tooling module is not available")
        
        # Get the remote control module
        remote_control_module = self.kernel.get_module("remote_control")
        
        # Get the audit logs
        logs = remote_control_module.get_remote_audit_log(limit=limit)
        
        return logs
        
"""
            # Insert the new method
            content = content[:insertion_point] + new_method + content[insertion_point:]
    
    # Write the updated content back to the file
    with open(tooling_system_path, 'w') as f:
        f.write(content)
    
    print(f"✅ Updated {tooling_system_path} with required methods")
    return True


if __name__ == "__main__":
    print("Fixing perception_action_demo.py script...")
    fix_demo_script()
    
    print("\nFixing tooling_system.py...")
    fix_tooling_system()
    
    print("\nFixes have been applied. You can now run the perception_action_demo.py script.")
