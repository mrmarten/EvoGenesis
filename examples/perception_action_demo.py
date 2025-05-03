"""
EvoGenesis Perception-Action Tooling Integration Example

This script demonstrates how to use the new Perception-Action Tooling layer
to remotely control machines with different adapters.
"""

import asyncio
import logging
import json
import os
import sys

# Add the parent directory to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import the patch script to fix the tooling system
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from fix_perception_action import patch_tooling_system
patch_tooling_system()

from evogenesis_core.kernel import EvoGenesisKernel as Kernel
from evogenesis_core.modules.perception_action_tooling import RemoteAdapterType


async def demonstrate_remote_control():
    """Demonstrate the Perception-Action Tooling layer capabilities."""
    # Initialize the EvoGenesis kernel
    kernel = Kernel()
    
    # Use the synchronous start() method instead of async initialize()
    kernel.start()
    
    # Get the tooling system - use proper method from kernel
    tooling_system = kernel.tooling_system
    
    # Example 1: Discover capabilities of a remote Windows machine
    print("\n=== Example 1: Discover Remote Machine Capabilities ===")
    try:
        target_info = await tooling_system.discover_remote_target(
            host_id="win-dev-01",
            hostname="win-dev-01.example.com",
            ip_address="10.0.0.101"
        )
        print(f"Discovered target: {target_info.hostname}")
        print(f"Available adapters: {target_info.available_adapters}")
        print(f"OS type: {target_info.os_type}")
    except Exception as e:
        print(f"Error discovering target: {str(e)}")
    
    # Example 2: Generate a tool to restart a service on a remote machine
    print("\n=== Example 2: Generate Remote Control Tool ===")
    try:
        tool_id = await tooling_system.generate_remote_control_tool(
            host_id="win-dev-01",
            hostname="win-dev-01.example.com",
            description="restart the Print Spooler service",
            operation_type="script",
            parameters={
                "service_name": {
                    "type": "string",
                    "description": "Name of the service to restart",
                    "default": "spooler"
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds",
                    "default": 30
                }
            },
            returns={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "status": {"type": "string"},
                    "message": {"type": "string"},
                    "audit": {"type": "object"}
                }
            }
        )
        
        print(f"Generated tool ID: {tool_id}")
    except Exception as e:
        print(f"Error generating tool: {str(e)}")
    
    # Example 3: Execute a remote control tool with audit logging
    print("\n=== Example 3: Execute Remote Control Tool ===")
    try:
        # Assuming we have a tool with ID "remote-restart-service"
        tool_id = "remote-restart-service"  # Use the tool ID from Example 2 in a real scenario
        
        result = await tooling_system.execute_remote_tool(
            tool_id=tool_id,
            args={
                "service_name": "spooler",
                "timeout": 60
            },
            record_video=True
        )
        
        print(f"Execution success: {result['success']}")
        print(f"Output: {json.dumps(result['output'], indent=2)}")
        print(f"Audit ID: {result['audit_id']}")
    except Exception as e:
        print(f"Error executing tool: {str(e)}")
    
    # Example 4: Retrieve audit logs for compliance
    print("\n=== Example 4: Retrieve Audit Logs ===")
    try:
        audit_logs = await tooling_system.get_remote_audit_log(limit=5)
        
        print(f"Retrieved {len(audit_logs)} audit logs")
        for log in audit_logs:
            print(f"Operation: {log['tool_name']} on {log['hostname']}")
            print(f"Timestamp: {log['timestamp_start']}")
            print(f"Status: {'Success' if log['success'] else 'Failed'}")
            print("---")
    except Exception as e:
        print(f"Error retrieving audit logs: {str(e)}")
      # Clean up - use the synchronous stop() method
    kernel.stop()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(demonstrate_remote_control())
