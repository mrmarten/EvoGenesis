"""
Tooling System WebSocket Handler for EvoGenesis Web UI

This module connects the Tooling System to the WebSocketManager for real-time updates.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional

class ToolingSystemWebSocketHandler:
    """
    Handles WebSocket events for the Tooling System.
    
    This class bridges the Tooling System with the WebSocketManager to provide
    real-time updates about tools to the web UI.
    """
    
    def __init__(self, tooling_system, ws_manager):
        """
        Initialize the Tooling System WebSocket Handler.
        
        Args:
            tooling_system: The Tooling System instance
            ws_manager: The WebSocket Manager instance
        """
        self.tooling_system = tooling_system
        self.ws_manager = ws_manager
        self.logger = logging.getLogger(__name__)
        
        # Track the last status to avoid sending duplicates
        self.last_status = None
        self.last_tools_hash = None
        
        # Register event handlers
        self._register_event_handlers()
        
        # Start the status update task
        asyncio.create_task(self._status_update_task())
    
    def _register_event_handlers(self):
        """Register handlers for tool-related WebSocket events."""
        # Register component-specific handlers
        self.ws_manager.register_component_handler("tools.action", self._handle_tool_action)
    
    async def _handle_tool_action(self, message: Dict[str, Any]):
        """
        Handle tool action messages from the WebSocket.
        
        Args:
            message: The action message
        """
        if not isinstance(message, dict):
            return
        
        action = message.get("action")
        tool_id = message.get("tool_id")
        
        if not action or not tool_id:
            return
        
        try:
            if action == "execute":
                # Get execution parameters
                params = message.get("params", {})
                
                # Execute the tool
                result = await self.tooling_system.execute_tool(tool_id, params)
                
                # Broadcast result
                await self._broadcast_tool_execution(tool_id, result)
            
            elif action == "enable":
                success = self.tooling_system.enable_tool(tool_id)
                if success:
                    await self._broadcast_tool_update(tool_id, "enabled")
            
            elif action == "disable":
                success = self.tooling_system.disable_tool(tool_id)
                if success:
                    await self._broadcast_tool_update(tool_id, "disabled")
            
            elif action == "delete":
                success = self.tooling_system.delete_tool(tool_id)
                if success:
                    await self._broadcast_tool_update(tool_id, "deleted")
        
        except Exception as e:
            self.logger.error(f"Error handling tool action {action} for tool {tool_id}: {str(e)}")
            await self.ws_manager.broadcast_to_topic("tools.errors", {
                "error": "action_failed",
                "action": action,
                "tool_id": tool_id,
                "message": str(e)
            })
    
    async def _broadcast_tool_execution(self, tool_id, result):
        """Broadcast tool execution result."""
        try:
            # Get the tool data
            tool = self.tooling_system.get_tool(tool_id)
            
            if not tool:
                return
            
            # Create tool execution data
            execution_data = {
                "tool_id": tool_id,
                "tool_name": tool.name,
                "result": result,
                "success": result.get("success", False) if isinstance(result, dict) else False,
                "timestamp": time.time(),
                "event": "tool_executed"
            }
            
            # Broadcast to tools.executions topic
            await self.ws_manager.broadcast_to_topic("tools.executions", execution_data)
            
            # Also broadcast to specific tool topic
            await self.ws_manager.broadcast_to_topic(f"tools.{tool_id}.executions", execution_data)
            
        except Exception as e:
            self.logger.error(f"Error broadcasting tool execution for {tool_id}: {str(e)}")
    
    async def _broadcast_tool_update(self, tool_id, event_type, extra_data=None):
        """Broadcast a tool update event."""
        try:
            # Get the tool data
            tool = self.tooling_system.get_tool(tool_id)
            
            if not tool and event_type != "deleted":
                return
            
            # Create tool data dict
            tool_data = {
                "id": tool_id,
                "event": f"tool_{event_type}",
                "timestamp": time.time()
            }
            
            if tool and event_type != "deleted":
                tool_data.update({
                    "name": tool.name,
                    "description": tool.description,
                    "status": tool.status,
                    "scope": tool.scope,
                    "sandbox_type": tool.sandbox_type,
                    "auto_generated": tool.auto_generated,
                    "execution_count": tool.execution_count,
                    "success_count": tool.success_count,
                    "error_count": tool.error_count,
                    "average_execution_time": tool.average_execution_time
                })
            
            # Add extra data if provided
            if extra_data:
                tool_data.update(extra_data)
            
            # Broadcast to tools topic
            await self.ws_manager.broadcast_to_topic("tools", tool_data)
            
            # Also broadcast to specific tool topic
            await self.ws_manager.broadcast_to_topic(f"tools.{tool_id}", tool_data)
            
            # Broadcast to tool status topic
            await self.ws_manager.broadcast_to_topic("tools.status", {
                "event": f"tool_{event_type}",
                "tool_id": tool_id,
                "status": tool.status if tool and hasattr(tool, "status") else "deleted",
                "timestamp": time.time()
            })
            
        except Exception as e:
            self.logger.error(f"Error broadcasting tool update for {tool_id}: {str(e)}")
    
    async def _status_update_task(self):
        """Background task to update tooling system status and broadcast it."""
        while True:
            try:
                # Get current status
                status = self.tooling_system.get_status()
                
                # Check if status has changed to avoid spamming
                if status != self.last_status:
                    # Broadcast to subscribers
                    await self.ws_manager.broadcast_to_topic("tools.system.status", status)
                    self.last_status = status
                
                # Update tool list if changed
                await self._update_tools_if_changed()
                
                # Wait before checking again
                await asyncio.sleep(5)  # Updates for tools
            except Exception as e:
                self.logger.error(f"Error in tooling system status update task: {str(e)}")
                await asyncio.sleep(10)  # Back off on error
    
    async def _update_tools_if_changed(self):
        """Update and broadcast tool list if it has changed."""
        try:
            # Get current tools
            tools = self.tooling_system.tool_registry
            
            # Generate a simple hash to check for changes
            tools_hash = hash(frozenset([
                (tool_id, 
                 getattr(tool, 'status', 'unknown'),
                 getattr(tool, 'execution_count', 0),
                 getattr(tool, 'success_count', 0),
                 getattr(tool, 'error_count', 0))
                for tool_id, tool in tools.items()
            ]))
            
            if tools_hash != self.last_tools_hash:
                # Format tool data for the frontend
                tool_list = []
                for tool_id, tool in tools.items():
                    tool_list.append({
                        "id": tool_id,
                        "name": tool.name,
                        "description": tool.description,
                        "status": tool.status,
                        "scope": tool.scope,
                        "sandbox_type": tool.sandbox_type,
                        "auto_generated": tool.auto_generated,
                        "execution_count": tool.execution_count,
                        "success_count": tool.success_count,
                        "error_count": tool.error_count,
                        "average_execution_time": tool.average_execution_time
                    })
                
                # Broadcast tool list
                await self.ws_manager.broadcast_to_topic("tools.list", {
                    "event": "tools_updated",
                    "tools": tool_list,
                    "timestamp": time.time()
                })
                
                # Update the hash
                self.last_tools_hash = tools_hash
        except Exception as e:
            self.logger.error(f"Error updating tools: {str(e)}")

def connect_tooling_system(tooling_system, ws_manager):
    """
    Connect the Tooling System to the WebSocket Manager.
    
    Args:
        tooling_system: The Tooling System instance
        ws_manager: The WebSocket Manager instance
        
    Returns:
        The handler instance
    """
    handler = ToolingSystemWebSocketHandler(tooling_system, ws_manager)
    return handler
