"""
Swarm Module WebSocket Handler for EvoGenesis Web UI

This module connects the Swarm Module to the WebSocketManager for real-time updates.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional

class SwarmWebSocketHandler:
    """
    Handles WebSocket events for the Swarm Module.
    
    This class bridges the Swarm Module with the WebSocketManager to provide
    real-time updates about swarm operations to the web UI.
    """
    
    def __init__(self, swarm_module, ws_manager):
        """
        Initialize the Swarm WebSocket Handler.
        
        Args:
            swarm_module: The Swarm Module instance
            ws_manager: The WebSocket Manager instance
        """
        self.swarm_module = swarm_module
        self.ws_manager = ws_manager
        self.logger = logging.getLogger(__name__)
        
        # Track the last status to avoid sending duplicates
        self.last_status = None
        self.last_swarms_hash = None
        self.last_nodes_hash = None
        
        # Register event handlers
        self._register_event_handlers()
        
        # Start the status update task
        asyncio.create_task(self._status_update_task())
    
    def _register_event_handlers(self):
        """Register handlers for swarm-related WebSocket events."""
        # Register component-specific handlers
        self.ws_manager.register_component_handler("swarm.action", self._handle_swarm_action)
        self.ws_manager.register_component_handler("swarm.node.action", self._handle_node_action)
    
    async def _handle_swarm_action(self, message: Dict[str, Any]):
        """
        Handle swarm action messages from the WebSocket.
        
        Args:
            message: The action message
        """
        if not isinstance(message, dict):
            return
        
        action = message.get("action")
        swarm_id = message.get("swarm_id")
        
        if not action:
            return
        
        try:
            if action == "create_swarm":
                swarm_data = message.get("swarm_data", {})
                
                if swarm_data:
                    swarm_id = await self.swarm_module.create_swarm(swarm_data)
                    if swarm_id:
                        await self._broadcast_swarm_update(swarm_id, "created")
            
            elif action == "start_swarm" and swarm_id:
                success = await self.swarm_module.start_swarm(swarm_id)
                if success:
                    await self._broadcast_swarm_update(swarm_id, "started")
            
            elif action == "stop_swarm" and swarm_id:
                success = await self.swarm_module.stop_swarm(swarm_id)
                if success:
                    await self._broadcast_swarm_update(swarm_id, "stopped")
            
            elif action == "delete_swarm" and swarm_id:
                success = await self.swarm_module.delete_swarm(swarm_id)
                if success:
                    await self._broadcast_swarm_update(swarm_id, "deleted")
        
        except Exception as e:
            self.logger.error(f"Error handling swarm action {action}: {str(e)}")
            await self.ws_manager.broadcast_to_topic("swarm.errors", {
                "error": "action_failed",
                "action": action,
                "swarm_id": swarm_id if swarm_id else None,
                "message": str(e)
            })
    
    async def _handle_node_action(self, message: Dict[str, Any]):
        """
        Handle swarm node action messages from the WebSocket.
        
        Args:
            message: The action message
        """
        if not isinstance(message, dict):
            return
        
        action = message.get("action")
        node_id = message.get("node_id")
        
        if not action or not node_id:
            return
        
        try:
            if action == "add_node":
                swarm_id = message.get("swarm_id")
                node_data = message.get("node_data", {})
                
                if swarm_id and node_data:
                    success = await self.swarm_module.add_node(swarm_id, node_data)
                    if success:
                        await self._broadcast_node_update(node_id, "added", {"swarm_id": swarm_id})
            
            elif action == "remove_node":
                swarm_id = message.get("swarm_id")
                
                if swarm_id:
                    success = await self.swarm_module.remove_node(swarm_id, node_id)
                    if success:
                        await self._broadcast_node_update(node_id, "removed", {"swarm_id": swarm_id})
            
            elif action == "update_node":
                node_data = message.get("node_data", {})
                
                if node_data:
                    success = await self.swarm_module.update_node(node_id, node_data)
                    if success:
                        await self._broadcast_node_update(node_id, "updated")
        
        except Exception as e:
            self.logger.error(f"Error handling node action {action} for node {node_id}: {str(e)}")
            await self.ws_manager.broadcast_to_topic("swarm.node.errors", {
                "error": "action_failed",
                "action": action,
                "node_id": node_id,
                "message": str(e)
            })
    
    async def _broadcast_swarm_update(self, swarm_id, event_type, extra_data=None):
        """Broadcast a swarm update event."""
        try:
            # Get the swarm data
            swarm = self.swarm_module.get_swarm(swarm_id)
            
            if not swarm and event_type not in ["deleted"]:
                return
            
            # Create swarm data dict
            swarm_data = {
                "id": swarm_id,
                "event": f"swarm_{event_type}",
                "timestamp": time.time()
            }
            
            if swarm and event_type not in ["deleted"]:
                swarm_data.update({
                    "name": swarm.name,
                    "description": swarm.description,
                    "status": swarm.status,
                    "node_count": len(swarm.nodes) if hasattr(swarm, "nodes") else 0,
                    "created_at": swarm.created_at,
                    "updated_at": swarm.updated_at
                })
            
            # Add extra data if provided
            if extra_data:
                swarm_data.update(extra_data)
            
            # Broadcast to swarms topic
            await self.ws_manager.broadcast_to_topic("swarm", swarm_data)
            
            # Also broadcast to specific swarm topic
            await self.ws_manager.broadcast_to_topic(f"swarm.{swarm_id}", swarm_data)
            
            # Broadcast to swarm status topic
            await self.ws_manager.broadcast_to_topic("swarm.status", {
                "event": f"swarm_{event_type}",
                "swarm_id": swarm_id,
                "status": swarm.status if swarm and hasattr(swarm, "status") else event_type,
                "timestamp": time.time()
            })
            
        except Exception as e:
            self.logger.error(f"Error broadcasting swarm update for {swarm_id}: {str(e)}")
    
    async def _broadcast_node_update(self, node_id, event_type, extra_data=None):
        """Broadcast a node update event."""
        try:
            # Get the node data
            node = self.swarm_module.get_node(node_id)
            
            if not node and event_type not in ["removed", "deleted"]:
                return
            
            # Create node data dict
            node_data = {
                "id": node_id,
                "event": f"node_{event_type}",
                "timestamp": time.time()
            }
            
            if node and event_type not in ["removed", "deleted"]:
                node_data.update({
                    "name": node.name,
                    "type": node.type,
                    "ip_address": node.ip_address,
                    "port": node.port,
                    "status": node.status,
                    "capabilities": node.capabilities,
                    "swarm_id": node.swarm_id,
                    "created_at": node.created_at,
                    "updated_at": node.updated_at
                })
            
            # Add extra data if provided
            if extra_data:
                node_data.update(extra_data)
            
            # Broadcast to nodes topic
            await self.ws_manager.broadcast_to_topic("swarm.nodes", node_data)
            
            # Also broadcast to specific node topic
            await self.ws_manager.broadcast_to_topic(f"swarm.nodes.{node_id}", node_data)
            
            # If swarm_id is provided, broadcast to swarm-specific node topic
            if extra_data and "swarm_id" in extra_data:
                await self.ws_manager.broadcast_to_topic(f"swarm.{extra_data['swarm_id']}.nodes", node_data)
            
        except Exception as e:
            self.logger.error(f"Error broadcasting node update for {node_id}: {str(e)}")
    
    async def _status_update_task(self):
        """Background task to update swarm module status and broadcast it."""
        while True:
            try:
                # Get current status
                status = self.swarm_module.get_status()
                
                # Check if status has changed to avoid spamming
                if status != self.last_status:
                    # Broadcast to subscribers
                    await self.ws_manager.broadcast_to_topic("swarm.module.status", status)
                    self.last_status = status
                
                # Update swarms if changed
                await self._update_swarms_if_changed()
                
                # Update nodes if changed
                await self._update_nodes_if_changed()
                
                # Wait before checking again
                await asyncio.sleep(5)  # Updates for swarms and nodes
            except Exception as e:
                self.logger.error(f"Error in swarm module status update task: {str(e)}")
                await asyncio.sleep(10)  # Back off on error
    
    async def _update_swarms_if_changed(self):
        """Update and broadcast swarm list if it has changed."""
        try:
            # Get current swarms
            swarms = self.swarm_module.swarms
            
            # Generate a simple hash to check for changes
            swarms_hash = hash(frozenset([
                (swarm_id, 
                 getattr(swarm, 'status', 'unknown'),
                 len(swarm.nodes) if hasattr(swarm, "nodes") else 0,
                 getattr(swarm, 'updated_at', 0))
                for swarm_id, swarm in swarms.items()
            ]))
            
            if swarms_hash != self.last_swarms_hash:
                # Format swarm data for the frontend
                swarm_list = []
                for swarm_id, swarm in swarms.items():
                    swarm_list.append({
                        "id": swarm_id,
                        "name": swarm.name,
                        "description": swarm.description,
                        "status": swarm.status,
                        "node_count": len(swarm.nodes) if hasattr(swarm, "nodes") else 0,
                        "created_at": swarm.created_at,
                        "updated_at": swarm.updated_at
                    })
                
                # Broadcast swarm list
                await self.ws_manager.broadcast_to_topic("swarm.list", {
                    "event": "swarms_updated",
                    "swarms": swarm_list,
                    "timestamp": time.time()
                })
                
                # Update the hash
                self.last_swarms_hash = swarms_hash
        except Exception as e:
            self.logger.error(f"Error updating swarms: {str(e)}")
    
    async def _update_nodes_if_changed(self):
        """Update and broadcast node list if it has changed."""
        try:
            # Get current nodes
            nodes = self.swarm_module.nodes
            
            # Generate a simple hash to check for changes
            nodes_hash = hash(frozenset([
                (node_id, 
                 getattr(node, 'status', 'unknown'),
                 getattr(node, 'updated_at', 0))
                for node_id, node in nodes.items()
            ]))
            
            if nodes_hash != self.last_nodes_hash:
                # Format node data for the frontend
                node_list = []
                for node_id, node in nodes.items():
                    node_list.append({
                        "id": node_id,
                        "name": node.name,
                        "type": node.type,
                        "ip_address": node.ip_address,
                        "port": node.port,
                        "status": node.status,
                        "capabilities": node.capabilities,
                        "swarm_id": node.swarm_id,
                        "created_at": node.created_at,
                        "updated_at": node.updated_at
                    })
                
                # Broadcast node list
                await self.ws_manager.broadcast_to_topic("swarm.nodes.list", {
                    "event": "nodes_updated",
                    "nodes": node_list,
                    "timestamp": time.time()
                })
                
                # Update the hash
                self.last_nodes_hash = nodes_hash
        except Exception as e:
            self.logger.error(f"Error updating nodes: {str(e)}")

def connect_swarm_module(swarm_module, ws_manager):
    """
    Connect the Swarm Module to the WebSocket Manager.
    
    Args:
        swarm_module: The Swarm Module instance
        ws_manager: The WebSocket Manager instance
        
    Returns:
        The handler instance
    """
    handler = SwarmWebSocketHandler(swarm_module, ws_manager)
    return handler
