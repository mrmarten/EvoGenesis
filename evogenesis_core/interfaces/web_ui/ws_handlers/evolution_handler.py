"""
Self Evolution Engine WebSocket Handler for EvoGenesis Web UI

This module connects the Self Evolution Engine to the WebSocketManager for real-time updates.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional

class SelfEvolutionWebSocketHandler:
    """
    Handles WebSocket events for the Self Evolution Engine.
    
    This class bridges the Self Evolution Engine with the WebSocketManager to provide
    real-time updates about system evolution to the web UI.
    """
    
    def __init__(self, evolution_engine, ws_manager):
        """
        Initialize the Self Evolution Engine WebSocket Handler.
        
        Args:
            evolution_engine: The Self Evolution Engine instance
            ws_manager: The WebSocket Manager instance
        """
        self.evolution_engine = evolution_engine
        self.ws_manager = ws_manager
        self.logger = logging.getLogger(__name__)
        
        # Track the last status to avoid sending duplicates
        self.last_status = None
        self.last_improvements_hash = None
        
        # Register event handlers
        self._register_event_handlers()
        
        # Start the status update task
        asyncio.create_task(self._status_update_task())
    
    def _register_event_handlers(self):
        """Register handlers for evolution-related WebSocket events."""
        # Register component-specific handlers
        self.ws_manager.register_component_handler("evolution.action", self._handle_evolution_action)
    
    async def _handle_evolution_action(self, message: Dict[str, Any]):
        """
        Handle evolution action messages from the WebSocket.
        
        Args:
            message: The action message
        """
        if not isinstance(message, dict):
            return
        
        action = message.get("action")
        
        if not action:
            return
        
        try:
            if action == "start_improvement":
                component = message.get("component")
                improvement_type = message.get("type")
                
                if component and improvement_type:
                    improvement_id = await self.evolution_engine.start_improvement(component, improvement_type)
                    if improvement_id:
                        await self._broadcast_improvement_update(improvement_id, "started", {
                            "component": component,
                            "type": improvement_type
                        })
            
            elif action == "cancel_improvement":
                improvement_id = message.get("improvement_id")
                
                if improvement_id:
                    success = await self.evolution_engine.cancel_improvement(improvement_id)
                    if success:
                        await self._broadcast_improvement_update(improvement_id, "cancelled")
            
            elif action == "apply_improvement":
                improvement_id = message.get("improvement_id")
                
                if improvement_id:
                    success = await self.evolution_engine.apply_improvement(improvement_id)
                    if success:
                        await self._broadcast_improvement_update(improvement_id, "applied")
            
            elif action == "reject_improvement":
                improvement_id = message.get("improvement_id")
                
                if improvement_id:
                    success = await self.evolution_engine.reject_improvement(improvement_id)
                    if success:
                        await self._broadcast_improvement_update(improvement_id, "rejected")
        
        except Exception as e:
            self.logger.error(f"Error handling evolution action {action}: {str(e)}")
            await self.ws_manager.broadcast_to_topic("evolution.errors", {
                "error": "action_failed",
                "action": action,
                "message": str(e)
            })
    
    async def _broadcast_improvement_update(self, improvement_id, event_type, extra_data=None):
        """Broadcast an improvement update event."""
        try:
            # Get the improvement data
            improvement = self.evolution_engine.get_improvement(improvement_id)
            
            if not improvement and event_type not in ["deleted", "rejected", "cancelled"]:
                return
            
            # Create improvement data dict
            improvement_data = {
                "id": improvement_id,
                "event": f"improvement_{event_type}",
                "timestamp": time.time()
            }
            
            if improvement and event_type not in ["deleted", "rejected", "cancelled"]:
                improvement_data.update({
                    "component": improvement.component,
                    "type": improvement.type,
                    "status": improvement.status,
                    "progress": improvement.progress,
                    "description": improvement.description,
                    "created_at": improvement.created_at,
                    "updated_at": improvement.updated_at
                })
            
            # Add extra data if provided
            if extra_data:
                improvement_data.update(extra_data)
            
            # Broadcast to improvements topic
            await self.ws_manager.broadcast_to_topic("evolution.improvements", improvement_data)
            
            # Also broadcast to specific improvement topic
            await self.ws_manager.broadcast_to_topic(f"evolution.improvements.{improvement_id}", improvement_data)
            
            # Broadcast to improvement status topic
            await self.ws_manager.broadcast_to_topic("evolution.improvements.status", {
                "event": f"improvement_{event_type}",
                "improvement_id": improvement_id,
                "status": improvement.status if improvement and hasattr(improvement, "status") else event_type,
                "timestamp": time.time()
            })
            
        except Exception as e:
            self.logger.error(f"Error broadcasting improvement update for {improvement_id}: {str(e)}")
    
    async def _status_update_task(self):
        """Background task to update self-evolution engine status and broadcast it."""
        while True:
            try:
                # Get current status
                status = self.evolution_engine.get_status()
                
                # Check if status has changed to avoid spamming
                if status != self.last_status:
                    # Broadcast to subscribers
                    await self.ws_manager.broadcast_to_topic("evolution.status", status)
                    self.last_status = status
                
                # Update improvements if changed
                await self._update_improvements_if_changed()
                
                # Wait before checking again
                await asyncio.sleep(5)  # Updates for improvements
            except Exception as e:
                self.logger.error(f"Error in self-evolution status update task: {str(e)}")
                await asyncio.sleep(10)  # Back off on error
    
    async def _update_improvements_if_changed(self):
        """Update and broadcast improvements list if it has changed."""
        try:
            # Get current improvements
            improvements = self.evolution_engine.improvements
            
            # Generate a simple hash to check for changes
            improvements_hash = hash(frozenset([
                (imp_id, 
                 getattr(imp, 'status', 'unknown'),
                 getattr(imp, 'progress', 0),
                 getattr(imp, 'updated_at', 0))
                for imp_id, imp in improvements.items()
            ]))
            
            if improvements_hash != self.last_improvements_hash:
                # Format improvement data for the frontend
                improvement_list = []
                for imp_id, imp in improvements.items():
                    improvement_list.append({
                        "id": imp_id,
                        "component": imp.component,
                        "type": imp.type,
                        "status": imp.status,
                        "progress": imp.progress,
                        "description": imp.description,
                        "created_at": imp.created_at,
                        "updated_at": imp.updated_at
                    })
                
                # Broadcast improvement list
                await self.ws_manager.broadcast_to_topic("evolution.improvements.list", {
                    "event": "improvements_updated",
                    "improvements": improvement_list,
                    "timestamp": time.time()
                })
                
                # Update the hash
                self.last_improvements_hash = improvements_hash
        except Exception as e:
            self.logger.error(f"Error updating improvements: {str(e)}")

def connect_evolution_engine(evolution_engine, ws_manager):
    """
    Connect the Self Evolution Engine to the WebSocket Manager.
    
    Args:
        evolution_engine: The Self Evolution Engine instance
        ws_manager: The WebSocket Manager instance
        
    Returns:
        The handler instance
    """
    handler = SelfEvolutionWebSocketHandler(evolution_engine, ws_manager)
    return handler
