"""
Mission Scheduler WebSocket Handler for EvoGenesis Web UI

This module connects the Mission Scheduler to the WebSocketManager for real-time updates.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional

class MissionSchedulerWebSocketHandler:
    """
    Handles WebSocket events for the Mission Scheduler.
    
    This class bridges the Mission Scheduler with the WebSocketManager to provide
    real-time updates about mission scheduling to the web UI.
    """
    
    def __init__(self, mission_scheduler, ws_manager):
        """
        Initialize the Mission Scheduler WebSocket Handler.
        
        Args:
            mission_scheduler: The Mission Scheduler instance
            ws_manager: The WebSocket Manager instance
        """
        self.mission_scheduler = mission_scheduler
        self.ws_manager = ws_manager
        self.logger = logging.getLogger(__name__)
        
        # Track the last status to avoid sending duplicates
        self.last_status = None
        self.last_missions_hash = None
        
        # Register event handlers
        self._register_event_handlers()
        
        # Start the status update task
        asyncio.create_task(self._status_update_task())
    
    def _register_event_handlers(self):
        """Register handlers for mission-related WebSocket events."""
        # Register component-specific handlers
        self.ws_manager.register_component_handler("missions.action", self._handle_mission_action)
    
    async def _handle_mission_action(self, message: Dict[str, Any]):
        """
        Handle mission action messages from the WebSocket.
        
        Args:
            message: The action message
        """
        if not isinstance(message, dict):
            return
        
        action = message.get("action")
        mission_id = message.get("mission_id")
        
        if not action:
            return
        
        try:
            if action == "create_mission":
                mission_data = message.get("mission_data", {})
                
                if mission_data:
                    mission_id = await self.mission_scheduler.create_mission(mission_data)
                    if mission_id:
                        await self._broadcast_mission_update(mission_id, "created")
            
            elif action == "start_mission" and mission_id:
                success = await self.mission_scheduler.start_mission(mission_id)
                if success:
                    await self._broadcast_mission_update(mission_id, "started")
            
            elif action == "pause_mission" and mission_id:
                success = await self.mission_scheduler.pause_mission(mission_id)
                if success:
                    await self._broadcast_mission_update(mission_id, "paused")
            
            elif action == "resume_mission" and mission_id:
                success = await self.mission_scheduler.resume_mission(mission_id)
                if success:
                    await self._broadcast_mission_update(mission_id, "resumed")
            
            elif action == "complete_mission" and mission_id:
                success = await self.mission_scheduler.complete_mission(mission_id)
                if success:
                    await self._broadcast_mission_update(mission_id, "completed")
            
            elif action == "cancel_mission" and mission_id:
                success = await self.mission_scheduler.cancel_mission(mission_id)
                if success:
                    await self._broadcast_mission_update(mission_id, "cancelled")
            
            elif action == "delete_mission" and mission_id:
                success = await self.mission_scheduler.delete_mission(mission_id)
                if success:
                    await self._broadcast_mission_update(mission_id, "deleted")
            
            elif action == "reschedule_mission" and mission_id:
                schedule_data = message.get("schedule_data", {})
                
                if schedule_data:
                    success = await self.mission_scheduler.reschedule_mission(mission_id, schedule_data)
                    if success:
                        await self._broadcast_mission_update(mission_id, "rescheduled")
        
        except Exception as e:
            self.logger.error(f"Error handling mission action {action}: {str(e)}")
            await self.ws_manager.broadcast_to_topic("missions.errors", {
                "error": "action_failed",
                "action": action,
                "mission_id": mission_id if mission_id else None,
                "message": str(e)
            })
    
    async def _broadcast_mission_update(self, mission_id, event_type, extra_data=None):
        """Broadcast a mission update event."""
        try:
            # Get the mission data
            mission = self.mission_scheduler.get_mission(mission_id)
            
            if not mission and event_type not in ["deleted", "cancelled"]:
                return
            
            # Create mission data dict
            mission_data = {
                "id": mission_id,
                "event": f"mission_{event_type}",
                "timestamp": time.time()
            }
            
            if mission and event_type not in ["deleted"]:
                mission_data.update({
                    "name": mission.name,
                    "description": mission.description,
                    "objective": mission.objective,
                    "status": mission.status,
                    "priority": mission.priority,
                    "progress": mission.progress,
                    "scheduled_start": mission.scheduled_start,
                    "scheduled_end": mission.scheduled_end,
                    "actual_start": mission.actual_start,
                    "actual_end": mission.actual_end,
                    "created_at": mission.created_at,
                    "updated_at": mission.updated_at
                })
            
            # Add extra data if provided
            if extra_data:
                mission_data.update(extra_data)
            
            # Broadcast to missions topic
            await self.ws_manager.broadcast_to_topic("missions", mission_data)
            
            # Also broadcast to specific mission topic
            await self.ws_manager.broadcast_to_topic(f"missions.{mission_id}", mission_data)
            
            # Broadcast to mission status topic
            await self.ws_manager.broadcast_to_topic("missions.status", {
                "event": f"mission_{event_type}",
                "mission_id": mission_id,
                "status": mission.status if mission and hasattr(mission, "status") else event_type,
                "timestamp": time.time()
            })
            
        except Exception as e:
            self.logger.error(f"Error broadcasting mission update for {mission_id}: {str(e)}")
    
    async def _status_update_task(self):
        """Background task to update mission scheduler status and broadcast it."""
        while True:
            try:
                # Get current status
                status = self.mission_scheduler.get_status()
                
                # Check if status has changed to avoid spamming
                if status != self.last_status:
                    # Broadcast to subscribers
                    await self.ws_manager.broadcast_to_topic("missions.scheduler.status", status)
                    self.last_status = status
                
                # Update missions if changed
                await self._update_missions_if_changed()
                
                # Wait before checking again
                await asyncio.sleep(3)  # Updates for missions
            except Exception as e:
                self.logger.error(f"Error in mission scheduler status update task: {str(e)}")
                await asyncio.sleep(10)  # Back off on error
    
    async def _update_missions_if_changed(self):
        """Update and broadcast mission list if it has changed."""
        try:
            # Get current missions
            missions = self.mission_scheduler.missions
            
            # Generate a simple hash to check for changes
            missions_hash = hash(frozenset([
                (mission_id, 
                 getattr(mission, 'status', 'unknown'),
                 getattr(mission, 'progress', 0),
                 getattr(mission, 'updated_at', 0))
                for mission_id, mission in missions.items()
            ]))
            
            if missions_hash != self.last_missions_hash:
                # Format mission data for the frontend
                mission_list = []
                for mission_id, mission in missions.items():
                    mission_list.append({
                        "id": mission_id,
                        "name": mission.name,
                        "description": mission.description,
                        "objective": mission.objective,
                        "status": mission.status,
                        "priority": mission.priority,
                        "progress": mission.progress,
                        "scheduled_start": mission.scheduled_start,
                        "scheduled_end": mission.scheduled_end,
                        "actual_start": mission.actual_start,
                        "actual_end": mission.actual_end,
                        "created_at": mission.created_at,
                        "updated_at": mission.updated_at
                    })
                
                # Broadcast mission list
                await self.ws_manager.broadcast_to_topic("missions.list", {
                    "event": "missions_updated",
                    "missions": mission_list,
                    "timestamp": time.time()
                })
                
                # Update the hash
                self.last_missions_hash = missions_hash
        except Exception as e:
            self.logger.error(f"Error updating missions: {str(e)}")

def connect_mission_scheduler(mission_scheduler, ws_manager):
    """
    Connect the Mission Scheduler to the WebSocket Manager.
    
    Args:
        mission_scheduler: The Mission Scheduler instance
        ws_manager: The WebSocket Manager instance
        
    Returns:
        The handler instance
    """
    handler = MissionSchedulerWebSocketHandler(mission_scheduler, ws_manager)
    return handler
