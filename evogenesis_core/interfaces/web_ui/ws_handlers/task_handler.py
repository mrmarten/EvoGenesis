"""
Task Planner WebSocket Handler for EvoGenesis Web UI

This module connects the Task Planner to the WebSocketManager for real-time updates.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional

class TaskPlannerWebSocketHandler:
    """
    Handles WebSocket events for the Task Planner.
    
    This class bridges the Task Planner with the WebSocketManager to provide
    real-time updates about tasks to the web UI.
    """
    
    def __init__(self, task_planner, ws_manager):
        """
        Initialize the Task Planner WebSocket Handler.
        
        Args:
            task_planner: The Task Planner instance
            ws_manager: The WebSocket Manager instance
        """
        self.task_planner = task_planner
        self.ws_manager = ws_manager
        self.logger = logging.getLogger(__name__)
        
        # Track the last status to avoid sending duplicates
        self.last_status = None
        self.last_tasks_hash = None
        
        # Register event handlers
        self._register_event_handlers()
        
        # Start the status update task
        asyncio.create_task(self._status_update_task())
    
    def _register_event_handlers(self):
        """Register handlers for task-related WebSocket events."""
        # Register component-specific handlers
        self.ws_manager.register_component_handler("tasks.action", self._handle_task_action)
    
    async def _handle_task_action(self, message: Dict[str, Any]):
        """
        Handle task action messages from the WebSocket.
        
        Args:
            message: The action message
        """
        if not isinstance(message, dict):
            return
        
        action = message.get("action")
        task_id = message.get("task_id")
        
        if not action or not task_id:
            return
        
        try:
            if action == "start":
                success = self.task_planner.start_task(task_id)
                if success:
                    await self._broadcast_task_update(task_id, "started")
            
            elif action == "pause":
                success = self.task_planner.pause_task(task_id)
                if success:
                    await self._broadcast_task_update(task_id, "paused")
            
            elif action == "resume":
                success = self.task_planner.resume_task(task_id)
                if success:
                    await self._broadcast_task_update(task_id, "resumed")
            
            elif action == "cancel":
                success = self.task_planner.cancel_task(task_id)
                if success:
                    await self._broadcast_task_update(task_id, "cancelled")
            
            elif action == "complete":
                success = self.task_planner.complete_task(task_id)
                if success:
                    await self._broadcast_task_update(task_id, "completed")
            
            elif action == "reassign":
                agent_id = message.get("agent_id")
                team_id = message.get("team_id")
                
                if agent_id:
                    success = self.task_planner.assign_task_to_agent(task_id, agent_id)
                    if success:
                        await self._broadcast_task_update(task_id, "reassigned", {"agent_id": agent_id})
                
                elif team_id:
                    success = self.task_planner.assign_task_to_team(task_id, team_id)
                    if success:
                        await self._broadcast_task_update(task_id, "reassigned", {"team_id": team_id})
        
        except Exception as e:
            self.logger.error(f"Error handling task action {action} for task {task_id}: {str(e)}")
            await self.ws_manager.broadcast_to_topic("tasks.errors", {
                "error": "action_failed",
                "action": action,
                "task_id": task_id,
                "message": str(e)
            })
    
    async def _broadcast_task_update(self, task_id, event_type, extra_data=None):
        """Broadcast a task update event."""
        try:
            # Get the task data
            task = self.task_planner.get_task(task_id)
            
            if not task:
                return
            
            # Create task data dict
            task_data = {
                "id": task_id,
                "name": task.name,
                "description": task.description,
                "status": task.status,
                "progress": task.progress,
                "assigned_agent_id": task.assigned_agent_id,
                "assigned_team_id": task.assigned_team_id,
                "parent_id": task.parent_id,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at,
                "event": f"task_{event_type}"
            }
            
            # Add extra data if provided
            if extra_data:
                task_data.update(extra_data)
            
            # Broadcast to tasks topic
            await self.ws_manager.broadcast_to_topic("tasks", task_data)
            
            # Also broadcast to specific task topic
            await self.ws_manager.broadcast_to_topic(f"tasks.{task_id}", task_data)
            
            # Broadcast to task status topic
            await self.ws_manager.broadcast_to_topic("tasks.status", {
                "event": f"task_{event_type}",
                "task_id": task_id,
                "status": task.status,
                "timestamp": time.time()
            })
            
        except Exception as e:
            self.logger.error(f"Error broadcasting task update for {task_id}: {str(e)}")
    
    async def _status_update_task(self):
        """Background task to update task planner status and broadcast it."""
        while True:
            try:
                # Get current status
                status = self.task_planner.get_status()
                
                # Check if status has changed to avoid spamming
                if status != self.last_status:
                    # Broadcast to subscribers
                    await self.ws_manager.broadcast_to_topic("tasks.planner.status", status)
                    self.last_status = status
                
                # Update task list if changed
                await self._update_tasks_if_changed()
                
                # Wait before checking again
                await asyncio.sleep(2)  # Frequent updates for tasks
            except Exception as e:
                self.logger.error(f"Error in task planner status update task: {str(e)}")
                await asyncio.sleep(10)  # Back off on error
    
    async def _update_tasks_if_changed(self):
        """Update and broadcast task list if it has changed."""
        try:
            # Get current tasks
            tasks = self.task_planner.tasks
            
            # Generate a simple hash to check for changes
            tasks_hash = hash(frozenset([
                (task_id, 
                 getattr(task, 'status', 'unknown'),
                 getattr(task, 'progress', 0),
                 getattr(task, 'updated_at', 0) if hasattr(task, 'updated_at') else 0)
                for task_id, task in tasks.items()
            ]))
            
            if tasks_hash != self.last_tasks_hash:
                # Format task data for the frontend
                task_list = []
                for task_id, task in tasks.items():
                    task_list.append({
                        "id": task_id,
                        "name": task.name,
                        "description": task.description,
                        "status": task.status,
                        "progress": task.progress,
                        "assigned_agent_id": task.assigned_agent_id,
                        "assigned_team_id": task.assigned_team_id,
                        "parent_id": task.parent_id,
                        "created_at": task.created_at,
                        "started_at": task.started_at,
                        "completed_at": task.completed_at
                    })
                
                # Broadcast task list
                await self.ws_manager.broadcast_to_topic("tasks.list", {
                    "event": "tasks_updated",
                    "tasks": task_list,
                    "timestamp": time.time()
                })
                
                # Update the hash
                self.last_tasks_hash = tasks_hash
        except Exception as e:
            self.logger.error(f"Error updating tasks: {str(e)}")

def connect_task_planner(task_planner, ws_manager):
    """
    Connect the Task Planner to the WebSocket Manager.
    
    Args:
        task_planner: The Task Planner instance
        ws_manager: The WebSocket Manager instance
        
    Returns:
        The handler instance
    """
    handler = TaskPlannerWebSocketHandler(task_planner, ws_manager)
    return handler
