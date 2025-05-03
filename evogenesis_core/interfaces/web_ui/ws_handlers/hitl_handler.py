"""
HITL Interface WebSocket Handler for EvoGenesis Web UI

This module connects the Human-In-The-Loop interface to the WebSocketManager for real-time updates.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional

class HITLWebSocketHandler:
    """
    Handles WebSocket events for the Human-In-The-Loop (HITL) Interface.
    
    This class bridges the HITL Interface with the WebSocketManager to provide
    real-time updates about human interactions to the web UI.
    """
    
    def __init__(self, hitl_interface, ws_manager):
        """
        Initialize the HITL WebSocket Handler.
        
        Args:
            hitl_interface: The HITL Interface instance
            ws_manager: The WebSocket Manager instance
        """
        self.hitl_interface = hitl_interface
        self.ws_manager = ws_manager
        self.logger = logging.getLogger(__name__)
        
        # Track the last status to avoid sending duplicates
        self.last_status = None
        self.last_requests_hash = None
        
        # Register event handlers
        self._register_event_handlers()
        
        # Start the status update task
        asyncio.create_task(self._status_update_task())
    
    def _register_event_handlers(self):
        """Register handlers for HITL-related WebSocket events."""
        # Register component-specific handlers
        self.ws_manager.register_component_handler("hitl.action", self._handle_hitl_action)
        self.ws_manager.register_component_handler("hitl.request.response", self._handle_request_response)
    
    async def _handle_hitl_action(self, message: Dict[str, Any]):
        """
        Handle HITL action messages from the WebSocket.
        
        Args:
            message: The action message
        """
        if not isinstance(message, dict):
            return
        
        action = message.get("action")
        
        if not action:
            return
        
        try:
            if action == "pause_all":
                success = await self.hitl_interface.pause_all_automated_actions()
                if success:
                    await self.ws_manager.broadcast_to_topic("hitl.status", {
                        "event": "hitl_paused",
                        "timestamp": time.time()
                    })
            
            elif action == "resume_all":
                success = await self.hitl_interface.resume_all_automated_actions()
                if success:
                    await self.ws_manager.broadcast_to_topic("hitl.status", {
                        "event": "hitl_resumed",
                        "timestamp": time.time()
                    })
            
            elif action == "set_intervention_level":
                level = message.get("level")
                if level is not None:
                    success = await self.hitl_interface.set_intervention_level(level)
                    if success:
                        await self.ws_manager.broadcast_to_topic("hitl.status", {
                            "event": "intervention_level_changed",
                            "level": level,
                            "timestamp": time.time()
                        })
        
        except Exception as e:
            self.logger.error(f"Error handling HITL action {action}: {str(e)}")
            await self.ws_manager.broadcast_to_topic("hitl.errors", {
                "error": "action_failed",
                "action": action,
                "message": str(e)
            })
    
    async def _handle_request_response(self, message: Dict[str, Any]):
        """
        Handle responses to HITL requests from the WebSocket.
        
        Args:
            message: The response message
        """
        if not isinstance(message, dict):
            return
        
        request_id = message.get("request_id")
        response = message.get("response")
        
        if not request_id or response is None:
            return
        
        try:
            # Submit the human response
            success = await self.hitl_interface.submit_response(request_id, response)
            
            if success:
                await self.ws_manager.broadcast_to_topic("hitl.requests", {
                    "event": "request_responded",
                    "request_id": request_id,
                    "status": "responded",
                    "timestamp": time.time()
                })
        
        except Exception as e:
            self.logger.error(f"Error handling HITL request response for {request_id}: {str(e)}")
            await self.ws_manager.broadcast_to_topic("hitl.errors", {
                "error": "response_failed",
                "request_id": request_id,
                "message": str(e)
            })
    
    async def _broadcast_new_request(self, request):
        """Broadcast a new HITL request event."""
        try:
            # Create request data dict
            request_data = {
                "id": request.id,
                "event": "new_request",
                "type": request.type,
                "source": request.source,
                "description": request.description,
                "options": request.options,
                "urgency": request.urgency,
                "created_at": request.created_at,
                "expires_at": request.expires_at,
                "status": request.status,
                "timestamp": time.time()
            }
            
            # Broadcast to requests topic
            await self.ws_manager.broadcast_to_topic("hitl.requests", request_data)
            
            # Also broadcast to specific request topic
            await self.ws_manager.broadcast_to_topic(f"hitl.requests.{request.id}", request_data)
            
        except Exception as e:
            self.logger.error(f"Error broadcasting new request: {str(e)}")
    
    async def _status_update_task(self):
        """Background task to update HITL interface status and broadcast it."""
        while True:
            try:
                # Get current status
                status = self.hitl_interface.get_status()
                
                # Check if status has changed to avoid spamming
                if status != self.last_status:
                    # Broadcast to subscribers
                    await self.ws_manager.broadcast_to_topic("hitl.status", {
                        "status": status,
                        "intervention_level": self.hitl_interface.intervention_level,
                        "timestamp": time.time()
                    })
                    self.last_status = status
                
                # Update pending requests if changed
                await self._update_requests_if_changed()
                
                # Check for new requests and handle any new requests that came in
                await self._handle_new_requests()
                
                # Wait before checking again
                await asyncio.sleep(1)  # Faster updates for HITL requests
            except Exception as e:
                self.logger.error(f"Error in HITL status update task: {str(e)}")
                await asyncio.sleep(5)  # Back off on error
    
    async def _handle_new_requests(self):
        """Check for and broadcast new HITL requests."""
        try:
            # Get new requests
            new_requests = self.hitl_interface.get_new_requests()
            
            # Broadcast each new request
            for request in new_requests:
                await self._broadcast_new_request(request)
        
        except Exception as e:
            self.logger.error(f"Error handling new HITL requests: {str(e)}")
    
    async def _update_requests_if_changed(self):
        """Update and broadcast requests list if it has changed."""
        try:
            # Get current pending requests
            requests = self.hitl_interface.get_pending_requests()
            
            # Generate a simple hash to check for changes
            requests_hash = hash(frozenset([
                (req.id, req.status, req.updated_at) for req in requests
            ]))
            
            if requests_hash != self.last_requests_hash:
                # Format request data for the frontend
                request_list = []
                for req in requests:
                    request_list.append({
                        "id": req.id,
                        "type": req.type,
                        "source": req.source,
                        "description": req.description,
                        "options": req.options,
                        "urgency": req.urgency,
                        "created_at": req.created_at,
                        "expires_at": req.expires_at,
                        "status": req.status
                    })
                
                # Broadcast request list
                await self.ws_manager.broadcast_to_topic("hitl.requests.list", {
                    "event": "requests_updated",
                    "requests": request_list,
                    "timestamp": time.time()
                })
                
                # Update the hash
                self.last_requests_hash = requests_hash
        except Exception as e:
            self.logger.error(f"Error updating HITL requests: {str(e)}")

def connect_hitl_interface(hitl_interface, ws_manager):
    """
    Connect the HITL Interface to the WebSocket Manager.
    
    Args:
        hitl_interface: The HITL Interface instance
        ws_manager: The WebSocket Manager instance
        
    Returns:
        The handler instance
    """
    handler = HITLWebSocketHandler(hitl_interface, ws_manager)
    
    # Allow the HITL interface to directly trigger broadcasts
    hitl_interface.register_new_request_callback(handler._broadcast_new_request)
    
    return handler
