"""
Strategic Observatory WebSocket Handler for EvoGenesis Web UI

This module connects the Strategic Opportunity Observatory to the WebSocketManager for real-time updates.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional

class ObservatoryWebSocketHandler:
    """
    Handles WebSocket events for the Strategic Opportunity Observatory.
    
    This class bridges the Observatory with the WebSocketManager to provide
    real-time updates about strategic opportunities to the web UI.
    """
    
    def __init__(self, observatory, ws_manager):
        """
        Initialize the Observatory WebSocket Handler.
        
        Args:
            observatory: The Strategic Opportunity Observatory instance
            ws_manager: The WebSocket Manager instance
        """
        self.observatory = observatory
        self.ws_manager = ws_manager
        self.logger = logging.getLogger(__name__)
        
        # Track the last status to avoid sending duplicates
        self.last_status = None
        
        # Register event handlers
        self._register_event_handlers()
        
        # Start the status update task
        asyncio.create_task(self._status_update_task())
    
    def _register_event_handlers(self):
        """Register handlers for observatory-related WebSocket events."""
        # Register component-specific handlers
        self.ws_manager.register_component_handler("observatory.query", self._handle_observatory_query)
    
    async def _handle_observatory_query(self, message: Dict[str, Any]):
        """
        Handle observatory query messages from the WebSocket.
        
        Args:
            message: The query message
        """
        if not isinstance(message, dict):
            return
        
        query_type = message.get("query_type")
        params = message.get("params", {})
        
        if query_type == "opportunities":
            # Handle opportunities query
            limit = params.get("limit", 10)
            category = params.get("category", None)
            
            try:
                # Call the observatory to get opportunities
                opportunities = self.observatory.get_opportunities(limit=limit, category=category)
                
                # Broadcast results
                await self.ws_manager.broadcast_to_topic("observatory.opportunities", {
                    "opportunities": opportunities,
                    "timestamp": time.time()
                })
            except Exception as e:
                self.logger.error(f"Error getting opportunities: {str(e)}")
                await self.ws_manager.broadcast_to_topic("observatory.errors", {
                    "error": "query_failed",
                    "message": str(e)
                })
        
        elif query_type == "signals":
            # Handle signals query
            signal_source = params.get("source", None)
            limit = params.get("limit", 10)
            
            try:
                # Call the observatory to get signals
                signals = self.observatory.get_signals(source=signal_source, limit=limit)
                
                # Broadcast results
                await self.ws_manager.broadcast_to_topic("observatory.signals", {
                    "signals": signals,
                    "timestamp": time.time()
                })
            except Exception as e:
                self.logger.error(f"Error getting signals: {str(e)}")
                await self.ws_manager.broadcast_to_topic("observatory.errors", {
                    "error": "query_failed",
                    "message": str(e)
                })
    
    async def _status_update_task(self):
        """Background task to update observatory status and broadcast it."""
        while True:
            try:
                # Get current status
                status = self.observatory.get_status()
                
                # Check if status has changed to avoid spamming
                if status != self.last_status:
                    # Broadcast to subscribers
                    await self.ws_manager.broadcast_to_topic("observatory.status", status)
                    self.last_status = status
                
                # Update opportunity statistics
                await self._update_opportunity_stats()
                
                # Wait before checking again
                await asyncio.sleep(10)  # Less frequent updates for observatory
            except Exception as e:
                self.logger.error(f"Error in observatory status update task: {str(e)}")
                await asyncio.sleep(20)  # Back off on error
    
    async def _update_opportunity_stats(self):
        """Update and broadcast opportunity statistics."""
        try:
            # Get opportunity statistics
            opportunities = self.observatory.get_opportunities(limit=5)
            signals = self.observatory.get_signal_sources()
            
            # Count by category
            categories = {}
            for opp in opportunities:
                category = opp.get("category", "Uncategorized")
                if category not in categories:
                    categories[category] = 0
                categories[category] += 1
            
            # Broadcast stats
            await self.ws_manager.broadcast_to_topic("observatory.statistics", {
                "opportunity_count": len(opportunities),
                "signal_sources": len(signals),
                "categories": categories,
                "recent_opportunities": opportunities,
                "timestamp": time.time()
            })
        except Exception as e:
            self.logger.error(f"Error updating opportunity stats: {str(e)}")

def connect_observatory(observatory, ws_manager):
    """
    Connect the Strategic Opportunity Observatory to the WebSocket Manager.
    
    Args:
        observatory: The Strategic Opportunity Observatory instance
        ws_manager: The WebSocket Manager instance
        
    Returns:
        The handler instance
    """
    handler = ObservatoryWebSocketHandler(observatory, ws_manager)
    return handler
