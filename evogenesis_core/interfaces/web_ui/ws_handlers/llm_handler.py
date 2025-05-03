"""
LLM Orchestrator WebSocket Handler for EvoGenesis Web UI

This module connects the LLM Orchestrator to the WebSocketManager for real-time updates.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional

class LLMWebSocketHandler:
    """
    Handles WebSocket events for the LLM Orchestrator.
    
    This class bridges the LLM Orchestrator with the WebSocketManager to provide
    real-time updates about LLM operations to the web UI.
    """
    
    def __init__(self, llm_orchestrator, ws_manager):
        """
        Initialize the LLM WebSocket Handler.
        
        Args:
            llm_orchestrator: The LLM Orchestrator instance
            ws_manager: The WebSocket Manager instance
        """
        self.llm_orchestrator = llm_orchestrator
        self.ws_manager = ws_manager
        self.logger = logging.getLogger(__name__)
        
        # Track the last LLM status to avoid sending duplicates
        self.last_llm_status = None
        
        # Register event handlers
        self._register_event_handlers()
        
        # Start the LLM status update task
        asyncio.create_task(self._llm_status_update_task())
    
    def _register_event_handlers(self):
        """Register handlers for LLM-related WebSocket events."""
        # Register component-specific handlers
        self.ws_manager.register_component_handler("llm.query", self._handle_llm_query)
    
    async def _handle_llm_query(self, message: Dict[str, Any]):
        """
        Handle LLM query messages from the WebSocket.
        
        Args:
            message: The query message
        """
        if not isinstance(message, dict):
            return
        
        query_type = message.get("query_type")
        params = message.get("params", {})
        
        if query_type == "generate":
            # Handle text generation query
            prompt = params.get("prompt", "")
            model = params.get("model", "default")
            temperature = params.get("temperature", 0.7)
            max_tokens = params.get("max_tokens", 100)
            
            try:
                # Call the LLM orchestrator's generate method
                response = await self.llm_orchestrator.generate_text(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Broadcast results
                await self.ws_manager.broadcast_to_topic("llm.generation_results", {
                    "prompt": prompt,
                    "response": response,
                    "model": model,
                    "timestamp": time.time()
                })
            except Exception as e:
                self.logger.error(f"Error generating text: {str(e)}")
                await self.ws_manager.broadcast_to_topic("llm.errors", {
                    "error": "generation_failed",
                    "message": str(e)
                })
    
    async def _llm_status_update_task(self):
        """Background task to update LLM status and broadcast it."""
        while True:
            try:
                # Get current LLM status
                status = self.llm_orchestrator.get_status()
                
                # Check if status has changed to avoid spamming
                if status != self.last_llm_status:
                    # Broadcast to subscribers
                    await self.ws_manager.broadcast_to_topic("llm.status", status)
                    self.last_llm_status = status
                
                # Update LLM usage statistics
                await self._update_llm_stats()
                
                # Wait before checking again
                await asyncio.sleep(5)
            except Exception as e:
                self.logger.error(f"Error in LLM status update task: {str(e)}")
                await asyncio.sleep(10)  # Back off on error
    
    async def _update_llm_stats(self):
        """Update and broadcast LLM usage statistics."""
        try:
            # Gather LLM stats
            providers = self.llm_orchestrator.list_providers()
            models = self.llm_orchestrator.list_models()
            
            # Get usage statistics
            usage_stats = self.llm_orchestrator.get_usage_stats()
            
            # Broadcast detailed LLM stats
            await self.ws_manager.broadcast_to_topic("llm.statistics", {
                "providers": providers,
                "models": models,
                "usage": usage_stats,
                "timestamp": time.time()
            })
        except Exception as e:
            self.logger.error(f"Error updating LLM stats: {str(e)}")

def connect_llm_orchestrator(llm_orchestrator, ws_manager):
    """
    Connect the LLM Orchestrator to the WebSocket Manager.
    
    Args:
        llm_orchestrator: The LLM Orchestrator instance
        ws_manager: The WebSocket Manager instance
        
    Returns:
        The handler instance
    """
    handler = LLMWebSocketHandler(llm_orchestrator, ws_manager)
    return handler
