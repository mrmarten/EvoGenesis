"""
Memory WebSocket Handler for EvoGenesis Web UI

This module connects the Memory Manager to the WebSocketManager for real-time updates.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional

class MemoryWebSocketHandler:
    """
    Handles WebSocket events for the Memory Manager.
    
    This class bridges the Memory Manager with the WebSocketManager to provide
    real-time updates about memory operations to the web UI.
    """
    
    def __init__(self, memory_manager, ws_manager):
        """
        Initialize the Memory WebSocket Handler.
        
        Args:
            memory_manager: The Memory Manager instance
            ws_manager: The WebSocket Manager instance
        """
        self.memory_manager = memory_manager
        self.ws_manager = ws_manager
        self.logger = logging.getLogger(__name__)
        
        # Track the last memory status to avoid sending duplicates
        self.last_memory_status = None
        
        # Register event handlers
        self._register_event_handlers()
        
        # Start the memory status update task
        asyncio.create_task(self._memory_status_update_task())
    
    def _register_event_handlers(self):
        """Register handlers for memory-related WebSocket events."""
        # Register component-specific handlers
        self.ws_manager.register_component_handler("memory.query", self._handle_memory_query)
    
    async def _handle_memory_query(self, message: Dict[str, Any]):
        """
        Handle memory query messages from the WebSocket.
        
        Args:
            message: The query message
        """
        if not isinstance(message, dict):
            return
        
        query_type = message.get("query_type")
        params = message.get("params", {})
        
        if query_type == "search":
            # Handle search query
            namespace = params.get("namespace", "default")
            query = params.get("query", "")
            limit = params.get("limit", 10)
            
            try:
                # Call the memory manager's search method
                results = self.memory_manager.search(namespace, query, limit=limit)
                
                # Broadcast results
                await self.ws_manager.broadcast_to_topic("memory.search_results", {
                    "query": query,
                    "namespace": namespace,
                    "results": results
                })
            except Exception as e:
                self.logger.error(f"Error searching memory: {str(e)}")
                await self.ws_manager.broadcast_to_topic("memory.errors", {
                    "error": "search_failed",
                    "message": str(e)
                })
    
    async def _memory_status_update_task(self):
        """Background task to update memory status and broadcast it."""
        while True:
            try:
                # Get current memory status
                status = self.memory_manager.get_status()
                
                # Check if status has changed to avoid spamming
                if status != self.last_memory_status:
                    # Broadcast to subscribers
                    await self.ws_manager.broadcast_to_topic("memory.status", status)
                    self.last_memory_status = status
                
                # Update memory usage statistics
                await self._update_memory_stats()
                
                # Wait before checking again
                await asyncio.sleep(5)
            except Exception as e:
                self.logger.error(f"Error in memory status update task: {str(e)}")
                await asyncio.sleep(10)  # Back off on error
    
    async def _update_memory_stats(self):
        """Update and broadcast memory usage statistics."""
        try:
            # Get namespaces and counts
            namespaces = self.memory_manager.long_term.list_namespaces()
            
            namespace_stats = {}
            for namespace in namespaces:
                # Get count and size information
                count = self.memory_manager.long_term.count_documents(namespace)
                
                # Get some example entries (just metadata)
                examples = []
                sample_docs = self.memory_manager.long_term.get_documents(
                    namespace, limit=3, include_metadata=True, include_embeddings=False
                )
                
                if sample_docs:
                    for doc in sample_docs:
                        if isinstance(doc, dict):
                            # Extract just the ID and metadata
                            examples.append({
                                "id": doc.get("id", "unknown"),
                                "metadata": doc.get("metadata", {})
                            })
                
                namespace_stats[namespace] = {
                    "count": count,
                    "examples": examples,
                    "last_updated": time.time()
                }
            
            # Short-term memory stats
            contexts = self.memory_manager.list_short_term_contexts()
            short_term_stats = {}
            
            for context_id in contexts:
                keys = self.memory_manager.list_short_term_keys(context_id)
                short_term_stats[context_id] = {
                    "keys": keys,
                    "count": len(keys)
                }
            
            # Broadcast detailed memory stats
            await self.ws_manager.broadcast_to_topic("memory.statistics", {
                "long_term": namespace_stats,
                "short_term": short_term_stats,
                "timestamp": time.time()
            })
        except Exception as e:
            self.logger.error(f"Error updating memory stats: {str(e)}")

def connect_memory_manager(memory_manager, ws_manager):
    """
    Connect the Memory Manager to the WebSocket Manager.
    
    Args:
        memory_manager: The Memory Manager instance
        ws_manager: The WebSocket Manager instance
        
    Returns:
        The handler instance
    """
    handler = MemoryWebSocketHandler(memory_manager, ws_manager)
    return handler
