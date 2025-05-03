"""
WebSocket Manager for EvoGenesis Web UI

This module manages WebSocket connections for real-time updates in the Web UI,
providing a robust communication layer between backend components and frontend interfaces.
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Set, Optional, Callable, Awaitable
from fastapi import WebSocket, WebSocketDisconnect

class WebSocketManager:
    """
    Manages WebSocket connections and message broadcasting.
    
    This class handles:
    - Client connection/disconnection
    - Topic subscription/unsubscription
    - Message broadcasting to specific topics or individual clients
    - Event history tracking for reconnecting clients
    - Real-time system metrics broadcasting
    - Component-specific event handling
    """
    
    def __init__(self):
        """Initialize the WebSocket Manager."""
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_topics: Dict[str, Set[str]] = {}
        self.topic_connections: Dict[str, Set[str]] = {}
        self.logger = logging.getLogger(__name__)
        
        # Event history for reconnecting clients (topic -> list of recent events)
        self.event_history: Dict[str, List[Dict[str, Any]]] = {}
        self.max_history_per_topic = 50  # Maximum events to keep per topic
        
        # Component event handlers
        self.component_handlers: Dict[str, Callable[[Dict[str, Any]], Awaitable[None]]] = {}
        
        # Track connection statistics
        self.connection_stats = {
            "total_connections": 0,
            "peak_concurrent": 0,
            "current_concurrent": 0,
            "last_connection": None,
            "total_messages_sent": 0,
            "messages_by_topic": {}
        }

    async def connect(self, websocket: WebSocket, client_id: str) -> None:
        """
        Accept a new WebSocket connection.
        
        Args:
            websocket: The WebSocket connection to accept
            client_id: Unique identifier for this client
        """
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_topics[client_id] = set()
        
        # Update connection statistics
        self.connection_stats["total_connections"] += 1
        self.connection_stats["current_concurrent"] = len(self.active_connections)
        self.connection_stats["last_connection"] = datetime.now().isoformat()
        
        if self.connection_stats["current_concurrent"] > self.connection_stats["peak_concurrent"]:
            self.connection_stats["peak_concurrent"] = self.connection_stats["current_concurrent"]
        self.logger.info(f"WebSocket client {client_id} connected (Total active: {self.connection_stats['current_concurrent']})")

    def disconnect(self, client_id: str) -> None:
        """
        Handle client disconnection.
        Handle client disconnection.
        
        Args:
            client_id: ID of the client that disconnected
        """
        # Remove from active connections
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        
        # Unsubscribe from all topics
        if client_id in self.connection_topics:
            topics = list(self.connection_topics[client_id])
            for topic in topics:
                self._unsubscribe_from_topic(client_id, topic)
            
            del self.connection_topics[client_id]
        
        # Update connection statistics
        self.connection_stats["current_concurrent"] = len(self.active_connections)
        self.logger.info(f"WebSocket client {client_id} disconnected (Total active: {self.connection_stats['current_concurrent']})")

    async def subscribe_to_topics(self, client_id: str, topics: List[str]) -> None:
        """
        Subscribe a client to specific topics.
        Subscribe a client to specific topics.
        
        Args:
            client_id: ID of the client
            topics: List of topics to subscribe to
        """
        for topic in topics:
            if topic not in self.topic_connections:
                self.topic_connections[topic] = set()
            
            self.topic_connections[topic].add(client_id)
            self.connection_topics[client_id].add(topic)
        
        self.logger.debug(f"Client {client_id} subscribed to topics: {topics}")
        
        # Send confirmation
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json({
                "type": "subscription_confirmation",
                "topics": topics
            })
            
            # Send recent events for these topics to bring client up to date
            for topic in topics:
                if topic in self.event_history and self.event_history[topic]:
                    # Send the last 10 events for this topic
                    recent_events = self.event_history[topic][-10:]
                    for event in recent_events:
                        try:
                            await self.active_connections[client_id].send_json({
                                "topic": topic,
                                "data": event
                            })
                        except Exception as e:
                            self.logger.error(f"Error sending history to client {client_id}: {str(e)}")

    def _unsubscribe_from_topic(self, client_id: str, topic: str) -> None:
        """
        Unsubscribe a client from a topic.
        
        Args:
            client_id: ID of the client
            topic: Topic to unsubscribe from
        """
        if client_id in self.connection_topics:
            self.connection_topics[client_id].discard(topic)
        
        if topic in self.topic_connections:
            self.topic_connections[topic].discard(client_id)
            
            # Clean up empty topic sets
            if not self.topic_connections[topic]:
                del self.topic_connections[topic]

    async def unsubscribe_from_topics(self, client_id: str, topics: List[str]) -> None:
        """
        Unsubscribe a client from specific topics.
        
        Args:
            client_id: ID of the client
            topics: List of topics to unsubscribe from
        """
        for topic in topics:
            self._unsubscribe_from_topic(client_id, topic)
        
        self.logger.debug(f"Client {client_id} unsubscribed from topics: {topics}")
        
        # Send confirmation
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json({
                    "type": "unsubscription_confirmation",
                    "topics": topics
                })
            except Exception as e:
                 self.logger.error(f"Error sending unsubscription confirmation to client {client_id}: {str(e)}")

    async def broadcast_to_topic(self, topic: str, message: Any) -> None:
        """
        Broadcast a message to all clients subscribed to a topic.
        
        Args:
            topic: Topic to broadcast to
            message: Message to broadcast
        """
        # Store in event history
        if topic not in self.event_history:
            self.event_history[topic] = []
        
        # Add timestamp if not present
        if isinstance(message, dict) and "timestamp" not in message:
            message["timestamp"] = time.time()
            
        # Store in history
        self.event_history[topic].append(message)
        
        # Trim history if needed
        if len(self.event_history[topic]) > self.max_history_per_topic:
            self.event_history[topic] = self.event_history[topic][-self.max_history_per_topic:]
        
        # Update message statistics
        self.connection_stats["total_messages_sent"] += 1
        if topic not in self.connection_stats["messages_by_topic"]:
            self.connection_stats["messages_by_topic"][topic] = 0
        self.connection_stats["messages_by_topic"][topic] += 1
        
        if topic not in self.topic_connections:
            return
        
        # Get client IDs subscribed to this topic
        client_ids = list(self.topic_connections[topic])
        
        # Prepare the message
        formatted_message = {
            "topic": topic,
            "data": message
        }
        
        # Broadcast to all subscribed clients
        for client_id in client_ids:
            if client_id in self.active_connections:
                try:
                    await self.active_connections[client_id].send_json(formatted_message)
                except Exception as e:
                    self.logger.error(f"Error sending message to client {client_id}: {str(e)}")
                    # Connection might be broken, disconnect the client
                    self.disconnect(client_id)
        
        # Call component-specific handlers if registered
        if topic in self.component_handlers:
            try:
                await self.component_handlers[topic](message)
            except Exception as e:
                self.logger.error(f"Error in component handler for topic {topic}: {str(e)}")

    async def broadcast_system_status(self, status_data: Dict[str, Any]) -> None:
        """
        Broadcast system status to all connected clients.
        
        Args:
            status_data: System status data
        """
        await self.broadcast_to_topic("system.status", status_data)

    async def broadcast_agent_update(self, agent_data: Dict[str, Any]) -> None:
        """
        Broadcast agent update to subscribed clients.
        
        Args:
            agent_data: Agent data
        """
        await self.broadcast_to_topic("agents", agent_data)
        
        # Also broadcast specific agent events
        if "event" in agent_data:
            event = agent_data["event"]
            if event in ["agent_started", "agent_paused", "agent_stopped"]:
                await self.broadcast_to_topic("agents.status", agent_data)
    async def broadcast_task_update(self, task_data: Dict[str, Any]) -> None:
        """
        Broadcast task update to subscribed clients.
        
        Args:
            task_data: Task data
        """
        await self.broadcast_to_topic("tasks", task_data)
        
        if "event" in task_data and task_data["event"] in ["task_started", "task_paused", "task_stopped", "task_completed"]:
            await self.broadcast_to_topic("tasks.status", task_data)

    async def broadcast_memory_update(self, memory_data: Dict[str, Any]) -> None:
        """
        Broadcast memory system update to subscribed clients.
        
        Args:
            memory_data: Memory data
        """
        await self.broadcast_to_topic("memory", memory_data)
        
        # Also broadcast specific memory events if relevant
        if "namespace" in memory_data:
            namespace = memory_data["namespace"]
            await self.broadcast_to_topic(f"memory.{namespace}", memory_data)

    async def broadcast_to_all(self, message: Any) -> None:
        """
        Broadcast a message to all connected clients.
        
        Args:
            message: Message to broadcast
        """
        # Update message statistics
        self.connection_stats["total_messages_sent"] += len(self.active_connections)
        
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                self.logger.error(f"Error sending message to client {client_id}: {str(e)}")
                # Connection might be broken, disconnect the client
                self.disconnect(client_id)
    def register_component_handler(self, topic: str, handler: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        """
        Register a component-specific handler for a topic.
        
        Args:
            topic: The topic to register the handler for.
            handler: An awaitable function that takes a message dictionary.
        """
        self.component_handlers[topic] = handler

    def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get connection statistics.
        
        Returns:
            Dictionary with connection statistics
        """
        return self.connection_stats
