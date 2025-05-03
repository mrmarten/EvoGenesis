"""
WebSocket Handlers Manager for EvoGenesis Web UI

This module initializes and manages all WebSocket handlers for different components.
"""

import logging
from typing import Dict, Any, Optional

def init_ws_handlers(kernel, ws_manager):
    """
    Initialize and connect all WebSocket handlers for the kernel components.
    
    Args:
        kernel: The EvoGenesis Kernel instance
        ws_manager: The WebSocket Manager instance
        
    Returns:
        Dictionary of initialized handlers
    """
    # Create a manager instance which will handle the initialization
    manager = WebSocketHandlersManager(kernel, ws_manager)
    
    # Return the handlers dictionary
    return manager.handlers

class WebSocketHandlersManager:
    """
    Manages WebSocket handlers for all system components.
    
    This class coordinates the initialization and connection of all 
    component-specific WebSocket handlers to ensure proper communication
    between backend components and the web UI.
    """
    
    def __init__(self, kernel, ws_manager):
        """
        Initialize the WebSocket Handlers Manager.
        
        Args:
            kernel: The EvoGenesis Kernel instance
            ws_manager: The WebSocket Manager instance
        """
        self.kernel = kernel
        self.ws_manager = ws_manager
        self.logger = logging.getLogger(__name__)
        self.handlers = {}
        
        # Initialize all handlers
        self._initialize_handlers()
      def _initialize_handlers(self):
        """Initialize all component-specific WebSocket handlers."""
        self.logger.info("Initializing WebSocket handlers for system components")
        
        try:
            # Import handlers
            from evogenesis_core.interfaces.web_ui.ws_handlers.memory_handler import connect_memory_manager
            from evogenesis_core.interfaces.web_ui.ws_handlers.llm_handler import connect_llm_orchestrator
            from evogenesis_core.interfaces.web_ui.ws_handlers.observatory_handler import connect_observatory
            from evogenesis_core.interfaces.web_ui.ws_handlers.task_handler import connect_task_planner
            from evogenesis_core.interfaces.web_ui.ws_handlers.agent_handler import connect_agent_factory
            from evogenesis_core.interfaces.web_ui.ws_handlers.tool_handler import connect_tooling_system
            from evogenesis_core.interfaces.web_ui.ws_handlers.hitl_handler import connect_hitl_interface
            from evogenesis_core.interfaces.web_ui.ws_handlers.evolution_handler import connect_evolution_engine
            from evogenesis_core.interfaces.web_ui.ws_handlers.mission_handler import connect_mission_scheduler
            from evogenesis_core.interfaces.web_ui.ws_handlers.swarm_handler import connect_swarm_module
            
            # Connect Memory Manager
            if hasattr(self.kernel, "memory_manager"):
                self.handlers["memory"] = connect_memory_manager(
                    self.kernel.memory_manager, self.ws_manager
                )
                self.logger.info("Memory Manager WebSocket handler initialized")
            
            # Connect LLM Orchestrator
            if hasattr(self.kernel, "llm_orchestrator"):
                self.handlers["llm"] = connect_llm_orchestrator(
                    self.kernel.llm_orchestrator, self.ws_manager
                )
                self.logger.info("LLM Orchestrator WebSocket handler initialized")
            
            # Connect Strategic Observatory
            if hasattr(self.kernel, "strategic_observatory"):
                self.handlers["observatory"] = connect_observatory(
                    self.kernel.strategic_observatory, self.ws_manager
                )
                self.logger.info("Strategic Observatory WebSocket handler initialized")
            
            # Connect Task Planner
            if hasattr(self.kernel, "task_planner"):
                self.handlers["task"] = connect_task_planner(
                    self.kernel.task_planner, self.ws_manager
                )
                self.logger.info("Task Planner WebSocket handler initialized")
            
            # Connect Agent Factory
            if hasattr(self.kernel, "agent_factory"):
                self.handlers["agent"] = connect_agent_factory(
                    self.kernel.agent_factory, self.ws_manager
                )
                self.logger.info("Agent Factory WebSocket handler initialized")
            
            # Connect Tooling System
            if hasattr(self.kernel, "tooling_system"):
                self.handlers["tool"] = connect_tooling_system(
                    self.kernel.tooling_system, self.ws_manager
                )
                self.logger.info("Tooling System WebSocket handler initialized")
            
            # Connect HITL Interface
            if hasattr(self.kernel, "hitl_interface"):
                self.handlers["hitl"] = connect_hitl_interface(
                    self.kernel.hitl_interface, self.ws_manager
                )
                self.logger.info("HITL Interface WebSocket handler initialized")
            
            # Connect Self Evolution Engine
            if hasattr(self.kernel, "evolution_engine"):
                self.handlers["evolution"] = connect_evolution_engine(
                    self.kernel.evolution_engine, self.ws_manager
                )
                self.logger.info("Self Evolution Engine WebSocket handler initialized")
            
            # Connect Mission Scheduler
            if hasattr(self.kernel, "mission_scheduler"):
                self.handlers["mission"] = connect_mission_scheduler(
                    self.kernel.mission_scheduler, self.ws_manager
                )
                self.logger.info("Mission Scheduler WebSocket handler initialized")
            
            # Connect Swarm Module
            if hasattr(self.kernel, "swarm_module"):
                self.handlers["swarm"] = connect_swarm_module(
                    self.kernel.swarm_module, self.ws_manager
                )
                self.logger.info("Swarm Module WebSocket handler initialized")
            
            self.logger.info(f"Initialized {len(self.handlers)} WebSocket handlers")
            
        except Exception as e:
            self.logger.error(f"Error initializing WebSocket handlers: {str(e)}")
    
    def get_handler(self, component_name):
        """
        Get a specific component handler.
        
        Args:
            component_name: Name of the component
            
        Returns:
            The handler instance or None if not found
        """
        return self.handlers.get(component_name)
    
    def broadcast_system_event(self, event_type, data):
        """
        Broadcast a system-wide event to all relevant topics.
        
        Args:
            event_type: Type of event
            data: Event data
        """
        import asyncio
        
        async def _broadcast():
            event_message = {
                "event": event_type,
                "data": data,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            # Broadcast to system topic
            await self.ws_manager.broadcast_to_topic("system.events", event_message)
            
            # Also broadcast to component-specific topic if applicable
            if "." in event_type:
                component, _ = event_type.split(".", 1)
                await self.ws_manager.broadcast_to_topic(f"{component}.events", event_message)
        
        # Create a task for the broadcast
        asyncio.create_task(_broadcast())

def initialize_ws_handlers(kernel, ws_manager):
    """
    Initialize all WebSocket handlers for the system components.
    
    Args:
        kernel: The EvoGenesis Kernel instance
        ws_manager: The WebSocket Manager instance
        
    Returns:
        The WebSocket Handlers Manager instance
    """
    return WebSocketHandlersManager(kernel, ws_manager)
