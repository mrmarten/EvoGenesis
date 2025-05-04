"""
WebSocket Handlers Manager for EvoGenesis Web UI

This module initializes and manages all WebSocket handlers for different components.
"""

import asyncio
import logging
from typing import Dict, Any, Optional

# Explicitly import connect functions and handler classes
from evogenesis_core.interfaces.web_ui.ws_handlers.memory_handler import connect_memory_manager, MemoryWebSocketHandler
from evogenesis_core.interfaces.web_ui.ws_handlers.llm_handler import connect_llm_orchestrator, LLMWebSocketHandler
from evogenesis_core.interfaces.web_ui.ws_handlers.observatory_handler import connect_observatory, ObservatoryWebSocketHandler
# Import others as they are created/needed
# from evogenesis_core.interfaces.web_ui.ws_handlers.task_handler import connect_task_planner, TaskPlannerWebSocketHandler
# from evogenesis_core.interfaces.web_ui.ws_handlers.agent_handler import connect_agent_factory, AgentFactoryWebSocketHandler
# from evogenesis_core.interfaces.web_ui.ws_handlers.tool_handler import connect_tooling_system, ToolingSystemWebSocketHandler
# from evogenesis_core.interfaces.web_ui.ws_handlers.hitl_handler import connect_hitl_interface, HiTLWebSocketHandler
# from evogenesis_core.interfaces.web_ui.ws_handlers.evolution_handler import connect_evolution_engine, EvolutionWebSocketHandler
# from evogenesis_core.interfaces.web_ui.ws_handlers.mission_handler import connect_mission_scheduler, MissionWebSocketHandler
# from evogenesis_core.interfaces.web_ui.ws_handlers.swarm_handler import connect_swarm_module, SwarmWebSocketHandler

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
        self.handlers: Dict[str, Any] = {} # Type hint for clarity
        # Initialize all handlers (without starting background tasks)
        self._initialize_handlers()

    def _initialize_handlers(self):
        """Initialize all component-specific WebSocket handlers."""
        self.logger.info("Initializing WebSocket handlers for system components")

        # Define connection functions and expected handler types
        # Tuple: (connect_function, HandlerClass, kernel_attribute_name)
        handler_connectors = {
            "memory": (connect_memory_manager, MemoryWebSocketHandler, "memory_manager"),
            "llm": (connect_llm_orchestrator, LLMWebSocketHandler, "llm_orchestrator"),
            "observatory": (connect_observatory, ObservatoryWebSocketHandler, "strategic_observatory"),
            # Add other handlers here following the pattern:
            # "task": (connect_task_planner, TaskPlannerWebSocketHandler, "task_planner"),
            # "agent": (connect_agent_factory, AgentFactoryWebSocketHandler, "agent_factory"),
            # "tool": (connect_tooling_system, ToolingSystemWebSocketHandler, "tooling_system"),
            # "hitl": (connect_hitl_interface, HiTLWebSocketHandler, "hitl_interface"),
            # "evolution": (connect_evolution_engine, EvolutionWebSocketHandler, "evolution_engine"),
            # "mission": (connect_mission_scheduler, MissionWebSocketHandler, "mission_scheduler"),
            # "swarm": (connect_swarm_module, SwarmWebSocketHandler, "swarm_module"),
        }

        # Dynamically import and connect handlers based on kernel attributes
        for key, (connect_func, _, kernel_attr) in handler_connectors.items():
             if hasattr(self.kernel, kernel_attr):
                component = getattr(self.kernel, kernel_attr)
                if component:
                    try:
                        self.handlers[key] = connect_func(component, self.ws_manager)
                        self.logger.info(f"{kernel_attr.replace('_', ' ').title()} WebSocket handler initialized")
                    except ImportError as ie:
                         self.logger.error(f"Failed to import handler for {key}: {ie}")
                    except Exception as e:
                        self.logger.error(f"Failed to initialize {key} WebSocket handler: {str(e)}", exc_info=True)
                else:
                    self.logger.warning(f"Kernel attribute '{kernel_attr}' is None, skipping WebSocket handler initialization.")
             else:
                 self.logger.warning(f"Kernel does not have attribute '{kernel_attr}', skipping WebSocket handler initialization.")

        self.logger.info(f"Initialized {len(self.handlers)} WebSocket handlers")

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
        # Ensure this runs in an event loop if called from sync code
        # Or expect it to be called from async code
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

        try:
            loop = asyncio.get_running_loop()
            loop.create_task(_broadcast())
        except RuntimeError:
            # No running loop, maybe log a warning or handle differently
            self.logger.warning("No running event loop to broadcast system event.")
            # Consider asyncio.run(_broadcast()) if appropriate, but might block

    async def start_all_handlers(self):
        """Starts background tasks for all initialized handlers."""
        self.logger.info("Starting background tasks for WebSocket handlers...")
        start_tasks = []
        for handler_key, handler in self.handlers.items():
            if hasattr(handler, 'start') and callable(handler.start):
                # Ensure start is awaitable
                if asyncio.iscoroutinefunction(handler.start):
                    start_tasks.append(asyncio.create_task(handler.start()))
                else:
                     self.logger.warning(f"Handler '{handler_key}' start method is not a coroutine.")
            else:
                self.logger.debug(f"Handler '{handler_key}' does not have an async start method.")

        if start_tasks:
            results = await asyncio.gather(*start_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    # Find corresponding handler key - this is a bit fragile
                    key = list(self.handlers.keys())[i] # Assumes order is preserved
                    self.logger.error(f"Error starting handler '{key}': {result}", exc_info=result)
            self.logger.info("Finished starting WebSocket handler background tasks.")
        else:
            self.logger.info("No WebSocket handler background tasks to start.")

    async def stop_all_handlers(self):
        """Stops background tasks for all initialized handlers."""
        self.logger.info("Stopping background tasks for WebSocket handlers...")
        stop_tasks = []
        for handler_key, handler in self.handlers.items():
            if hasattr(handler, 'stop') and callable(handler.stop):
                 # Ensure stop is awaitable
                if asyncio.iscoroutinefunction(handler.stop):
                    stop_tasks.append(asyncio.create_task(handler.stop()))
                else:
                    self.logger.warning(f"Handler '{handler_key}' stop method is not a coroutine.")
            else:
                 self.logger.debug(f"Handler '{handler_key}' does not have an async stop method.")

        if stop_tasks:
            results = await asyncio.gather(*stop_tasks, return_exceptions=True)
            for i, result in enumerate(results):
                 if isinstance(result, Exception):
                    key = list(self.handlers.keys())[i]
                    self.logger.error(f"Error stopping handler '{key}': {result}", exc_info=result)
            self.logger.info("Finished stopping WebSocket handler background tasks.")
        else:
            self.logger.info("No WebSocket handler background tasks to stop.")
