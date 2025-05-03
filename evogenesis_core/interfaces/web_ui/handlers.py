"""
WebSocket Handler Initialization Module

This module provides utilities for initializing WebSocket handlers for the web UI server.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def init_ws_handlers(kernel, ws_manager):
    """
    Initialize and connect all WebSocket handlers for the kernel components.
    
    Args:
        kernel: The EvoGenesis Kernel instance
        ws_manager: The WebSocket Manager instance
        
    Returns:
        Dictionary of initialized handlers
    """
    logger.info("Initializing WebSocket handlers for kernel components")
    
    handlers = {}
    
    # Import handlers - using try/except for each to ensure one failure doesn't break everything
    try:
        from evogenesis_core.interfaces.web_ui.ws_handlers.memory_handler import connect_memory_manager
        if hasattr(kernel, "memory_manager"):
            handlers["memory"] = connect_memory_manager(kernel.memory_manager, ws_manager)
            logger.info("Memory Manager WebSocket handler initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Memory Manager WebSocket handler: {str(e)}")
    
    try:
        from evogenesis_core.interfaces.web_ui.ws_handlers.llm_handler import connect_llm_orchestrator
        if hasattr(kernel, "llm_orchestrator"):
            handlers["llm"] = connect_llm_orchestrator(kernel.llm_orchestrator, ws_manager)
            logger.info("LLM Orchestrator WebSocket handler initialized")
    except Exception as e:
        logger.error(f"Failed to initialize LLM Orchestrator WebSocket handler: {str(e)}")
    
    try:
        from evogenesis_core.interfaces.web_ui.ws_handlers.observatory_handler import connect_observatory
        if hasattr(kernel, "strategic_observatory"):
            handlers["observatory"] = connect_observatory(kernel.strategic_observatory, ws_manager)
            logger.info("Strategic Observatory WebSocket handler initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Strategic Observatory WebSocket handler: {str(e)}")
    
    # Add more handler initializations here
    
    return handlers

def broadcast_system_event(ws_manager, event_type, data):
    """
    Broadcast a system event to all subscribed clients.
    
    Args:
        ws_manager: The WebSocket Manager instance
        event_type: Type of event
        data: Event data
    """
    import asyncio
    
    async def _broadcast():
        try:
            event_message = {
                "event": event_type,
                "data": data,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            # Broadcast to system topic
            await ws_manager.broadcast_to_topic("system.events", event_message)
            
            # Also broadcast to component-specific topic if applicable
            if "." in event_type:
                component, _ = event_type.split(".", 1)
                await ws_manager.broadcast_to_topic(f"{component}.events", event_message)
                
            logger.debug(f"Broadcast system event: {event_type}")
        except Exception as e:
            logger.error(f"Error broadcasting system event: {str(e)}")
    
    # Create a task for the broadcast
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_broadcast())
        else:
            logger.warning("AsyncIO loop not running, could not broadcast system event")
    except Exception as e:
        logger.error(f"Error setting up broadcast task: {str(e)}")

def get_component_status(kernel):
    """
    Get status information from all kernel components.
    
    Args:
        kernel: The EvoGenesis Kernel instance
        
    Returns:
        Dictionary with component status information
    """
    if not kernel:
        return {"error": "No kernel available"}
        
    status = {}
    
    # Core components
    for component_name in [
        "memory_manager", 
        "llm_orchestrator", 
        "tooling_system",
        "agent_factory", 
        "task_planner", 
        "hitl_interface",
        "self_evolution_engine",
        "mission_scheduler",
        "strategic_observatory"
    ]:
        # Get the component
        component = getattr(kernel, component_name, None)
        if component and hasattr(component, "get_status"):
            try:
                component_status = component.get_status()
                status[component_name] = component_status
            except Exception as e:
                status[component_name] = {"error": str(e)}
        else:
            status[component_name] = {"status": "not_available"}
    
    # Swarm module (optional)
    if hasattr(kernel, "swarm_module") and kernel.swarm_module:
        try:
            status["swarm_module"] = kernel.swarm_module.get_status()
        except Exception as e:
            status["swarm_module"] = {"error": str(e)}
    else:
        status["swarm_module"] = {"status": "not_enabled"}
    
    return status
