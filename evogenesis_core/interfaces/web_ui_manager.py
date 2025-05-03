"""
Web UI Module for EvoGenesis Control Panel

This module provides the web UI interface for the EvoGenesis system, connecting
the kernel and its components to the web frontend through a robust WebSocket infrastructure.
"""

import os
import threading
import logging
import asyncio
import time
from typing import Optional, Dict, Any, List

from .web_ui.server import start_server

class WebUI:
    """
    Web UI for the EvoGenesis Control Panel.
    
    This class manages the web-based control panel that provides a user interface for:
    - Monitoring system status and component health
    - Managing agents and agent teams
    - Creating and tracking tasks and missions
    - Visualizing memory contents and operations
    - Configuring system settings and components
    - Observing strategic opportunities and insights
    - Interacting with the self-evolution engine
    - Viewing system logs and activity history
    - Real-time monitoring and control via WebSockets
    """

    def __init__(self, kernel=None):
        """Initialize the Web UI."""
        self.kernel = kernel
        self.server_thread = None
        self.host = "0.0.0.0"  # Listen on all interfaces
        self.port = 5000       # Default port
        self.is_running = False
        self.logger = logging.getLogger(__name__)
        self.dev_mode = False  # Development mode flag
        
        # System status tracking
        self.last_status_update = 0
        self.status_update_interval = 5  # seconds
        self.component_status = {}
        
        # Initialize activity tracking
        if kernel and hasattr(kernel, 'log_activity'):
            kernel.log_activity(
                activity_type="system.initialization",
                title="Web UI Initialized",
                message="Web UI module initialized and ready",
                data={"timestamp": time.time()}
            )
        
    def start(self, host: Optional[str] = None, port: Optional[int] = None) -> bool:
        """
        Start the Web UI server.
        
        Args:
            host: Host to bind to (default: 0.0.0.0)
            port: Port to listen on (default: 5000)
            
        Returns:
            True if started successfully, False otherwise
        """
        if self.is_running:
            self.logger.warning("Web UI server is already running")
            return True
            
        try:
            self.host = host or self.host
            self.port = port or self.port
            
            # Log the start attempt
            self.logger.info(f"Starting Web UI server on {self.host}:{self.port}")
            
            if self.kernel and hasattr(self.kernel, 'log_activity'):
                self.kernel.log_activity(
                    activity_type="web_ui.startup",
                    title="Web UI Starting",
                    message=f"Starting Web UI server on http://{self.host}:{self.port}",
                    data={"host": self.host, "port": self.port}
                )
            
            # Start the server in a separate thread
            self.server_thread = threading.Thread(
                target=start_server,
                args=(self.kernel, self.host, self.port),
                daemon=True
            )
            self.server_thread.start()
            
            self.is_running = True
            self.logger.info(f"Web UI server started on http://{self.host}:{self.port}")
            
            # Log successful start
            if self.kernel and hasattr(self.kernel, 'log_activity'):
                self.kernel.log_activity(
                    activity_type="web_ui.started",
                    title="Web UI Started",
                    message=f"Web UI server successfully started on http://{self.host}:{self.port}",
                    data={"host": self.host, "port": self.port, "url": self.get_url()}
                )
            return True
        except Exception as e:
            self.logger.error(f"Failed to start Web UI server: {str(e)}")
            
            # Log the error
            if self.kernel and hasattr(self.kernel, 'log_activity'):
                self.kernel.log_activity(
                    activity_type="web_ui.error",
                    title="Web UI Start Failed",
                    message=f"Failed to start Web UI server: {str(e)}",
                    data={"error": str(e)}
                )
                
            return False
    
    def stop(self) -> bool:
        """
        Stop the Web UI server.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if not self.is_running:
            self.logger.warning("Web UI server is not running")
            return True
            
        try:
            # Log the stop attempt
            if self.kernel and hasattr(self.kernel, 'log_activity'):
                self.kernel.log_activity(
                    activity_type="web_ui.shutdown",
                    title="Web UI Stopping",
                    message="Web UI server is shutting down",
                    data={"host": self.host, "port": self.port}
                )
            
            # The server is running in a daemon thread, so it will be terminated
            # when the main thread exits. We just need to update the flag.
            self.is_running = False
            self.logger.info("Web UI server stopped")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop Web UI server: {str(e)}")
            return False
    
    def get_url(self) -> str:
        """
        Get the URL for the Web UI.
        
        Returns:
            URL string for the Web UI
        """
        host_display = "localhost" if self.host == "0.0.0.0" else self.host
        return f"http://{host_display}:{self.port}"
        
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the Web UI.
        
        Returns:
            Dictionary with status information
        """
        return {
            "running": self.is_running,
            "host": self.host,
            "port": self.port,
            "url": self.get_url() if self.is_running else None,
            "mode": "development" if self.dev_mode else "production"
        }
    
    def collect_component_status(self) -> Dict[str, Any]:
        """
        Collect status information from all kernel components.
        
        Returns:
            Dictionary with component status information
        """
        now = time.time()
        
        # Only update status periodically to avoid overhead
        if now - self.last_status_update < self.status_update_interval:
            return self.component_status
            
        if not self.kernel:
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
            component = getattr(self.kernel, component_name, None)
            if component and hasattr(component, "get_status"):
                try:
                    component_status = component.get_status()
                    status[component_name] = component_status
                except Exception as e:
                    status[component_name] = {"error": str(e)}
            else:
                status[component_name] = {"status": "not_available"}
        
        # Swarm module (optional)
        if hasattr(self.kernel, "swarm_module") and self.kernel.swarm_module:
            try:
                status["swarm_module"] = self.kernel.swarm_module.get_status()
            except Exception as e:
                status["swarm_module"] = {"error": str(e)}
        else:
            status["swarm_module"] = {"status": "not_enabled"}
        
        # Update cached status
        self.component_status = status
        self.last_status_update = now
        
        return status
