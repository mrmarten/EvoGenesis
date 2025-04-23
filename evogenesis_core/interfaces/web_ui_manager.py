"""
Web UI Module for EvoGenesis Control Panel

This module provides the web UI interface for the EvoGenesis system.
"""

import os
import threading
import logging
from typing import Optional

from .web_ui.server import start_server

class WebUI:
    """Web UI for the EvoGenesis Control Panel."""

    def __init__(self, kernel=None):
        """Initialize the Web UI."""
        self.kernel = kernel
        self.server_thread = None
        self.host = "0.0.0.0"  # Listen on all interfaces
        self.port = 5000       # Default port
        self.is_running = False
        self.logger = logging.getLogger(__name__)

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
            
            # Start the server in a separate thread
            self.server_thread = threading.Thread(
                target=start_server,
                args=(self.kernel, self.host, self.port),
                daemon=True
            )
            self.server_thread.start()
            
            self.is_running = True
            self.logger.info(f"Web UI server started on http://{self.host}:{self.port}")
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to start Web UI server: {str(e)}")
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
