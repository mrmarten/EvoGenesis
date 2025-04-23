"""
EvoGenesis Kernel - The central coordinator for the EvoGenesis framework.

The Kernel initializes and manages all core modules, facilitates communication
between them, and ensures the overall system integrity.
"""

import logging
from typing import Dict, Any, Optional, List

from evogenesis_core.modules.agent_manager import AgentManager
from evogenesis_core.modules.task_planner import TaskPlanner
from evogenesis_core.modules.llm_orchestrator import LLMOrchestrator
from evogenesis_core.modules.tooling_system import ToolingSystem
from evogenesis_core.modules.memory_manager import MemoryManager
from evogenesis_core.modules.hitl_interface import HiTLInterface
from evogenesis_core.modules.self_evolution_engine import SelfEvolutionEngine
from evogenesis_core.adapters.framework_adapter_manager import FrameworkAdapterManager
from evogenesis_core.interfaces.web_ui_manager import WebUI

class EvoGenesisKernel:
    """
    The central kernel of the EvoGenesis framework.
    
    Responsible for initializing all modules, managing their lifecycle,
    and facilitating communication between them.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the EvoGenesis Kernel with optional configuration.
        
        Args:
            config: Configuration dictionary for the kernel and its modules
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Initialize core modules
        self.memory_manager = MemoryManager(self)
        self.llm_orchestrator = LLMOrchestrator(self)
        self.tooling_system = ToolingSystem(self)
        
        # Initialize the framework adapter manager
        adapter_config = self.config.get("adapters", {})
        self.framework_adapter_manager = FrameworkAdapterManager(config=adapter_config)
        
        self.agent_manager = AgentManager(self)
        self.task_planner = TaskPlanner(self)
        self.hitl_interface = HiTLInterface(self)
        self.self_evolution_engine = SelfEvolutionEngine(self)
        
        # Initialize the Web UI (but don't start it yet)
        self.web_ui = WebUI(self)
        
        self.logger.info("EvoGenesis Kernel initialized")
    
    def start(self) -> None:
        """
        Start the EvoGenesis Kernel and all its modules.
        """
        self.logger.info("Starting EvoGenesis Kernel")
        
        # Start modules in dependency order
        self.memory_manager.start()
        self.llm_orchestrator.start()
        self.tooling_system.start()
        self.framework_adapter_manager.initialize_all_adapters()
        self.agent_manager.start()
        self.task_planner.start()
        self.hitl_interface.start()
        self.self_evolution_engine.start()
        
        self.logger.info("EvoGenesis Kernel started")
    
    def stop(self) -> None:
        """
        Stop the EvoGenesis Kernel and all its modules.
        """
        self.logger.info("Stopping EvoGenesis Kernel")
        
        # Stop modules in reverse dependency order
        self.self_evolution_engine.stop()
        self.hitl_interface.stop()
        self.task_planner.stop()
        self.agent_manager.stop()
        self.framework_adapter_manager.shutdown_all_adapters()
        self.tooling_system.stop()
        self.llm_orchestrator.stop()
        self.memory_manager.stop()
        
        # Stop the Web UI if it's running
        if hasattr(self, 'web_ui') and self.web_ui.is_running:
            self.web_ui.stop()
        
        self.logger.info("EvoGenesis Kernel stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the kernel and all its modules.
        
        Returns:
            A dictionary containing status information
        """
        return {
            "kernel": "active",
            "modules": {
                "memory_manager": self.memory_manager.get_status(),
                "llm_orchestrator": self.llm_orchestrator.get_status(),
                "tooling_system": self.tooling_system.get_status(),
                "framework_adapter_manager": {
                    "available_adapters": list(self.framework_adapter_manager.available_adapters.keys()),
                    "initialized_adapters": list(self.framework_adapter_manager.initialized_adapters.keys()),
                    "framework_registry": self.framework_adapter_manager.framework_registry
                },
                "agent_manager": self.agent_manager.get_status(),
                "task_planner": self.task_planner.get_status(),
                "hitl_interface": self.hitl_interface.get_status(),
                "self_evolution_engine": self.self_evolution_engine.get_status()
            },
            "web_ui": {
                "running": self.web_ui.is_running if hasattr(self, 'web_ui') else False,
                "url": self.web_ui.get_url() if hasattr(self, 'web_ui') and self.web_ui.is_running else None
            }
        }
    
    def start_web_ui(self, host: Optional[str] = None, port: Optional[int] = None, dev_mode: bool = False) -> bool:
        """
        Start the Web UI for the EvoGenesis Control Panel.
        
        Args:
            host: Host to bind to (default: from config or 0.0.0.0)
            port: Port to listen on (default: from config or 5000)
            dev_mode: Whether to start in development mode
            
        Returns:
            True if started successfully, False otherwise
        """
        if not hasattr(self, 'web_ui'):
            self.logger.error("Web UI not initialized")
            return False
            
        # Get configuration from config file
        web_ui_config = self.config.get("web_ui", {})
        
        # Use parameters or config values
        host = host or web_ui_config.get("host", "0.0.0.0")
        port = port or web_ui_config.get("port", 5000)
        
        # If in dev mode, override debug setting
        if dev_mode:
            web_ui_config["debug"] = True
            
        self.logger.info(f"Starting Web UI on http://{host}:{port}")
        return self.web_ui.start(host=host, port=port)
        
    def stop_web_ui(self) -> bool:
        """
        Stop the Web UI if it's running.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if not hasattr(self, 'web_ui'):
            self.logger.error("Web UI not initialized")
            return False
            
        return self.web_ui.stop()
        
    def get_web_ui_url(self) -> Optional[str]:
        """
        Get the URL for the Web UI.
        
        Returns:
            URL string for the Web UI if running, None otherwise
        """
        if not hasattr(self, 'web_ui') or not self.web_ui.is_running:
            return None
            
        return self.web_ui.get_url()
