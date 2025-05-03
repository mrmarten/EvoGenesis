"""
EvoGenesis Kernel - The central coordinator for the EvoGenesis framework.

The Kernel initializes and manages all core modules, facilitates communication
between them, and ensures the overall system integrity.
"""

import logging
from typing import Dict, Any, Optional, List

from evogenesis_core.modules.agent_factory import AgentFactory
from evogenesis_core.modules.task_planner import TaskPlanner
from evogenesis_core.modules.llm_orchestrator import LLMOrchestrator
from evogenesis_core.modules.tooling_system import ToolingSystem
from evogenesis_core.modules.memory_manager import MemoryManager
from evogenesis_core.modules.hitl_interface import HiTLInterface
from evogenesis_core.modules.self_evolution_engine import SelfEvolutionEngine
from evogenesis_core.modules.mission_scheduler import MissionScheduler
from evogenesis_core.modules.strategic_opportunity_observatory import StrategicOpportunityObservatory
from evogenesis_core.swarm.coordinator import SwarmCoordinator
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
        
        # System activity tracking
        self.activities = []
        self.max_activities = 100  # Maximum number of activities to keep in memory
        
        # Initialize core modules
        self.memory_manager = MemoryManager(self)
        self.llm_orchestrator = LLMOrchestrator(self)
        self.tooling_system = ToolingSystem(self)
        
        # Initialize the framework adapter manager
        adapter_config = self.config.get("adapters", {})
        self.framework_adapter_manager = FrameworkAdapterManager(config=adapter_config)
        self.agent_factory = AgentFactory(self)
        self.task_planner = TaskPlanner(self)
        self.hitl_interface = HiTLInterface(self)
        self.self_evolution_engine = SelfEvolutionEngine(self)
        self.mission_scheduler = MissionScheduler(self)
        self.strategic_observatory = StrategicOpportunityObservatory(self)
        
        # Initialize the Web UI (but don't start it yet)
        self.web_ui = WebUI(self)
        
        # Initialize swarm module (but don't start it yet)
        self.swarm_module = None
        swarm_config = self.config.get("swarm", {})
        if swarm_config.get("enabled", False):
            try:
                self.swarm_module = SwarmCoordinator(self)
                self.logger.info("Swarm module initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize swarm module: {str(e)}")
        
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
        self.agent_factory.start()
        self.task_planner.start()
        self.hitl_interface.start()
        self.self_evolution_engine.start()
        self.mission_scheduler.start()
        self.strategic_observatory.start()
        
        # Start swarm module if initialized
        if self.swarm_module is not None:
            try:
                self.swarm_module.start()
                self.logger.info("Swarm module started")
            except Exception as e:
                self.logger.error(f"Failed to start swarm module: {str(e)}")
        
        self._start_periodic_tasks()
        
        self.logger.info("EvoGenesis Kernel started")
    
    def stop(self) -> None:
        """
        Stop the EvoGenesis Kernel and all its modules.
        """
        self.logger.info("Stopping EvoGenesis Kernel")
        
        # Stop modules in reverse dependency order
        self.self_evolution_engine.stop()
        self.strategic_observatory.stop()
        self.hitl_interface.stop()
        self.mission_scheduler.stop()
        self.task_planner.stop()
        self.agent_factory.stop()
        
        # Stop swarm module if initialized
        if hasattr(self, 'swarm_module') and self.swarm_module is not None:
            try:
                self.swarm_module.stop()
                self.logger.info("Swarm module stopped")
            except Exception as e:
                self.logger.error(f"Failed to stop swarm module: {str(e)}")
        
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
                "agent_factory": self.agent_factory.get_status(),
                "task_planner": self.task_planner.get_status(),
                "hitl_interface": self.hitl_interface.get_status(),
                "self_evolution_engine": self.self_evolution_engine.get_status(),
                "mission_scheduler": {
                    "missions": len(self.mission_scheduler.missions),
                    "schedules": len(self.mission_scheduler.schedules),
                    "persistent_agents": len(self.mission_scheduler.persistent_agents)
                },
                "swarm_module": "active" if self.swarm_module is not None else "inactive"
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
    
    def log_activity(self, activity_type: str, title: str, message: str, data: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a system activity.
        
        Args:
            activity_type: Type of activity (e.g., 'agent_created', 'task_completed')
            title: Short title for the activity
            message: Detailed message about the activity
            data: Additional data associated with the activity
        """
        import time
        
        activity = {
            "type": activity_type,
            "title": title,
            "message": message,
            "timestamp": time.time(),
            "data": data or {}
        }
        
        self.activities.insert(0, activity)  # Add to front of list
        
        # Trim list if it gets too long
        if len(self.activities) > self.max_activities:
            self.activities = self.activities[:self.max_activities]
        
        # Also log to standard logger
        self.logger.info(f"Activity: {activity_type} - {title} - {message}")
    
    def get_recent_activities(self, limit: int = 10, activity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent system activities.
        
        Args:
            limit: Maximum number of activities to return
            activity_type: Filter by activity type
            
        Returns:
            List of activity records
        """
        if activity_type:
            filtered = [a for a in self.activities if a["type"] == activity_type]
            return filtered[:limit]
        
        return self.activities[:limit]
    
    def _start_periodic_tasks(self):
        """Start periodic tasks that run in the background."""
        import threading
        import time
        
        def mission_scheduler_check():
            """Periodic task to check for mission schedules that need to run."""
            while hasattr(self, 'mission_scheduler') and self.mission_scheduler:
                try:
                    self.mission_scheduler.check_due_schedules()
                except Exception as e:
                    self.logger.error(f"Error in mission scheduler check: {str(e)}")
                time.sleep(60)  # Check every minute, the scheduler will handle rate limiting
        
        # Start the mission scheduler check thread
        if hasattr(self, 'mission_scheduler'):
            mission_checker = threading.Thread(
                target=mission_scheduler_check,
                name="MissionSchedulerChecker",
                daemon=True
            )
            mission_checker.start()
            self.logger.info("Mission scheduler checker thread started")
    
    def get_module(self, module_name: str) -> Optional[Any]:
        """
        Get a module by name.
        
        Args:
            module_name: The name of the module to get
            
        Returns:
            The module if found, None otherwise
        """
        if module_name == "agent_factory":
            return self.agent_factory
        elif module_name == "task_planner":
            return self.task_planner
        elif module_name == "memory_manager":
            return self.memory_manager
        elif module_name == "tooling_system":
            return self.tooling_system
        elif module_name == "llm_orchestrator":
            return self.llm_orchestrator
        elif module_name == "hitl_interface":
            return self.hitl_interface
        elif module_name == "self_evolution_engine":
            return self.self_evolution_engine
        elif module_name == "mission_scheduler":
            return self.mission_scheduler
        elif module_name == "strategic_observatory":
            return self.strategic_observatory
        elif module_name == "swarm_coordinator" and hasattr(self, "swarm_coordinator"):
            return self.swarm_coordinator
        elif module_name == "agent_manager":  # Add compatibility with agent_manager
            return self.agent_factory  # Agent factory serves as agent manager
        
        return None

# For backward compatibility with existing code that imports 'Kernel'
Kernel = EvoGenesisKernel
