"""
EvoGenesis Kernel - The central coordinator for the EvoGenesis framework.

The Kernel initializes and manages all core modules, facilitates communication
between them, and ensures the overall system integrity.
"""

import logging
from typing import Dict, Any, Optional, List
import os

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
from evogenesis_core.swarm.bus import create_message_bus, BusImplementation, MessageBus
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
        self._initialize_modules(self.config.get("modules", {}))

    def _initialize_modules(self, module_configs: Dict[str, Any]):
        """Initialize all core modules based on configuration."""
        self.logger.info("Initializing EvoGenesis modules...")
        
        # --- Message Bus Initialization (Moved earlier) ---
        bus_config = module_configs.get("message_bus", {})
        bus_implementation_str = bus_config.get("implementation", "MEMORY").upper() # Default to MEMORY
        try:
            bus_implementation = BusImplementation[bus_implementation_str]
            self.message_bus = create_message_bus(bus_implementation, **bus_config.get("config", {}))
            self.logger.info(f"Message Bus initialized with implementation: {bus_implementation.value}")
        except (KeyError, Exception) as e:
            self.logger.error(f"Failed to initialize Message Bus with implementation '{bus_implementation_str}': {e}. Defaulting to MEMORY.")
            # Fallback to in-memory bus if configured one fails
            bus_implementation = BusImplementation.MEMORY # Use the correct enum member
            self.message_bus = create_message_bus(bus_implementation)
            self.logger.info("Defaulted to in-memory Message Bus.")

        # --- Module Initializations ---
        # Agent Factory
        agent_config = module_configs.get("agent_factory", {})
        if agent_config.get("enabled", True):
            try:
                # Pass only the kernel instance, not the config dict directly
                self.agent_factory = AgentFactory(self)
                self.logger.info("Agent Factory initialized.")
                # Attempt to start the agent factory if a start method exists
                if hasattr(self.agent_factory, 'start') and callable(getattr(self.agent_factory, 'start')):
                    self.agent_factory.start()
            except Exception as e:
                self.logger.error(f"Failed to initialize Agent Factory: {e}", exc_info=True)
                self.agent_factory = None
        else:
            self.logger.warning("Agent Factory module is disabled in config.")
            self.agent_factory = None

        # LLM Orchestrator
        llm_config = module_configs.get("llm_orchestrator", {})
        if llm_config.get("enabled", True):
            try:
                # Pass only the kernel instance
                self.llm_orchestrator = LLMOrchestrator(self)
                self.logger.info("LLM Orchestrator initialized.")
                # Attempt to start if a start method exists
                if hasattr(self.llm_orchestrator, 'start') and callable(getattr(self.llm_orchestrator, 'start')):
                    self.llm_orchestrator.start()
            except Exception as e:
                self.logger.error(f"Failed to initialize LLM Orchestrator: {e}", exc_info=True)
                self.llm_orchestrator = None
        else:
            self.llm_orchestrator = None
            self.logger.warning("LLM Orchestrator is disabled.")
            
        # Tooling System
        tooling_config = module_configs.get("tooling_system", {})
        if tooling_config.get("enabled", True):
            try:
                # Pass only the kernel instance
                self.tooling_system = ToolingSystem(self)
                self.logger.info("Tooling System initialized.")
                # Attempt to start if a start method exists
                if hasattr(self.tooling_system, 'start') and callable(getattr(self.tooling_system, 'start')):
                    self.tooling_system.start()
            except Exception as e:
                self.logger.error(f"Failed to initialize Tooling System: {e}", exc_info=True)
                self.tooling_system = None
        else:
            self.tooling_system = None
            self.logger.warning("Tooling System is disabled.")

        # Memory Manager
        memory_config = module_configs.get("memory_manager", {})
        if memory_config.get("enabled", True):
            try:
                # Pass only the kernel instance
                self.memory_manager = MemoryManager(self)
                self.logger.info("Memory Manager initialized.")
                # Attempt to start if a start method exists
                if hasattr(self.memory_manager, 'start') and callable(getattr(self.memory_manager, 'start')):
                    self.memory_manager.start()
            except Exception as e:
                self.logger.error(f"Failed to initialize Memory Manager: {e}", exc_info=True)
                self.memory_manager = None
        else:
            self.memory_manager = None
            self.logger.warning("Memory Manager is disabled.")

        # Task Planner
        task_config = module_configs.get("task_planner", {})
        if task_config.get("enabled", True):
            try:
                # Pass only the kernel instance
                self.task_planner = TaskPlanner(self)
                self.logger.info("Task Planner initialized.")
                # Attempt to start if a start method exists
                if hasattr(self.task_planner, 'start') and callable(getattr(self.task_planner, 'start')):
                    self.task_planner.start()
            except Exception as e:
                self.logger.error(f"Failed to initialize Task Planner: {e}", exc_info=True)
                self.task_planner = None
        else:
            self.task_planner = None
            self.logger.warning("Task Planner is disabled.")

        # Swarm Coordinator (Depends on Message Bus)
        swarm_config = module_configs.get("swarm_coordinator", {})
        if swarm_config.get("enabled", False): # Default to disabled unless explicitly enabled
            if self.message_bus:
                try:
                    # Pass kernel and message_bus
                    self.swarm_coordinator = SwarmCoordinator(self, self.message_bus)
                    self.logger.info("Swarm Coordinator initialized.")
                    # Attempt to start if a start method exists
                    if hasattr(self.swarm_coordinator, 'start') and callable(getattr(self.swarm_coordinator, 'start')):
                        self.swarm_coordinator.start()
                except Exception as e:
                    self.logger.error(f"Failed to initialize Swarm Coordinator: {e}", exc_info=True)
                    self.swarm_coordinator = None
            else:
                self.logger.error("Cannot initialize Swarm Coordinator: Message Bus is not available.")
                self.swarm_coordinator = None
        else:
            self.swarm_coordinator = None
            self.logger.warning("Swarm Coordinator is disabled.")
            
        # HiTL Interface
        hitl_config = module_configs.get("hitl_interface", {})
        if hitl_config.get("enabled", True):
            try:
                # Pass only the kernel instance
                self.hitl_interface = HiTLInterface(self)
                self.logger.info("HiTL Interface initialized.")
                # Attempt to start if a start method exists
                if hasattr(self.hitl_interface, 'start') and callable(getattr(self.hitl_interface, 'start')):
                    self.hitl_interface.start()
            except Exception as e:
                self.logger.error(f"Failed to initialize HiTL Interface: {e}", exc_info=True)
                self.hitl_interface = None
        else:
            self.hitl_interface = None
            self.logger.warning("HiTL Interface is disabled.")

        # Self-Evolution Engine
        evo_config = module_configs.get("self_evolution_engine", {})
        if evo_config.get("enabled", True):
            try:
                # Pass only the kernel instance
                self.self_evolution_engine = SelfEvolutionEngine(self)
                self.logger.info("Self-Evolution Engine initialized.")
                # Attempt to start if a start method exists
                if hasattr(self.self_evolution_engine, 'start') and callable(getattr(self.self_evolution_engine, 'start')):
                    self.self_evolution_engine.start()
            except Exception as e:
                self.logger.error(f"Failed to initialize Self-Evolution Engine: {e}", exc_info=True)
                self.self_evolution_engine = None
        else:
            self.self_evolution_engine = None
            self.logger.warning("Self-Evolution Engine is disabled.")
            
        # Mission Scheduler
        mission_config = module_configs.get("mission_scheduler", {})
        if mission_config.get("enabled", True):
            try:
                # Pass only the kernel instance
                self.mission_scheduler = MissionScheduler(self)
                self.logger.info("Mission Scheduler initialized.")
                # Attempt to start if a start method exists
                if hasattr(self.mission_scheduler, 'start') and callable(getattr(self.mission_scheduler, 'start')):
                    self.mission_scheduler.start()
            except Exception as e:
                self.logger.error(f"Failed to initialize Mission Scheduler: {e}", exc_info=True)
                self.mission_scheduler = None
        else:
            self.mission_scheduler = None
            self.logger.warning("Mission Scheduler is disabled.")

        # Strategic Opportunity Observatory
        observatory_config = module_configs.get("strategic_observatory", {})
        if observatory_config.get("enabled", True):
            try:
                # Pass only the kernel instance
                self.strategic_observatory = StrategicOpportunityObservatory(self)
                self.logger.info("Strategic Opportunity Observatory initialized.")
                # Attempt to start if a start method exists
                if hasattr(self.strategic_observatory, 'start') and callable(getattr(self.strategic_observatory, 'start')):
                    self.strategic_observatory.start()
            except Exception as e:
                self.logger.error(f"Failed to initialize Strategic Opportunity Observatory: {e}", exc_info=True)
                self.strategic_observatory = None
        else:
            self.strategic_observatory = None
            self.logger.warning("Strategic Opportunity Observatory is disabled.")

        # Framework Adapter Manager
        framework_adapter_config = module_configs.get("framework_adapter_manager", {})
        if framework_adapter_config.get("enabled", True):
            try:
                # Determine the adapters directory path (relative to this file)
                adapters_dir_path = os.path.join(os.path.dirname(__file__), 'adapters')
                # Pass the adapters directory path and the specific config
                self.framework_adapter_manager = FrameworkAdapterManager(
                    adapters_dir=adapters_dir_path, 
                    config=framework_adapter_config
                )
                self.logger.info("Framework Adapter Manager initialized.")
                # Attempt to start if a start method exists
                if hasattr(self.framework_adapter_manager, 'start') and callable(getattr(self.framework_adapter_manager, 'start')):
                    self.framework_adapter_manager.start()
            except Exception as e:
                self.logger.error(f"Failed to initialize Framework Adapter Manager: {e}", exc_info=True)
                self.framework_adapter_manager = None
        else:
            self.framework_adapter_manager = None
            self.logger.warning("Framework Adapter Manager is disabled.")
            
        # --- Web UI Initialization (Moved to the end) ---
        webui_config = module_configs.get("web_ui_manager", {})
        if webui_config.get("enabled", True):
            try:
                # Pass only the kernel instance
                self.web_ui_manager = WebUI(self) # Assuming WebUI is the correct class name
                self.logger.info("Web UI Manager initialized.")
                # Note: Web UI is typically started explicitly via start_web_ui, not automatically here.
            except Exception as e:
                self.logger.error(f"Failed to initialize Web UI Manager: {e}", exc_info=True)
                self.web_ui_manager = None
        else:
            self.web_ui_manager = None
            self.logger.warning("Web UI Manager is disabled.")

        self.logger.info("All enabled modules initialized.")

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
        if self.swarm_coordinator is not None:
            try:
                self.swarm_coordinator.start()
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
        if hasattr(self, 'swarm_coordinator') and self.swarm_coordinator is not None:
            try:
                self.swarm_coordinator.stop()
                self.logger.info("Swarm module stopped")
            except Exception as e:
                self.logger.error(f"Failed to stop swarm module: {str(e)}")
        
        self.framework_adapter_manager.shutdown_all_adapters()
        self.tooling_system.stop()
        self.llm_orchestrator.stop()
        self.memory_manager.stop()
        
        # Stop the Web UI if it's running
        if hasattr(self, 'web_ui_manager') and self.web_ui_manager.is_running:
            self.web_ui_manager.stop()
        
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
                "swarm_module": "active" if self.swarm_coordinator is not None else "inactive"
            },
            "web_ui": {
                "running": self.web_ui_manager.is_running if hasattr(self, 'web_ui_manager') else False,
                "url": self.web_ui_manager.get_url() if hasattr(self, 'web_ui_manager') and self.web_ui_manager.is_running else None
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
        if not hasattr(self, 'web_ui_manager'):
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
        return self.web_ui_manager.start(host=host, port=port)
        
    def stop_web_ui(self) -> bool:
        """
        Stop the Web UI if it's running.
        
        Returns:
            True if stopped successfully, False otherwise
        """
        if not hasattr(self, 'web_ui_manager'):
            self.logger.error("Web UI not initialized")
            return False
            
        return self.web_ui_manager.stop()
        
    def get_web_ui_url(self) -> Optional[str]:
        """
        Get the URL for the Web UI.
        
        Returns:
            URL string for the Web UI if running, None otherwise
        """
        if not hasattr(self, 'web_ui_manager') or not self.web_ui_manager.is_running:
            return None
            
        return self.web_ui_manager.get_url()
    
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
        
        return None

# For backward compatibility with existing code that imports 'Kernel'
Kernel = EvoGenesisKernel
