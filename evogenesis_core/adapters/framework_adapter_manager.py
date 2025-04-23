"""
Framework Adapter Manager - Manages the lifecycle of framework adapters.

This module provides a centralized management system for framework adapters,
handling discovery, initialization, and selection of appropriate adapters.
"""

import os
import importlib
import inspect
import logging
import json
import asyncio
from typing import Dict, Any, List, Optional, Union, Type, Set
import pkgutil
import re
from datetime import datetime, timedelta

from evogenesis_core.adapters.base_adapter import AgentExecutionAdapter
from evogenesis_core.adapters.adapter_factory import AdapterFactory


class FrameworkAdapterManager:
    """
    Manages the lifecycle of framework adapters in the EvoGenesis system.
    
    Responsible for:
    - Discovering available adapters
    - Initializing adapters with configuration
    - Selecting appropriate adapters for tasks
    - Monitoring adapter health
    - Updating adapters when frameworks change
    """
    
    def __init__(self, adapters_dir: str = None, config: Dict[str, Any] = None):
        """
        Initialize the Framework Adapter Manager.
        
        Args:
            adapters_dir: Directory containing adapter modules
            config: Configuration for adapters
        """
        self.adapters_dir = adapters_dir or os.path.dirname(__file__)
        self.config = config or {}
        self.adapter_factory = AdapterFactory(
            templates_dir=os.path.join(self.adapters_dir, "templates"),
            adapters_dir=self.adapters_dir
        )
        
        # Adapter registry
        self.available_adapters: Dict[str, Type[AgentExecutionAdapter]] = {}
        self.initialized_adapters: Dict[str, AgentExecutionAdapter] = {}
        self.adapter_health: Dict[str, Dict[str, Any]] = {}
        self.adapter_capabilities: Dict[str, Dict[str, Any]] = {}
        
        # Framework registry
        self.framework_registry: Dict[str, Dict[str, Any]] = {}
        
        # Adapter usage statistics
        self.adapter_usage: Dict[str, Dict[str, Any]] = {}
        
        # Last update check
        self.last_update_check = datetime.now() - timedelta(days=2)  # Force initial check
        
        # Load adapters
        self.discover_adapters()
    
    def discover_adapters(self) -> Dict[str, Type[AgentExecutionAdapter]]:
        """
        Discover available adapters in the adapters directory.
        
        Returns:
            Dictionary of adapter name -> adapter class
        """
        discovered = {}
        
        # Find all adapter modules
        adapter_modules = self._find_adapter_modules()
        
        for module_name in adapter_modules:
            try:
                # Import the module
                module = importlib.import_module(f"evogenesis_core.adapters.{module_name}")
                
                # Find adapter classes in the module
                for name, obj in inspect.getmembers(module):
                    # Look for classes that inherit from AgentExecutionAdapter
                    if (inspect.isclass(obj) and 
                        issubclass(obj, AgentExecutionAdapter) and 
                        obj != AgentExecutionAdapter):
                        
                        adapter_name = self._normalize_adapter_name(name)
                        discovered[adapter_name] = obj
                        logging.info(f"Discovered adapter: {adapter_name}")
            except ImportError as e:
                logging.warning(f"Could not import adapter module {module_name}: {str(e)}")
            except Exception as e:
                logging.error(f"Error discovering adapters in {module_name}: {str(e)}")
        
        self.available_adapters = discovered
        return discovered
    
    def initialize_adapter(self, adapter_name: str, adapter_config: Dict[str, Any] = None) -> Optional[AgentExecutionAdapter]:
        """
        Initialize an adapter with configuration.
        
        Args:
            adapter_name: Name of the adapter to initialize
            adapter_config: Configuration for the adapter
            
        Returns:
            Initialized adapter instance or None if initialization failed
        """
        if adapter_name not in self.available_adapters:
            logging.error(f"Adapter {adapter_name} not found")
            return None
        
        # Create adapter instance
        adapter_class = self.available_adapters[adapter_name]
        adapter = adapter_class()
        
        # Merge with global config
        config = self.config.copy()
        if adapter_config:
            # Only override adapter-specific settings
            if "adapters" in config and adapter_name in config["adapters"]:
                config["adapters"][adapter_name].update(adapter_config)
            else:
                config.setdefault("adapters", {})[adapter_name] = adapter_config
        
        # Initialize the adapter
        try:
            success = adapter.initialize(config)
            if success:
                self.initialized_adapters[adapter_name] = adapter
                
                # Update health status
                self.adapter_health[adapter_name] = {
                    "status": "healthy",
                    "last_check": datetime.now(),
                    "initialization_time": datetime.now(),
                    "error_count": 0
                }
                
                # Initialize usage stats
                self.adapter_usage[adapter_name] = {
                    "task_count": 0,
                    "success_count": 0,
                    "error_count": 0,
                    "last_used": None
                }
                
                logging.info(f"Successfully initialized adapter: {adapter_name}")
                # Fetch adapter capabilities synchronously using the sync wrapper
                try:
                    self.adapter_capabilities[adapter_name] = adapter.get_framework_capabilities()
                except Exception as e:
                    logging.warning(f"Could not fetch adapter capabilities: {str(e)}")
                return adapter
            else:
                logging.error(f"Failed to initialize adapter: {adapter_name}")
                return None
        except Exception as e:
            logging.error(f"Error initializing adapter {adapter_name}: {str(e)}")
            return None
    
    def initialize_all_adapters(self) -> Dict[str, AgentExecutionAdapter]:
        """
        Initialize all available adapters.
        
        Returns:
            Dictionary of initialized adapters
        """
        for adapter_name in self.available_adapters:
            # Check if this adapter has specific config
            adapter_config = None
            if "adapters" in self.config and adapter_name in self.config["adapters"]:
                adapter_config = self.config["adapters"][adapter_name]
            
            # Initialize the adapter if not already initialized
            if adapter_name not in self.initialized_adapters:
                self.initialize_adapter(adapter_name, adapter_config)
        
        return self.initialized_adapters
    
    def get_adapter(self, adapter_name: str) -> Optional[AgentExecutionAdapter]:
        """
        Get an initialized adapter by name.
        
        Args:
            adapter_name: Name of the adapter
            
        Returns:
            Adapter instance or None if not found/initialized
        """
        if adapter_name in self.initialized_adapters:
            return self.initialized_adapters[adapter_name]
        
        # Try to initialize it if available but not initialized
        if adapter_name in self.available_adapters:
            return self.initialize_adapter(adapter_name)
        
        return None
    
    def select_adapter_for_task(self, task_requirements: Dict[str, Any]) -> Optional[str]:
        """
        Select the most appropriate adapter for a task based on requirements.
        
        Args:
            task_requirements: Dictionary with task requirements
            
        Returns:
            Name of the selected adapter or None if no suitable adapter found
        """
        # Check if a specific adapter is requested
        if "adapter" in task_requirements:
            requested_adapter = task_requirements["adapter"]
            if requested_adapter in self.initialized_adapters:
                return requested_adapter
            # Try to initialize it if available
            if requested_adapter in self.available_adapters:
                if self.initialize_adapter(requested_adapter):
                    return requested_adapter
        
        # Check if a specific framework is requested
        if "framework" in task_requirements:
            requested_framework = task_requirements["framework"]
            # Find adapters for this framework
            for adapter_name, capabilities in self.adapter_capabilities.items():
                if capabilities.get("name", "").lower() == requested_framework.lower():
                    if adapter_name in self.initialized_adapters:
                        return adapter_name
                    # Try to initialize it
                    if self.initialize_adapter(adapter_name):
                        return adapter_name
        
        # Score adapters based on requirements
        candidates = []
        for adapter_name, adapter in self.initialized_adapters.items():
            # Skip unhealthy adapters
            if self.adapter_health.get(adapter_name, {}).get("status") != "healthy":
                continue
            
            # Get capabilities
            capabilities = self.adapter_capabilities.get(adapter_name, {})
            
            # Calculate match score
            score = self._calculate_adapter_match_score(adapter_name, capabilities, task_requirements)
            candidates.append((adapter_name, score))
        
        # Sort by score (higher is better)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Return the best match if any
        if candidates and candidates[0][1] > 0:
            return candidates[0][0]
        
        # No suitable adapter found
        return None
    
    def generate_adapter(self, framework_name: str, module_name: str = None) -> Optional[str]:
        """
        Generate a new adapter for a framework using the adapter factory.
        
        Args:
            framework_name: Name of the framework
            module_name: Optional module name if different from framework name
            
        Returns:
            Name of the generated adapter or None if generation failed
        """
        try:
            # Determine template to use
            template_name = "generic_framework_adapter"
            
            # Generate the adapter
            output_name = f"{framework_name.lower()}_adapter.py"
            adapter_path = self.adapter_factory.generate_adapter(
                template_name=template_name,
                framework_name=framework_name,
                module_name=module_name,
                output_name=output_name
            )
            
            # Reload adapters to include the new one
            self.discover_adapters()
            
            # Return the normalized adapter name
            return self._normalize_adapter_name(f"{framework_name}Adapter")
        except Exception as e:
            logging.error(f"Failed to generate adapter for {framework_name}: {str(e)}")
            return None
    
    def check_for_updates(self) -> Dict[str, Any]:
        """
        Check for updates to frameworks and adapters.
        
        Returns:
            Dictionary with update information
        """
        current_time = datetime.now()
        # Only check once per day
        if (current_time - self.last_update_check).days < 1:
            return {"status": "skipped", "last_check": self.last_update_check}
        
        self.last_update_check = current_time
        update_results = {
            "checked": [],
            "updates_needed": [],
            "errors": []
        }
        
        # Check each initialized adapter
        for adapter_name, adapter in self.initialized_adapters.items():
            try:
                # Get framework info from adapter
                capabilities = self.adapter_capabilities.get(adapter_name, {})
                framework_name = capabilities.get("name")
                
                if not framework_name:
                    continue
                
                update_results["checked"].append(adapter_name)
                
                # Find the adapter file
                adapter_class = self.available_adapters.get(adapter_name)
                if not adapter_class:
                    continue
                
                adapter_path = inspect.getfile(adapter_class)
                
                # Check for updates
                update_info = self.adapter_factory.adapt_to_framework_changes(
                    adapter_path=adapter_path,
                    framework_name=framework_name,
                    auto_fix=False
                )
                
                if update_info["status"] == "updated":
                    update_results["updates_needed"].append({
                        "adapter": adapter_name,
                        "framework": framework_name,
                        "differences": update_info["differences"],
                        "updated_path": update_info["updated_path"]
                    })
            except Exception as e:
                update_results["errors"].append({
                    "adapter": adapter_name,
                    "error": str(e)
                })
        
        return update_results
    
    def apply_updates(self, adapter_names: List[str] = None) -> Dict[str, Any]:
        """
        Apply updates to adapters.
        
        Args:
            adapter_names: Optional list of adapter names to update (all if None)
            
        Returns:
            Dictionary with update results
        """
        update_results = {
            "updated": [],
            "failed": []
        }
        
        # Determine which adapters to update
        to_update = adapter_names or list(self.initialized_adapters.keys())
        
        for adapter_name in to_update:
            try:
                adapter_class = self.available_adapters.get(adapter_name)
                if not adapter_class:
                    update_results["failed"].append({
                        "adapter": adapter_name,
                        "error": "Adapter not found"
                    })
                    continue
                
                adapter_path = inspect.getfile(adapter_class)
                
                # Get framework info
                capabilities = self.adapter_capabilities.get(adapter_name, {})
                framework_name = capabilities.get("name")
                
                if not framework_name:
                    update_results["failed"].append({
                        "adapter": adapter_name,
                        "error": "Could not determine framework name"
                    })
                    continue
                
                # Update the adapter
                update_info = self.adapter_factory.adapt_to_framework_changes(
                    adapter_path=adapter_path,
                    framework_name=framework_name,
                    auto_fix=True
                )
                
                if update_info["status"] == "auto_fixed":
                    update_results["updated"].append({
                        "adapter": adapter_name,
                        "framework": framework_name,
                        "differences": update_info["differences"]
                    })
                    
                    # Reload the adapter
                    if adapter_name in self.initialized_adapters:
                        del self.initialized_adapters[adapter_name]
                    
                    # Reload the module
                    module_name = adapter_class.__module__
                    importlib.reload(importlib.import_module(module_name))
                    
                    # Rediscover adapters
                    self.discover_adapters()
                    
                    # Reinitialize the adapter
                    self.initialize_adapter(adapter_name)
            except Exception as e:
                update_results["failed"].append({
                    "adapter": adapter_name,
                    "error": str(e)
                })
        
        return update_results
    
    def monitor_adapter_health(self) -> Dict[str, Dict[str, Any]]:
        """
        Check the health of all initialized adapters.
        
        Returns:
            Dictionary of adapter health information
        """
        for adapter_name, adapter in self.initialized_adapters.items():
            try:
                # Create a simple async task to test adapter responsiveness
                async def test_adapter():
                    try:
                        # Simple capability check
                        await adapter.get_framework_capabilities()
                        return True
                    except Exception as e:
                        logging.warning(f"Health check failed for adapter {adapter_name}: {str(e)}")
                        return False
                
                # Run the test
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(test_adapter())
                
                if result:
                    self.adapter_health[adapter_name] = {
                        "status": "healthy",
                        "last_check": datetime.now(),
                        "error_count": 0
                    }
                else:
                    # Increment error count
                    error_count = self.adapter_health.get(adapter_name, {}).get("error_count", 0) + 1
                    self.adapter_health[adapter_name] = {
                        "status": "unhealthy" if error_count > 3 else "warning",
                        "last_check": datetime.now(),
                        "error_count": error_count
                    }
            except Exception as e:
                logging.error(f"Error monitoring adapter {adapter_name}: {str(e)}")
                self.adapter_health[adapter_name] = {
                    "status": "error",
                    "last_check": datetime.now(),
                    "error": str(e)
                }
        
        return self.adapter_health
    
    def record_adapter_usage(self, adapter_name: str, success: bool) -> None:
        """
        Record usage statistics for an adapter.
        
        Args:
            adapter_name: Name of the adapter
            success: Whether the usage was successful
        """
        if adapter_name not in self.adapter_usage:
            self.adapter_usage[adapter_name] = {
                "task_count": 0,
                "success_count": 0,
                "error_count": 0,
                "last_used": None
            }
        
        stats = self.adapter_usage[adapter_name]
        stats["task_count"] += 1
        if success:
            stats["success_count"] += 1
        else:
            stats["error_count"] += 1
        stats["last_used"] = datetime.now()
    
    def shutdown_adapter(self, adapter_name: str) -> bool:
        """
        Shut down an adapter cleanly.
        
        Args:
            adapter_name: Name of the adapter to shut down
            
        Returns:
            True if successful, False otherwise
        """
        if adapter_name not in self.initialized_adapters:
            return False
        
        adapter = self.initialized_adapters[adapter_name]
        try:
            # Shutdown asynchronously and wait for result
            loop = asyncio.get_event_loop()
            success = loop.run_until_complete(adapter.shutdown())
            
            if success:
                del self.initialized_adapters[adapter_name]
                logging.info(f"Successfully shut down adapter: {adapter_name}")
            else:
                logging.warning(f"Adapter reported unsuccessful shutdown: {adapter_name}")
            
            return success
        except Exception as e:
            logging.error(f"Error shutting down adapter {adapter_name}: {str(e)}")
            # Remove it from initialized adapters anyway
            if adapter_name in self.initialized_adapters:
                del self.initialized_adapters[adapter_name]
            return False
    
    def shutdown_all_adapters(self) -> Dict[str, bool]:
        """
        Shut down all initialized adapters.
        
        Returns:
            Dictionary of adapter name -> shutdown success
        """
        results = {}
        adapters_to_shutdown = list(self.initialized_adapters.keys())
        
        for adapter_name in adapters_to_shutdown:
            results[adapter_name] = self.shutdown_adapter(adapter_name)
        
        return results
    
    async def _fetch_adapter_capabilities(self, adapter_name: str) -> Dict[str, Any]:
        """
        Fetch and store capabilities of an adapter.
        
        Args:
            adapter_name: Name of the adapter
            
        Returns:
            Dictionary of capabilities
        """
        if adapter_name not in self.initialized_adapters:
            return {}
        
        adapter = self.initialized_adapters[adapter_name]
        try:
            capabilities = await adapter.get_framework_capabilities()
            self.adapter_capabilities[adapter_name] = capabilities
            
            # Update framework registry
            framework_name = capabilities.get("name")
            if framework_name:
                self.framework_registry[framework_name.lower()] = {
                    "adapter": adapter_name,
                    "version": capabilities.get("version", "unknown"),
                    "capabilities": capabilities
                }
            
            return capabilities
        except Exception as e:
            logging.error(f"Error fetching capabilities for adapter {adapter_name}: {str(e)}")
            return {}
    
    def _find_adapter_modules(self) -> List[str]:
        """
        Find adapter modules in the adapters directory.
        
        Returns:
            List of module names
        """
        modules = []
        
        # Pattern to match adapter filenames
        pattern = re.compile(r'^(\w+)_adapter\.py$')
        
        # Find all Python files in adapters directory
        for filename in os.listdir(self.adapters_dir):
            if filename.endswith(".py"):
                match = pattern.match(filename)
                if match:
                    modules.append(match.group(0)[:-3])  # Remove .py extension
        
        return modules
    
    def _normalize_adapter_name(self, name: str) -> str:
        """
        Normalize adapter class name to a consistent format.
        
        Args:
            name: Original adapter name
            
        Returns:
            Normalized adapter name
        """
        # Remove 'Adapter' suffix if present
        if name.endswith("Adapter"):
            name = name[:-7]
        
        # Convert to lowercase
        return name.lower()
    
    def _calculate_adapter_match_score(self, adapter_name: str, 
                                     capabilities: Dict[str, Any], 
                                     requirements: Dict[str, Any]) -> float:
        """
        Calculate how well an adapter matches task requirements.
        
        Args:
            adapter_name: Name of the adapter
            capabilities: Adapter capabilities
            requirements: Task requirements
            
        Returns:
            Match score (higher is better)
        """
        score = 0.0
        
        # Check required features
        required_features = requirements.get("required_features", [])
        if required_features:
            features = capabilities.get("features", {})
            for feature in required_features:
                if features.get(feature):
                    score += 1.0
                else:
                    # Missing a required feature is a deal-breaker
                    return 0.0
        
        # Check preferred features
        preferred_features = requirements.get("preferred_features", [])
        if preferred_features:
            features = capabilities.get("features", {})
            for feature in preferred_features:
                if features.get(feature):
                    score += 0.5
        
        # Check model support if specified
        if "model" in requirements:
            model = requirements["model"]
            supported_models = capabilities.get("supported_models", [])
            if model in supported_models:
                score += 1.0
            elif any(m.startswith(model.split("-")[0]) for m in supported_models):
                # Partial match (e.g. gpt-4 vs gpt-4-1106-preview)
                score += 0.5
        
        # Consider adapter usage statistics (prefer more successful adapters)
        usage_stats = self.adapter_usage.get(adapter_name, {})
        task_count = usage_stats.get("task_count", 0)
        if task_count > 0:
            success_rate = usage_stats.get("success_count", 0) / task_count
            score += success_rate
        
        return score
