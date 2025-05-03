"""
Semantic Kernel Adapter - Implements adapter for Microsoft Semantic Kernel.

This adapter enables EvoGenesis to use Microsoft Semantic Kernel for agent execution,
mapping EvoGenesis concepts to Semantic Kernel plugins, functions, and execution patterns.
"""

import asyncio
import logging
from typing import Dict, Any
import uuid
import threading

# Conditionally import Semantic Kernel
try:
    import semantic_kernel as sk
    from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
    try:
        # Try the old location first (as of SK <= 1.27)
        from semantic_kernel.planning import StepwisePlanner  # type: ignore
    except ImportError:
        try:
            # Try the new location for SK >= 1.28
            from semantic_kernel.planners.stepwise_planner import StepwisePlanner  # type: ignore
        except ImportError:
            StepwisePlanner = None
    SEMANTIC_KERNEL_AVAILABLE = True
except ImportError:
    SEMANTIC_KERNEL_AVAILABLE = False

from evogenesis_core.adapters.base_adapter import AgentExecutionAdapter


def run_async(coro):
    """
    Helper function to run async coroutine in a sync context.
    Safely handles the event loop to avoid conflicts with existing loops.
    """
    if not asyncio.iscoroutine(coro):
        logging.error(f"Expected a coroutine object, but got {type(coro)}")
        if isinstance(coro, dict):
            return coro
        return {}
    try:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_running():
            # Run in a new thread to avoid deadlock
            result_container = {}
            def runner():
                try:
                    result_container['result'] = asyncio.run(coro)
                except Exception as e:
                    logging.error(f"Error in thread running async function: {str(e)}", exc_info=True)
                    result_container['result'] = {"name": "Semantic Kernel", "version": "unknown", "features": {}}
            t = threading.Thread(target=runner)
            t.start()
            t.join(timeout=120)
            if t.is_alive():
                logging.error("Async function timed out in thread.")
                return {"name": "Semantic Kernel", "version": "unknown", "features": {}}
            return result_container.get('result', {})
        else:
            return loop.run_until_complete(coro)
    except Exception as e:
        logging.error(f"Error running async function: {str(e)}", exc_info=True)
        return {"name": "Semantic Kernel", "version": "unknown", "features": {}}
        


class SemanticKernelAdapter(AgentExecutionAdapter):
    """
    Adapter for Microsoft Semantic Kernel.
    
    Maps EvoGenesis agents and tasks to Semantic Kernel concepts:
    - Agents → Semantic Kernel instances with specific plugins
    - Agent capabilities → Semantic Kernel plugins
    - Tasks → Semantic Kernel requests with planning
    """
    
    def __init__(self):
        """Initialize the Semantic Kernel adapter."""
        if not SEMANTIC_KERNEL_AVAILABLE:
            raise ImportError("Semantic Kernel is not available. Install with 'pip install semantic-kernel'")
        
        self.kernel_instances = {}  # agent_id -> sk.Kernel
        self.agent_configs = {}  # agent_id -> config
        self.active_tasks = {}  # task_id -> task_info
        self.agent_status = {}  # agent_id -> status
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """
        Initialize the Semantic Kernel adapter with configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.global_config = config
            logging.info("Semantic Kernel adapter initialized successfully")
            return True
        except Exception as e:
            logging.error(f"Failed to initialize Semantic Kernel adapter: {str(e)}")
            return False
    
    # Synchronous wrapper for create_agent
    def create_agent(self, agent_spec: Dict[str, Any]) -> str:
        """
        Create an agent using Semantic Kernel (synchronous version).
        
        Args:
            agent_spec: Specification of the agent to create
            
        Returns:
            Agent ID
        """
        return run_async(self._create_agent_async(agent_spec))
    
    async def _create_agent_async(self, agent_spec: Dict[str, Any]) -> str:
        """
        Create an agent using Semantic Kernel (async implementation).
        
        Args:
            agent_spec: Specification of the agent to create
            
        Returns:
            Agent ID
        """
        try:
            # Generate an ID for this agent
            agent_id = str(uuid.uuid4())
            
            # Create new Semantic Kernel instance
            kernel = sk.Kernel()
            
            # Configure LLM service
            llm_config = agent_spec.get("llm_config", {})
            model = llm_config.get("model_name", "gpt-4o")
            api_key = llm_config.get("api_key", self.global_config.get("openai_api_key"))
            
            # Add OpenAI chat completion service
            kernel.add_chat_service(
                "chat_completion", 
                OpenAIChatCompletion(model, api_key)
            )
              # Load any specified plugins
            if agent_spec.get("semantic_plugins"):
                # Load built-in plugins from the plugin directory
                plugin_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plugins")
                if os.path.exists(plugin_dir):
                    for plugin_folder in os.listdir(plugin_dir):
                        plugin_path = os.path.join(plugin_dir, plugin_folder)
                        if os.path.isdir(plugin_path):
                            try:
                                kernel.import_plugin_from_directory(plugin_path, plugin_folder)
                                self.logger.info(f"Loaded plugin {plugin_folder} from directory")
                            except Exception as e:
                                self.logger.error(f"Failed to load plugin {plugin_folder}: {str(e)}")
            
            # Load semantic plugins if specified with custom configuration
            for plugin_name, plugin_config in agent_spec.get("semantic_plugins", {}).items():
                try:
                    # Check if plugin config has a directory path
                    if "path" in plugin_config:
                        kernel.import_plugin_from_directory(plugin_config["path"], plugin_name)
                    # Check if plugin config has a Python module
                    elif "module" in plugin_config:
                        import importlib
                        module = importlib.import_module(plugin_config["module"])
                        kernel.import_plugin_from_object(module, plugin_name)
                    # Use semantic functions defined in the config
                    elif "functions" in plugin_config:
                        # Create a new plugin
                        plugin = kernel.create_plugin(plugin_name)
                        
                        # Add each function from the config
                        for func_name, func_def in plugin_config["functions"].items():
                            prompt = func_def.get("prompt", "")
                            system_prompt = func_def.get("system_prompt", "")
                            
                            # Create and register the function
                            plugin.add_semantic_function(prompt, func_name, system_prompt)
                            
                    self.logger.info(f"Loaded semantic plugin: {plugin_name}")
                except Exception as e:
                    self.logger.error(f"Failed to load semantic plugin {plugin_name}: {str(e)}")
            
            # Store the kernel instance
            self.kernel_instances[agent_id] = kernel
            self.agent_configs[agent_id] = agent_spec
            self.agent_status[agent_id] = {
                "status": "initialized",
                "tasks_completed": 0,
                "current_task": None,
                "last_active": asyncio.get_event_loop().time()
            }
            
            logging.info(f"Created Semantic Kernel agent {agent_id}")
            return agent_id
            
        except Exception as e:
            logging.error(f"Failed to create Semantic Kernel agent: {str(e)}")
            raise
    
    # Synchronous wrapper for run_agent_task
    def run_agent_task(self, agent_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task using Semantic Kernel (synchronous version).
        
        Args:
            agent_id: ID of the agent to use
            task: Task specification
            
        Returns:
            Task execution results
        """
        return run_async(self._run_agent_task_async(agent_id, task))
    
    async def _run_agent_task_async(self, agent_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task using Semantic Kernel (async implementation).
        
        Args:
            agent_id: ID of the agent to use
            task: Task specification
            
        Returns:
            Task execution results
        """
        if agent_id not in self.kernel_instances:
            raise ValueError(f"Agent {agent_id} not found")
        
        kernel = self.kernel_instances[agent_id]
        task_id = task.get("task_id", str(uuid.uuid4()))
        
        try:
            # Update agent status
            self.agent_status[agent_id]["status"] = "running"
            self.agent_status[agent_id]["current_task"] = task_id
            self.agent_status[agent_id]["last_active"] = asyncio.get_event_loop().time()
            
            # Store task information
            self.active_tasks[task_id] = {
                "agent_id": agent_id,
                "task": task,
                "status": "running",
                "start_time": asyncio.get_event_loop().time()
            }
            
            # Extract task information
            task_type = task.get("type", "general")
            goal = task.get("goal", "")
            context = task.get("context", {})
            
            # Create SK context
            sk_context = kernel.create_new_context()
            for key, value in context.items():
                sk_context[key] = value
            
            # Choose execution method based on task type
            if task_type == "planning":
                # Use StepwisePlanner for complex tasks
                planner = StepwisePlanner(kernel)
                plan = await planner.create_plan(goal)
                result = await plan.invoke_async(sk_context)
                output = str(result)
                
            elif task_type == "function_call":
                # Direct function call
                function_name = task.get("function")
                if "." in function_name:
                    plugin_name, function_name = function_name.split(".")
                    function = kernel.plugins[plugin_name][function_name]
                    result = await function.invoke_async(sk_context)
                    output = str(result)
                else:
                    raise ValueError(f"Invalid function name: {function_name}")
            
            else:
                # Simple completion
                result = await kernel.invoke_semantic_function_async(goal, sk_context)
                output = str(result)
            
            # Update task and agent status
            self.active_tasks[task_id]["status"] = "completed"
            self.active_tasks[task_id]["end_time"] = asyncio.get_event_loop().time()
            self.active_tasks[task_id]["result"] = output
            
            self.agent_status[agent_id]["status"] = "idle"
            self.agent_status[agent_id]["current_task"] = None
            self.agent_status[agent_id]["tasks_completed"] += 1
            self.agent_status[agent_id]["last_active"] = asyncio.get_event_loop().time()
            
            return {
                "task_id": task_id,
                "agent_id": agent_id,
                "status": "completed",
                "result": output,
                "execution_time": self.active_tasks[task_id]["end_time"] - self.active_tasks[task_id]["start_time"]
            }
            
        except Exception as e:
            # Update task and agent status on error
            if task_id in self.active_tasks:
                self.active_tasks[task_id]["status"] = "failed"
                self.active_tasks[task_id]["error"] = str(e)
                self.active_tasks[task_id]["end_time"] = asyncio.get_event_loop().time()
            
            self.agent_status[agent_id]["status"] = "error"
            self.agent_status[agent_id]["current_task"] = None
            self.agent_status[agent_id]["last_active"] = asyncio.get_event_loop().time()
            
            logging.error(f"Failed to execute task {task_id} with agent {agent_id}: {str(e)}")
            
            return {
                "task_id": task_id,
                "agent_id": agent_id,
                "status": "failed",
                "error": str(e)
            }
    
    # Synchronous wrapper for get_agent_status
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """
        Get the current status of a Semantic Kernel agent (synchronous version).
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Dictionary with agent status information
        """
        return run_async(self._get_agent_status_async(agent_id))
    
    async def _get_agent_status_async(self, agent_id: str) -> Dict[str, Any]:
        """
        Get the current status of a Semantic Kernel agent (async implementation).
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Dictionary with agent status information
        """
        if agent_id not in self.agent_status:
            raise ValueError(f"Agent {agent_id} not found")
        
        return self.agent_status[agent_id]
    
    # Synchronous wrapper for terminate_agent
    def terminate_agent(self, agent_id: str) -> bool:
        """
        Terminate a Semantic Kernel agent (synchronous version).
        
        Args:
            agent_id: ID of the agent to terminate
            
        Returns:
            True if successful, False otherwise
        """
        return run_async(self._terminate_agent_async(agent_id))
    
    async def _terminate_agent_async(self, agent_id: str) -> bool:
        """
        Terminate a Semantic Kernel agent (async implementation).
        
        Args:
            agent_id: ID of the agent to terminate
            
        Returns:
            True if successful, False otherwise
        """
        if agent_id not in self.kernel_instances:
            return False
        
        try:
            # Clean up the kernel instance
            del self.kernel_instances[agent_id]
            del self.agent_configs[agent_id]
            self.agent_status[agent_id] = {"status": "terminated"}
            
            return True
        except Exception as e:
            logging.error(f"Failed to terminate agent {agent_id}: {str(e)}")
            return False
    
    # Synchronous wrapper for pause_agent
    def pause_agent(self, agent_id: str) -> bool:
        """
        Pause a Semantic Kernel agent (synchronous version).
        
        Args:
            agent_id: ID of the agent to pause
            
        Returns:
            True if successful, False otherwise
        """
        return run_async(self._pause_agent_async(agent_id))
    
    async def _pause_agent_async(self, agent_id: str) -> bool:
        """
        Pause a Semantic Kernel agent (async implementation).
        
        Args:
            agent_id: ID of the agent to pause
            
        Returns:
            True if successful, False otherwise
        """
        if agent_id not in self.agent_status:
            return False
        
        # SK doesn't have direct pause/resume, but we can track status
        self.agent_status[agent_id]["status"] = "paused"
        return True
    
    # Synchronous wrapper for resume_agent
    def resume_agent(self, agent_id: str) -> bool:
        """
        Resume a paused Semantic Kernel agent (synchronous version).
        
        Args:
            agent_id: ID of the agent to resume
            
        Returns:
            True if successful, False otherwise
        """
        return run_async(self._resume_agent_async(agent_id))
    
    async def _resume_agent_async(self, agent_id: str) -> bool:
        """
        Resume a paused Semantic Kernel agent (async implementation).
        
        Args:
            agent_id: ID of the agent to resume
            
        Returns:
            True if successful, False otherwise
        """
        if agent_id not in self.agent_status:
            return False
        
        if self.agent_status[agent_id]["status"] == "paused":
            self.agent_status[agent_id]["status"] = "idle"
            return True
        
        return False
    
    # Synchronous wrapper for update_agent
    def update_agent(self, agent_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a Semantic Kernel agent's configuration (synchronous version).
        
        Args:
            agent_id: ID of the agent to update
            updates: Dictionary of updates to apply
            
        Returns:
            True if successful, False otherwise
        """
        return run_async(self._update_agent_async(agent_id, updates))
    
    async def _update_agent_async(self, agent_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a Semantic Kernel agent's configuration (async implementation).
        
        Args:
            agent_id: ID of the agent to update
            updates: Dictionary of updates to apply
            
        Returns:
            True if successful, False otherwise
        """
        if agent_id not in self.kernel_instances:
            return False
        
        try:
            # Update the agent configuration
            self.agent_configs[agent_id].update(updates)
            
            # Handle specific updates that need to be applied to the kernel
            if "llm_config" in updates:
                llm_config = updates["llm_config"]
                model = llm_config.get("model_name", "gpt-4o")
                api_key = llm_config.get("api_key", self.global_config.get("openai_api_key"))
                
                kernel = self.kernel_instances[agent_id]
                # Replace the chat service
                kernel.add_chat_service(
                    "chat_completion", 
                    OpenAIChatCompletion(model, api_key)
                )
            
            # Handle plugin updates separately
            if "add_plugins" in updates:
                kernel = self.kernel_instances[agent_id]
                for plugin_name, plugin_config in updates["add_plugins"].items():
                    # Implementation would depend on plugin type
                    logging.info(f"Adding plugin {plugin_name} to agent {agent_id}")
                    # Add actual plugin loading logic here
                    pass
            
            # Add other update handlers as needed (e.g., remove_plugins)
            
            logging.info(f"Agent {agent_id} updated successfully.")
            return True
            
        except Exception as e:
            logging.error(f"Failed to update agent {agent_id}: {str(e)}")
            return False
    
    # Synchronous wrapper for create_team
    def create_team(self, team_spec: Dict[str, Any]) -> str:
        """
        Create a team of Semantic Kernel agents (synchronous version).
        
        Args:
            team_spec: Specification of the team to create
            
        Returns:
            Team ID
        """
        return run_async(self._create_team_async(team_spec))
    
    async def _create_team_async(self, team_spec: Dict[str, Any]) -> str:
        """
        Create a team of Semantic Kernel agents (async implementation).
        
        Args:
            team_spec: Specification of the team to create
            
        Returns:
            Team ID
        """
        # SK doesn't have native team support, so we'd need to implement
        # custom coordination logic here
        team_id = str(uuid.uuid4())
        
        # In a full implementation, this would:
        # 1. Create individual agents for team members
        # 2. Set up communication channels between them        # 3. Create a coordination mechanism
        
        return team_id      # Synchronous wrapper for get_framework_capabilities
    def get_framework_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities of Semantic Kernel (synchronous version).
        
        Returns:
            Dictionary describing Semantic Kernel's capabilities
        """        # Use the async version through run_async
        return run_async(self._get_framework_capabilities_async())
    
    async def _get_framework_capabilities_async(self) -> Dict[str, Any]:
        """
        Get the capabilities of Semantic Kernel (async implementation).
        This method is kept for API compatibility but not used in the sync version.
        
        Returns:
            Dictionary describing Semantic Kernel's capabilities
        """
        try:
            # Use await to make this a proper coroutine
            await asyncio.sleep(0)
            
            # Return the capabilities dictionary
            return {
                "name": "Semantic Kernel",
                "version": sk.__version__ if hasattr(sk, "__version__") else "unknown",
                "features": {
                    "planning": True,
                    "function_calling": True,
                    "memory": True,
                    "native_plugins": True,
                    "semantic_plugins": True,
                    "team_coordination": False  # Not built into SK
                },
                "max_concurrent_agents": 100,  # Theoretical limit
                "supported_models": ["gpt-4", "gpt-4o", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet"]
            }
        except Exception as e:
            logging.error(f"Error getting framework capabilities: {str(e)}")
            # Return a minimal dictionary if there's an error
            return {
                "name": "Semantic Kernel",
                "version": "unknown",
                "features": {}
            }
            # Return a minimal dictionary if there's an error
            return {
                "name": "Semantic Kernel",
                "version": "unknown",
                "features": {}
            }
    
    # Synchronous wrapper for shutdown
    def shutdown(self) -> bool:
        """
        Shut down the Semantic Kernel adapter cleanly (synchronous version).
        
        Returns:
            True if successful, False otherwise
        """
        return run_async(self._shutdown_async())
    
    async def _shutdown_async(self) -> bool:
        """
        Shut down the Semantic Kernel adapter cleanly (async implementation).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Clean up all kernel instances
            self.kernel_instances.clear()
            self.agent_configs.clear()
            self.agent_status.clear()
            self.active_tasks.clear()
            
            return True
        except Exception as e:
            logging.error(f"Error shutting down Semantic Kernel adapter: {str(e)}")
            return False
