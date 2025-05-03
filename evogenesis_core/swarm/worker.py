# filepath: c:\dev\evoorg\evogenesis_core\swarm\worker.py
"""
Worker Agent Runner - Enables any EvoGenesis instance to join the swarm as a worker.

This module provides the capability for an EvoGenesis instance to operate in
worker mode, subscribing to a task queue, executing tasks, and publishing
results back to the swarm.

Key Components:
- WorkerRunner: Main class for operating in worker mode
- TaskExecutor: Executes individual tasks within the worker
"""

import logging
import threading
import time
import uuid
import sys
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from datetime import datetime
import json
import asyncio
import os

# Import necessary EvoGenesis components
from evogenesis_core.swarm.bus import (
    MessageBus, TaskConsumer, TaskProducer, EventPublisher, EventSubscriber,
    TaskSpec, TaskStatus, create_message_bus, BusImplementation
)

# Import the memory mesh
from evogenesis_core.swarm.memory import (
    VectorMemorySwarm, DistributedMemoryConnector, AuthToken,
    MemoryAccessLevel, create_memory_mesh
)


class WorkerMode(str):
    """Operating modes for worker agents."""
    DEDICATED = "dedicated"  # Work exclusively for the swarm
    OPPORTUNISTIC = "opportunistic"  # Work when idle
    PROPORTIONAL = "proportional"  # Split resources between local and swarm tasks


class WorkerCapability:
    """Capabilities that a worker can advertise to the swarm."""
    # Task types
    RESEARCH = "research"
    CODE_GENERATION = "code_gen"
    REASONING = "reasoning"
    MATH = "math"
    SCIENCE = "science"
    DATA_ANALYSIS = "data_analysis"
    PLANNING = "planning"
    LANGUAGE_PROCESSING = "language_processing"
    
    # Domain-specific capabilities
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    LEGAL = "legal"
    BIOLOGY = "biology"
    CHEMISTRY = "chemistry"
    PHYSICS = "physics"
    
    # Resource-based capabilities
    GPU_COMPUTE = "gpu_compute"
    HIGH_MEMORY = "high_memory"
    HIGH_STORAGE = "high_storage"
    LOW_LATENCY = "low_latency"
    
    # Tool-based capabilities
    CODE_EXECUTION = "code_execution"
    WEB_ACCESS = "web_access"
    DATABASE_ACCESS = "database_access"
    FILE_SYSTEM_ACCESS = "file_system_access"
    
    @classmethod
    def from_kernel(cls, kernel) -> List[str]:
        """
        Derive capabilities from a kernel instance.
        
        Args:
            kernel: EvoGenesis kernel instance
            
        Returns:
            List of capability strings
        """
        capabilities = []
        
        # Check for base capabilities
        capabilities.append(cls.REASONING)
        capabilities.append(cls.PLANNING)
        capabilities.append(cls.LANGUAGE_PROCESSING)
        
        # Check for tool availability
        if hasattr(kernel, "tooling_system"):
            tools = kernel.tooling_system.get_tools() if hasattr(kernel.tooling_system, "get_tools") else []
            
            for tool in tools:
                tool_name = tool.name.lower() if hasattr(tool, "name") else ""
                
                # Map tool names to capabilities
                if "code" in tool_name or "programming" in tool_name:
                    capabilities.append(cls.CODE_GENERATION)
                    capabilities.append(cls.CODE_EXECUTION)
                
                if "web" in tool_name or "browser" in tool_name or "http" in tool_name:
                    capabilities.append(cls.WEB_ACCESS)
                
                if "database" in tool_name or "sql" in tool_name:
                    capabilities.append(cls.DATABASE_ACCESS)
                
                if "file" in tool_name or "disk" in tool_name:
                    capabilities.append(cls.FILE_SYSTEM_ACCESS)
                
                if "math" in tool_name or "calculator" in tool_name:
                    capabilities.append(cls.MATH)
                
                if "research" in tool_name or "search" in tool_name:
                    capabilities.append(cls.RESEARCH)
                
                if "data" in tool_name or "analysis" in tool_name:
                    capabilities.append(cls.DATA_ANALYSIS)
        
        # Check for GPU availability
        try:
            import torch
            if torch.cuda.is_available():
                capabilities.append(cls.GPU_COMPUTE)
        except ImportError:
            pass
        
        # Return unique capabilities
        return list(set(capabilities))


class TaskResult:
    """Represents the result of a task execution."""
    
    def __init__(self, 
                 task_id: str,
                 status: str,
                 result: Optional[Dict[str, Any]] = None,
                 error: Optional[str] = None,
                 execution_time: float = 0.0,
                 memory_entries: Optional[List[str]] = None,
                 worker_id: str = ""):
        """
        Initialize a task result.
        
        Args:
            task_id: ID of the executed task
            status: Execution status
            result: Task execution result
            error: Error message if execution failed
            execution_time: Time taken to execute the task in seconds
            memory_entries: IDs of memory entries created during execution
            worker_id: ID of the worker that executed the task
        """
        self.task_id = task_id
        self.status = status
        self.result = result
        self.error = error
        self.execution_time = execution_time
        self.memory_entries = memory_entries or []
        self.worker_id = worker_id
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_id": self.task_id,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "execution_time": self.execution_time,
            "memory_entries": self.memory_entries,
            "worker_id": self.worker_id,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskResult':
        """Create from dictionary."""
        return cls(
            task_id=data.get("task_id", ""),
            status=data.get("status", ""),
            result=data.get("result"),
            error=data.get("error"),
            execution_time=data.get("execution_time", 0.0),
            memory_entries=data.get("memory_entries", []),
            worker_id=data.get("worker_id", "")
        )


class TaskExecutor:
    """
    Executes individual tasks within a worker.
    
    This class is responsible for:
    - Executing a task using the EvoGenesis kernel
    - Managing task resources and timeouts
    - Recording and reporting results
    """
    
    def __init__(self, 
                 kernel,
                 task: TaskSpec,
                 memory_connector: Optional[DistributedMemoryConnector] = None):
        """
        Initialize a task executor.
        
        Args:
            kernel: EvoGenesis kernel instance
            task: Task specification
            memory_connector: Connection to the distributed memory mesh
        """
        self.kernel = kernel
        self.task = task
        self.memory_connector = memory_connector
        self.logger = logging.getLogger(__name__)
        
        # Task state
        self.start_time = None
        self.end_time = None
        self.status = TaskStatus.PENDING
        self.result = None
        self.error = None
        self.memory_entries = []
        
        # Set up logging context
        self.logger = logging.getLogger(f"{__name__}.task.{task.task_id}")
    
    async def execute(self) -> TaskResult:
        """
        Execute the task.
        
        Returns:
            TaskResult with execution results
        """
        self.start_time = time.time()
        self.status = TaskStatus.IN_PROGRESS
        self.logger.info(f"Executing task {self.task.task_id}: {self.task.name}")
        
        try:
            # Set up memory access if available
            memory_token = None
            if self.memory_connector:
                # Get a token for the task's memory namespace
                memory_token = self.memory_connector.get_auth_token(
                    agent_id=self.task.assigned_to,
                    namespaces=[self.task.memory_namespace],
                    access_level=MemoryAccessLevel.READ_WRITE,
                    ttl_seconds=self.task.timeout_seconds
                )
                
                # Set the token on the memory connector
                self.memory_connector.distributed_memory.auth_token = memory_token
            
            # Determine the task type and dispatch accordingly
            result = await self._dispatch_task()
            
            # Record success
            self.status = TaskStatus.COMPLETED
            self.result = result
            
            # Extract created memory entries if any
            if "memory_entries" in result:
                self.memory_entries = result["memory_entries"]
            
            self.logger.info(f"Task {self.task.task_id} completed successfully")
            
        except asyncio.TimeoutError:
            self.logger.error(f"Task {self.task.task_id} timed out")
            self.status = TaskStatus.FAILED
            self.error = "Task execution timed out"
            
        except Exception as e:
            self.logger.error(f"Error executing task {self.task.task_id}: {str(e)}")
            self.status = TaskStatus.FAILED
            self.error = str(e)
            
            # Log the full exception for debugging
            import traceback
            self.logger.debug(f"Task execution exception: {traceback.format_exc()}")
        
        finally:
            self.end_time = time.time()
            execution_time = self.end_time - self.start_time
            
            # Create and return the task result
            return TaskResult(
                task_id=self.task.task_id,
                status=self.status,
                result=self.result,
                error=self.error,
                execution_time=execution_time,
                memory_entries=self.memory_entries,
                worker_id=self.task.assigned_to
            )
    
    async def _dispatch_task(self) -> Dict[str, Any]:
        """
        Dispatch the task to the appropriate handler based on context.
        
        Returns:
            Task result dictionary
        """
        # Extract task context
        task_type = self.task.execution_context.get("task_type", "default")
        
        # Dispatch based on task type
        if task_type == "agent_execution":
            return await self._execute_agent_task()
        
        elif task_type == "llm_call":
            return await self._execute_llm_task()
        
        elif task_type == "tool_execution":
            return await self._execute_tool_task()
        
        else:
            # Default execution using task planner
            return await self._execute_default_task()
    
    async def _execute_agent_task(self) -> Dict[str, Any]:
        """Execute a task using a specific agent."""
        # Extract agent ID from context
        agent_id = self.task.execution_context.get("agent_id")
        
        if not agent_id:
            raise ValueError("No agent ID specified for agent execution task")
        
        # Get the agent from the agent manager
        agent = self.kernel.agent_manager.get_agent(agent_id)
        if not agent:
            raise ValueError(f"Agent {agent_id} not found")
        
        # Create a task for the agent to execute
        agent_task = {
            "type": "direct",
            "content": self.task.description,
            "task_id": self.task.task_id
        }
        
        # Execute the task using the agent
        result = await self.kernel.agent_manager.execute_agent_task(
            agent_id=agent_id,
            task=agent_task
        )
        
        return result
    
    async def _execute_llm_task(self) -> Dict[str, Any]:
        """Execute a task using the LLM directly."""
        # Extract LLM parameters from context
        model = self.task.execution_context.get("model")
        prompt = self.task.description
        system_prompt = self.task.execution_context.get("system_prompt", "")
        
        # Execute using the LLM orchestrator
        result = await self.kernel.llm_orchestrator.generate_text(
            prompt=prompt,
            system_prompt=system_prompt,
            model=model
        )
        
        return {"result": result}
    
    async def _execute_tool_task(self) -> Dict[str, Any]:
        """Execute a task using a specific tool."""
        # Extract tool info from context
        tool_name = self.task.execution_context.get("tool_name")
        tool_args = self.task.execution_context.get("tool_args", {})
        
        if not tool_name:
            raise ValueError("No tool name specified for tool execution task")
        
        # Execute the tool
        tool_result = await self.kernel.tooling_system.execute_tool(
            tool_name=tool_name,
            args=tool_args
        )
        
        return {"result": tool_result}
    
    async def _execute_default_task(self) -> Dict[str, Any]:
        """Execute a task using the task planner."""
        # Create an internal task in the task planner
        internal_task = {
            "name": self.task.name,
            "description": self.task.description,
            "priority": self.task.priority
        }
        
        # Execute the task
        task_id = self.kernel.task_planner.create_task(**internal_task)
        result = await self.kernel.task_planner.execute_task(task_id)
        
        # Store any memory entries created during execution
        memory_entries = []
        if hasattr(self.kernel.task_planner, "get_task_memory_entries"):
            memory_entries = self.kernel.task_planner.get_task_memory_entries(task_id)
        
        return {
            "result": result,
            "memory_entries": memory_entries
        }


class WorkerRunner:
    """
    Enables an EvoGenesis instance to operate as a worker in the swarm.
    
    This class manages the worker lifecycle, including:
    - Connecting to the message bus
    - Subscribing to task queues
    - Executing tasks using the local EvoGenesis kernel
    - Publishing results back to the swarm
    """
    
    def __init__(self, 
                 kernel,
                 message_bus: MessageBus,
                 worker_id: Optional[str] = None,
                 capabilities: Optional[List[str]] = None,
                 projects: Optional[List[str]] = None,
                 mode: WorkerMode = WorkerMode.DEDICATED,
                 max_concurrent_tasks: int = 1,
                 **kwargs):
        """
        Initialize a worker runner.
        
        Args:
            kernel: EvoGenesis kernel instance
            message_bus: Message bus for communication
            worker_id: Unique identifier for this worker
            capabilities: List of capabilities this worker provides
            projects: List of projects to subscribe to
            mode: Worker operating mode
            max_concurrent_tasks: Maximum number of concurrent tasks
            **kwargs: Additional parameters
        """
        self.kernel = kernel
        self.message_bus = message_bus
        self.worker_id = worker_id or f"worker-{uuid.uuid4()}"
        self.capabilities = capabilities or WorkerCapability.from_kernel(kernel)
        self.projects = projects or ["default"]
        self.mode = mode
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Task tracking
        self.active_tasks = {}  # task_id -> TaskSpec
        self.task_executors = {}  # task_id -> TaskExecutor
        self.task_results = {}  # task_id -> result
        
        # Resource management
        self.resource_usage = 0.0  # 0.0 to 1.0, represents CPU/memory usage
        self.task_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        # Create communication components
        self.task_consumer = TaskConsumer(message_bus, self.worker_id)
        self.task_producer = TaskProducer(message_bus)
        self.event_publisher = EventPublisher(message_bus)
        self.event_subscriber = EventSubscriber(message_bus)
        
        # Set up memory mesh connection
        self.memory_connector = None
        if kwargs.get("enable_memory_mesh", True):
            memory_config = kwargs.get("memory_config", {})
            self.memory_connector = DistributedMemoryConnector(
                local_memory_manager=kernel.memory_manager,
                message_bus=message_bus,
                config={
                    "instance_id": self.worker_id,
                    "namespaces": ["default"] + [f"goal_{p}" for p in self.projects],
                    "vector_memory": memory_config
                }
            )
        
        # Set up logger
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"WorkerRunner initialized with ID {self.worker_id}")
        
        # Runtime state
        self.running = False
        self.executor_task = None
        self.heartbeat_task = None
    
    def start(self) -> None:
        """Start the worker."""
        if self.running:
            return
        
        self.running = True
        
        # Start memory connector if available
        if self.memory_connector:
            self.memory_connector.start()
        
        # Set up task handler
        self.task_consumer.set_callback(self._handle_incoming_task)
        
        # Subscribe to relevant projects
        for project in self.projects:
            self.task_consumer.subscribe(project)
        
        # Start the task consumer
        self.task_consumer.start()
        
        # Start the executor task
        loop = asyncio.get_event_loop() if asyncio.get_event_loop().is_running() else None
        if not loop:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        self.executor_task = asyncio.ensure_future(self._task_executor_loop())
        
        # Start the heartbeat task
        self.heartbeat_task = asyncio.ensure_future(self._send_heartbeats())
        
        # Announce worker availability
        self._announce_availability()
        
        self.logger.info(f"Worker {self.worker_id} started")
    
    def stop(self) -> None:
        """Stop the worker."""
        if not self.running:
            return
        
        self.running = False
        
        # Stop the task consumer
        self.task_consumer.stop()
        
        # Cancel async tasks
        if self.executor_task:
            self.executor_task.cancel()
        
        if self.heartbeat_task:
            self.heartbeat_task.cancel()
        
        # Stop memory connector if available
        if self.memory_connector:
            self.memory_connector.stop()
        
        # Announce worker departure
        self._announce_departure()
        
        self.logger.info(f"Worker {self.worker_id} stopped")
    
    def _handle_incoming_task(self, task_spec_dict: Dict[str, Any]) -> None:
        """
        Handle an incoming task from the message bus.
        
        Args:
            task_spec_dict: Task specification dictionary
        """
        try:
            # Convert dictionary to TaskSpec
            task_spec = TaskSpec.from_dict(task_spec_dict)
            
            # Check if we're already at capacity
            current_tasks = len(self.active_tasks)
            if current_tasks >= self.max_concurrent_tasks:
                self.logger.warning(f"Worker at capacity ({current_tasks}/{self.max_concurrent_tasks}), rejecting task {task_spec.task_id}")
                return
            
            # Check if we have the required capabilities
            if task_spec.required_capabilities:
                missing_capabilities = [cap for cap in task_spec.required_capabilities if cap not in self.capabilities]
                if missing_capabilities:
                    self.logger.debug(f"Rejecting task {task_spec.task_id} due to missing capabilities: {missing_capabilities}")
                    return
            
            # Accept the task
            self.logger.info(f"Accepting task {task_spec.task_id}: {task_spec.name}")
            
            # Mark as assigned to this worker
            task_spec.status = TaskStatus.ASSIGNED
            task_spec.assigned_to = self.worker_id
            task_spec.updated_at = datetime.now().isoformat()
            
            # Store in active tasks
            self.active_tasks[task_spec.task_id] = task_spec
            
            # Publish acceptance
            self.event_publisher.publish_system_event(
                event_type="task_assigned",
                data={
                    "task_id": task_spec.task_id,
                    "worker_id": self.worker_id,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error handling incoming task: {str(e)}")
    
    async def _task_executor_loop(self) -> None:
        """Main loop for executing tasks."""
        try:
            while self.running:
                # Check if there are tasks to execute
                if not self.active_tasks:
                    await asyncio.sleep(1)
                    continue
                
                # Get tasks in ASSIGNED state
                assigned_tasks = {
                    task_id: task for task_id, task in self.active_tasks.items()
                    if task.status == TaskStatus.ASSIGNED
                }
                
                # Execute each assigned task
                for task_id, task in assigned_tasks.items():
                    # Skip if already being executed
                    if task_id in self.task_executors:
                        continue
                    
                    # Try to acquire a semaphore slot
                    if self.task_semaphore._value <= 0:
                        continue
                    
                    # Start task execution
                    asyncio.create_task(self._execute_task(task))
                
                # Sleep briefly
                await asyncio.sleep(0.1)
                
        except asyncio.CancelledError:
            self.logger.info("Task executor loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in task executor loop: {str(e)}")
    
    async def _execute_task(self, task: TaskSpec) -> None:
        """
        Execute a single task.
        
        Args:
            task: Task to execute
        """
        async with self.task_semaphore:
            try:
                # Update task status
                task.status = TaskStatus.IN_PROGRESS
                task.started_at = datetime.now().isoformat()
                
                # Create executor
                executor = TaskExecutor(
                    kernel=self.kernel,
                    task=task,
                    memory_connector=self.memory_connector
                )
                
                # Store executor
                self.task_executors[task.task_id] = executor
                
                # Publish status update
                self.event_publisher.publish_system_event(
                    event_type="task_update",
                    data={
                        "task_id": task.task_id,
                        "status": task.status,
                        "worker_id": self.worker_id,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                # Execute with timeout
                result = await asyncio.wait_for(
                    executor.execute(),
                    timeout=task.timeout_seconds
                )
                
                # Store the result
                self.task_results[task.task_id] = result
                
                # Update task status
                task.status = result.status
                task.completed_at = datetime.now().isoformat()
                
                if result.status == TaskStatus.COMPLETED:
                    task.result = result.result
                else:
                    task.error = result.error
                
                # Publish result
                self.event_publisher.publish_system_event(
                    event_type="task_update",
                    data={
                        "task_id": task.task_id,
                        "status": task.status,
                        "result": result.result if result.status == TaskStatus.COMPLETED else None,
                        "error": result.error if result.status == TaskStatus.FAILED else None,
                        "worker_id": self.worker_id,
                        "execution_time": result.execution_time,
                        "memory_entries": result.memory_entries,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                self.logger.info(
                    f"Task {task.task_id} completed with status {task.status} "
                    f"in {result.execution_time:.2f} seconds"
                )
                
            except asyncio.TimeoutError:
                # Task timed out
                task.status = TaskStatus.FAILED
                task.error = "Task execution timed out"
                
                # Publish timeout error
                self.event_publisher.publish_system_event(
                    event_type="task_update",
                    data={
                        "task_id": task.task_id,
                        "status": task.status,
                        "error": task.error,
                        "worker_id": self.worker_id,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                self.logger.warning(f"Task {task.task_id} timed out after {task.timeout_seconds} seconds")
                
            except Exception as e:
                # Task failed
                task.status = TaskStatus.FAILED
                task.error = str(e)
                
                # Publish failure
                self.event_publisher.publish_system_event(
                    event_type="task_update",
                    data={
                        "task_id": task.task_id,
                        "status": task.status,
                        "error": task.error,
                        "worker_id": self.worker_id,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                self.logger.error(f"Error executing task {task.task_id}: {str(e)}")
                
            finally:
                # Clean up
                if task.task_id in self.task_executors:
                    del self.task_executors[task.task_id]
                
                # Remove from active tasks
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
    
    async def _send_heartbeats(self) -> None:
        """Send periodic heartbeats to the swarm."""
        try:
            while self.running:
                # Publish heartbeat
                self.event_publisher.publish_system_event(
                    event_type="worker_heartbeat",
                    data={
                        "worker_id": self.worker_id,
                        "status": "active",
                        "active_tasks": len(self.active_tasks),
                        "resource_usage": self._get_resource_usage(),
                        "capabilities": self.capabilities,
                        "projects": self.projects,
                        "timestamp": datetime.now().isoformat()
                    }
                )
                
                # Wait for next heartbeat interval
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds
                
        except asyncio.CancelledError:
            self.logger.info("Heartbeat task cancelled")
        except Exception as e:
            self.logger.error(f"Error sending heartbeat: {str(e)}")
    
    def _get_resource_usage(self) -> Dict[str, float]:
        """
        Get current resource usage metrics.
        
        Returns:
            Dictionary with resource usage metrics
        """
        import psutil
        
        try:
            # Get CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Get disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            
            return {
                "cpu": cpu_percent / 100.0,
                "memory": memory_percent / 100.0,
                "disk": disk_percent / 100.0
            }
            
        except Exception as e:
            self.logger.error(f"Error getting resource usage: {str(e)}")
            return {
                "cpu": 0.5,  # Default to 50% as fallback
                "memory": 0.5,
                "disk": 0.5
            }
    
    def _announce_availability(self) -> None:
        """Announce worker availability to the swarm."""
        self.event_publisher.publish_system_event(
            event_type="worker_joined",
            data={
                "worker_id": self.worker_id,
                "capabilities": self.capabilities,
                "projects": self.projects,
                "max_concurrent_tasks": self.max_concurrent_tasks,
                "mode": self.mode,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def _announce_departure(self) -> None:
        """Announce worker departure from the swarm."""
        self.event_publisher.publish_system_event(
            event_type="worker_left",
            data={
                "worker_id": self.worker_id,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def update_capabilities(self, capabilities: List[str]) -> None:
        """
        Update worker capabilities.
        
        Args:
            capabilities: New list of capabilities
        """
        self.capabilities = capabilities
        
        # Announce capability update
        self.event_publisher.publish_system_event(
            event_type="worker_updated",
            data={
                "worker_id": self.worker_id,
                "capabilities": self.capabilities,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        self.logger.info(f"Updated worker capabilities: {len(capabilities)} capabilities")
    
    def add_projects(self, projects: List[str]) -> None:
        """
        Add projects to subscribe to.
        
        Args:
            projects: List of projects to add
        """
        # Subscribe to new projects
        for project in projects:
            if project not in self.projects:
                self.projects.append(project)
                self.task_consumer.subscribe(project)
        
        # Announce project update
        self.event_publisher.publish_system_event(
            event_type="worker_updated",
            data={
                "worker_id": self.worker_id,
                "projects": self.projects,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        self.logger.info(f"Added projects: {projects}")


def create_worker(kernel, config: Dict[str, Any]) -> WorkerRunner:
    """
    Factory function to create a worker runner.
    
    Args:
        kernel: EvoGenesis kernel instance
        config: Worker configuration
        
    Returns:
        WorkerRunner instance
    """
    # Extract configuration values
    worker_id = config.get("worker_id")
    capabilities = config.get("capabilities")
    projects = config.get("projects")
    mode = config.get("mode", WorkerMode.DEDICATED)
    max_concurrent_tasks = config.get("max_concurrent_tasks", 1)
    
    # Create message bus
    bus_impl = config.get("message_bus", {}).get("implementation", "redis_streams")
    bus_config = config.get("message_bus", {}).get("config", {})
    
    message_bus = create_message_bus(
        implementation=bus_impl,
        config=bus_config
    )
    
    # Create worker
    worker = WorkerRunner(
        kernel=kernel,
        message_bus=message_bus,
        worker_id=worker_id,
        capabilities=capabilities,
        projects=projects,
        mode=mode,
        max_concurrent_tasks=max_concurrent_tasks,
        enable_memory_mesh=config.get("enable_memory_mesh", True),
        memory_config=config.get("memory_mesh", {})
    )
    
    return worker


def run_worker_mode(kernel, args) -> None:
    """
    Run EvoGenesis in worker mode.
    
    Args:
        kernel: EvoGenesis kernel instance
        args: Command line arguments
    """
    import signal
    
    # Parse configuration from args or environment
    config = {
        "worker_id": args.worker_id,
        "capabilities": args.capabilities.split(",") if args.capabilities else None,
        "projects": args.projects.split(",") if args.projects else None,
        "mode": args.mode,
        "max_concurrent_tasks": args.max_concurrent_tasks,
        "message_bus": {
            "implementation": args.bus_impl,
            "config": {
                "redis_url": args.redis_url
            }
        },
        "enable_memory_mesh": not args.disable_memory_mesh,
        "memory_mesh": {
            "store_type": args.memory_store,
            "connection": {
                "url": args.memory_url
            }
        }
    }
    
    # Create worker
    worker = create_worker(kernel, config)
    
    # Set up signal handling
    def signal_handler(sig, frame):
        print(f"Received signal {sig}, shutting down worker...")
        worker.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start the worker
    worker.start()
    
    print(f"EvoGenesis worker started with ID: {worker.worker_id}")
    print(f"Capabilities: {worker.capabilities}")
    print(f"Subscribed projects: {worker.projects}")
    print(f"Press Ctrl+C to stop")
    
    # Keep the main thread alive
    try:
        # Run forever
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down worker...")
        worker.stop()


def add_worker_args(parser) -> None:
    """
    Add worker mode arguments to a parser.
    
    Args:
        parser: ArgumentParser to add arguments to
    """
    worker_group = parser.add_argument_group("Worker Mode Options")
    
    worker_group.add_argument(
        "--worker-id",
        type=str,
        help="Unique identifier for this worker (generated if not provided)"
    )
    
    worker_group.add_argument(
        "--capabilities",
        type=str,
        help="Comma-separated list of worker capabilities"
    )
    
    worker_group.add_argument(
        "--projects",
        type=str,
        default="default",
        help="Comma-separated list of projects to subscribe to"
    )
    
    worker_group.add_argument(
        "--mode",
        type=str,
        choices=["dedicated", "opportunistic", "proportional"],
        default="dedicated",
        help="Worker operating mode"
    )
    
    worker_group.add_argument(
        "--max-concurrent-tasks",
        type=int,
        default=1,
        help="Maximum number of concurrent tasks"
    )
    
    worker_group.add_argument(
        "--bus-impl",
        type=str,
        choices=["redis_streams", "memory", "nats", "kafka"],
        default="redis_streams",
        help="Message bus implementation to use"
    )
    
    worker_group.add_argument(
        "--redis-url",
        type=str,
        default="redis://localhost:6379",
        help="Redis URL for redis_streams bus implementation"
    )
    
    worker_group.add_argument(
        "--disable-memory-mesh",
        action="store_true",
        help="Disable connection to distributed memory mesh"
    )
    
    worker_group.add_argument(
        "--memory-store",
        type=str,
        choices=["weaviate", "pinecone", "milvus"],
        default="weaviate",
        help="Vector store type for distributed memory mesh"
    )
    
    worker_group.add_argument(
        "--memory-url",
        type=str,
        default="http://localhost:8080",
        help="URL for the vector store"
    )
