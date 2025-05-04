"""
Message Bus - Core communication fabric for distributed EvoGenesis swarm.

This module provides producers and consumers for task distribution across 
multiple EvoGenesis instances. It abstracts the underlying message bus 
implementation (Redis Streams, NATS, Kafka, etc.).

Key Components:
- TaskProducer: Publishes tasks to the message bus
- TaskConsumer: Subscribes to and processes tasks from the message bus
- EventPublisher: Publishes events (memory updates, status changes) to the bus
- EventSubscriber: Subscribes to events from other components
"""

import json
import time
import uuid
import threading
import logging
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime

# Optional Redis dependency for Redis Streams implementation
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Optional NATS dependency for potential future implementation
try:
    import nats
    NATS_AVAILABLE = True
except ImportError:
    NATS_AVAILABLE = False


class BusImplementation(str, Enum):
    """Supported message bus implementations."""
    REDIS_STREAMS = "redis_streams"  # Using Redis Streams
    MEMORY = "memory"  # In-memory implementation for single-machine testing
    NATS = "nats"  # NATS implementation (future)
    KAFKA = "kafka"  # Kafka implementation (future)


class TaskStatus(str, Enum):
    """Status values for tasks in the swarm."""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRY = "retry"


class TaskSpec:
    """Specification for a distributed task in the swarm."""
    
    def __init__(self, 
                 task_id: Optional[str] = None,
                 name: str = "",
                 description: str = "",
                 parent_task_id: Optional[str] = None,
                 priority: int = 5,  # 1 (highest) to 10 (lowest)
                 dependencies: Optional[List[str]] = None,
                 required_capabilities: Optional[List[str]] = None,
                 memory_namespace: str = "default",
                 execution_context: Optional[Dict[str, Any]] = None,
                 timeout_seconds: int = 3600,
                 retries: int = 3,
                 created_by: str = "system"):
        """
        Initialize a task specification.
        
        Args:
            task_id: Unique ID for the task (generated if not provided)
            name: Human-readable name for the task
            description: Detailed description of the task
            parent_task_id: ID of parent task if this is a subtask
            priority: Priority level (1-10, 1 being highest)
            dependencies: List of task IDs that must complete before this one
            required_capabilities: Capabilities required by worker to process this task
            memory_namespace: Namespace in the global memory to use
            execution_context: Additional context for task execution
            timeout_seconds: Maximum allowed execution time
            retries: Maximum number of retry attempts
            created_by: ID of the agent/coordinator creating the task
        """
        self.task_id = task_id or f"task-{uuid.uuid4()}"
        self.name = name
        self.description = description
        self.parent_task_id = parent_task_id
        self.priority = priority
        self.dependencies = dependencies or []
        self.required_capabilities = required_capabilities or []
        self.memory_namespace = memory_namespace
        self.execution_context = execution_context or {}
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.created_by = created_by
        
        # Set by the system during processing
        self.status = TaskStatus.PENDING
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self.assigned_to = None
        self.started_at = None
        self.completed_at = None
        self.attempt = 0
        self.result = None
        self.error = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert TaskSpec to a dictionary."""
        return {
            "task_id": self.task_id,
            "name": self.name,
            "description": self.description,
            "parent_task_id": self.parent_task_id,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "required_capabilities": self.required_capabilities,
            "memory_namespace": self.memory_namespace,
            "execution_context": self.execution_context,
            "timeout_seconds": self.timeout_seconds,
            "retries": self.retries,
            "created_by": self.created_by,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "assigned_to": self.assigned_to,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "attempt": self.attempt,
            "result": self.result,
            "error": self.error
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskSpec':
        """Create a TaskSpec from a dictionary."""
        task = cls(
            task_id=data.get("task_id"),
            name=data.get("name", ""),
            description=data.get("description", ""),
            parent_task_id=data.get("parent_task_id"),
            priority=data.get("priority", 5),
            dependencies=data.get("dependencies", []),
            required_capabilities=data.get("required_capabilities", []),
            memory_namespace=data.get("memory_namespace", "default"),
            execution_context=data.get("execution_context", {}),
            timeout_seconds=data.get("timeout_seconds", 3600),
            retries=data.get("retries", 3),
            created_by=data.get("created_by", "system")
        )
        
        # Set status fields
        task.status = data.get("status", TaskStatus.PENDING)
        task.created_at = data.get("created_at", task.created_at)
        task.updated_at = data.get("updated_at", task.updated_at)
        task.assigned_to = data.get("assigned_to")
        task.started_at = data.get("started_at")
        task.completed_at = data.get("completed_at")
        task.attempt = data.get("attempt", 0)
        task.result = data.get("result")
        task.error = data.get("error")
        
        return task


class MessageBus(ABC):
    """Abstract base class for message bus implementations."""
    
    @abstractmethod
    def publish(self, stream: str, message: Dict[str, Any], **kwargs) -> str:
        """
        Publish a message to the specified stream.
        
        Args:
            stream: Name of the stream to publish to
            message: Message content as a dictionary
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            Message ID or confirmation
        """
        pass
    
    @abstractmethod
    def subscribe(self, stream: str, callback: Callable[[Dict[str, Any]], None], **kwargs) -> Any:
        """
        Subscribe to messages from the specified stream.
        
        Args:
            stream: Name of the stream to subscribe to
            callback: Function to call when a message is received
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            Subscription reference or ID
        """
        pass
    
    @abstractmethod
    def unsubscribe(self, subscription: Any) -> bool:
        """
        Unsubscribe from a previously created subscription.
        
        Args:
            subscription: Subscription reference or ID
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the connection to the message bus."""
        pass


class RedisStreamsBus(MessageBus):
    """Redis Streams implementation of the message bus."""
    
    def __init__(self, connection_params: Dict[str, Any], **kwargs):
        """
        Initialize Redis Streams message bus.
        
        Args:
            connection_params: Redis connection parameters
            **kwargs: Additional parameters
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis package is required for RedisStreamsBus. "
                             "Install the 'redis' dependency.")
        
        self.redis_client = redis.Redis(**connection_params)
        self.consumer_group = kwargs.get("consumer_group", "evogenesis-swarm")
        self.consumer_id = kwargs.get("consumer_id", f"consumer-{uuid.uuid4()}")
        self.subscription_threads = {}
        self.running = True
        self.logger = logging.getLogger(__name__)
    
    def publish(self, stream: str, message: Dict[str, Any], **kwargs) -> str:
        """Publish a message to a Redis stream."""
        # Convert any non-string values to JSON strings
        message_data = {}
        for key, value in message.items():
            if not isinstance(value, (str, bytes, int, float, bool)) or value is None:
                message_data[key] = json.dumps(value)
            else:
                message_data[key] = value
        
        # Add timestamp if not present
        if "timestamp" not in message_data:
            message_data["timestamp"] = time.time()
        
        # Publish to Redis stream
        message_id = self.redis_client.xadd(stream, message_data)
        return message_id.decode('utf-8') if isinstance(message_id, bytes) else message_id
    
    def subscribe(self, stream: str, callback: Callable[[Dict[str, Any]], None], **kwargs) -> str:
        """Subscribe to messages from a Redis stream."""
        # Create consumer group if it doesn't exist
        try:
            self.redis_client.xgroup_create(stream, self.consumer_group, id='0', mkstream=True)
            self.logger.info(f"Created consumer group {self.consumer_group} for stream {stream}")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP Consumer Group name already exists" not in str(e):
                raise
        
        # Generate a unique subscription ID
        subscription_id = str(uuid.uuid4())
        
        # Start a background thread to poll for messages
        def stream_listener():
            self.logger.info(f"Starting listener for stream {stream}, subscription {subscription_id}")
            last_id = '0'  # Start with all messages
            
            while self.running and subscription_id in self.subscription_threads:
                try:
                    # Read new messages from the stream
                    streams = {stream: last_id}
                    response = self.redis_client.xreadgroup(
                        groupname=self.consumer_group,
                        consumername=self.consumer_id,
                        streams=streams,
                        count=kwargs.get("batch_size", 10),
                        block=kwargs.get("block_ms", 1000)
                    )
                    
                    if response:
                        for stream_data in response:
                            stream_name, messages = stream_data
                            for message_id, data in messages:
                                # Process the message
                                message_dict = {}
                                for k, v in data.items():
                                    key = k.decode('utf-8') if isinstance(k, bytes) else k
                                    val = v.decode('utf-8') if isinstance(v, bytes) else v
                                    
                                    # Try to parse JSON for complex values
                                    if isinstance(val, str):
                                        try:
                                            message_dict[key] = json.loads(val)
                                        except json.JSONDecodeError:
                                            message_dict[key] = val
                                    else:
                                        message_dict[key] = val
                                
                                # Call the callback function with the message
                                try:
                                    callback(message_dict)
                                    # Acknowledge the message
                                    self.redis_client.xack(stream, self.consumer_group, message_id)
                                except Exception as e:
                                    self.logger.error(f"Error processing message {message_id}: {str(e)}")
                    
                except Exception as e:
                    self.logger.error(f"Error in Redis stream listener: {str(e)}")
                    time.sleep(1)  # Avoid tight error loop
        
        # Start the listener thread
        thread = threading.Thread(target=stream_listener, daemon=True)
        thread.start()
        self.subscription_threads[subscription_id] = thread
        
        return subscription_id
    
    def unsubscribe(self, subscription: str) -> bool:
        """Unsubscribe from a Redis stream."""
        if subscription in self.subscription_threads:
            del self.subscription_threads[subscription]
            return True
        return False
    
    def close(self) -> None:
        """Close the Redis connection and stop all listeners."""
        self.running = False
        time.sleep(2)  # Give time for threads to exit
        self.redis_client.close()


class InMemoryBus(MessageBus):
    """
    In-memory implementation of the message bus for testing or single-instance use.
    This is useful for development and testing without external dependencies.
    """
    
    def __init__(self, **kwargs):
        """Initialize in-memory message bus."""
        self.streams = {}  # Stream name -> list of messages
        self.subscribers = {}  # Stream name -> list of (subscription_id, callback) tuples
        self.lock = threading.RLock()
        self.running = True
        self.logger = logging.getLogger(__name__)
    
    def publish(self, stream: str, message: Dict[str, Any], **kwargs) -> str:
        """Publish a message to an in-memory stream."""
        with self.lock:
            if stream not in self.streams:
                self.streams[stream] = []
            
            # Generate a unique message ID
            message_id = f"{time.time()}-{uuid.uuid4()}"
            
            # Add timestamp if not present
            if "timestamp" not in message:
                message["timestamp"] = time.time()
            
            # Add the message to the stream
            self.streams[stream].append((message_id, message))
            
            # Notify subscribers
            if stream in self.subscribers:
                for _, callback in self.subscribers[stream]:
                    try:
                        threading.Thread(target=callback, args=(message,), daemon=True).start()
                    except Exception as e:
                        self.logger.error(f"Error notifying subscriber: {str(e)}")
            
            return message_id
    
    def subscribe(self, stream: str, callback: Callable[[Dict[str, Any]], None], **kwargs) -> str:
        """Subscribe to messages from an in-memory stream."""
        with self.lock:
            if stream not in self.subscribers:
                self.subscribers[stream] = []
            
            # Generate a unique subscription ID
            subscription_id = str(uuid.uuid4())
            
            # Add the subscriber
            self.subscribers[stream].append((subscription_id, callback))
            
            # If requested, replay existing messages
            if kwargs.get("replay_existing", False) and stream in self.streams:
                for _, message in self.streams[stream]:
                    try:
                        threading.Thread(target=callback, args=(message,), daemon=True).start()
                    except Exception as e:
                        self.logger.error(f"Error replaying message to subscriber: {str(e)}")
            
            return subscription_id
    
    def unsubscribe(self, subscription: str) -> bool:
        """Unsubscribe from an in-memory stream."""
        with self.lock:
            for stream, subscribers in self.subscribers.items():
                for i, (sub_id, _) in enumerate(subscribers):
                    if sub_id == subscription:
                        subscribers.pop(i)
                        return True
            return False
    
    def close(self) -> None:
        """Close the in-memory message bus."""
        with self.lock:
            self.running = False
            self.streams.clear()
            self.subscribers.clear()


class TaskProducer:
    """Publishes tasks to the message bus for distributed processing."""
    
    def __init__(self, bus: MessageBus, stream_prefix: str = "tasks"):
        """
        Initialize a task producer.
        
        Args:
            bus: The message bus implementation to use
            stream_prefix: Prefix for task streams (e.g., "tasks" -> "tasks.project_name")
        """
        self.bus = bus
        self.stream_prefix = stream_prefix
        self.logger = logging.getLogger(__name__)
    
    def publish_task(self, task: TaskSpec, project: str = "default") -> str:
        """
        Publish a task to the message bus.
        
        Args:
            task: The task specification to publish
            project: Project identifier used in the stream name
            
        Returns:
            Message ID from the message bus
        """
        stream = f"{self.stream_prefix}.{project}"
        self.logger.info(f"Publishing task {task.task_id} to stream {stream}")
        
        # Convert TaskSpec to a dictionary
        task_dict = task.to_dict()
        
        # Publish to the message bus
        message_id = self.bus.publish(stream, task_dict)
        
        # Also publish to the global task registry
        self.bus.publish("tasks.registry", {
            "task_id": task.task_id,
            "stream": stream,
            "project": project,
            "name": task.name,
            "priority": task.priority,
            "status": task.status,
            "created_at": task.created_at
        })
        
        return message_id
    
    def update_task_status(self, task_id: str, status: TaskStatus, 
                          worker_id: Optional[str] = None,
                          result: Optional[Dict[str, Any]] = None,
                          error: Optional[str] = None,
                          project: str = "default") -> str:
        """
        Update the status of a task.
        
        Args:
            task_id: ID of the task to update
            status: New status for the task
            worker_id: ID of the worker updating the status
            result: Task result data (for completed tasks)
            error: Error message (for failed tasks)
            project: Project identifier used in the stream name
            
        Returns:
            Message ID from the message bus
        """
        stream = f"task_updates.{project}"
        
        update = {
            "task_id": task_id,
            "status": status,
            "updated_at": datetime.now().isoformat(),
            "updated_by": worker_id or "system"
        }
        
        if status == TaskStatus.COMPLETED and result is not None:
            update["result"] = result
        
        if status == TaskStatus.FAILED and error is not None:
            update["error"] = error
        
        self.logger.info(f"Publishing task update for {task_id}: {status}")
        return self.bus.publish(stream, update)


class TaskConsumer:
    """Consumes tasks from the message bus for processing."""
    
    def __init__(self, bus: MessageBus, worker_id: str, stream_prefix: str = "tasks"):
        """
        Initialize a task consumer.
        
        Args:
            bus: The message bus implementation to use
            worker_id: Unique identifier for this worker
            stream_prefix: Prefix for task streams (e.g., "tasks" -> "tasks.project_name")
        """
        self.bus = bus
        self.worker_id = worker_id
        self.stream_prefix = stream_prefix
        self.subscriptions = []
        self.task_handlers = {}  # project -> callback
        self.capabilities = []
        self.logger = logging.getLogger(__name__)
        self.running = False
    
    def register_capabilities(self, capabilities: List[str]) -> None:
        """
        Register the capabilities of this worker.
        
        Args:
            capabilities: List of capability strings
        """
        self.capabilities = capabilities
    
    def subscribe_to_project(self, project: str, 
                            handler: Callable[[TaskSpec], None]) -> str:
        """
        Subscribe to tasks for a specific project.
        
        Args:
            project: Project identifier used in the stream name
            handler: Callback function to process tasks
            
        Returns:
            Subscription ID
        """
        stream = f"{self.stream_prefix}.{project}"
        self.task_handlers[project] = handler
        
        def message_handler(message: Dict[str, Any]) -> None:
            try:
                # Convert dictionary to TaskSpec
                task = TaskSpec.from_dict(message)
                
                # Check if we have the required capabilities
                if task.required_capabilities:
                    if not all(cap in self.capabilities for cap in task.required_capabilities):
                        self.logger.info(f"Skipping task {task.task_id} due to missing capabilities")
                        return
                
                # Check if the task is already assigned to another worker
                if task.status != TaskStatus.PENDING and task.assigned_to != self.worker_id:
                    self.logger.info(f"Skipping task {task.task_id}, assigned to {task.assigned_to}")
                    return
                
                # Process the task
                self.logger.info(f"Processing task {task.task_id}")
                task.status = TaskStatus.ASSIGNED
                task.assigned_to = self.worker_id
                task.updated_at = datetime.now().isoformat()
                
                # Call the task handler
                handler(task)
                
            except Exception as e:
                self.logger.error(f"Error processing task message: {str(e)}")
        
        # Subscribe to the task stream
        subscription_id = self.bus.subscribe(stream, message_handler)
        self.subscriptions.append(subscription_id)
        
        self.logger.info(f"Subscribed to project {project} tasks on stream {stream}")
        return subscription_id
    
    def start(self, projects: Optional[List[str]] = None) -> None:
        """
        Start consuming tasks.
        
        Args:
            projects: List of projects to subscribe to (if None, uses existing subscriptions)
        """
        if self.running:
            return
        
        self.running = True
        
        # Register with the swarm registry
        self._register_with_registry()
        
        # Start heartbeat thread
        self._start_heartbeat()
    
    def stop(self) -> None:
        """Stop consuming tasks and unsubscribe from all streams."""
        self.running = False
        
        # Unsubscribe from all streams
        for subscription_id in self.subscriptions:
            self.bus.unsubscribe(subscription_id)
        
        self.subscriptions = []
        self.logger.info(f"Worker {self.worker_id} stopped")
    
    def _register_with_registry(self) -> None:
        """Register this worker with the swarm registry."""
        self.bus.publish("swarm.registry", {
            "worker_id": self.worker_id,
            "capabilities": self.capabilities,
            "status": "active",
            "registered_at": datetime.now().isoformat(),
            "projects": list(self.task_handlers.keys())
        })
        self.logger.info(f"Worker {self.worker_id} registered with registry")
    
    def _start_heartbeat(self) -> None:
        """Start a background thread to send periodic heartbeats."""
        def heartbeat_sender():
            while self.running:
                try:
                    self.bus.publish("swarm.heartbeats", {
                        "worker_id": self.worker_id,
                        "timestamp": datetime.now().isoformat(),
                        "status": "active",
                        "load": 0.0  # TODO: Add actual load metrics
                    })
                    time.sleep(10)  # Send heartbeat every 10 seconds
                except Exception as e:
                    self.logger.error(f"Error sending heartbeat: {str(e)}")
                    time.sleep(5)
        
        thread = threading.Thread(target=heartbeat_sender, daemon=True)
        thread.start()
        self.logger.info(f"Started heartbeat sender for worker {self.worker_id}")


class EventPublisher:
    """Publishes events to the message bus."""
    
    def __init__(self, bus: MessageBus):
        """
        Initialize an event publisher.
        
        Args:
            bus: The message bus implementation to use
        """
        self.bus = bus
        self.logger = logging.getLogger(__name__)
    
    def publish_memory_event(self, 
                           namespace: str,
                           event_type: str,
                           data: Dict[str, Any],
                           origin_id: str = "system") -> str:
        """
        Publish a memory-related event.
        
        Args:
            namespace: Memory namespace
            event_type: Type of event (e.g., "added", "updated", "removed")
            data: Event data
            origin_id: ID of the component publishing the event
            
        Returns:
            Message ID from the message bus
        """
        stream = f"memory.events.{namespace}"
        
        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "namespace": namespace,
            "data": data,
            "origin_id": origin_id,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.debug(f"Publishing memory event: {event_type} in {namespace}")
        return self.bus.publish(stream, event)
    
    def publish_system_event(self,
                           event_type: str,
                           data: Dict[str, Any],
                           origin_id: str = "system") -> str:
        """
        Publish a system-wide event.
        
        Args:
            event_type: Type of event
            data: Event data
            origin_id: ID of the component publishing the event
            
        Returns:
            Message ID from the message bus
        """
        stream = "system.events"
        
        event = {
            "event_id": str(uuid.uuid4()),
            "event_type": event_type,
            "data": data,
            "origin_id": origin_id,
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.debug(f"Publishing system event: {event_type}")
        return self.bus.publish(stream, event)


class EventSubscriber:
    """Subscribes to events from the message bus."""
    
    def __init__(self, bus: MessageBus):
        """
        Initialize an event subscriber.
        
        Args:
            bus: The message bus implementation to use
        """
        self.bus = bus
        self.subscriptions = []
        self.logger = logging.getLogger(__name__)
    
    def subscribe_to_memory_events(self,
                                 namespace: str,
                                 callback: Callable[[Dict[str, Any]], None],
                                 event_types: Optional[List[str]] = None) -> str:
        """
        Subscribe to memory-related events.
        
        Args:
            namespace: Memory namespace
            callback: Function to call when an event is received
            event_types: Optional list of event types to filter by
            
        Returns:
            Subscription ID
        """
        stream = f"memory.events.{namespace}"
        
        def event_handler(event: Dict[str, Any]) -> None:
            # Filter by event type if specified
            if event_types and event.get("event_type") not in event_types:
                return
            
            # Call the callback
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Error in memory event handler: {str(e)}")
        
        # Subscribe to the event stream
        subscription_id = self.bus.subscribe(stream, event_handler)
        self.subscriptions.append(subscription_id)
        
        self.logger.info(f"Subscribed to memory events in namespace {namespace}")
        return subscription_id
    
    def subscribe_to_system_events(self,
                                 callback: Callable[[Dict[str, Any]], None],
                                 event_types: Optional[List[str]] = None) -> str:
        """
        Subscribe to system-wide events.
        
        Args:
            callback: Function to call when an event is received
            event_types: Optional list of event types to filter by
            
        Returns:
            Subscription ID
        """
        stream = "system.events"
        
        def event_handler(event: Dict[str, Any]) -> None:
            # Filter by event type if specified
            if event_types and event.get("event_type") not in event_types:
                return
            
            # Call the callback
            try:
                callback(event)
            except Exception as e:
                self.logger.error(f"Error in system event handler: {str(e)}")
        
        # Subscribe to the event stream
        subscription_id = self.bus.subscribe(stream, event_handler)
        self.subscriptions.append(subscription_id)
        
        self.logger.info(f"Subscribed to system events")
        return subscription_id
    
    def unsubscribe_all(self) -> None:
        """Unsubscribe from all event streams."""
        for subscription_id in self.subscriptions:
            self.bus.unsubscribe(subscription_id)
        
        self.subscriptions = []
        self.logger.info("Unsubscribed from all event streams")


# Factory function to create a message bus instance
def create_message_bus(implementation: BusImplementation, **kwargs) -> MessageBus:
    """
    Create a message bus instance for the specified implementation.
    
    Args:
        implementation: Message bus implementation to use
        **kwargs: Implementation-specific parameters
        
    Returns:
        Message bus instance
    """
    if implementation == BusImplementation.REDIS_STREAMS:
        # Extract Redis connection parameters
        connection_params = {
            "host": kwargs.get("host", "localhost"),
            "port": kwargs.get("port", 6379),
            "db": kwargs.get("db", 0),
        }
        
        # Add password if specified
        if "password" in kwargs:
            connection_params["password"] = kwargs["password"]
            
        # Add SSL parameters if specified
        if kwargs.get("ssl", False):
            connection_params["ssl"] = True
            connection_params["ssl_ca_certs"] = kwargs.get("ssl_ca_certs")
            connection_params["ssl_certfile"] = kwargs.get("ssl_certfile")
            connection_params["ssl_keyfile"] = kwargs.get("ssl_keyfile")
        
        return RedisStreamsBus(connection_params, **kwargs)
    
    elif implementation == BusImplementation.MEMORY:
        return InMemoryBus(**kwargs)
    
    elif implementation == BusImplementation.NATS:
        raise NotImplementedError("NATS implementation not yet available")
    
    elif implementation == BusImplementation.KAFKA:
        raise NotImplementedError("Kafka implementation not yet available")
    
    else:
        raise ValueError(f"Unknown message bus implementation: {implementation}")
