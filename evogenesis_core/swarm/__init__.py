# filepath: c:\dev\evoorg\evogenesis_core\swarm\__init__.py
"""
EvoGenesis Swarm - Distributed architecture for coordinated EvoGenesis instances.

This package enables EvoGenesis to operate as a distributed system with multiple
instances working together in a coordinated swarm. Key components include:

- Message Bus: Communication fabric for distributed task execution
- Coordinator: Manages task distribution and goal completion
- Worker: Executes distributed tasks using local capabilities
- Memory Mesh: Distributed knowledge sharing and vector database
"""

from evogenesis_core.swarm.bus import (
    BusImplementation, TaskStatus, TaskSpec,
    TaskProducer, TaskConsumer, EventPublisher, EventSubscriber,
    create_message_bus
)

from evogenesis_core.swarm.coordinator import (
    SwarmCoordinator, CoordinationStrategy, ConflictResolutionStrategy,
    SwarmConflictResolver
)

from evogenesis_core.swarm.worker import (
    WorkerRunner, WorkerMode, WorkerCapability, TaskExecutor, TaskResult,
    create_worker, run_worker_mode, add_worker_args
)

from evogenesis_core.swarm.memory import (
    VectorStoreType, MemoryAccessLevel, MemoryNamespaceConfig,
    VectorMemorySwarm, MemoryEventBroadcaster, MemoryEventListener,
    DistributedMemoryConnector, AuthToken, MemoryChangeEvent,
    create_memory_mesh
)

__all__ = [
    # Bus module
    'BusImplementation', 'TaskStatus', 'TaskSpec',
    'TaskProducer', 'TaskConsumer', 'EventPublisher', 'EventSubscriber',
    'create_message_bus',
    
    # Coordinator module
    'SwarmCoordinator', 'CoordinationStrategy', 'ConflictResolutionStrategy',
    'SwarmConflictResolver',
    
    # Worker module
    'WorkerRunner', 'WorkerMode', 'WorkerCapability', 'TaskExecutor', 'TaskResult',
    'create_worker', 'run_worker_mode', 'add_worker_args',
    
    # Memory module
    'VectorStoreType', 'MemoryAccessLevel', 'MemoryNamespaceConfig',
    'VectorMemorySwarm', 'MemoryEventBroadcaster', 'MemoryEventListener',
    'DistributedMemoryConnector', 'AuthToken', 'MemoryChangeEvent',
    'create_memory_mesh'
]
