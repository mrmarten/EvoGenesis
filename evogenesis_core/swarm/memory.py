# filepath: c:\dev\evoorg\evogenesis_core\swarm\memory.py
"""
Distributed Memory Mesh - Extends EvoGenesis memory system for swarm intelligence.

This module enables distributed knowledge sharing across multiple EvoGenesis instances
by implementing a global memory adapter that connects to clustered vector databases
and broadcasts memory changes to other instances.

Key Components:
- VectorMemorySwarm: Vector store adapter for distributed memory
- MemoryEventBroadcaster: Sends memory change events to the message bus
- MemoryEventListener: Receives memory changes from other instances
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import json
import uuid
import time
import logging
import threading
from datetime import datetime, timedelta
from enum import Enum
import os
import numpy as np
import asyncio

# Import the base memory system
from evogenesis_core.modules.memory_manager import MemoryType, VectorMemoryBase

# Import the message bus
from evogenesis_core.swarm.bus import (
    EventPublisher, EventSubscriber, MessageBus
)

# Vectorstore dependencies - import conditionally
try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    import milvus
    from pymilvus import Collection, DataType, FieldSchema, CollectionSchema
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False


class VectorStoreType(str, Enum):
    """Supported distributed vector store backends."""
    WEAVIATE = "weaviate"
    PINECONE = "pinecone"
    MILVUS = "milvus"
    

class MemoryAccessLevel(str, Enum):
    """Access control levels for memory namespaces."""
    READ_WRITE = "read_write"  # Full access
    READ_ONLY = "read_only"    # Read-only access
    NONE = "none"              # No access
    

class MemoryNamespaceConfig:
    """Configuration for a memory namespace with access control."""
    
    def __init__(self, 
                 namespace: str,
                 description: str = "",
                 access_level: MemoryAccessLevel = MemoryAccessLevel.READ_WRITE,
                 ttl_days: Optional[int] = None,
                 encryption_enabled: bool = False,
                 authorized_teams: Optional[List[str]] = None,
                 authorized_agents: Optional[List[str]] = None,
                 tags: Optional[List[str]] = None):
        """
        Initialize a memory namespace configuration.
        
        Args:
            namespace: Namespace identifier
            description: Human-readable description
            access_level: Default access level for this namespace
            ttl_days: Time-to-live in days (None for permanent)
            encryption_enabled: Whether to encrypt data in this namespace
            authorized_teams: Teams that can access this namespace
            authorized_agents: Agents that can access this namespace
            tags: Tags for categorizing the namespace
        """
        self.namespace = namespace
        self.description = description
        self.access_level = access_level
        self.ttl_days = ttl_days
        self.encryption_enabled = encryption_enabled
        self.authorized_teams = authorized_teams or []
        self.authorized_agents = authorized_agents or []
        self.tags = tags or []
        self.created_at = datetime.now().isoformat()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "namespace": self.namespace,
            "description": self.description,
            "access_level": self.access_level,
            "ttl_days": self.ttl_days,
            "encryption_enabled": self.encryption_enabled,
            "authorized_teams": self.authorized_teams,
            "authorized_agents": self.authorized_agents,
            "tags": self.tags,
            "created_at": self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryNamespaceConfig':
        """Create from dictionary."""
        return cls(
            namespace=data.get("namespace", ""),
            description=data.get("description", ""),
            access_level=data.get("access_level", MemoryAccessLevel.READ_WRITE),
            ttl_days=data.get("ttl_days"),
            encryption_enabled=data.get("encryption_enabled", False),
            authorized_teams=data.get("authorized_teams", []),
            authorized_agents=data.get("authorized_agents", []),
            tags=data.get("tags", [])
        )


class MemoryChangeEvent:
    """Event representing a change to the shared memory."""
    
    def __init__(self, 
                 event_id: Optional[str] = None,
                 namespace: str = "default",
                 operation: str = "add",  # add, update, delete
                 memory_id: str = "",
                 memory_type: str = MemoryType.KNOWLEDGE,
                 content: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 agent_id: str = "",
                 team_id: Optional[str] = None,
                 timestamp: Optional[str] = None):
        """
        Initialize a memory change event.
        
        Args:
            event_id: Unique ID for this event
            namespace: Memory namespace
            operation: Operation type (add/update/delete)
            memory_id: ID of the affected memory entry
            memory_type: Type of memory (knowledge, experience, etc.)
            content: Memory content
            metadata: Additional metadata
            agent_id: ID of the agent making the change
            team_id: Optional team ID
            timestamp: Event timestamp
        """
        self.event_id = event_id or f"mem-event-{uuid.uuid4()}"
        self.namespace = namespace
        self.operation = operation
        self.memory_id = memory_id
        self.memory_type = memory_type
        self.content = content or {}
        self.metadata = metadata or {}
        self.agent_id = agent_id
        self.team_id = team_id
        self.timestamp = timestamp or datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "namespace": self.namespace,
            "operation": self.operation,
            "memory_id": self.memory_id,
            "memory_type": self.memory_type,
            "content": self.content,
            "metadata": self.metadata,
            "agent_id": self.agent_id,
            "team_id": self.team_id,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryChangeEvent':
        """Create from dictionary."""
        return cls(
            event_id=data.get("event_id"),
            namespace=data.get("namespace", "default"),
            operation=data.get("operation", "add"),
            memory_id=data.get("memory_id", ""),
            memory_type=data.get("memory_type", MemoryType.KNOWLEDGE),
            content=data.get("content", {}),
            metadata=data.get("metadata", {}),
            agent_id=data.get("agent_id", ""),
            team_id=data.get("team_id"),
            timestamp=data.get("timestamp")
        )


class AuthToken:
    """Represents an authentication token for memory access."""
    
    def __init__(self, 
                 token_id: Optional[str] = None,
                 subject: str = "",
                 namespace_access: Optional[Dict[str, MemoryAccessLevel]] = None,
                 expires_at: Optional[str] = None,
                 issuer: str = "system",
                 scopes: Optional[List[str]] = None):
        """
        Initialize an authentication token.
        
        Args:
            token_id: Unique token identifier
            subject: Subject (agent_id or team_id)
            namespace_access: Map of namespace->access_level
            expires_at: Expiration timestamp
            issuer: Token issuer
            scopes: Authorization scopes
        """
        self.token_id = token_id or f"token-{uuid.uuid4()}"
        self.subject = subject
        self.namespace_access = namespace_access or {"default": MemoryAccessLevel.READ_WRITE}
        self.expires_at = expires_at
        self.issuer = issuer
        self.scopes = scopes or ["memory:read", "memory:write"]
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "token_id": self.token_id,
            "subject": self.subject,
            "namespace_access": self.namespace_access,
            "expires_at": self.expires_at,
            "issuer": self.issuer,
            "scopes": self.scopes,
            "created_at": self.created_at
        }
    
    def can_access(self, namespace: str, required_level: MemoryAccessLevel) -> bool:
        """
        Check if token has sufficient access to a namespace.
        
        Args:
            namespace: The namespace to check
            required_level: The required access level
            
        Returns:
            True if access is granted, False otherwise
        """
        # Check token expiration
        if self.expires_at:
            if datetime.fromisoformat(self.expires_at) < datetime.now():
                return False
        
        # Check for global access
        if "*" in self.namespace_access:
            granted_level = self.namespace_access["*"]
            if granted_level == MemoryAccessLevel.READ_WRITE:
                return True
            if granted_level == MemoryAccessLevel.READ_ONLY and required_level == MemoryAccessLevel.READ_ONLY:
                return True
            return False
        
        # Check specific namespace access
        if namespace in self.namespace_access:
            granted_level = self.namespace_access[namespace]
            if granted_level == MemoryAccessLevel.READ_WRITE:
                return True
            if granted_level == MemoryAccessLevel.READ_ONLY and required_level == MemoryAccessLevel.READ_ONLY:
                return True
        
        # Default to no access
        return False


class MemoryEventBroadcaster:
    """Broadcasts memory changes to the message bus for other instances to consume."""
    
    def __init__(self, message_bus: MessageBus, namespace: str = "default"):
        """
        Initialize the memory event broadcaster.
        
        Args:
            message_bus: Message bus instance
            namespace: Default memory namespace
        """
        self.message_bus = message_bus
        self.namespace = namespace
        self.publisher = EventPublisher(message_bus)
        self.logger = logging.getLogger(__name__)
    
    def broadcast_change(self, event: MemoryChangeEvent) -> bool:
        """
        Broadcast a memory change event to the message bus.
        
        Args:
            event: The memory change event
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Publish to the memory events stream
            event_data = event.to_dict()
            stream_name = f"memory.{event.namespace}"
            
            self.publisher.publish_event(
                stream=stream_name,
                event_type=f"memory_{event.operation}",
                data=event_data
            )
            
            self.logger.debug(f"Broadcast memory {event.operation} event: {event.memory_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error broadcasting memory change: {str(e)}")
            return False


class MemoryEventListener:
    """Listens for memory changes from other instances and updates local cache."""
    
    def __init__(self, 
                 message_bus: MessageBus, 
                 memory_manager,
                 namespaces: Optional[List[str]] = None):
        """
        Initialize the memory event listener.
        
        Args:
            message_bus: Message bus instance
            memory_manager: The local memory manager to update
            namespaces: List of namespaces to subscribe to
        """
        self.message_bus = message_bus
        self.memory_manager = memory_manager
        self.namespaces = namespaces or ["default"]
        self.subscriber = EventSubscriber(message_bus)
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self.subscriptions = []
    
    def start(self) -> None:
        """Start listening for memory events."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Subscribe to each namespace
        for namespace in self.namespaces:
            stream_name = f"memory.{namespace}"
            
            # Register callback for memory events
            subscription = self.subscriber.subscribe_to_events(
                stream=stream_name,
                callback=self._handle_memory_event
            )
            
            self.subscriptions.append(subscription)
            self.logger.info(f"Subscribed to memory events for namespace: {namespace}")
    
    def stop(self) -> None:
        """Stop listening for memory events."""
        if not self.is_running:
            return
        
        # Unsubscribe from all streams
        for subscription in self.subscriptions:
            self.subscriber.unsubscribe(subscription)
        
        self.subscriptions = []
        self.is_running = False
        self.logger.info("Stopped listening for memory events")
    
    def _handle_memory_event(self, event_data: Dict[str, Any]) -> None:
        """
        Handle an incoming memory event.
        
        Args:
            event_data: Event data from the message bus
        """
        try:
            # Extract event details
            memory_event = MemoryChangeEvent.from_dict(event_data)
            
            # Skip if this is our own event
            if memory_event.metadata.get("origin_instance_id") == self.memory_manager.instance_id:
                return
            
            # Process based on operation type
            if memory_event.operation == "add" or memory_event.operation == "update":
                # Get the actual content and vector
                content = memory_event.content.get("content", "")
                metadata = memory_event.content.get("metadata", {})
                vector = memory_event.content.get("vector")
                
                # Add to local memory if vector is provided
                if vector:
                    self.memory_manager.add_memory(
                        content=content,
                        memory_type=memory_event.memory_type,
                        metadata=metadata,
                        namespace=memory_event.namespace,
                        memory_id=memory_event.memory_id,
                        skip_broadcast=True,  # Don't re-broadcast
                        vector=vector  # Use the provided vector
                    )
                    self.logger.debug(f"Added/updated memory from event: {memory_event.memory_id}")
                
            elif memory_event.operation == "delete":
                # Delete from local memory
                self.memory_manager.delete_memory(
                    memory_id=memory_event.memory_id,
                    namespace=memory_event.namespace,
                    skip_broadcast=True  # Don't re-broadcast
                )
                self.logger.debug(f"Deleted memory from event: {memory_event.memory_id}")
            
        except Exception as e:
            self.logger.error(f"Error processing memory event: {str(e)}")


class VectorMemorySwarm(VectorMemoryBase):
    """
    Distributed vector memory implementation that connects to clustered vector databases
    and synchronizes memory changes across the swarm.
    """
    
    def __init__(self, 
                 config: Dict[str, Any], 
                 message_bus: Optional[MessageBus] = None,
                 instance_id: Optional[str] = None):
        """
        Initialize the distributed vector memory.
        
        Args:
            config: Configuration dictionary
            message_bus: Message bus for event broadcasting
            instance_id: Unique ID for this instance
        """
        super().__init__()
        
        self.config = config
        self.instance_id = instance_id or f"instance-{uuid.uuid4()}"
        
        # Vector store configuration
        self.store_type = VectorStoreType(config.get("store_type", VectorStoreType.WEAVIATE))
        self.embedding_dim = config.get("embedding_dim", 1536)  # Default for OpenAI embeddings
        
        # Auth and security
        self.auth_token = config.get("auth_token")
        self.namespace_configs = {}
        
        # In-memory cache for faster retrieval
        self.memory_cache = {}  # namespace -> {memory_id -> content}
        self.max_cache_items = config.get("max_cache_items", 1000)
        
        # Set up the message bus if provided
        self.message_bus = message_bus
        if self.message_bus:
            self.broadcaster = MemoryEventBroadcaster(self.message_bus)
        else:
            self.broadcaster = None
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize the vector store connection
        self._initialize_vector_store()
        
        # Load namespace configurations
        self._load_namespace_configs()
        
        self.logger.info(f"Initialized VectorMemorySwarm with {self.store_type} backend")
    
    def _initialize_vector_store(self) -> None:
        """Initialize the connection to the vector store."""
        if self.store_type == VectorStoreType.WEAVIATE:
            self._initialize_weaviate()
        elif self.store_type == VectorStoreType.PINECONE:
            self._initialize_pinecone()
        elif self.store_type == VectorStoreType.MILVUS:
            self._initialize_milvus()
        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")
    
    def _initialize_weaviate(self) -> None:
        """Initialize Weaviate connection."""
        if not WEAVIATE_AVAILABLE:
            raise ImportError("Weaviate is not installed. Install the 'weaviate-client' dependency.")
        
        try:
            # Get connection details from config
            url = self.config.get("connection", {}).get("url", "http://localhost:8080")
            api_key = self.config.get("connection", {}).get("api_key")
            
            # Create client
            auth_config = None
            if api_key:
                auth_config = weaviate.auth.AuthApiKey(api_key)
            
            self.client = weaviate.Client(url=url, auth_client_secret=auth_config)
            
            # Create schema if it doesn't exist
            class_name = "Memory"
            if not self.client.schema.contains(class_name):
                class_obj = {
                    "class": class_name,
                    "description": "Memory entries for EvoGenesis",
                    "vectorizer": "none",  # We provide our own vectors
                    "properties": [
                        {
                            "name": "content",
                            "dataType": ["text"],
                            "description": "The memory content"
                        },
                        {
                            "name": "memory_type",
                            "dataType": ["string"],
                            "description": "Type of memory"
                        },
                        {
                            "name": "namespace",
                            "dataType": ["string"],
                            "description": "Memory namespace"
                        },
                        {
                            "name": "metadata",
                            "dataType": ["object"],
                            "description": "Additional metadata"
                        },
                        {
                            "name": "created_at",
                            "dataType": ["date"],
                            "description": "Creation timestamp"
                        }
                    ]
                }
                self.client.schema.create_class(class_obj)
            
            self.logger.info("Connected to Weaviate")
            
        except Exception as e:
            self.logger.error(f"Error connecting to Weaviate: {str(e)}")
            raise
    
    def _initialize_pinecone(self) -> None:
        """Initialize Pinecone connection."""
        if not PINECONE_AVAILABLE:
            # Suggest installing the correct package
            raise ImportError("Pinecone is not installed. Run 'pip install pinecone'")
        
        try:
            # Get connection details from config
            api_key = self.config.get("connection", {}).get("api_key")
            environment = self.config.get("connection", {}).get("environment", "us-west1-gcp")
            index_name = self.config.get("connection", {}).get("index", "evogenesis")
            
            # Initialize Pinecone
            pinecone.init(api_key=api_key, environment=environment)
            
            # Check if index exists, create it if not
            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index_name, 
                    dimension=self.embedding_dim,
                    metric="cosine"
                )
            
            # Connect to the index
            self.index = pinecone.Index(index_name)
            
            self.logger.info(f"Connected to Pinecone index: {index_name}")
            
        except Exception as e:
            self.logger.error(f"Error connecting to Pinecone: {str(e)}")
            raise
    
    def _initialize_milvus(self) -> None:
        """Initialize Milvus connection."""
        if not MILVUS_AVAILABLE:
            raise ImportError("Milvus is not installed. Run 'pip install pymilvus'")
        
        try:
            # Get connection details from config
            host = self.config.get("connection", {}).get("host", "localhost")
            port = self.config.get("connection", {}).get("port", 19530)
            collection_name = self.config.get("connection", {}).get("collection", "evogenesis_memory")
            
            # Connect to Milvus
            self.milvus_client = milvus.connections.connect(host=host, port=port)
            
            # Check if collection exists, create it if not
            if not milvus.utility.has_collection(collection_name):
                # Define fields for the collection
                fields = [
                    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=100),
                    FieldSchema(name="namespace", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="memory_type", dtype=DataType.VARCHAR, max_length=50),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="metadata", dtype=DataType.JSON),
                    FieldSchema(name="created_at", dtype=DataType.VARCHAR, max_length=30),
                    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim)
                ]
                
                # Create schema and collection
                schema = CollectionSchema(fields)
                self.collection = Collection(name=collection_name, schema=schema)
                
                # Create index for vector field
                index_params = {
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "params": {"M": 8, "efConstruction": 64}
                }
                self.collection.create_index(field_name="embedding", index_params=index_params)
            else:
                # Use existing collection
                self.collection = Collection(name=collection_name)
                self.collection.load()
                
            self.logger.info(f"Connected to Milvus collection: {collection_name}")
        except Exception as e:
            self.logger.error(f"Error connecting to Milvus: {str(e)}")
            raise
            
    def _load_namespace_configs(self) -> None:
        """Load namespace configurations from storage."""
        try:
            # Load configurations from database or file storage
            config_path = os.path.join(self.storage_path, "namespace_configs.json")
            
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    configs_data = json.load(f)
                
                # Deserialize the configs
                for namespace, config_data in configs_data.items():
                    namespace_config = MemoryNamespaceConfig(
                        namespace=namespace,
                        description=config_data.get("description", ""),
                        access_level=MemoryAccessLevel(config_data.get("access_level", MemoryAccessLevel.READ_WRITE.value)),
                        ttl_days=config_data.get("ttl_days")
                    )
                    self.namespace_configs[namespace] = namespace_config
                
                self.logger.info(f"Loaded {len(self.namespace_configs)} namespace configurations from storage")
                return
            
            # If no configs found, initialize with default namespaces
            default_config = MemoryNamespaceConfig(
                namespace="default",
                description="Default namespace for general memories",
                access_level=MemoryAccessLevel.READ_WRITE,
                ttl_days=None
            )
            
            system_config = MemoryNamespaceConfig(
                namespace="system",
                description="System-level memories and configurations",
                access_level=MemoryAccessLevel.READ_ONLY,
                authorized_agents=["system"],
                encryption_enabled=True
            )
            
            self.namespace_configs["default"] = default_config
            self.namespace_configs["system"] = system_config
            
            self.logger.debug(f"Loaded {len(self.namespace_configs)} namespace configurations")
            
        except Exception as e:
            self.logger.error(f"Error loading namespace configurations: {str(e)}")
            # Initialize with just the default namespace
            self.namespace_configs["default"] = MemoryNamespaceConfig(namespace="default")
    def add_namespace(self, config: MemoryNamespaceConfig) -> bool:
        """
        Add a new memory namespace configuration.
        
        Args:
            config: Namespace configuration
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate the namespace name
            if not config.namespace or config.namespace in ["", "*"] or not isinstance(config.namespace, str):
                self.logger.error(f"Invalid namespace name: {config.namespace}")
                return False
            
            # Check for existing namespace
            if config.namespace in self.namespace_configs:
                self.logger.warning(f"Namespace already exists: {config.namespace}")
                # Update existing namespace rather than failing
                self.namespace_configs[config.namespace] = config
                self.logger.info(f"Updated existing namespace configuration: {config.namespace}")
            else:
                # Add to configurations
                self.namespace_configs[config.namespace] = config
                self.logger.info(f"Added new namespace configuration: {config.namespace}")
            
            # Persist to storage
            self._save_namespace_configs()
            
            # Broadcast namespace change to swarm if message bus is available
            if hasattr(self, 'broadcaster') and self.broadcaster:
                try:
                    # Create a system event for namespace configuration change
                    event = MemoryChangeEvent(
                        namespace="system",
                        operation="update",
                        memory_id=f"namespace_config_{config.namespace}",
                        memory_type="system_config",
                        content={"namespace_config": config.to_dict()},
                        metadata={"event_type": "namespace_added"}
                    )
                    event.metadata["origin_instance_id"] = self.instance_id
                    self.broadcaster.broadcast_change(event)
                except Exception as broadcast_err:
                    self.logger.warning(f"Failed to broadcast namespace change: {str(broadcast_err)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding namespace configuration: {str(e)}", exc_info=True)
            return False
    
    def _save_namespace_configs(self) -> bool:
        """Save namespace configurations to persistent storage."""
        try:
            # Ensure storage path exists
            os.makedirs(self.storage_path, exist_ok=True)
            config_path = os.path.join(self.storage_path, "namespace_configs.json")
            
            # Convert configs to serializable dict
            configs_data = {}
            for namespace, config in self.namespace_configs.items():
                configs_data[namespace] = config.to_dict()
            
            # Write to file with atomic replacement
            temp_path = f"{config_path}.tmp"
            with open(temp_path, "w") as f:
                json.dump(configs_data, f, indent=2)
            
            # Atomic replace
            os.replace(temp_path, config_path)
            
            self.logger.debug(f"Saved {len(self.namespace_configs)} namespace configurations to {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save namespace configurations: {str(e)}", exc_info=True)
            return False
    
    def get_namespace_config(self, namespace: str) -> Optional[MemoryNamespaceConfig]:
        """
        Get configuration for a namespace.
        
        Args:
            namespace: Namespace identifier
            
        Returns:
            NamespaceConfig if found, None otherwise
        """
        return self.namespace_configs.get(namespace)
    
    def _check_access(self, namespace: str, required_level: MemoryAccessLevel) -> bool:
        """
        Check if current auth token has access to a namespace.
        
        Args:
            namespace: The namespace to check
            required_level: Required access level
            
        Returns:
            True if access is allowed, False otherwise
        """
        # Skip access check if no auth token is configured
        if not self.auth_token:
            return True
        
        return self.auth_token.can_access(namespace, required_level)
    
    def add_memory(self, 
                  content: str, 
                  memory_type: str = MemoryType.KNOWLEDGE,
                  metadata: Optional[Dict[str, Any]] = None, 
                  namespace: str = "default",
                  memory_id: Optional[str] = None,
                  vector: Optional[List[float]] = None,
                  agent_id: str = "system",
                  team_id: Optional[str] = None,
                  skip_broadcast: bool = False) -> Optional[str]:
        """
        Add a memory entry to the distributed vector store.
        
        Args:
            content: Memory content text
            memory_type: Type of memory
            metadata: Additional metadata
            namespace: Memory namespace
            memory_id: Optional memory ID (generated if not provided)
            vector: Optional embedding vector (computed if not provided)
            agent_id: ID of the agent adding the memory
            team_id: Optional team ID
            skip_broadcast: Whether to skip broadcasting the change
            
        Returns:
            Memory ID if successful, None otherwise
        """
        try:
            # Check access level
            if not self._check_access(namespace, MemoryAccessLevel.READ_WRITE):
                self.logger.warning(f"Access denied to add memory to namespace: {namespace}")
                return None
            
            # Generate memory ID if not provided
            if not memory_id:
                memory_id = f"mem-{str(uuid.uuid4())}"
            
            # Generate embedding if not provided
            if not vector:
                vector = self._generate_embedding(content)
            
            # Prepare metadata
            metadata = metadata or {}
            metadata["memory_type"] = memory_type
            metadata["agent_id"] = agent_id
            if team_id:
                metadata["team_id"] = team_id
            
            metadata["instance_id"] = self.instance_id
            metadata["created_at"] = datetime.now().isoformat()
            
            # Store in the appropriate vector store
            if self.store_type == VectorStoreType.WEAVIATE:
                self._add_to_weaviate(memory_id, content, namespace, metadata, vector)
            elif self.store_type == VectorStoreType.PINECONE:
                self._add_to_pinecone(memory_id, content, namespace, metadata, vector)
            elif self.store_type == VectorStoreType.MILVUS:
                self._add_to_milvus(memory_id, content, namespace, metadata, vector)
            
            # Add to local cache
            self._add_to_cache(memory_id, content, namespace, metadata)
            
            # Broadcast the change event if enabled
            if not skip_broadcast and self.broadcaster:
                event = MemoryChangeEvent(
                    namespace=namespace,
                    operation="add",
                    memory_id=memory_id,
                    memory_type=memory_type,
                    content={
                        "content": content,
                        "metadata": metadata,
                        "vector": vector
                    },
                    agent_id=agent_id,
                    team_id=team_id
                )
                
                # Add origin instance to avoid reprocessing our own events
                event.metadata["origin_instance_id"] = self.instance_id
                
                self.broadcaster.broadcast_change(event)
            
            self.logger.debug(f"Added memory {memory_id} to namespace {namespace}")
            return memory_id
            
        except Exception as e:
            self.logger.error(f"Error adding memory: {str(e)}")
            return None
    
    def _add_to_weaviate(self, 
                        memory_id: str, 
                        content: str, 
                        namespace: str, 
                        metadata: Dict[str, Any], 
                        vector: List[float]) -> None:
        """Add memory to Weaviate."""
        # Prepare data object
        data_object = {
            "content": content,
            "namespace": namespace,
            "memory_type": metadata.get("memory_type", MemoryType.KNOWLEDGE),
            "metadata": metadata,
            "created_at": metadata.get("created_at")
        }
        
        # Add with vector
        self.client.data_object.create(
            data_object=data_object,
            class_name="Memory",
            uuid=memory_id,
            vector=vector
        )
    
    def _add_to_pinecone(self, 
                        memory_id: str, 
                        content: str, 
                        namespace: str, 
                        metadata: Dict[str, Any], 
                        vector: List[float]) -> None:
        """Add memory to Pinecone."""
        # Add namespace to metadata
        metadata["namespace"] = namespace
        metadata["content"] = content
        
        # Create vector record
        self.index.upsert(
            vectors=[(memory_id, vector, metadata)],
            namespace=namespace
        )
    
    def _add_to_milvus(self, 
                      memory_id: str, 
                      content: str, 
                      namespace: str, 
                      metadata: Dict[str, Any], 
                      vector: List[float]) -> None:
        """Add memory to Milvus."""
        # Prepare data
        data = [
            [memory_id],  # id
            [namespace],  # namespace
            [metadata.get("memory_type", MemoryType.KNOWLEDGE)],  # memory_type
            [content],  # content
            [json.dumps(metadata)],  # metadata
            [metadata.get("created_at", datetime.now().isoformat())],  # created_at
            [vector]  # embedding
        ]
        
        # Insert the data
        self.collection.insert(data)
    
    def _add_to_cache(self, 
                     memory_id: str, 
                     content: str, 
                     namespace: str, 
                     metadata: Dict[str, Any]) -> None:
        """Add memory to local cache."""
        # Initialize namespace cache if needed
        if namespace not in self.memory_cache:
            self.memory_cache[namespace] = {}
        
        # Add to cache
        self.memory_cache[namespace][memory_id] = {
            "content": content,
            "metadata": metadata
        }
        
        # Check cache size and evict if necessary
        if len(self.memory_cache[namespace]) > self.max_cache_items:
            # Remove oldest item (simple LRU approximation)
            oldest_id = next(iter(self.memory_cache[namespace]))
            del self.memory_cache[namespace][oldest_id]
    
    def search_memory(self, 
                     query: str, 
                     namespace: str = "default",
                     limit: int = 5,
                     threshold: float = 0.7,
                     filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for memories similar to the query.
        
        Args:
            query: Search query
            namespace: Memory namespace
            limit: Maximum number of results
            threshold: Similarity threshold (0-1)
            filter_metadata: Optional metadata filters
            
        Returns:
            List of matching memories
        """
        try:            # Check access level
            if not self._check_access(namespace, MemoryAccessLevel.READ_ONLY):
                self.logger.warning(f"Access denied to search namespace: {namespace}")
                return []
            
            # Generate query embedding
            query_vector = self._generate_embedding(query)
            
            # Search in the appropriate vector store
            if self.store_type == VectorStoreType.WEAVIATE:
                return self._search_weaviate(query_vector, namespace, limit, threshold, filter_metadata)
            elif self.store_type == VectorStoreType.PINECONE:
                return self._search_pinecone(query_vector, namespace, limit, threshold, filter_metadata)
            elif self.store_type == VectorStoreType.MILVUS:
                return self._search_milvus(query_vector, namespace, limit, threshold, filter_metadata)
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error searching memory: {str(e)}")
            return []
    
    def _search_weaviate(self, 
                        query_vector: List[float], 
                        namespace: str, 
                        limit: int, 
                        threshold: float,
                        filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search in Weaviate."""
        # Build filter if needed
        where_filter = None
        if filter_metadata or namespace != "default":
            where_clauses = []
            
            # Always filter by namespace
            where_clauses.append({
                "path": ["namespace"],
                "operator": "Equal",
                "valueString": namespace
            })
            
            # Add other filters
            if filter_metadata:
                for key, value in filter_metadata.items():
                    if key == "memory_type":
                        where_clauses.append({
                            "path": ["memory_type"],
                            "operator": "Equal",
                            "valueString": value
                        })
                    else:
                        # For other metadata keys, we'd need custom handling
                        # For simplicity, we'll skip complex metadata filtering here
                        pass
            
            where_filter = {"operator": "And", "operands": where_clauses}
        
        # Perform the search
        result = self.client.query.get(
            class_name="Memory",
            properties=["content", "memory_type", "namespace", "metadata", "created_at"]
        ).with_near_vector({
            "vector": query_vector,
            "certainty": threshold
        }).with_where(where_filter).with_limit(limit).do()
        
        # Format results
        memories = []
        if "data" in result and "Get" in result["data"] and "Memory" in result["data"]["Get"]:
            for item in result["data"]["Get"]["Memory"]:
                memories.append({
                    "id": item["id"],
                    "content": item["content"],
                    "metadata": item["metadata"],
                    "similarity": item.get("_additional", {}).get("certainty", 0.0)
                })
        
        return memories
    
    def _search_pinecone(self, 
                        query_vector: List[float], 
                        namespace: str, 
                        limit: int, 
                        threshold: float,
                        filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search in Pinecone."""
        # Build filter if needed
        filter_dict = {}
        if filter_metadata:
            for key, value in filter_metadata.items():
                filter_dict[key] = {"$eq": value}
        
        # Perform the search
        result = self.index.query(
            vector=query_vector,
            namespace=namespace,
            top_k=limit,
            include_metadata=True,
            filter=filter_dict if filter_dict else None
        )
        
        # Format results
        memories = []
        for match in result.get("matches", []):
            if match["score"] >= threshold:
                metadata = match.get("metadata", {})
                content = metadata.pop("content", "")
                
                memories.append({
                    "id": match["id"],
                    "content": content,
                    "metadata": metadata,
                    "similarity": match["score"]
                })
        
        return memories
    
    def _search_milvus(self, 
                      query_vector: List[float], 
                      namespace: str, 
                      limit: int, 
                      threshold: float,
                      filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search in Milvus."""
        # Build filter if needed
        expr = f'namespace == "{namespace}"'
        if filter_metadata:
            for key, value in filter_metadata.items():
                if key == "memory_type":
                    expr += f' && memory_type == "{value}"'
        
        # Perform the search
        search_params = {"metric_type": "COSINE", "params": {"ef": 32}}
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param=search_params,
            limit=limit,
            expr=expr,
            output_fields=["content", "memory_type", "namespace", "metadata", "created_at"]
        )
        
        # Format results
        memories = []
        for hits in results:
            for hit in hits:
                if hit.score >= threshold:
                    metadata = json.loads(hit.entity.get("metadata", "{}"))
                    memories.append({
                        "id": hit.id,
                        "content": hit.entity.get("content", ""),
                        "metadata": metadata,
                        "similarity": hit.score
                    })
        
        return memories
    
    def delete_memory(self, 
                     memory_id: str, 
                     namespace: str = "default",
                     skip_broadcast: bool = False) -> bool:
        """
        Delete a memory entry.
        
        Args:
            memory_id: Memory ID to delete
            namespace: Memory namespace
            skip_broadcast: Whether to skip broadcasting the change
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check access level
            if not self._check_access(namespace, MemoryAccessLevel.READ_WRITE):
                self.logger.warning(f"Access denied to delete memory from namespace: {namespace}")
                return False
            
            # Delete from the appropriate vector store
            if self.store_type == VectorStoreType.WEAVIATE:
                self.client.data_object.delete(memory_id, class_name="Memory")
            elif self.store_type == VectorStoreType.PINECONE:
                self.index.delete(ids=[memory_id], namespace=namespace)
            elif self.store_type == VectorStoreType.MILVUS:
                self.collection.delete(f"id == '{memory_id}'")
            
            # Remove from cache
            if namespace in self.memory_cache and memory_id in self.memory_cache[namespace]:
                del self.memory_cache[namespace][memory_id]
            
            # Broadcast the change event if enabled
            if not skip_broadcast and self.broadcaster:
                event = MemoryChangeEvent(
                    namespace=namespace,
                    operation="delete",
                    memory_id=memory_id
                )
                
                # Add origin instance to avoid reprocessing our own events
                event.metadata["origin_instance_id"] = self.instance_id
                
                self.broadcaster.broadcast_change(event)
            
            self.logger.debug(f"Deleted memory {memory_id} from namespace {namespace}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting memory: {str(e)}")
            return False
    
    def get_memory(self, memory_id: str, namespace: str = "default") -> Optional[Dict[str, Any]]:
        """
        Get a specific memory by ID.
        
        Args:
            memory_id: Memory ID to retrieve
            namespace: Memory namespace
            
        Returns:
            Memory data if found, None otherwise
        """
        try:
            # Check access level
            if not self._check_access(namespace, MemoryAccessLevel.READ_ONLY):
                self.logger.warning(f"Access denied to read memory from namespace: {namespace}")
                return None
            
            # Check cache first
            if namespace in self.memory_cache and memory_id in self.memory_cache[namespace]:
                cache_item = self.memory_cache[namespace][memory_id]
                return {
                    "id": memory_id,
                    "content": cache_item["content"],
                    "metadata": cache_item["metadata"]
                }
            
            # Retrieve from the appropriate vector store
            if self.store_type == VectorStoreType.WEAVIATE:
                result = self.client.data_object.get_by_id(memory_id, class_name="Memory")
                if result:
                    return {
                        "id": memory_id,
                        "content": result["properties"]["content"],
                        "metadata": result["properties"]["metadata"]
                    }
            
            elif self.store_type == VectorStoreType.PINECONE:
                result = self.index.fetch(ids=[memory_id], namespace=namespace)
                if result and "vectors" in result and memory_id in result["vectors"]:
                    vector_data = result["vectors"][memory_id]
                    metadata = vector_data.get("metadata", {})
                    content = metadata.pop("content", "")
                    return {
                        "id": memory_id,
                        "content": content,
                        "metadata": metadata
                    }
            
            elif self.store_type == VectorStoreType.MILVUS:
                result = self.collection.query(
                    expr=f"id == '{memory_id}'",
                    output_fields=["content", "metadata", "namespace"]
                )
                if result:
                    metadata = json.loads(result[0]["metadata"])
                    return {
                        "id": memory_id,
                        "content": result[0]["content"],
                        "metadata": metadata
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting memory: {str(e)}")
            return None
    
    def subscribe_to_namespace(self, namespace: str) -> bool:
        """
        Subscribe to memory changes in a namespace.
        
        Args:
            namespace: Namespace to subscribe to
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check access level for the namespace
            if not self._check_access(namespace, MemoryAccessLevel.READ_ONLY):
                self.logger.warning(f"Access denied to subscribe to namespace: {namespace}")
                return False
            
            # Skip if we don't have a message bus or broadcaster
            if not hasattr(self, 'message_bus') or not self.message_bus:
                self.logger.warning(f"Cannot subscribe to namespace {namespace}: no message bus configured")
                return False
            
            # Create a subscriber if we don't have one yet
            if not hasattr(self, 'subscriber'):
                self.subscriber = EventSubscriber(self.message_bus)
            
            # Define callback to handle memory events
            def _handle_memory_event(event_data):
                try:
                    # Parse the event
                    memory_event = MemoryChangeEvent.from_dict(event_data)
                    
                    # Skip if this is our own event
                    if memory_event.metadata.get("origin_instance_id") == self.instance_id:
                        return
                    
                    # Process the event based on operation type
                    if memory_event.operation in ["add", "update"]:
                        # Extract content and metadata
                        content = memory_event.content.get("content", "")
                        metadata = memory_event.content.get("metadata", {})
                        vector = memory_event.content.get("vector")
                        
                        # Update local storage
                        if vector:
                            self._add_to_cache(
                                memory_id=memory_event.memory_id,
                                content=content,
                                namespace=memory_event.namespace,
                                metadata=metadata
                            )
                            
                    elif memory_event.operation == "delete":
                        # Remove from cache if present
                        if namespace in self.memory_cache and memory_event.memory_id in self.memory_cache[namespace]:
                            del self.memory_cache[namespace][memory_event.memory_id]
                    
                except Exception as e:
                    self.logger.error(f"Error handling memory event: {str(e)}")
            
            # Subscribe to the stream
            stream_name = f"memory.{namespace}"
            subscription = self.subscriber.subscribe_to_events(
                stream=stream_name,
                callback=_handle_memory_event
            )
            
            # Store the subscription
            if not hasattr(self, 'subscriptions'):
                self.subscriptions = {}
            
            self.subscriptions[namespace] = subscription
            
            self.logger.info(f"Subscribed to memory events for namespace: {namespace}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error subscribing to namespace {namespace}: {str(e)}")
            return False

    def unsubscribe_from_namespace(self, namespace: str) -> bool:
        """
        Unsubscribe from memory changes in a namespace.
        
        Args:
            namespace: Namespace to unsubscribe from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Skip if we don't have subscriptions
            if not hasattr(self, 'subscriptions') or namespace not in self.subscriptions:
                self.logger.warning(f"No active subscription for namespace: {namespace}")
                return False
            
            # Get the subscription
            subscription = self.subscriptions[namespace]
            
            # Unsubscribe
            if hasattr(self, 'subscriber'):
                self.subscriber.unsubscribe(subscription)
            
            # Remove from our subscription list
            del self.subscriptions[namespace]
            
            self.logger.info(f"Unsubscribed from memory events for namespace: {namespace}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error unsubscribing from namespace {namespace}: {str(e)}")
            return False
            
    def get_subscribed_namespaces(self) -> List[str]:
        """
        Get the list of namespaces currently subscribed to.
        
        Returns:
            List of namespace names
        """
        if hasattr(self, 'subscriptions'):
            return list(self.subscriptions.keys())
        return []
        
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding vector for the text using the configured embedding model.
        
        Uses the LLM orchestrator to access different embedding models based on 
        configured providers (OpenAI, Azure, etc.)
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            # Check if we have a configuration for embeddings
            embedding_config = self.config.get("embeddings", {})
            provider = embedding_config.get("provider", "openai")
            model = embedding_config.get("model", "text-embedding-3-small")
            
            # Use OpenAI embeddings (default)
            if provider.lower() == "openai":
                return self._get_openai_embedding(text, model)
                
            # Use Azure OpenAI embeddings
            elif provider.lower() == "azure_openai":
                return self._get_azure_embedding(text, embedding_config)
                
            # Use local embeddings if configured
            elif provider.lower() == "local":
                return self._get_local_embedding(text, embedding_config)
                
            # Fallback to a simple embedding method if all else fails
            self.logger.warning(f"Unsupported embedding provider: {provider}, using fallback method")
            return self._get_fallback_embedding(text)
            
        except Exception as e:
            self.logger.error(f"Error generating embedding: {str(e)}")
            # Fallback to simple embedding in case of error
            return self._get_fallback_embedding(text)
    
    def _get_openai_embedding(self, text: str, model: str = "text-embedding-3-small") -> List[float]:
        """Get embeddings using OpenAI's API"""
        try:
            from openai import OpenAI
            
            # Get API key from configuration
            api_key = None
            try:
                # Try to use the existing API key manager if available
                from evogenesis_core.modules.llm_orchestrator import APIKeyManager, ModelProvider
                key_manager = APIKeyManager()
                api_key = key_manager.get_api_key(ModelProvider.OPENAI)
            except Exception:
                # Fall back to config
                api_key = self.config.get("connection", {}).get("openai_api_key")
            
            if not api_key:
                raise ValueError("No OpenAI API key found")
            
            client = OpenAI(api_key=api_key)
            
            # Get the embedding from OpenAI
            response = client.embeddings.create(
                model=model,
                input=text,
                encoding_format="float"
            )
            
            # Return the embedding
            return response.data[0].embedding
            
        except Exception as e:
            self.logger.error(f"Error getting OpenAI embedding: {str(e)}")
            raise
    
    def _get_azure_embedding(self, text: str, config: Dict[str, Any]) -> List[float]:
        """Get embeddings using Azure OpenAI"""
        try:
            from openai import AzureOpenAI
            
            # Get Azure configuration
            api_key = config.get("api_key")
            azure_endpoint = config.get("azure_endpoint")
            deployment = config.get("deployment", "text-embedding-ada-002")
            api_version = config.get("api_version", "2023-05-15")
            
            # Try to use the existing API key manager if available
            if not api_key:
                try:
                    from evogenesis_core.modules.llm_orchestrator import APIKeyManager, ModelProvider
                    key_manager = APIKeyManager()
                    api_key_data = key_manager.get_api_key(ModelProvider.AZURE_OPENAI)
                    
                    # Parse the API key data
                    if isinstance(api_key_data, str):
                        try:
                            api_key_config = json.loads(api_key_data)
                            api_key = api_key_config.get("api_key")
                            azure_endpoint = api_key_config.get("azure_endpoint") or azure_endpoint
                            api_version = api_key_config.get("api_version") or api_version
                        except json.JSONDecodeError:
                            # If it's not JSON, just use as-is
                            api_key = api_key_data
                    else:
                        api_key = api_key_data
                
                except Exception as e:
                    self.logger.error(f"Error getting Azure API key: {str(e)}")
                    # Continue with config-based keys
            
            if not api_key or not azure_endpoint:
                raise ValueError("Missing required Azure OpenAI configuration")
            
            # Create Azure OpenAI client
            client = AzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version
            )
            
            # Get the embedding from Azure
            response = client.embeddings.create(
                model=deployment,
                input=text,
                encoding_format="float"
            )
            
            # Return the embedding
            return response.data[0].embedding
            
        except Exception as e:
            self.logger.error(f"Error getting Azure embedding: {str(e)}")
            raise
    
    def _get_local_embedding(self, text: str, config: Dict[str, Any]) -> List[float]:
        """Get embeddings using a local model"""
        try:
            # Check what local embedding library is configured
            library = config.get("library", "sentence_transformers")
            model_name = config.get("model_name", "all-MiniLM-L6-v2")
            
            if library == "sentence_transformers":
                # Use sentence_transformers
                try:
                    from sentence_transformers import SentenceTransformer
                    
                    # Load or get model
                    if not hasattr(self, '_embedding_model'):
                        self._embedding_model = SentenceTransformer(model_name)
                    
                    # Generate embedding
                    embedding = self._embedding_model.encode(text)
                    
                    # Normalize the embedding
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
                        
                    return embedding.tolist()
                except ImportError:
                    self.logger.error("sentence_transformers not installed. Install with pip install sentence-transformers")
                    raise
            
            # Fall back to simple embedding
            self.logger.warning(f"Unsupported local embedding library: {library}")
            return self._get_fallback_embedding(text)
            
        except Exception as e:
            self.logger.error(f"Error getting local embedding: {str(e)}")
            raise
    
    def _get_fallback_embedding(self, text: str) -> List[float]:
        """Generate a fallback embedding when other methods fail"""
        # Use a simple hash-based method to generate semi-meaningful embeddings
        # Not suitable for production but better than completely random values
        import hashlib
        
        # Create a seed from the text hash
        hash_object = hashlib.md5(text.encode())
        seed = int(hash_object.hexdigest(), 16) % 2**32
        np.random.seed(seed)
        
        # Generate a random vector
        embedding = np.random.rand(self.embedding_dim).astype(np.float32)
        
        # Add some semantic meaning by using character frequencies for the first few dimensions
        # This way similar texts will have somewhat similar embeddings
        char_freq = {}
        for char in text.lower():
            if char.isalnum():
                char_freq[char] = char_freq.get(char, 0) + 1
        
        # Normalize frequencies and use for first dimensions (if we have enough text)
        if char_freq:
            total_chars = sum(char_freq.values())
            for i, (char, freq) in enumerate(sorted(char_freq.items())[:min(100, self.embedding_dim)]):
                if i < self.embedding_dim:
                    embedding[i] = freq / total_chars
        
        # Normalize the vector
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.tolist()


class DistributedMemoryConnector:
    """
    Connects the local memory manager to the distributed memory mesh.
    
    This class acts as a bridge between the local memory manager and the
    distributed memory system, handling synchronization and caching.
    """
    
    def __init__(self, 
                 local_memory_manager,
                 message_bus: MessageBus,
                 config: Dict[str, Any]):
        """
        Initialize the distributed memory connector.
        
        Args:
            local_memory_manager: Local memory manager instance
            message_bus: Message bus for communication
            config: Configuration dictionary
        """
        self.local_memory = local_memory_manager
        self.message_bus = message_bus
        self.config = config
        self.instance_id = config.get("instance_id", f"instance-{uuid.uuid4()}")
        
        # Create the distributed memory adapter
        self.distributed_memory = VectorMemorySwarm(
            config=config.get("vector_memory", {}),
            message_bus=message_bus,
            instance_id=self.instance_id
        )
        
        # Create the event listener
        self.event_listener = MemoryEventListener(
            message_bus=message_bus,
            memory_manager=local_memory_manager,
            namespaces=config.get("namespaces", ["default"])
        )
        
        # Set up logging
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized DistributedMemoryConnector for instance {self.instance_id}")
    
    def start(self) -> None:
        """Start the memory connector."""
        # Start listening for memory events
        self.event_listener.start()
        self.logger.info("Started distributed memory connector")
    
    def stop(self) -> None:
        """Stop the memory connector."""
        # Stop listening for memory events
        self.event_listener.stop()
        self.logger.info("Stopped distributed memory connector")
    
    def get_auth_token(self, 
                      agent_id: str,
                      namespaces: Optional[List[str]] = None,
                      access_level: MemoryAccessLevel = MemoryAccessLevel.READ_ONLY,
                      ttl_seconds: Optional[int] = None) -> AuthToken:
        """
        Generate an authentication token for memory access.
        
        Args:
            agent_id: Agent ID requesting access
            namespaces: Namespaces to grant access to
            access_level: Access level to grant
            ttl_seconds: Token lifetime in seconds
            
        Returns:
            Authentication token
        """
        # Generate expiration if needed
        expires_at = None
        if ttl_seconds:
            expires_at = (datetime.now() + timedelta(seconds=ttl_seconds)).isoformat()
        
        # Set up namespace access
        namespace_access = {}
        if namespaces:
            for ns in namespaces:
                namespace_access[ns] = access_level
        else:
            namespace_access["default"] = access_level
        
        # Create the token
        token = AuthToken(
            subject=agent_id,
            namespace_access=namespace_access,
            expires_at=expires_at,
            issuer=self.instance_id
        )
        
        self.logger.debug(f"Generated auth token for agent {agent_id} with {len(namespace_access)} namespaces")
        return token


def create_memory_mesh(config: Dict[str, Any], message_bus: MessageBus) -> VectorMemorySwarm:
    """
    Factory function to create a distributed memory mesh instance.
    
    Args:
        config: Memory mesh configuration
        message_bus: Message bus for communication
        
    Returns:
        VectorMemorySwarm instance
    """
    return VectorMemorySwarm(config=config, message_bus=message_bus)
