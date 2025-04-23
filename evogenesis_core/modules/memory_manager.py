"""
Memory Management Module - Manages short-term and long-term memory for agents and the system.

This module provides the cognitive backbone for local adaptation, storing:
- Short-term memory: Immediate context, conversation history, task state
- Long-term memory: Learned knowledge, experiences, user preferences, effective patterns
"""

from typing import Dict, Any, List, Optional, Union, Tuple
import os
import json
import time
import uuid
import logging
import threading
import pickle
from datetime import datetime, timedelta
from enum import Enum
import hashlib

# Optional imports - these will be imported conditionally
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

try:
    import pinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False


class MemoryType(str, Enum):
    """Types of memory stored in the system."""
    CONVERSATION = "conversation"
    TASK_STATE = "task_state"
    AGENT_STATE = "agent_state"
    USER_PREFERENCE = "user_preference"
    EXPERIENCE = "experience"
    KNOWLEDGE = "knowledge"
    PATTERN = "pattern"
    PROMPT = "prompt"
    REFLECTION = "reflection"
    FEEDBACK = "feedback"
    METRIC = "metric"
    ERROR = "error"
    STRATEGY = "strategy"


class MemoryProvider(str, Enum):
    """Memory storage providers."""
    IN_MEMORY = "in_memory"
    FILE = "file"
    REDIS = "redis"
    CHROMADB = "chromadb"
    PINECONE = "pinecone"
    WEAVIATE = "weaviate"
    CUSTOM = "custom"


class ShortTermMemoryStore:
    """
    Manages short-term memory storage with TTL (Time To Live) functionality.
    
    This store is optimized for fast access to temporary data like conversation 
    context, active task state, etc.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the short-term memory store.
        
        Args:
            config: Configuration options for the store
        """
        self.config = config or {}
        self.provider = self.config.get("provider", MemoryProvider.IN_MEMORY)
        
        # Initialize the selected provider
        if self.provider == MemoryProvider.IN_MEMORY:
            self.store = {}  # context_id -> {key -> (value, expiry)}
            self._cleanup_thread = None
            self._running = False
            self._start_cleanup_thread()
            
        elif self.provider == MemoryProvider.REDIS:
            if not REDIS_AVAILABLE:
                raise ImportError("Redis is not available. Install with 'pip install redis'")
            
            redis_config = self.config.get("redis", {})
            self.redis_client = redis.Redis(
                host=redis_config.get("host", "localhost"),
                port=redis_config.get("port", 6379),
                db=redis_config.get("db", 0),
                password=redis_config.get("password"),
                decode_responses=redis_config.get("decode_responses", True)
            )
            
        elif self.provider == MemoryProvider.FILE:
            file_config = self.config.get("file", {})
            self.file_dir = file_config.get("directory", "memory/short_term")
            os.makedirs(self.file_dir, exist_ok=True)
            
            # Load existing data from files
            self.store = {}
            self._load_from_files()
            
            # Start cleanup thread for file-based storage
            self._cleanup_thread = None
            self._running = False
            self._start_cleanup_thread()
            
        elif self.provider == MemoryProvider.CUSTOM:
            # Use a custom provider provided in the config
            custom_provider = self.config.get("custom_provider")
            if not custom_provider:
                raise ValueError("Custom provider not specified in config")
            
            self.custom_store = custom_provider
            
        else:
            raise ValueError(f"Unsupported short-term memory provider: {self.provider}")
    
    def _start_cleanup_thread(self):
        """Start a background thread to cleanup expired items."""
        if self._cleanup_thread is not None:
            return
            
        self._running = True
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        self._cleanup_thread.start()
    
    def _cleanup_loop(self):
        """Background loop for cleaning up expired items."""
        while self._running:
            if self.provider == MemoryProvider.IN_MEMORY:
                self._cleanup_memory()
            elif self.provider == MemoryProvider.FILE:
                self._cleanup_files()
                
            # Sleep for a bit to avoid consuming too much CPU
            time.sleep(5)
    
    def _cleanup_memory(self):
        """Clean up expired items from in-memory store."""
        current_time = time.time()
        for context_id in list(self.store.keys()):
            context_data = self.store[context_id]
            for key in list(context_data.keys()):
                value, expiry = context_data[key]
                if expiry and expiry < current_time:
                    del context_data[key]
            
            # Remove empty contexts
            if not context_data:
                del self.store[context_id]
    
    def _cleanup_files(self):
        """Clean up expired items from file-based store."""
        current_time = time.time()
        for context_id in list(self.store.keys()):
            context_data = self.store[context_id]
            modified = False
            
            for key in list(context_data.keys()):
                value, expiry = context_data[key]
                if expiry and expiry < current_time:
                    del context_data[key]
                    modified = True
            
            # Save changes to file if needed
            if modified:
                if context_data:
                    context_file = os.path.join(self.file_dir, f"{context_id}.json")
                    with open(context_file, 'w') as f:
                        json.dump(context_data, f)
                else:
                    # Remove empty context file
                    context_file = os.path.join(self.file_dir, f"{context_id}.json")
                    if os.path.exists(context_file):
                        os.remove(context_file)
                    del self.store[context_id]
    
    def _load_from_files(self):
        """Load short-term memory data from files."""
        for filename in os.listdir(self.file_dir):
            if filename.endswith(".json"):
                context_id = filename[:-5]  # Remove .json extension
                context_file = os.path.join(self.file_dir, filename)
                
                try:
                    with open(context_file, 'r') as f:
                        context_data = json.load(f)
                    
                    # Convert loaded data to proper format
                    processed_data = {}
                    for key, (value, expiry) in context_data.items():
                        processed_data[key] = (value, expiry)
                    
                    self.store[context_id] = processed_data
                except Exception as e:
                    logging.error(f"Error loading short-term memory file {filename}: {str(e)}")
    
    def store(self, context_id: str, key: str, value: Any, 
             ttl: Optional[int] = None) -> bool:
        """
        Store a key-value pair in short-term memory.
        
        Args:
            context_id: Context identifier (e.g., conversation_id, agent_id)
            key: Key to store the value under
            value: Value to store
            ttl: Time to live in seconds (None means no expiration)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.provider == MemoryProvider.IN_MEMORY:
                # Calculate expiry time if ttl is provided
                expiry = time.time() + ttl if ttl is not None else None
                
                # Create context if it doesn't exist
                if context_id not in self.store:
                    self.store[context_id] = {}
                
                # Store value with expiry
                self.store[context_id][key] = (value, expiry)
                
            elif self.provider == MemoryProvider.REDIS:
                # Serialize value if it's not a string
                if not isinstance(value, (str, int, float, bool)):
                    value = json.dumps(value)
                
                # Create a Redis key with context_id and key
                redis_key = f"{context_id}:{key}"
                
                # Store in Redis with optional TTL
                if ttl is not None:
                    self.redis_client.setex(redis_key, ttl, value)
                else:
                    self.redis_client.set(redis_key, value)
                    
            elif self.provider == MemoryProvider.FILE:
                # Calculate expiry time if ttl is provided
                expiry = time.time() + ttl if ttl is not None else None
                
                # Create context if it doesn't exist
                if context_id not in self.store:
                    self.store[context_id] = {}
                
                # Store value with expiry
                self.store[context_id][key] = (value, expiry)
                
                # Save to file
                context_file = os.path.join(self.file_dir, f"{context_id}.json")
                with open(context_file, 'w') as f:
                    json.dump(self.store[context_id], f)
                
            elif self.provider == MemoryProvider.CUSTOM:
                # Use custom provider
                return self.custom_store.store(context_id, key, value, ttl)
            
            return True
            
        except Exception as e:
            logging.error(f"Error storing in short-term memory: {str(e)}")
            return False
    
    def retrieve(self, context_id: str, key: str) -> Optional[Any]:
        """
        Retrieve a value from short-term memory.
        
        Args:
            context_id: Context identifier
            key: Key to retrieve
            
        Returns:
            The stored value, or None if not found or expired
        """
        try:
            if self.provider == MemoryProvider.IN_MEMORY:
                # Check if context and key exist
                if context_id in self.store and key in self.store[context_id]:
                    value, expiry = self.store[context_id][key]
                    
                    # Check if expired
                    if expiry is not None and expiry < time.time():
                        del self.store[context_id][key]
                        return None
                    
                    return value
                
                return None
                
            elif self.provider == MemoryProvider.REDIS:
                # Create a Redis key with context_id and key
                redis_key = f"{context_id}:{key}"
                
                # Get from Redis
                value = self.redis_client.get(redis_key)
                
                if value is None:
                    return None
                
                # Try to parse as JSON if it might be a complex object
                if isinstance(value, str) and (value.startswith('{') or value.startswith('[')):
                    try:
                        return json.loads(value)
                    except:
                        pass
                
                return value
                
            elif self.provider == MemoryProvider.FILE:
                # Check if context and key exist
                if context_id in self.store and key in self.store[context_id]:
                    value, expiry = self.store[context_id][key]
                    
                    # Check if expired
                    if expiry is not None and expiry < time.time():
                        del self.store[context_id][key]
                        
                        # Save to file
                        context_file = os.path.join(self.file_dir, f"{context_id}.json")
                        with open(context_file, 'w') as f:
                            json.dump(self.store[context_id], f)
                        
                        return None
                    
                    return value
                
                return None
                
            elif self.provider == MemoryProvider.CUSTOM:
                # Use custom provider
                return self.custom_store.retrieve(context_id, key)
            
        except Exception as e:
            logging.error(f"Error retrieving from short-term memory: {str(e)}")
            return None
    
    def delete(self, context_id: str, key: Optional[str] = None) -> bool:
        """
        Delete a key or an entire context from short-term memory.
        
        Args:
            context_id: Context identifier
            key: Key to delete (if None, delete entire context)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.provider == MemoryProvider.IN_MEMORY:
                if key is None:
                    # Delete entire context
                    if context_id in self.store:
                        del self.store[context_id]
                else:
                    # Delete specific key
                    if context_id in self.store and key in self.store[context_id]:
                        del self.store[context_id][key]
                
                return True
                
            elif self.provider == MemoryProvider.REDIS:
                if key is None:
                    # Delete all keys matching the context pattern
                    keys = self.redis_client.keys(f"{context_id}:*")
                    if keys:
                        self.redis_client.delete(*keys)
                else:
                    # Delete specific key
                    redis_key = f"{context_id}:{key}"
                    self.redis_client.delete(redis_key)
                
                return True
                
            elif self.provider == MemoryProvider.FILE:
                if key is None:
                    # Delete entire context
                    if context_id in self.store:
                        del self.store[context_id]
                        
                        # Remove file
                        context_file = os.path.join(self.file_dir, f"{context_id}.json")
                        if os.path.exists(context_file):
                            os.remove(context_file)
                else:
                    # Delete specific key
                    if context_id in self.store and key in self.store[context_id]:
                        del self.store[context_id][key]
                        
                        # Save to file
                        context_file = os.path.join(self.file_dir, f"{context_id}.json")
                        with open(context_file, 'w') as f:
                            json.dump(self.store[context_id], f)
                
                return True
                
            elif self.provider == MemoryProvider.CUSTOM:
                # Use custom provider
                return self.custom_store.delete(context_id, key)
            
            return True
            
        except Exception as e:
            logging.error(f"Error deleting from short-term memory: {str(e)}")
            return False
    
    def list_keys(self, context_id: str) -> List[str]:
        """
        List all keys in a context.
        
        Args:
            context_id: Context identifier
            
        Returns:
            List of keys in the context
        """
        try:
            if self.provider == MemoryProvider.IN_MEMORY:
                if context_id in self.store:
                    # Filter out expired keys
                    current_time = time.time()
                    return [
                        key for key, (_, expiry) in self.store[context_id].items()
                        if expiry is None or expiry > current_time
                    ]
                
                return []
                
            elif self.provider == MemoryProvider.REDIS:
                # Get all keys matching the context pattern
                redis_keys = self.redis_client.keys(f"{context_id}:*")
                
                # Extract just the key part after the context_id:
                return [key.split(':', 1)[1] for key in redis_keys]
                
            elif self.provider == MemoryProvider.FILE:
                if context_id in self.store:
                    # Filter out expired keys
                    current_time = time.time()
                    return [
                        key for key, (_, expiry) in self.store[context_id].items()
                        if expiry is None or expiry > current_time
                    ]
                
                return []
                
            elif self.provider == MemoryProvider.CUSTOM:
                # Use custom provider
                return self.custom_store.list_keys(context_id)
            
            return []
            
        except Exception as e:
            logging.error(f"Error listing keys from short-term memory: {str(e)}")
            return []
    
    def close(self):
        """Close connections and clean up resources."""
        if self.provider == MemoryProvider.IN_MEMORY or self.provider == MemoryProvider.FILE:
            # Stop the cleanup thread
            self._running = False
            if self._cleanup_thread:
                self._cleanup_thread.join(timeout=1)
                
        elif self.provider == MemoryProvider.REDIS:
            # Close Redis connection
            self.redis_client.close()
            
        elif self.provider == MemoryProvider.CUSTOM:
            # Close custom provider if it has a close method
            if hasattr(self.custom_store, 'close'):
                self.custom_store.close()


class LongTermMemoryStore:
    """
    Manages long-term memory storage with vector search capabilities.
    
    This store is optimized for semantic search over accumulated knowledge,
    experiences, patterns, etc.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the long-term memory store.
        
        Args:
            config: Configuration options for the store
        """
        self.config = config or {}
        self.provider = self.config.get("provider", MemoryProvider.CHROMADB)
        
        # Initialize the selected provider
        if self.provider == MemoryProvider.CHROMADB:
            if not CHROMADB_AVAILABLE:
                raise ImportError("ChromaDB is not available. Install with 'pip install chromadb'")
            
            chromadb_config = self.config.get("chromadb", {})
            persist_directory = chromadb_config.get("persist_directory", "memory/long_term/chromadb")
            os.makedirs(persist_directory, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(path=persist_directory)
            self.collections = {}  # Cache of ChromaDB collections by namespace
            
        elif self.provider == MemoryProvider.PINECONE:
            if not PINECONE_AVAILABLE:
                raise ImportError("Pinecone is not available. Install with 'pip install pinecone-client'")
            
            pinecone_config = self.config.get("pinecone", {})
            api_key = pinecone_config.get("api_key")
            environment = pinecone_config.get("environment")
            
            if not api_key or not environment:
                raise ValueError("Pinecone API key and environment must be provided in config")
            
            pinecone.init(api_key=api_key, environment=environment)
            
            # Get or create indexes for each namespace
            self.indexes = {}  # Pinecone indexes by namespace
            
        elif self.provider == MemoryProvider.WEAVIATE:
            if not WEAVIATE_AVAILABLE:
                raise ImportError("Weaviate is not available. Install with 'pip install weaviate-client'")
            
            weaviate_config = self.config.get("weaviate", {})
            url = weaviate_config.get("url")
            auth_config = weaviate_config.get("auth_config")
            
            if not url:
                raise ValueError("Weaviate URL must be provided in config")
            
            self.weaviate_client = weaviate.Client(url=url, auth_client_secret=auth_config)
            
        elif self.provider == MemoryProvider.FILE:
            file_config = self.config.get("file", {})
            self.file_dir = file_config.get("directory", "memory/long_term/files")
            os.makedirs(self.file_dir, exist_ok=True)
            
            # Simple file-based vector store
            self.vector_store = {}  # namespace -> [{"id": id, "embedding": [...], "metadata": {...}, "content": "..."}]
            self._load_file_vector_store()
            
        elif self.provider == MemoryProvider.IN_MEMORY:
            # Simple in-memory vector store
            self.vector_store = {}  # namespace -> [{"id": id, "embedding": [...], "metadata": {...}, "content": "..."}]
            
        elif self.provider == MemoryProvider.CUSTOM:
            # Use a custom provider provided in the config
            custom_provider = self.config.get("custom_provider")
            if not custom_provider:
                raise ValueError("Custom provider not specified in config")
            
            self.custom_store = custom_provider
            
        else:
            raise ValueError(f"Unsupported long-term memory provider: {self.provider}")
    
    def _load_file_vector_store(self):
        """Load vector store data from files."""
        if not os.path.exists(self.file_dir):
            return
            
        for namespace_dir in os.listdir(self.file_dir):
            namespace_path = os.path.join(self.file_dir, namespace_dir)
            if os.path.isdir(namespace_path):
                self.vector_store[namespace_dir] = []
                
                data_file = os.path.join(namespace_path, "data.json")
                if os.path.exists(data_file):
                    try:
                        with open(data_file, 'r') as f:
                            self.vector_store[namespace_dir] = json.load(f)
                    except Exception as e:
                        logging.error(f"Error loading vector store data for namespace {namespace_dir}: {str(e)}")
    
    def _save_file_vector_store(self, namespace: str):
        """Save vector store data to files."""
        namespace_path = os.path.join(self.file_dir, namespace)
        os.makedirs(namespace_path, exist_ok=True)
        
        data_file = os.path.join(namespace_path, "data.json")
        try:
            with open(data_file, 'w') as f:
                json.dump(self.vector_store.get(namespace, []), f)
        except Exception as e:
            logging.error(f"Error saving vector store data for namespace {namespace}: {str(e)}")
    
    def _get_chroma_collection(self, namespace: str):
        """Get or create a ChromaDB collection for a namespace."""
        if namespace not in self.collections:
            # Create collection if it doesn't exist
            self.collections[namespace] = self.chroma_client.get_or_create_collection(
                name=namespace
            )
        
        return self.collections[namespace]
    
    def _get_pinecone_index(self, namespace: str):
        """Get or create a Pinecone index for a namespace."""
        if namespace not in self.indexes:
            # Check if index exists
            if namespace not in pinecone.list_indexes():
                # Create index with default settings
                pinecone_config = self.config.get("pinecone", {})
                dimension = pinecone_config.get("dimension", 1536)  # Default for OpenAI embeddings
                pinecone.create_index(
                    name=namespace,
                    dimension=dimension,
                    metric="cosine"
                )
            
            # Connect to index
            self.indexes[namespace] = pinecone.Index(namespace)
        
        return self.indexes[namespace]
    
    def _cosine_similarity(self, v1, v2):
        """Calculate cosine similarity between two vectors."""
        import numpy as np
        v1 = np.array(v1)
        v2 = np.array(v2)
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    
    def store(self, namespace: str, embedding: List[float], 
             metadata: Dict[str, Any], content: str, 
             doc_id: Optional[str] = None) -> str:
        """
        Store a document in long-term memory.
        
        Args:
            namespace: Namespace for multi-tenancy or project separation
            embedding: Vector embedding of the content
            metadata: Additional metadata for the document
            content: Text content of the document
            doc_id: Optional document ID (generated if not provided)
            
        Returns:
            Document ID
        """
        # Generate ID if not provided
        if doc_id is None:
            doc_id = str(uuid.uuid4())
        
        try:
            if self.provider == MemoryProvider.CHROMADB:
                collection = self._get_chroma_collection(namespace)
                
                # Store document in ChromaDB
                collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    metadatas=[metadata],
                    documents=[content]
                )
                
            elif self.provider == MemoryProvider.PINECONE:
                index = self._get_pinecone_index(namespace)
                
                # Store document in Pinecone
                index.upsert([
                    (doc_id, embedding, {**metadata, "content": content})
                ])
                
            elif self.provider == MemoryProvider.WEAVIATE:
                # Ensure class exists for the namespace
                class_name = f"Memory{namespace.capitalize()}"
                
                if not self.weaviate_client.schema.exists(class_name):
                    # Create class for this namespace
                    class_schema = {
                        "class": class_name,
                        "vectorizer": "none",  # We provide our own vectors
                        "properties": [
                            {
                                "name": "content",
                                "dataType": ["text"]
                            }
                        ]
                    }
                    
                    # Add metadata fields to schema
                    for key, value in metadata.items():
                        data_type = "text"
                        if isinstance(value, bool):
                            data_type = "boolean"
                        elif isinstance(value, (int, float)):
                            data_type = "number"
                        elif isinstance(value, dict):
                            continue  # Skip complex objects for now
                            
                        class_schema["properties"].append({
                            "name": key,
                            "dataType": [data_type]
                        })
                    
                    self.weaviate_client.schema.create_class(class_schema)
                
                # Store document in Weaviate
                self.weaviate_client.data_object.create(
                    data_object={
                        "content": content,
                        **metadata
                    },
                    class_name=class_name,
                    uuid=doc_id,
                    vector=embedding
                )
                
            elif self.provider == MemoryProvider.FILE or self.provider == MemoryProvider.IN_MEMORY:
                # Initialize namespace if it doesn't exist
                if namespace not in self.vector_store:
                    self.vector_store[namespace] = []
                
                # Store document in vector store
                self.vector_store[namespace].append({
                    "id": doc_id,
                    "embedding": embedding,
                    "metadata": metadata,
                    "content": content
                })
                
                # Save to file if using file provider
                if self.provider == MemoryProvider.FILE:
                    self._save_file_vector_store(namespace)
                
            elif self.provider == MemoryProvider.CUSTOM:
                # Use custom provider
                return self.custom_store.store(namespace, embedding, metadata, content, doc_id)
            
            return doc_id
            
        except Exception as e:
            logging.error(f"Error storing in long-term memory: {str(e)}")
            return doc_id
    
    def search(self, namespace: str, query_embedding: List[float], 
              limit: int = 5, 
              filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents in long-term memory.
        
        Args:
            namespace: Namespace to search in
            query_embedding: Vector embedding of the query
            limit: Maximum number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of matching documents with scores
        """
        try:
            if self.provider == MemoryProvider.CHROMADB:
                collection = self._get_chroma_collection(namespace)
                
                # Search in ChromaDB
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=limit,
                    where=filter_metadata
                )
                
                # Format results
                formatted_results = []
                if results['documents'] and len(results['documents'][0]) > 0:
                    for i in range(len(results['documents'][0])):
                        formatted_results.append({
                            "id": results['ids'][0][i],
                            "content": results['documents'][0][i],
                            "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                            "score": results['distances'][0][i] if results['distances'] else 0.0
                        })
                
                return formatted_results
                
            elif self.provider == MemoryProvider.PINECONE:
                index = self._get_pinecone_index(namespace)
                
                # Search in Pinecone
                filter_dict = {}
                if filter_metadata:
                    filter_dict = filter_metadata
                
                results = index.query(
                    vector=query_embedding,
                    top_k=limit,
                    include_metadata=True,
                    filter=filter_dict
                )
                
                # Format results
                formatted_results = []
                for match in results['matches']:
                    metadata = match['metadata'].copy()
                    content = metadata.pop('content', "")
                    
                    formatted_results.append({
                        "id": match['id'],
                        "content": content,
                        "metadata": metadata,
                        "score": match['score']
                    })
                
                return formatted_results
                
            elif self.provider == MemoryProvider.WEAVIATE:
                # Convert namespace to class name
                class_name = f"Memory{namespace.capitalize()}"
                
                # Prepare filter if needed
                where_filter = None
                if filter_metadata:
                    where_clauses = []
                    for key, value in filter_metadata.items():
                        where_clauses.append({
                            "path": [key],
                            "operator": "Equal",
                            "valueString": str(value) if not isinstance(value, (bool, int, float)) else value
                        })
                    
                    if where_clauses:
                        where_filter = {"operator": "And", "operands": where_clauses}
                
                # Search in Weaviate
                results = (
                    self.weaviate_client.query
                    .get(class_name, ["content", "id", "_additional {certainty}"])
                    .with_near_vector({"vector": query_embedding})
                    .with_limit(limit)
                )
                
                if where_filter:
                    results = results.with_where(where_filter)
                
                query_results = results.do()
                
                # Format results
                formatted_results = []
                if query_results.get('data', {}).get('Get', {}).get(class_name):
                    for item in query_results['data']['Get'][class_name]:
                        # Extract content and metadata
                        content = item.pop('content', "")
                        item_id = item.pop('id', "")
                        score = item.get('_additional', {}).get('certainty', 0.0)
                        
                        # Remove internal fields
                        metadata = {k: v for k, v in item.items() if not k.startswith('_')}
                        
                        formatted_results.append({
                            "id": item_id,
                            "content": content,
                            "metadata": metadata,
                            "score": score
                        })
                
                return formatted_results
                
            elif self.provider == MemoryProvider.FILE or self.provider == MemoryProvider.IN_MEMORY:
                if namespace not in self.vector_store:
                    return []
                
                # Simple vector search implementation
                results = []
                for item in self.vector_store[namespace]:
                    # Apply metadata filter if provided
                    if filter_metadata:
                        match = True
                        for key, value in filter_metadata.items():
                            if key not in item['metadata'] or item['metadata'][key] != value:
                                match = False
                                break
                        
                        if not match:
                            continue
                    
                    # Calculate similarity score
                    score = self._cosine_similarity(query_embedding, item['embedding'])
                    
                    results.append({
                        "id": item['id'],
                        "content": item['content'],
                        "metadata": item['metadata'],
                        "score": score
                    })
                
                # Sort by score and limit
                results.sort(key=lambda x: x['score'], reverse=True)
                return results[:limit]
                
            elif self.provider == MemoryProvider.CUSTOM:
                # Use custom provider
                return self.custom_store.search(namespace, query_embedding, limit, filter_metadata)
            
            return []
            
        except Exception as e:
            logging.error(f"Error searching in long-term memory: {str(e)}")
            return []
    
    def delete(self, namespace: str, doc_id: str) -> bool:
        """
        Delete a document from long-term memory.
        
        Args:
            namespace: Namespace of the document
            doc_id: ID of the document to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.provider == MemoryProvider.CHROMADB:
                collection = self._get_chroma_collection(namespace)
                
                # Delete document from ChromaDB
                collection.delete(ids=[doc_id])
                
            elif self.provider == MemoryProvider.PINECONE:
                index = self._get_pinecone_index(namespace)
                
                # Delete document from Pinecone
                index.delete(ids=[doc_id])
                
            elif self.provider == MemoryProvider.WEAVIATE:
                # Convert namespace to class name
                class_name = f"Memory{namespace.capitalize()}"
                
                # Delete document from Weaviate
                self.weaviate_client.data_object.delete(
                    class_name=class_name,
                    uuid=doc_id
                )
                
            elif self.provider == MemoryProvider.FILE or self.provider == MemoryProvider.IN_MEMORY:
                if namespace in self.vector_store:
                    # Remove document from vector store
                    self.vector_store[namespace] = [
                        item for item in self.vector_store[namespace]
                        if item['id'] != doc_id
                    ]
                    
                    # Save to file if using file provider
                    if self.provider == MemoryProvider.FILE:
                        self._save_file_vector_store(namespace)
                
            elif self.provider == MemoryProvider.CUSTOM:
                # Use custom provider
                return self.custom_store.delete(namespace, doc_id)
            
            return True
            
        except Exception as e:
            logging.error(f"Error deleting from long-term memory: {str(e)}")
            return False
    
    def get(self, namespace: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document from long-term memory by ID.
        
        Args:
            namespace: Namespace of the document
            doc_id: ID of the document to retrieve
            
        Returns:
            Document if found, None otherwise
        """
        try:
            if self.provider == MemoryProvider.CHROMADB:
                collection = self._get_chroma_collection(namespace)
                
                # Get document from ChromaDB
                result = collection.get(ids=[doc_id])
                
                if result and len(result['documents']) > 0:
                    return {
                        "id": doc_id,
                        "content": result['documents'][0],
                        "metadata": result['metadatas'][0] if result['metadatas'] else {},
                        "embedding": result['embeddings'][0] if result['embeddings'] else []
                    }
                
                return None
                
            elif self.provider == MemoryProvider.PINECONE:
                index = self._get_pinecone_index(namespace)
                
                # Get document from Pinecone
                result = index.fetch([doc_id])
                
                if doc_id in result['vectors']:
                    vector = result['vectors'][doc_id]
                    metadata = vector['metadata'].copy()
                    content = metadata.pop('content', "")
                    
                    return {
                        "id": doc_id,
                        "content": content,
                        "metadata": metadata,
                        "embedding": vector['values']
                    }
                
                return None
                
            elif self.provider == MemoryProvider.WEAVIATE:
                # Convert namespace to class name
                class_name = f"Memory{namespace.capitalize()}"
                
                # Get document from Weaviate
                result = self.weaviate_client.data_object.get_by_id(
                    class_name=class_name,
                    uuid=doc_id,
                    with_vector=True
                )
                
                if result:
                    # Extract content and metadata
                    properties = result.get('properties', {})
                    content = properties.pop('content', "")
                    vector = result.get('vector', [])
                    
                    return {
                        "id": doc_id,
                        "content": content,
                        "metadata": properties,
                        "embedding": vector
                    }
                
                return None
                
            elif self.provider == MemoryProvider.FILE or self.provider == MemoryProvider.IN_MEMORY:
                if namespace in self.vector_store:
                    # Find document in vector store
                    for item in self.vector_store[namespace]:
                        if item['id'] == doc_id:
                            return item
                
                return None
                
            elif self.provider == MemoryProvider.CUSTOM:
                # Use custom provider
                return self.custom_store.get(namespace, doc_id)
            
            return None
            
        except Exception as e:
            logging.error(f"Error getting document from long-term memory: {str(e)}")
            return None
    
    def list_namespaces(self) -> List[str]:
        """
        List all namespaces in long-term memory.
        
        Returns:
            List of namespace names
        """
        try:
            if self.provider == MemoryProvider.CHROMADB:
                return [collection.name for collection in self.chroma_client.list_collections()]
                
            elif self.provider == MemoryProvider.PINECONE:
                return pinecone.list_indexes()
                
            elif self.provider == MemoryProvider.WEAVIATE:
                schema = self.weaviate_client.schema.get()
                # Extract namespace from class names (removing "Memory" prefix and lowercasing)
                return [
                    class_obj['class'][6:].lower() 
                    for class_obj in schema['classes'] 
                    if class_obj['class'].startswith('Memory')
                ]
                
            elif self.provider == MemoryProvider.FILE:
                # List directories in the file_dir
                return [
                    d for d in os.listdir(self.file_dir)
                    if os.path.isdir(os.path.join(self.file_dir, d))
                ]
                
            elif self.provider == MemoryProvider.IN_MEMORY:
                return list(self.vector_store.keys())
                
            elif self.provider == MemoryProvider.CUSTOM:
                # Use custom provider
                return self.custom_store.list_namespaces()
            
            return []
            
        except Exception as e:
            logging.error(f"Error listing namespaces in long-term memory: {str(e)}")
            return []
    
    def count_documents(self, namespace: str) -> int:
        """
        Count the number of documents in a namespace.
        
        Args:
            namespace: Namespace to count documents in
            
        Returns:
            Number of documents
        """
        try:
            if self.provider == MemoryProvider.CHROMADB:
                collection = self._get_chroma_collection(namespace)
                
                # Count documents in ChromaDB
                return collection.count()
                
            elif self.provider == MemoryProvider.PINECONE:
                index = self._get_pinecone_index(namespace)
                
                # Count documents in Pinecone
                stats = index.describe_index_stats()
                return stats['total_vector_count']
                
            elif self.provider == MemoryProvider.WEAVIATE:
                # Convert namespace to class name
                class_name = f"Memory{namespace.capitalize()}"
                
                # Count documents in Weaviate
                result = (
                    self.weaviate_client.query
                    .aggregate(class_name)
                    .with_meta_count()
                    .do()
                )
                
                return result.get('data', {}).get('Aggregate', {}).get(class_name, [{}])[0].get('meta', {}).get('count', 0)
                
            elif self.provider == MemoryProvider.FILE or self.provider == MemoryProvider.IN_MEMORY:
                if namespace in self.vector_store:
                    return len(self.vector_store[namespace])
                
                return 0
                
            elif self.provider == MemoryProvider.CUSTOM:
                # Use custom provider
                return self.custom_store.count_documents(namespace)
            
            return 0
            
        except Exception as e:
            logging.error(f"Error counting documents in long-term memory: {str(e)}")
            return 0
    
    def close(self):
        """Close connections and clean up resources."""
        if self.provider == MemoryProvider.CHROMADB:
            # Close ChromaDB client
            if hasattr(self.chroma_client, 'close'):
                self.chroma_client.close()
                
        elif self.provider == MemoryProvider.PINECONE:
            # Nothing to close for Pinecone
            pass
                
        elif self.provider == MemoryProvider.WEAVIATE:
            # Close Weaviate client
            if hasattr(self.weaviate_client, 'close'):
                self.weaviate_client.close()
                
        elif self.provider == MemoryProvider.CUSTOM:
            # Close custom provider if it has a close method
            if hasattr(self.custom_store, 'close'):
                self.custom_store.close()


class MemoryManager:
    """
    Manages short-term and long-term memory for the EvoGenesis framework.
    
    Provides the cognitive backbone for local adaptation, storing:
    - Short-term memory: Immediate context, conversation history, task state
    - Long-term memory: Learned knowledge, experiences, user preferences, effective patterns
    """
    
    def __init__(self, kernel, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Memory Manager.
        
        Args:
            kernel: The EvoGenesis kernel instance
            config: Configuration options for memory stores
        """
        self.kernel = kernel
        self.config = config or {}
        
        # Initialize short-term memory store
        short_term_config = self.config.get("short_term", {})
        self.short_term = ShortTermMemoryStore(config=short_term_config)
        
        # Initialize long-term memory store
        long_term_config = self.config.get("long_term", {})
        self.long_term = LongTermMemoryStore(config=long_term_config)
        
        # Memory access tracking
        self.access_stats = {
            "short_term_reads": 0,
            "short_term_writes": 0,
            "long_term_reads": 0,
            "long_term_writes": 0,
            "short_term_hits": 0,
            "long_term_hits": 0
        }
    
    def start(self):
        """Start the Memory Manager module."""
        logging.info("Memory Manager started")
    
    def stop(self):
        """Stop the Memory Manager module and clean up resources."""
        try:
            self.short_term.close()
            self.long_term.close()
            logging.info("Memory Manager stopped")
        except Exception as e:
            logging.error(f"Error stopping Memory Manager: {str(e)}")
    
    def get_status(self):
        """Get the current status of the Memory Manager."""
        # Get list of active namespaces in long-term memory
        namespaces = self.long_term.list_namespaces()
        
        # Count documents in each namespace
        namespace_counts = {}
        for namespace in namespaces:
            namespace_counts[namespace] = self.long_term.count_documents(namespace)
        
        return {
            "status": "active",
            "access_stats": self.access_stats,
            "long_term_namespaces": namespace_counts
        }
    
    # Short-term memory methods
    
    def store_short_term(self, context_id: str, key: str, value: Any, 
                        ttl: Optional[int] = None) -> bool:
        """
        Store a value in short-term memory.
        
        Args:
            context_id: Context identifier (e.g., conversation_id, agent_id)
            key: Key to store the value under
            value: Value to store
            ttl: Time to live in seconds (None means no expiration)
            
        Returns:
            True if successful, False otherwise
        """
        result = self.short_term.store(context_id, key, value, ttl)
        
        if result:
            self.access_stats["short_term_writes"] += 1
        
        return result
    
    def retrieve_short_term(self, context_id: str, key: str) -> Optional[Any]:
        """
        Retrieve a value from short-term memory.
        
        Args:
            context_id: Context identifier
            key: Key to retrieve
            
        Returns:
            The stored value, or None if not found or expired
        """
        self.access_stats["short_term_reads"] += 1
        
        result = self.short_term.retrieve(context_id, key)
        
        if result is not None:
            self.access_stats["short_term_hits"] += 1
        
        return result
    
    def delete_short_term(self, context_id: str, key: Optional[str] = None) -> bool:
        """
        Delete a key or an entire context from short-term memory.
        
        Args:
            context_id: Context identifier
            key: Key to delete (if None, delete entire context)
            
        Returns:
            True if successful, False otherwise
        """
        return self.short_term.delete(context_id, key)
    
    def list_short_term_keys(self, context_id: str) -> List[str]:
        """
        List all keys in a short-term memory context.
        
        Args:
            context_id: Context identifier
            
        Returns:
            List of keys in the context
        """
        return self.short_term.list_keys(context_id)
    
    # Long-term memory methods
    
    def store_long_term(self, namespace: str, content: str, 
                       metadata: Optional[Dict[str, Any]] = None,
                       doc_id: Optional[str] = None,
                       memory_type: Optional[MemoryType] = None) -> str:
        """
        Store a document in long-term memory.
        
        Args:
            namespace: Namespace for multi-tenancy or project separation
            content: Text content to store
            metadata: Additional metadata for the document
            doc_id: Optional document ID (generated if not provided)
            memory_type: Type of memory being stored
            
        Returns:
            Document ID
        """
        # Prepare metadata
        if metadata is None:
            metadata = {}
        
        if memory_type:
            metadata["memory_type"] = memory_type
            
        metadata["timestamp"] = time.time()
        
        # Generate embedding for the content
        embedding = self._get_embedding(content)
        
        if embedding is None:
            logging.error("Failed to generate embedding for content")
            return "" if doc_id is None else doc_id
        
        # Store in long-term memory
        result = self.long_term.store(
            namespace=namespace,
            embedding=embedding,
            metadata=metadata,
            content=content,
            doc_id=doc_id
        )
        
        self.access_stats["long_term_writes"] += 1
        
        return result
    
    def search_long_term(self, namespace: str, query: str, 
                        limit: int = 5,
                        filter_metadata: Optional[Dict[str, Any]] = None,
                        memory_type: Optional[MemoryType] = None) -> List[Dict[str, Any]]:
        """
        Search for similar content in long-term memory.
        
        Args:
            namespace: Namespace to search in
            query: Query text
            limit: Maximum number of results to return
            filter_metadata: Optional metadata filters
            memory_type: Type of memory to search for
            
        Returns:
            List of matching documents with scores
        """
        self.access_stats["long_term_reads"] += 1
        
        # Prepare metadata filter
        if filter_metadata is None:
            filter_metadata = {}
            
        if memory_type:
            filter_metadata["memory_type"] = memory_type
        
        # Generate embedding for the query
        query_embedding = self._get_embedding(query)
        
        if query_embedding is None:
            logging.error("Failed to generate embedding for query")
            return []
        
        # Search in long-term memory
        results = self.long_term.search(
            namespace=namespace,
            query_embedding=query_embedding,
            limit=limit,
            filter_metadata=filter_metadata
        )
        
        if results:
            self.access_stats["long_term_hits"] += 1
        
        return results
    
    def get_long_term(self, namespace: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a document from long-term memory by ID.
        
        Args:
            namespace: Namespace of the document
            doc_id: ID of the document to retrieve
            
        Returns:
            Document if found, None otherwise
        """
        self.access_stats["long_term_reads"] += 1
        
        result = self.long_term.get(namespace, doc_id)
        
        if result:
            self.access_stats["long_term_hits"] += 1
        
        return result
    
    def delete_long_term(self, namespace: str, doc_id: str) -> bool:
        """
        Delete a document from long-term memory.
        
        Args:
            namespace: Namespace of the document
            doc_id: ID of the document to delete
            
        Returns:
            True if successful, False otherwise
        """
        return self.long_term.delete(namespace, doc_id)
    
    def store_conversation(self, conversation_id: str, message: Dict[str, Any],
                          namespace: Optional[str] = None) -> str:
        """
        Store a conversation message in both short-term and long-term memory.
        
        Args:
            conversation_id: ID of the conversation
            message: Message data
            namespace: Namespace for long-term storage
            
        Returns:
            Document ID in long-term memory
        """
        # Store in short-term memory (24 hour TTL)
        message_id = message.get("id", str(uuid.uuid4()))
        self.store_short_term(
            context_id=conversation_id,
            key=f"message:{message_id}",
            value=message,
            ttl=86400  # 24 hours
        )
        
        # Store conversation history list
        history = self.retrieve_short_term(conversation_id, "message_history") or []
        history.append(message_id)
        self.store_short_term(
            context_id=conversation_id,
            key="message_history",
            value=history,
            ttl=86400  # 24 hours
        )
        
        # Store in long-term memory if namespace is provided
        if namespace:
            # Format message content
            content = f"{message.get('role', 'unknown')}: {message.get('content', '')}"
            
            metadata = {
                "conversation_id": conversation_id,
                "message_id": message_id,
                "role": message.get("role"),
                "timestamp": message.get("timestamp", time.time())
            }
            
            return self.store_long_term(
                namespace=namespace,
                content=content,
                metadata=metadata,
                doc_id=message_id,
                memory_type=MemoryType.CONVERSATION
            )
        
        return message_id
    
    def store_experience(self, namespace: str, experience_data: Dict[str, Any]) -> str:
        """
        Store an experience (success/failure) in long-term memory.
        
        Args:
            namespace: Namespace for storage
            experience_data: Experience data
            
        Returns:
            Document ID in long-term memory
        """
        # Extract key fields
        content = experience_data.get("description", "")
        if "task" in experience_data:
            content = f"Task: {experience_data['task']}\n{content}"
            
        if "outcome" in experience_data:
            content = f"{content}\nOutcome: {experience_data['outcome']}"
            
        metadata = {
            "success": experience_data.get("success", False),
            "task_id": experience_data.get("task_id"),
            "agent_id": experience_data.get("agent_id"),
            "tags": experience_data.get("tags", [])
        }
        
        return self.store_long_term(
            namespace=namespace,
            content=content,
            metadata=metadata,
            memory_type=MemoryType.EXPERIENCE
        )
    
    def store_knowledge(self, namespace: str, knowledge_data: Dict[str, Any]) -> str:
        """
        Store knowledge in long-term memory.
        
        Args:
            namespace: Namespace for storage
            knowledge_data: Knowledge data
            
        Returns:
            Document ID in long-term memory
        """
        content = knowledge_data.get("content", "")
        
        metadata = {
            "title": knowledge_data.get("title", ""),
            "source": knowledge_data.get("source", "agent"),
            "confidence": knowledge_data.get("confidence", 1.0),
            "tags": knowledge_data.get("tags", [])
        }
        
        return self.store_long_term(
            namespace=namespace,
            content=content,
            metadata=metadata,
            memory_type=MemoryType.KNOWLEDGE
        )
    
    def store_user_preference(self, user_id: str, preference_data: Dict[str, Any],
                             namespace: Optional[str] = None) -> str:
        """
        Store a user preference in both short-term and long-term memory.
        
        Args:
            user_id: ID of the user
            preference_data: Preference data
            namespace: Namespace for long-term storage
            
        Returns:
            Document ID in long-term memory
        """
        preference_key = preference_data.get("key")
        preference_value = preference_data.get("value")
        
        # Store in short-term memory (30 day TTL)
        self.store_short_term(
            context_id=f"user:{user_id}:preferences",
            key=preference_key,
            value=preference_value,
            ttl=2592000  # 30 days
        )
        
        # Store in long-term memory if namespace is provided
        if namespace:
            content = f"Preference: {preference_key} = {preference_value}"
            if "description" in preference_data:
                content = f"{content}\nDescription: {preference_data['description']}"
                
            metadata = {
                "user_id": user_id,
                "preference_key": preference_key,
                "preference_value": str(preference_value),
                "category": preference_data.get("category", "general")
            }
            
            return self.store_long_term(
                namespace=namespace,
                content=content,
                metadata=metadata,
                memory_type=MemoryType.USER_PREFERENCE
            )
        
        return preference_key
    
    def get_user_preferences(self, user_id: str) -> Dict[str, Any]:
        """
        Get all preferences for a user from short-term memory.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Dictionary of user preferences
        """
        preferences = {}
        
        # Get all keys in the user preferences context
        keys = self.list_short_term_keys(f"user:{user_id}:preferences")
        
        # Retrieve each preference
        for key in keys:
            value = self.retrieve_short_term(f"user:{user_id}:preferences", key)
            if value is not None:
                preferences[key] = value
        
        return preferences
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Generate an embedding for text using the LLM Orchestrator.
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embedding as a list of floats, or None if generation fails
        """
        # Use the LLM Orchestrator to generate an embedding if available
        if hasattr(self.kernel, "llm_orchestrator"):
            try:
                response = self.kernel.llm_orchestrator.execute_prompt(
                    task_type="embedding",
                    prompt_template="raw_text",
                    params={"text": text}
                )
                
                if response.get("success", False):
                    return response.get("result", {}).get("embedding")
            except Exception as e:
                logging.error(f"Error generating embedding: {str(e)}")
        
        # Fallback to a simple hashing-based embedding (not suitable for semantic search)
        try:
            import numpy as np
            import hashlib
            
            # Create a simple hash-based embedding (not semantic!)
            hash_values = []
            for i in range(128):  # Create a 128-dimensional embedding
                hash_input = f"{text}:{i}"
                hash_obj = hashlib.md5(hash_input.encode())
                hash_values.append(int(hash_obj.hexdigest(), 16) % 1000 / 500.0 - 1.0)
            
            # Normalize
            embedding = np.array(hash_values)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding.tolist()
            
        except Exception as e:
            logging.error(f"Error generating fallback embedding: {str(e)}")
            return None
    
    def archive_agent(self, agent_id: str, agent_data: Optional[Dict[str, Any]] = None) -> bool:
        """
        Archive data related to an agent being terminated.
        
        Args:
            agent_id: ID of the agent being archived
            agent_data: Optional additional data about the agent to archive
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get the agent's memory to archive
            agent_memories = self.get_memories(entity_id=agent_id)
              # Create archive entry
            archive_data = {
                "agent_id": agent_id,
                "archived_at": time.time(),
                "memory_count": len(agent_memories),
                "memories": agent_memories,
                "agent_data": agent_data or {}
            }
            
            # Store in archives collection
            if not hasattr(self, "archives"):
                self.archives = {}
            
            self.archives[f"agent_{agent_id}"] = archive_data
            
            logging.info(f"Archived agent {agent_id} with {len(agent_memories)} memories")
            return True
            
        except Exception as e:
            logging.error(f"Failed to archive agent {agent_id}: {str(e)}")
            return False
