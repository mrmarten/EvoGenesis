"""
Memory Routes for EvoGenesis Web UI

This module provides API routes for managing memory in the EvoGenesis system.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from pydantic import BaseModel


# Define models
class MemoryCreate(BaseModel):
    content: str
    type: str  # episodic, semantic, procedural
    metadata: Dict[str, Any] = {}
    owner: Optional[str] = None  # agent_id or "system"
    tags: List[str] = []


class MemoryQuery(BaseModel):
    query: str
    limit: int = 10
    memory_type: Optional[str] = None
    owner: Optional[str] = None
    tags: List[str] = []


class MemoryResponse(BaseModel):
    id: str
    content: str
    type: str
    metadata: Dict[str, Any]
    owner: str
    tags: List[str]
    created_at: str
    embedding: Optional[List[float]] = None


# Create router
router = APIRouter(
    prefix="/api/memory",
    tags=["memory"],
)


# Define routes
@router.get("/", response_model=Dict[str, Any])
async def get_memories(
    request: Request,
    limit: int = Query(50, description="Maximum number of memories to return"),
    offset: int = Query(0, description="Offset for pagination"),
    memory_type: Optional[str] = Query(None, description="Filter by memory type"),
    owner: Optional[str] = Query(None, description="Filter by owner (agent_id or 'system')"),
    tag: Optional[str] = Query(None, description="Filter by tag")
):
    """Get memories with optional filtering."""
    kernel = request.app.state.kernel
    
    try:
        memory_manager = kernel.get_module("memory_manager")
        
        # Get statistics about memory store
        stats = memory_manager.get_stats()
        
        # Get memories with filters
        filters = {}
        if memory_type:
            filters["type"] = memory_type
        if owner:
            filters["owner"] = owner
        if tag:
            filters["tag"] = tag
            
        memories = memory_manager.list_memories(
            limit=limit,
            offset=offset,
            filters=filters
        )
        
        # Transform to response format
        memory_items = []
        for memory_id, memory_data in memories.items():
            memory_items.append({
                "id": memory_id,
                "content": memory_data.content,
                "type": memory_data.type,
                "metadata": memory_data.metadata,
                "owner": memory_data.owner,
                "tags": memory_data.tags,
                "created_at": memory_data.created_at
            })
        
        return {
            "total": stats.get("total_memories", 0),
            "agent_memories": stats.get("agent_memories", 0),
            "system_memories": stats.get("system_memories", 0),
            "vector_store": {
                "type": stats.get("vector_store_type", "unknown"),
                "path": stats.get("vector_store_path", ""),
                "collections": stats.get("collections", 0),
                "embeddings": stats.get("total_embeddings", 0),
                "size": stats.get("size_mb", 0)
            },
            "items": memory_items
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get memories: {str(e)}")


@router.get("/{memory_id}", response_model=MemoryResponse)
async def get_memory(memory_id: str, request: Request):
    """Get a specific memory by ID."""
    kernel = request.app.state.kernel
    
    try:
        memory_manager = kernel.get_module("memory_manager")
        memory = memory_manager.get_memory(memory_id)
        
        if not memory:
            raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
        
        return {
            "id": memory_id,
            "content": memory.content,
            "type": memory.type,
            "metadata": memory.metadata,
            "owner": memory.owner,
            "tags": memory.tags,
            "created_at": memory.created_at,
            "embedding": memory.embedding if hasattr(memory, "embedding") else None
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get memory {memory_id}: {str(e)}")


@router.post("/", response_model=Dict[str, Any])
async def create_memory(memory: MemoryCreate, request: Request):
    """Create a new memory."""
    kernel = request.app.state.kernel
    
    try:
        memory_manager = kernel.get_module("memory_manager")
        
        memory_id = memory_manager.add_memory(
            content=memory.content,
            memory_type=memory.type,
            metadata=memory.metadata,
            owner=memory.owner,
            tags=memory.tags
        )
        
        return {
            "success": True,
            "message": "Memory created successfully",
            "memory_id": memory_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create memory: {str(e)}")


@router.delete("/{memory_id}", response_model=Dict[str, Any])
async def delete_memory(memory_id: str, request: Request):
    """Delete a memory."""
    kernel = request.app.state.kernel
    
    try:
        memory_manager = kernel.get_module("memory_manager")
        success = memory_manager.delete_memory(memory_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")
        
        return {
            "success": True,
            "message": f"Memory {memory_id} deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete memory {memory_id}: {str(e)}")


@router.post("/search", response_model=Dict[str, Any])
async def search_memories(query: MemoryQuery, request: Request):
    """Search memories using vector similarity."""
    kernel = request.app.state.kernel
    
    try:
        memory_manager = kernel.get_module("memory_manager")
        
        search_results = memory_manager.search_memories(
            query=query.query,
            limit=query.limit,
            memory_type=query.memory_type,
            owner=query.owner,
            tags=query.tags
        )
        
        # Transform to response format
        results = []
        for memory_id, memory_data, similarity in search_results:
            results.append({
                "id": memory_id,
                "content": memory_data.content,
                "type": memory_data.type,
                "metadata": memory_data.metadata,
                "owner": memory_data.owner,
                "tags": memory_data.tags,
                "created_at": memory_data.created_at,
                "similarity": similarity
            })
        
        return {
            "query": query.query,
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to search memories: {str(e)}")


@router.post("/generate", response_model=Dict[str, Any])
async def generate_memory(request: Request, prompt: str, memory_type: str = "semantic", agent_id: Optional[str] = None):
    """Generate and store a memory using the LLM."""
    kernel = request.app.state.kernel
    
    try:
        memory_manager = kernel.get_module("memory_manager")
        llm_orchestrator = kernel.get_module("llm_orchestrator")
        
        # Generate content using LLM
        content = await llm_orchestrator.generate(prompt)
        
        # Create memory
        memory_id = memory_manager.add_memory(
            content=content,
            memory_type=memory_type,
            metadata={"source": "generated", "prompt": prompt},
            owner=agent_id if agent_id else "system",
            tags=["generated"]
        )
        
        return {
            "success": True,
            "message": "Memory generated successfully",
            "memory_id": memory_id,
            "content": content
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate memory: {str(e)}")


def add_routes(app):
    """Add memory routes to the main app."""
    app.include_router(router)
