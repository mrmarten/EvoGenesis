"""
Swarm Routes for EvoGenesis Web UI

This module provides API routes for managing the EvoGenesis swarm functionality.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
import json


# Define models
class SwarmNodeBase(BaseModel):
    name: str
    role: str  # "coordinator", "worker"
    url: str
    status: Optional[str] = "disconnected"


class SwarmNodeCreate(SwarmNodeBase):
    pass


class SwarmNodeUpdate(BaseModel):
    name: Optional[str] = None
    role: Optional[str] = None
    url: Optional[str] = None
    status: Optional[str] = None


class SwarmNodeResponse(SwarmNodeBase):
    id: str
    connected_at: Optional[str] = None
    last_heartbeat: Optional[str] = None


class SwarmTaskBase(BaseModel):
    title: str
    description: str
    priority: Optional[str] = "medium"
    assigned_node_id: Optional[str] = None


class SwarmTaskCreate(SwarmTaskBase):
    pass


class SwarmTaskResponse(SwarmTaskBase):
    id: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    progress: Optional[float] = 0.0
    result: Optional[Dict[str, Any]] = None


# Create router
router = APIRouter(
    prefix="/api/swarm",
    tags=["swarm"],
)


# Define routes
@router.get("/status", response_model=Dict[str, Any])
async def get_swarm_status(request: Request):
    """Get the current status of the EvoGenesis swarm network."""
    kernel = request.app.state.kernel
    
    try:
        # Check if swarm module is available
        if not hasattr(kernel, "swarm_coordinator"):
            return {
                "status": "disabled",
                "message": "Swarm functionality is not enabled in this instance"
            }
        
        coordinator = kernel.swarm_coordinator
        
        # Get swarm statistics and status
        stats = coordinator.get_swarm_statistics()
        
        return {
            "status": "active" if coordinator.is_active() else "inactive",
            "node_role": coordinator.node_role,
            "node_id": coordinator.node_id,
            "connected_nodes": stats.get("connected_nodes", 0),
            "active_tasks": stats.get("active_tasks", 0),
            "completed_tasks": stats.get("completed_tasks", 0),
            "memory_sync_status": stats.get("memory_sync_status", "unknown")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get swarm status: {str(e)}")


@router.get("/nodes", response_model=List[SwarmNodeResponse])
async def get_swarm_nodes(request: Request):
    """Get all nodes in the EvoGenesis swarm."""
    kernel = request.app.state.kernel
    
    try:
        if not hasattr(kernel, "swarm_coordinator"):
            return []
        
        coordinator = kernel.swarm_coordinator
        nodes = coordinator.get_connected_nodes()
        
        node_responses = []
        for node_id, node_data in nodes.items():
            node_responses.append({
                "id": node_id,
                "name": node_data.get("name", f"Node-{node_id[:6]}"),
                "role": node_data.get("role", "worker"),
                "url": node_data.get("url", ""),
                "status": node_data.get("status", "unknown"),
                "connected_at": node_data.get("connected_at", ""),
                "last_heartbeat": node_data.get("last_heartbeat", "")
            })
        
        return node_responses
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get swarm nodes: {str(e)}")


@router.post("/nodes", response_model=Dict[str, Any])
async def add_swarm_node(node: SwarmNodeCreate, request: Request):
    """Add a new node to the EvoGenesis swarm."""
    kernel = request.app.state.kernel
    
    try:
        if not hasattr(kernel, "swarm_coordinator"):
            raise HTTPException(status_code=400, detail="Swarm functionality is not enabled in this instance")
        
        coordinator = kernel.swarm_coordinator
        
        # Register the new node
        node_id = coordinator.register_node(
            name=node.name,
            role=node.role,
            url=node.url
        )
        
        return {
            "success": True,
            "message": f"Node {node.name} added to swarm successfully",
            "node_id": node_id
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add swarm node: {str(e)}")


@router.delete("/nodes/{node_id}", response_model=Dict[str, Any])
async def remove_swarm_node(node_id: str, request: Request):
    """Remove a node from the EvoGenesis swarm."""
    kernel = request.app.state.kernel
    
    try:
        if not hasattr(kernel, "swarm_coordinator"):
            raise HTTPException(status_code=400, detail="Swarm functionality is not enabled in this instance")
        
        coordinator = kernel.swarm_coordinator
        success = coordinator.deregister_node(node_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found in swarm")
        
        return {
            "success": True,
            "message": f"Node {node_id} removed from swarm successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove swarm node: {str(e)}")


@router.get("/tasks", response_model=List[SwarmTaskResponse])
async def get_swarm_tasks(request: Request):
    """Get all tasks in the EvoGenesis swarm."""
    kernel = request.app.state.kernel
    
    try:
        if not hasattr(kernel, "swarm_coordinator"):
            return []
        
        coordinator = kernel.swarm_coordinator
        tasks = coordinator.get_active_tasks()
        
        task_responses = []
        for task_id, task_data in tasks.items():
            task_responses.append({
                "id": task_id,
                "title": task_data.get("title", f"Task-{task_id[:6]}"),
                "description": task_data.get("description", ""),
                "priority": task_data.get("priority", "medium"),
                "status": task_data.get("status", "unknown"),
                "assigned_node_id": task_data.get("assigned_node_id", None),
                "created_at": task_data.get("created_at", ""),
                "started_at": task_data.get("started_at", None),
                "completed_at": task_data.get("completed_at", None),
                "progress": task_data.get("progress", 0.0),
                "result": task_data.get("result", None)
            })
        
        return task_responses
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get swarm tasks: {str(e)}")


@router.post("/tasks", response_model=Dict[str, Any])
async def create_swarm_task(task: SwarmTaskCreate, request: Request):
    """Create a new distributed task in the EvoGenesis swarm."""
    kernel = request.app.state.kernel
    
    try:
        if not hasattr(kernel, "swarm_coordinator"):
            raise HTTPException(status_code=400, detail="Swarm functionality is not enabled in this instance")
        
        coordinator = kernel.swarm_coordinator
        
        # Create the task
        task_id = coordinator.create_task(
            title=task.title,
            description=task.description,
            priority=task.priority,
            assigned_node_id=task.assigned_node_id
        )
        
        return {
            "success": True,
            "message": f"Task {task.title} created successfully in swarm",
            "task_id": task_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create swarm task: {str(e)}")


@router.post("/connect", response_model=Dict[str, Any])
async def connect_to_swarm(request: Request):
    """Connect this instance to an existing EvoGenesis swarm network."""
    kernel = request.app.state.kernel
    
    try:
        data = await request.json()
        coordinator_url = data.get("coordinator_url")
        
        if not coordinator_url:
            raise HTTPException(status_code=400, detail="Coordinator URL is required")
        
        if not hasattr(kernel, "swarm_coordinator"):
            raise HTTPException(status_code=400, detail="Swarm functionality is not enabled in this instance")
        
        coordinator = kernel.swarm_coordinator
        
        # Connect to the swarm
        success = coordinator.connect_to_swarm(coordinator_url)
        
        if success:
            return {
                "success": True,
                "message": f"Successfully connected to swarm at {coordinator_url}",
                "node_id": coordinator.node_id,
                "role": coordinator.node_role
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to connect to swarm")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to connect to swarm: {str(e)}")


@router.post("/disconnect", response_model=Dict[str, Any])
async def disconnect_from_swarm(request: Request):
    """Disconnect this instance from the EvoGenesis swarm network."""
    kernel = request.app.state.kernel
    
    try:
        if not hasattr(kernel, "swarm_coordinator"):
            raise HTTPException(status_code=400, detail="Swarm functionality is not enabled in this instance")
        
        coordinator = kernel.swarm_coordinator
        
        # Disconnect from the swarm
        success = coordinator.disconnect_from_swarm()
        
        if success:
            return {
                "success": True,
                "message": "Successfully disconnected from swarm"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to disconnect from swarm")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to disconnect from swarm: {str(e)}")


@router.post("/sync-memory", response_model=Dict[str, Any])
async def sync_swarm_memory(request: Request):
    """Synchronize memory across all nodes in the EvoGenesis swarm."""
    kernel = request.app.state.kernel
    
    try:
        if not hasattr(kernel, "swarm_coordinator"):
            raise HTTPException(status_code=400, detail="Swarm functionality is not enabled in this instance")
        
        coordinator = kernel.swarm_coordinator
        
        # Sync memory
        result = coordinator.sync_memory()
        
        return {
            "success": result.get("success", False),
            "message": result.get("message", "Memory synchronization initiated"),
            "synced_nodes": result.get("synced_nodes", 0),
            "sync_id": result.get("sync_id", "")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to sync swarm memory: {str(e)}")


def add_routes(app):
    """Add swarm routes to the main app."""
    app.include_router(router)
