"""
Task Routes for EvoGenesis Web UI

This module provides API routes for managing tasks in the EvoGenesis system.
"""

import json
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel


# Define models
class TaskBase(BaseModel):
    title: str
    description: str
    priority: str
    assignee_id: Optional[str] = None
    team_id: Optional[str] = None
    deadline: Optional[str] = None


class TaskCreate(TaskBase):
    pass


class TaskUpdate(TaskBase):
    pass


class TaskResponse(TaskBase):
    id: str
    status: str
    progress: int
    created_at: str


# Create router
router = APIRouter(
    prefix="/api/tasks",
    tags=["tasks"],
)


# Define routes
@router.get("/", response_model=List[TaskResponse])
async def get_tasks(request: Request):
    """Get all tasks in the system."""
    kernel = request.app.state.kernel
    
    try:
        task_planner = kernel.get_module("task_planner")
        tasks = task_planner.list_tasks()
        
        # Transform to response format
        task_responses = []
        for task_id, task_data in tasks.items():
            task_responses.append({
                "id": task_id,
                "title": task_data.title if hasattr(task_data, "title") else f"Task-{task_id[:6]}",
                "description": task_data.description if hasattr(task_data, "description") else "",
                "priority": task_data.priority if hasattr(task_data, "priority") else "medium",
                "status": task_data.status if hasattr(task_data, "status") else "pending",
                "progress": task_data.progress if hasattr(task_data, "progress") else 0,
                "assignee_id": task_data.assignee_id if hasattr(task_data, "assignee_id") else None,
                "team_id": task_data.team_id if hasattr(task_data, "team_id") else None,
                "deadline": task_data.deadline if hasattr(task_data, "deadline") else None,
                "created_at": task_data.created_at if hasattr(task_data, "created_at") else ""
            })
        
        return task_responses
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tasks: {str(e)}")


@router.get("/{task_id}", response_model=Dict[str, Any])
async def get_task(task_id: str, request: Request):
    """Get details of a specific task."""
    kernel = request.app.state.kernel
    
    try:
        task_planner = kernel.get_module("task_planner")
        task = task_planner.get_task(task_id)
        
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        # Get additional information like subtasks, history, etc.
        task_details = {
            "id": task_id,
            "title": task.title if hasattr(task, "title") else f"Task-{task_id[:6]}",
            "description": task.description if hasattr(task, "description") else "",
            "priority": task.priority if hasattr(task, "priority") else "medium",
            "status": task.status if hasattr(task, "status") else "pending",
            "progress": task.progress if hasattr(task, "progress") else 0,
            "assignee_id": task.assignee_id if hasattr(task, "assignee_id") else None,
            "team_id": task.team_id if hasattr(task, "team_id") else None,
            "deadline": task.deadline if hasattr(task, "deadline") else None,
            "created_at": task.created_at if hasattr(task, "created_at") else "",
            "subtasks": task.subtasks if hasattr(task, "subtasks") else [],
            "history": task.history if hasattr(task, "history") else []
        }
        
        return task_details
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get task {task_id}: {str(e)}")


@router.post("/", response_model=Dict[str, Any])
async def create_task(task: TaskCreate, request: Request):
    """Create a new task."""
    kernel = request.app.state.kernel
    
    try:
        task_planner = kernel.get_module("task_planner")
        
        # Create the task
        task_id = task_planner.create_task(
            title=task.title,
            description=task.description,
            priority=task.priority,
            assignee_id=task.assignee_id,
            team_id=task.team_id,
            deadline=task.deadline
        )
        
        return {
            "success": True,
            "message": f"Task {task.title} created successfully",
            "task_id": task_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create task: {str(e)}")


@router.put("/{task_id}", response_model=Dict[str, Any])
async def update_task(task_id: str, task: TaskUpdate, request: Request):
    """Update a task."""
    kernel = request.app.state.kernel
    
    try:
        task_planner = kernel.get_module("task_planner")
        success = task_planner.update_task(
            task_id=task_id,
            title=task.title,
            description=task.description,
            priority=task.priority,
            assignee_id=task.assignee_id,
            team_id=task.team_id,
            deadline=task.deadline
        )
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        return {
            "success": True,
            "message": f"Task {task_id} updated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update task {task_id}: {str(e)}")


@router.delete("/{task_id}", response_model=Dict[str, Any])
async def delete_task(task_id: str, request: Request):
    """Delete a task."""
    kernel = request.app.state.kernel
    
    try:
        task_planner = kernel.get_module("task_planner")
        success = task_planner.delete_task(task_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        return {
            "success": True,
            "message": f"Task {task_id} deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete task {task_id}: {str(e)}")


@router.post("/auto-plan", response_model=Dict[str, Any])
async def auto_plan_tasks(request: Request, goal: str):
    """Auto-create tasks based on a natural language goal description."""
    kernel = request.app.state.kernel
    
    try:
        task_planner = kernel.get_module("task_planner")
        llm_orchestrator = kernel.get_module("llm_orchestrator")
        
        # Use LLM to break down the goal into tasks
        prompt = f"""
        Based on the following goal, break it down into specific tasks:
        
        Goal: {goal}
        
        Generate a JSON response with the following structure:
        {{
            "tasks": [
                {{
                    "title": "Task 1 title",
                    "description": "Detailed description of task 1",
                    "priority": "high|medium|low",
                    "estimated_duration": "Time estimate"
                }},
                // More tasks...
            ]
        }}
        """
        
        llm_response = await llm_orchestrator.generate(prompt)
        task_plan = json.loads(llm_response)
        
        # Create the tasks from the generated plan
        created_tasks = []
        for task_config in task_plan["tasks"]:
            task_id = task_planner.create_task(
                title=task_config["title"],
                description=task_config["description"],
                priority=task_config["priority"]
            )
            created_tasks.append({
                "task_id": task_id,
                "title": task_config["title"]
            })
        
        return {
            "success": True,
            "message": f"Created {len(created_tasks)} tasks for goal: {goal}",
            "tasks": created_tasks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to auto-plan tasks: {str(e)}")


@router.post("/{task_id}/status", response_model=Dict[str, Any])
async def update_task_status(task_id: str, status: str, request: Request):
    """Update the status of a task."""
    kernel = request.app.state.kernel
    
    try:
        task_planner = kernel.get_module("task_planner")
        success = task_planner.update_task_status(task_id, status)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        return {
            "success": True,
            "message": f"Task {task_id} status updated to {status}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update task status: {str(e)}")


@router.post("/{task_id}/progress", response_model=Dict[str, Any])
async def update_task_progress(task_id: str, progress: int, request: Request):
    """Update the progress of a task."""
    kernel = request.app.state.kernel
    
    if progress < 0 or progress > 100:
        raise HTTPException(status_code=400, detail="Progress must be between 0 and 100")
    
    try:
        task_planner = kernel.get_module("task_planner")
        success = task_planner.update_task_progress(task_id, progress)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        return {
            "success": True,
            "message": f"Task {task_id} progress updated to {progress}%"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update task progress: {str(e)}")


@router.post("/{task_id}/start", response_model=Dict[str, Any])
async def start_task(task_id: str, request: Request):
    """Start a task."""
    kernel = request.app.state.kernel
    
    try:
        task_planner = kernel.get_module("task_planner")
        success = task_planner.start_task(task_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found or cannot be started")
        
        # Broadcast task status change via WebSocket
        ws_manager = request.app.state.ws_manager
        await ws_manager.broadcast_to_topic(
            "tasks.status",
            {
                "event": "task_started",
                "task_id": task_id
            }
        )
        
        return {
            "success": True,
            "message": f"Task {task_id} started successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start task {task_id}: {str(e)}")


@router.post("/{task_id}/pause", response_model=Dict[str, Any])
async def pause_task(task_id: str, request: Request):
    """Pause a task."""
    kernel = request.app.state.kernel
    
    try:
        task_planner = kernel.get_module("task_planner")
        success = task_planner.pause_task(task_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found or cannot be paused")
        
        # Broadcast task status change via WebSocket
        ws_manager = request.app.state.ws_manager
        await ws_manager.broadcast_to_topic(
            "tasks.status",
            {
                "event": "task_paused",
                "task_id": task_id
            }
        )
        
        return {
            "success": True,
            "message": f"Task {task_id} paused successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to pause task {task_id}: {str(e)}")


@router.post("/{task_id}/stop", response_model=Dict[str, Any])
async def stop_task(task_id: str, request: Request):
    """Stop a task."""
    kernel = request.app.state.kernel
    
    try:
        task_planner = kernel.get_module("task_planner")
        success = task_planner.stop_task(task_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found or cannot be stopped")
        
        # Broadcast task status change via WebSocket
        ws_manager = request.app.state.ws_manager
        await ws_manager.broadcast_to_topic(
            "tasks.status",
            {
                "event": "task_stopped",
                "task_id": task_id
            }
        )
        
        return {
            "success": True,
            "message": f"Task {task_id} stopped successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop task {task_id}: {str(e)}")


def add_routes(app):
    """Add task routes to the main app."""
    app.include_router(router)
