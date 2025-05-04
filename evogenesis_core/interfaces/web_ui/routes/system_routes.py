"""
System Routes for EvoGenesis Web UI

This module provides API routes for system management and monitoring.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel


# Create router
router = APIRouter(
    prefix="/api/system",
    tags=["system"],
)


# Define routes
@router.get("/status", response_model=Dict[str, Any])
async def get_system_status(request: Request):
    """Get the current system status."""
    kernel = request.app.state.kernel
    
    try:
        # Get status information from various modules
        status = {
            "status": "active",  # or "paused", "stopped"
            "health": 98.5,      # percentage
            "uptime": 0,         # seconds
            "active_agents": 0,
            "running_tasks": 0,
            "memory_usage": 0,   # MB
            "module_status": {},
            "resources": {}
        }
        
        # Get basic system status from kernel
        kernel_status = kernel.get_status()
        if kernel_status:
            status["status"] = kernel_status.get("status", "unknown")
            status["uptime"] = kernel_status.get("uptime", 0)
        
        # Get agent status
        agent_factory = kernel.get_module("agent_factory") # Changed from agent_manager
        if agent_factory:
            agents = agent_factory.list_agents()
            status["active_agents"] = sum(1 for a in agents.values() 
                                         if hasattr(a, "status") and a.status == "active")
        
        # Get task status
        task_planner = kernel.get_module("task_planner")
        if task_planner:
            tasks = task_planner.list_tasks()
            status["running_tasks"] = sum(1 for t in tasks.values() 
                                         if hasattr(t, "status") and t.status == "running")
        
        # Get module status
        for module_name in [
            "agent_factory", "task_planner", "memory_manager", 
            "tooling_system", "llm_orchestrator", "hitl_interface",
            "self_evolution_engine", "strategic_opportunity_observatory"
        ]:
            module = kernel.get_module(module_name)
            if module:
                module_status = module.get_status() if hasattr(module, "get_status") else {}
                status["module_status"][module_name] = {
                    "status": module_status.get("status", "inactive"),
                    "health": module_status.get("health", 0),
                    "message": module_status.get("message", "")
                }
        
        # Calculate overall health based on module health
        if status["module_status"]:
            health_values = [m.get("health", 0) for m in status["module_status"].values()]
            if health_values:
                status["health"] = sum(health_values) / len(health_values)
        
        # Get resource usage
        import psutil
        status["resources"] = {
            "cpu": psutil.cpu_percent(),
            "memory": psutil.virtual_memory().percent,
            "disk": psutil.disk_usage('/').percent
        }
        
        # Calculate memory usage
        process = psutil.Process()
        status["memory_usage"] = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {str(e)}")


@router.post("/start", response_model=Dict[str, Any])
async def start_system(request: Request):
    """Start the EvoGenesis system."""
    kernel = request.app.state.kernel
    
    try:
        success = kernel.start()
        
        return {
            "success": success,
            "message": "System started successfully" if success else "Failed to start system"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start system: {str(e)}")


@router.post("/stop", response_model=Dict[str, Any])
async def stop_system(request: Request):
    """Stop the EvoGenesis system."""
    kernel = request.app.state.kernel
    
    try:
        success = kernel.stop()
        
        return {
            "success": success,
            "message": "System stopped successfully" if success else "Failed to stop system"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop system: {str(e)}")


@router.post("/pause", response_model=Dict[str, Any])
async def pause_system(request: Request):
    """Pause the EvoGenesis system."""
    kernel = request.app.state.kernel
    
    try:
        success = kernel.pause()
        
        return {
            "success": success,
            "message": "System paused successfully" if success else "Failed to pause system"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to pause system: {str(e)}")


@router.post("/restart", response_model=Dict[str, Any])
async def restart_system(request: Request):
    """Restart the EvoGenesis system."""
    kernel = request.app.state.kernel
    
    try:
        success = kernel.restart()
        
        return {
            "success": success,
            "message": "System restarted successfully" if success else "Failed to restart system"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restart system: {str(e)}")


@router.get("/modules", response_model=List[Dict[str, Any]])
async def get_modules(request: Request):
    """Get a list of all modules in the system."""
    kernel = request.app.state.kernel
    
    try:
        modules = []
        
        # Get all modules from kernel
        for module_name in kernel.list_modules():
            module = kernel.get_module(module_name)
            if module:
                module_info = {
                    "name": module_name,
                    "version": getattr(module, "version", "0.1.0"),
                    "status": "active" if getattr(module, "is_active", False) else "inactive",
                    "description": getattr(module, "description", "")
                }
                modules.append(module_info)
        
        return modules
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get modules: {str(e)}")


@router.get("/activities", response_model=List[Dict[str, Any]])
async def get_recent_activities(
    request: Request,
    limit: int = 10,
    activity_type: Optional[str] = None
):
    """Get recent system activities."""
    kernel = request.app.state.kernel
    
    try:
        # This assumes there's an activity log in the kernel
        # Adapt to actual implementation
        activities = kernel.get_recent_activities(limit, activity_type) if hasattr(kernel, "get_recent_activities") else []
        
        return activities
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get recent activities: {str(e)}")


@router.get("/performance", response_model=Dict[str, Any])
async def get_performance_metrics(
    request: Request,
    timeframe: str = "day"  # day, week, month
):
    """Get system performance metrics over time."""
    kernel = request.app.state.kernel
    
    try:
        # This assumes there's a performance monitoring system
        # Adapt to actual implementation
        metrics = kernel.get_performance_metrics(timeframe) if hasattr(kernel, "get_performance_metrics") else {
            "timestamps": [],
            "cpu_usage": [],
            "memory_usage": [],
            "task_completion_rate": [],
            "response_times": []
        }
        
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance metrics: {str(e)}")


def add_routes(app):
    """Add system routes to the main app."""
    app.include_router(router)
