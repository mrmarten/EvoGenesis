"""
Web UI Server for EvoGenesis Control Panel

This module provides a FastAPI-based web server for the EvoGenesis Control Panel.
"""

import os
import json
import logging
import asyncio
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Set

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Path configuration
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")

# Initialize FastAPI app
app = FastAPI(title="EvoGenesis Control Panel", 
              description="Web interface for monitoring and controlling the EvoGenesis system",
              version="0.1.0")

# Mount static files
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Initialize templates
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# Kernel reference (will be set when server is started)
kernel = None

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.client_subscriptions: Dict[WebSocket, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket, topics: List[str] = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        self.client_subscriptions[websocket] = set(topics or ["system"])
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if websocket in self.client_subscriptions:
            del self.client_subscriptions[websocket]
    
    async def broadcast(self, topic: str, message: Dict[str, Any]):
        """Broadcast a message to all connected clients subscribed to the topic."""
        data = {
            "topic": topic,
            "timestamp": datetime.now().isoformat(),
            "data": message
        }
        
        for connection in self.active_connections:
            if topic in self.client_subscriptions.get(connection, set()):
                try:
                    await connection.send_json(data)
                except Exception as e:
                    logging.error(f"Error sending message to client: {str(e)}")
                    # Connection might be dead, disconnect it
                    self.disconnect(connection)

# Initialize connection manager
manager = ConnectionManager()

# Routes
@app.get("/", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    """Render the main dashboard."""
    return templates.TemplateResponse(
        "dashboard.html", 
        {"request": request, "title": "EvoGenesis Control Panel"}
    )

@app.get("/agents", response_class=HTMLResponse)
async def get_agents_view(request: Request):
    """Render the agents view."""
    return templates.TemplateResponse(
        "agents.html", 
        {"request": request, "title": "EvoGenesis Agents"}
    )

@app.get("/teams", response_class=HTMLResponse)
async def get_teams_view(request: Request):
    """Render the teams view."""
    return templates.TemplateResponse(
        "teams.html", 
        {"request": request, "title": "EvoGenesis Teams"}
    )

@app.get("/tasks", response_class=HTMLResponse)
async def get_tasks_view(request: Request):
    """Render the tasks view."""
    return templates.TemplateResponse(
        "tasks.html", 
        {"request": request, "title": "EvoGenesis Tasks"}
    )

@app.get("/memory", response_class=HTMLResponse)
async def get_memory_view(request: Request):
    """Render the memory view."""
    return templates.TemplateResponse(
        "memory.html", 
        {"request": request, "title": "EvoGenesis Memory"}
    )

@app.get("/tools", response_class=HTMLResponse)
async def get_tools_view(request: Request):
    """Render the tools view."""
    return templates.TemplateResponse(
        "tools.html", 
        {"request": request, "title": "EvoGenesis Tools"}
    )

@app.get("/logs", response_class=HTMLResponse)
async def get_logs_view(request: Request):
    """Render the logs view."""
    return templates.TemplateResponse(
        "logs.html", 
        {"request": request, "title": "EvoGenesis Logs"}
    )

@app.get("/settings", response_class=HTMLResponse)
async def get_settings_view(request: Request):
    """Render the settings view."""
    return templates.TemplateResponse(
        "settings.html", 
        {"request": request, "title": "EvoGenesis Settings"}
    )

# API endpoints
@app.get("/api/system/status")
async def get_system_status():
    """Get the current system status."""
    if not kernel:
        return {"status": "not_initialized"}
    
    try:
        # Get status from each module
        agent_manager_status = kernel.agent_manager.get_status()
        task_planner_status = kernel.task_planner.get_status()
        memory_manager_status = {"status": "active"}  # Placeholder
        tooling_system_status = kernel.tooling_system.get_status()
        
        # Calculate system health
        components = [
            agent_manager_status.get("status") == "active",
            task_planner_status.get("status") == "active",
            memory_manager_status.get("status") == "active",
            tooling_system_status.get("status") == "active"
        ]
        health_percentage = (sum(1 for c in components if c) / len(components)) * 100
        
        return {
            "status": "active",
            "health": health_percentage,
            "active_agents": agent_manager_status.get("active_agents", 0),
            "running_tasks": task_planner_status.get("pending_tasks", 0),
            "components": {
                "agent_manager": agent_manager_status,
                "task_planner": task_planner_status,
                "memory_manager": memory_manager_status,
                "tooling_system": tooling_system_status
            }
        }
    except Exception as e:
        logging.error(f"Error getting system status: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/api/agents")
async def get_agents():
    """Get all agents in the system."""
    if not kernel:
        return []
    
    try:
        agents = kernel.agent_manager.agents
        result = []
        for agent_id, agent in agents.items():
            result.append({
                "id": agent_id,
                "name": agent.name if hasattr(agent, "name") else "Unknown",
                "type": agent.attributes.get("agent_type", "Unknown") if hasattr(agent, "attributes") else "Unknown",
                "status": agent.status if hasattr(agent, "status") else "Unknown",
                "capabilities": agent.capabilities if hasattr(agent, "capabilities") else [],
                "team_id": agent.attributes.get("team_id") if hasattr(agent, "attributes") else None
            })
        
        return result
    except Exception as e:
        logging.error(f"Error getting agents: {str(e)}")
        return []

@app.get("/api/teams")
async def get_teams():
    """Get all teams in the system."""
    if not kernel:
        return []
    try:
        teams = kernel.agent_manager.teams
        result = []
        for team_id, team in teams.items():
            result.append({
                "id": team_id,
                "name": team.name if hasattr(team, "name") else "Unknown",
                "goal": team.goals[0] if hasattr(team, "goals") and team.goals else "Unknown",
                "agent_count": len(team.agents) if hasattr(team, "agents") else 0,
                "status": team.status if hasattr(team, "status") else "Unknown"
            })
        
        return result
    except Exception as e:
        logging.error(f"Error getting teams: {str(e)}")
        return []

@app.get("/api/tasks")
async def get_tasks():
    """Get all tasks in the system."""
    if not kernel:
        return []
    
    try:
        tasks = kernel.task_planner.tasks
        result = []
        
        for task_id, task in tasks.items():
            result.append({
                "id": task_id,
                "name": task.name,
                "description": task.description,
                "status": task.status,
                "progress": task.progress,
                "assigned_agent_id": task.assigned_agent_id,
                "assigned_team_id": task.assigned_team_id,
                "parent_id": task.parent_id,
                "created_at": task.created_at,
                "started_at": task.started_at,
                "completed_at": task.completed_at
            })
        
        return result
    except Exception as e:
        logging.error(f"Error getting tasks: {str(e)}")
        return []

@app.get("/api/tools")
async def get_tools():
    """Get all tools in the system."""
    if not kernel:
        return []
    
    try:
        tools = kernel.tooling_system.tool_registry
        result = []
        
        for tool_id, tool in tools.items():
            result.append({
                "id": tool_id,
                "name": tool.name,
                "description": tool.description,
                "status": tool.status,
                "scope": tool.scope,
                "sandbox_type": tool.sandbox_type,
                "auto_generated": tool.auto_generated,
                "execution_count": tool.execution_count,
                "success_count": tool.success_count,
                "error_count": tool.error_count,
                "average_execution_time": tool.average_execution_time
            })
        
        return result
    except Exception as e:
        logging.error(f"Error getting tools: {str(e)}")
        return []

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Handle messages from the client (e.g., subscription updates)
            data = await websocket.receive_json()
            if "subscribe" in data:
                topics = data["subscribe"]
                manager.client_subscriptions[websocket] = set(topics)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logging.error(f"WebSocket error: {str(e)}")
        manager.disconnect(websocket)

# Control endpoints
@app.post("/api/system/start")
async def start_system():
    """Start the EvoGenesis system."""
    if not kernel:
        raise HTTPException(status_code=500, detail="Kernel not initialized")
    
    try:
        kernel.start()
        await manager.broadcast("system", {"event": "system_started"})
        return {"status": "started"}
    except Exception as e:
        logging.error(f"Error starting system: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/system/stop")
async def stop_system():
    """Stop the EvoGenesis system."""
    if not kernel:
        raise HTTPException(status_code=500, detail="Kernel not initialized")
    
    try:
        kernel.stop()
        await manager.broadcast("system", {"event": "system_stopped"})
        return {"status": "stopped"}
    except Exception as e:
        logging.error(f"Error stopping system: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/system/pause")
async def pause_system():
    """Pause the EvoGenesis system (if supported)."""
    if not kernel:
        raise HTTPException(status_code=500, detail="Kernel not initialized")
    
    try:
        # This would need implementation in the kernel
        # For now, just broadcast the event
        await manager.broadcast("system", {"event": "system_paused"})
        return {"status": "paused"}
    except Exception as e:
        logging.error(f"Error pausing system: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Event monitoring task
async def event_monitor():
    """Monitor events from the kernel and broadcast to clients."""
    while True:
        try:
            if kernel:
                # Here we would listen for events from the kernel
                # For now, just send a heartbeat
                await manager.broadcast("system", {"event": "heartbeat"})
                
                # Get latest system status and broadcast
                status = await get_system_status()
                await manager.broadcast("system.status", status)
            
            await asyncio.sleep(5)  # Check every 5 seconds
        except Exception as e:
            logging.error(f"Error in event monitor: {str(e)}")
            await asyncio.sleep(10)  # Back off on error

# Start the server
def start_server(kernel_instance, host="0.0.0.0", port=5000):
    """Start the web UI server."""
    global kernel
    kernel = kernel_instance
    
    # Start the event monitor
    @app.on_event("startup")
    async def startup_event():
        asyncio.create_task(event_monitor())
    
    # Run the server
    # Setting log_level to prevent duplicate logs
    uvicorn.run(app, host=host, port=port, log_level="info")
