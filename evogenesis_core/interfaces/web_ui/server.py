"""
Web UI Server for EvoGenesis Control Panel

This module provides a FastAPI-based web server for the EvoGenesis Control Panel.
"""

import os
import json
import logging
import asyncio
import uuid
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Set

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Import route modules
from evogenesis_core.interfaces.web_ui.routes.agent_routes import add_routes as add_agent_routes
from evogenesis_core.interfaces.web_ui.routes.task_routes import add_routes as add_task_routes
from evogenesis_core.interfaces.web_ui.routes.team_routes import add_routes as add_team_routes
from evogenesis_core.interfaces.web_ui.routes.memory_routes import add_routes as add_memory_routes
from evogenesis_core.interfaces.web_ui.routes.tool_routes import add_routes as add_tool_routes
from evogenesis_core.interfaces.web_ui.routes.llm_routes import add_routes as add_llm_routes
from evogenesis_core.interfaces.web_ui.routes.settings_routes import add_routes as add_settings_routes
from evogenesis_core.interfaces.web_ui.routes.system_routes import add_routes as add_system_routes
from evogenesis_core.interfaces.web_ui.routes.log_routes import add_routes as add_log_routes
from evogenesis_core.interfaces.web_ui.routes.self_evolution_routes import add_routes as add_self_evolution_routes
from evogenesis_core.interfaces.web_ui.routes.observatory_routes import add_routes as add_observatory_routes
from evogenesis_core.interfaces.web_ui.routes.swarm_routes import add_routes as add_swarm_routes
from evogenesis_core.interfaces.web_ui.routes.activity_routes import add_routes as add_activity_routes

# Import WebSocket manager
from evogenesis_core.interfaces.web_ui.ws_manager import WebSocketManager
from evogenesis_core.interfaces.web_ui.handlers import init_ws_handlers, broadcast_system_event, get_component_status

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

# Initialize WebSocket manager
ws_manager = WebSocketManager()

# Add all API routes 
def setup_routes(app):
    """Set up all API routes for the EvoGenesis Control Panel."""
    # Add routes from all modules
    add_agent_routes(app)
    add_task_routes(app)
    add_team_routes(app)
    add_memory_routes(app)
    add_tool_routes(app)
    add_llm_routes(app)
    add_settings_routes(app)
    add_system_routes(app)
    add_log_routes(app)
    add_self_evolution_routes(app)
    add_observatory_routes(app)
    add_swarm_routes(app)
    add_activity_routes(app)
    
    # Return app for chaining
    return app

# Set up all routes
setup_routes(app)

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

@app.get("/wizards/new-project", response_class=HTMLResponse)
async def get_new_project_wizard(request: Request):
    """Render the new project wizard."""
    return templates.TemplateResponse(
        "wizards/new_project.html", 
        {"request": request, "title": "Create New Project"}
    )

@app.get("/wizards/team", response_class=HTMLResponse)
async def get_team_wizard(request: Request):
    """Render the team creation wizard."""
    return templates.TemplateResponse(
        "wizards/team.html", 
        {"request": request, "title": "Create Agent Team"}
    )

@app.get("/wizards/observatory", response_class=HTMLResponse)
async def get_observatory_wizard(request: Request):
    """Render the Strategic Observatory wizard."""
    return templates.TemplateResponse(
        "wizards/observatory.html", 
        {"request": request, "title": "Configure Strategic Observatory"}
    )

@app.get("/wizards/memory-system", response_class=HTMLResponse)
async def get_memory_system_wizard(request: Request):
    """Render the Memory System wizard."""
    return templates.TemplateResponse(
        "wizards/memory_system.html", 
        {"request": request, "title": "Configure Memory System"}
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

@app.get("/activities", response_class=HTMLResponse)
async def get_activities_view(request: Request):
    """Render the activities view."""
    return templates.TemplateResponse(
        "activities.html", 
        {"request": request, "title": "System Activities"}
    )

@app.get("/observatory", response_class=HTMLResponse)
async def get_observatory_view(request: Request):
    """Render the Strategic Opportunity Observatory dashboard."""
    return templates.TemplateResponse(
        "soo_dashboard.html", 
        {"request": request, "title": "Strategic Opportunity Observatory"}
    )

@app.get("/self-evolution", response_class=HTMLResponse)
async def get_self_evolution_view(request: Request):
    """Render the Self-Evolution Engine view."""
    return templates.TemplateResponse(
        "self_evolution.html", 
        {"request": request, "title": "Self-Evolution Engine"}
    )

@app.get("/settings", response_class=HTMLResponse)
async def get_settings_view(request: Request):
    """Render the settings view."""
    return templates.TemplateResponse(
        "settings.html", 
        {"request": request, "title": "EvoGenesis Settings"}
    )

@app.get("/swarm", response_class=HTMLResponse)
async def get_swarm_view(request: Request):
    """Render the swarm view."""
    return templates.TemplateResponse(
        "swarm.html", 
        {"request": request, "title": "EvoGenesis Swarm Network"}
    )

# API endpoints
@app.get("/api/system/status")
async def get_system_status():
    """Get the current system status."""
    if not kernel:
        return {"status": "not_initialized"}
    
    try:
        # Get status from each module
        agent_factory_status = kernel.agent_factory.get_status()
        task_planner_status = kernel.task_planner.get_status()
        memory_manager_status = kernel.memory_manager.get_status()
        tooling_system_status = kernel.tooling_system.get_status()
        
        # Calculate system health
        components = [
            agent_factory_status.get("status") == "active",
            task_planner_status.get("status") == "active",
            memory_manager_status.get("status") == "active",
            tooling_system_status.get("status") == "active"
        ]
        health_percentage = (sum(1 for c in components if c) / len(components)) * 100 if components else 0
        
        # Get recent activities
        recent_activities = kernel.get_recent_activities(5)
        
        return {
            "status": "active",
            "health": health_percentage,
            "active_agents": agent_factory_status.get("active_agents", 0),
            "running_tasks": task_planner_status.get("pending_tasks", 0),
            "components": {
                "agent_manager": agent_factory_status,
                "task_planner": task_planner_status,
                "memory_manager": memory_manager_status,
                "tooling_system": tooling_system_status
            },
            "recent_activities": recent_activities
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
        agents = kernel.agent_factory.agents
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
        teams = kernel.agent_factory.teams
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
    # Generate a unique client ID
    client_id = str(uuid.uuid4())
    await ws_manager.connect(websocket, client_id)
    
    try:
        # Send initial connection info with available topics
        await websocket.send_json({
            "type": "connection_info",
            "client_id": client_id,
            "available_topics": [
                "system", "system.status", "agents", "agents.status", "agents.created",
                "tasks", "tasks.status", "memory", "tools", "self_evolution",
                "strategic_observatory", "swarm", "metrics", "logs"
            ],
            "server_time": datetime.now().isoformat()
        })
        
        while True:
            # Handle messages from the client
            data = await websocket.receive_json()
            
            if "subscribe" in data:
                topics = data["subscribe"]
                if isinstance(topics, list):
                    await ws_manager.subscribe_to_topics(client_id, topics)
            
            elif "unsubscribe" in data:
                topics = data["unsubscribe"]
                if isinstance(topics, list):
                    await ws_manager.unsubscribe_from_topics(client_id, topics)
                    
            elif "action" in data:
                # Handle client-initiated actions
                action = data["action"]
                action_data = data.get("data", {})
                
                # Log the action for auditing
                if kernel:
                    kernel.log_activity(
                        activity_type="client_action",
                        title=f"Client action: {action}",
                        message=f"Client {client_id} initiated action: {action}",
                        data={"action": action, "data": action_data}
                    )
                
                # Process specific actions
                if action == "refresh_agents":
                    agents_data = await get_agents()
                    await ws_manager.broadcast_to_topic(f"client.{client_id}", {
                        "event": "agents_refreshed",
                        "data": agents_data
                    })
                    
                elif action == "refresh_tasks":
                    tasks_data = await get_tasks()
                    await ws_manager.broadcast_to_topic(f"client.{client_id}", {
                        "event": "tasks_refreshed",
                        "data": tasks_data
                    })
                    
                elif action == "refresh_memory":
                    # Get memory data from the memory manager
                    if kernel and hasattr(kernel, "memory_manager"):
                        memory_data = kernel.memory_manager.get_status()
                        await ws_manager.broadcast_to_topic(f"client.{client_id}", {
                            "event": "memory_refreshed",
                            "data": memory_data
                        })
            
            elif "query" in data:
                # Handle client queries
                query_type = data.get("query_type")
                query_params = data.get("params", {})
                
                if query_type == "memory_search":
                    # Example: search memory based on query parameters
                    if kernel and hasattr(kernel, "memory_manager"):
                        namespace = query_params.get("namespace", "default")
                        query = query_params.get("query", "")
                        limit = query_params.get("limit", 10)
                        
                        try:
                            # This would be implemented in memory_manager
                            search_results = kernel.memory_manager.search(
                                namespace=namespace,
                                query=query,
                                limit=limit
                            )
                            
                            await websocket.send_json({
                                "type": "query_result",
                                "query_id": data.get("query_id"),
                                "results": search_results
                            })
                        except Exception as e:
                            await websocket.send_json({
                                "type": "query_error",
                                "query_id": data.get("query_id"),
                                "error": str(e)
                            })
                
    except WebSocketDisconnect:
        ws_manager.disconnect(client_id)
    except Exception as e:
        logging.error(f"WebSocket error: {str(e)}")
        ws_manager.disconnect(client_id)

# Self-Evolution Engine API endpoints
@app.get("/api/self-evolution/status")
async def get_self_evolution_status():
    """Get the current status of the Self-Evolution Engine."""
    if not kernel:
        return {"status": "not_initialized"}
    
    try:
        return kernel.self_evolution_engine.get_status()
    except Exception as e:
        logging.error(f"Error getting Self-Evolution Engine status: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.get("/api/self-evolution/updates")
async def get_self_evolution_updates():
    """Get all evolution updates."""
    if not kernel:
        return []
    
    try:
        updates = kernel.self_evolution_engine.updates
        result = []
        
        for update_id, update in updates.items():
            if hasattr(update, 'to_dict'):
                # Use the to_dict method if available
                update_dict = update.to_dict()
            else:
                # Manually extract the data
                update_dict = {
                    "update_id": update.update_id,
                    "title": update.title,
                    "description": update.description,
                    "affected_components": update.affected_components,
                    "priority": update.priority.value if hasattr(update.priority, 'value') else update.priority,
                    "proposed_by": update.proposed_by,
                    "status": update.status.value if hasattr(update.status, 'value') else update.status,
                    "created_at": update.created_at.isoformat(),
                    "updated_at": update.updated_at.isoformat(),
                    "deployed_at": update.deployed_at.isoformat() if update.deployed_at else None,
                    "votes": update.votes,
                }
            
            result.append(update_dict)
        
        return result
    except Exception as e:
        logging.error(f"Error getting evolution updates: {str(e)}")
        return []

@app.get("/api/self-evolution/abtests")
async def get_self_evolution_abtests():
    """Get all A/B tests."""
    if not kernel:
        return []
    
    try:
        active_tests = kernel.self_evolution_engine.active_ab_tests
        result = []
        
        for test_id, test_data in active_tests.items():
            result.append({
                "test_id": test_id,
                "feature": test_data.get("feature", "Unknown"),
                "status": test_data.get("status", "Unknown"),
                "started_at": test_data.get("started_at", "Unknown"),
                "duration": test_data.get("duration", 0),
                "version_a": test_data.get("version_a", {}).get("name", "Current"),
                "version_b": test_data.get("version_b", {}).get("name", "Experimental"),
                "results": test_data.get("results", None)
            })
        
        return result
    except Exception as e:
        logging.error(f"Error getting A/B tests: {str(e)}")
        return []

@app.post("/api/self-evolution/propose")
async def propose_update(request: Request):
    """Propose a new evolution update."""
    if not kernel:
        raise HTTPException(status_code=500, detail="Kernel not initialized")
    
    try:
        data = await request.json()
        
        # Required fields
        title = data.get("title")
        description = data.get("description")
        
        if not title or not description:
            raise HTTPException(status_code=400, detail="Title and description are required")
        
        # Optional fields with defaults
        affected_components = data.get("affected_components", [])
        if isinstance(affected_components, str):
            affected_components = [comp.strip() for comp in affected_components.split(",") if comp.strip()]
            
        code_changes = data.get("code_changes", {})
        priority = data.get("priority", "medium")
        proposed_by = data.get("proposed_by", "user")
        
        # Convert priority string to enum if necessary
        from evogenesis_core.modules.self_evolution_engine import UpdatePriority
        if isinstance(priority, str):
            priority_map = {
                "critical": UpdatePriority.CRITICAL,
                "high": UpdatePriority.HIGH,
                "medium": UpdatePriority.MEDIUM,
                "low": UpdatePriority.LOW
            }
            priority = priority_map.get(priority.lower(), UpdatePriority.MEDIUM)
        
        # Propose the update
        update_id = kernel.self_evolution_engine.propose_update(
            title=title,
            description=description,
            affected_components=affected_components,
            code_changes=code_changes,
            priority=priority,
            proposed_by=proposed_by
        )
          # Broadcast event
        await ws_manager.broadcast_to_topic("self_evolution.updates", {"event": "update_proposed", "update_id": update_id})
        
        return {"success": True, "update_id": update_id}
    except Exception as e:
        logging.error(f"Error proposing update: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/self-evolution/approve/{update_id}")
async def approve_update(update_id: str):
    """Approve an evolution update."""
    if not kernel:
        raise HTTPException(status_code=500, detail="Kernel not initialized")
    
    try:
        success = kernel.self_evolution_engine.approve_update(update_id)
        
        if success:
            await ws_manager.broadcast_to_topic("self_evolution.updates", {"event": "update_approved", "update_id": update_id})
            return {"success": True}
        else:
            return {"success": False, "message": "Failed to approve update"}
    except Exception as e:
        logging.error(f"Error approving update: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/self-evolution/rollback/{update_id}")
async def rollback_update(update_id: str):
    """Rollback an evolution update."""
    if not kernel:
        raise HTTPException(status_code=500, detail="Kernel not initialized")
    
    try:
        success = kernel.self_evolution_engine.rollback_update(update_id)
        
        if success:
            await ws_manager.broadcast_to_topic("self_evolution.updates", {"event": "update_rolled_back", "update_id": update_id})
            return {"success": True}
        else:
            return {"success": False, "message": "Failed to rollback update"}
    except Exception as e:
        logging.error(f"Error rolling back update: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/self-evolution/abtest/{update_id}")
async def start_abtest(update_id: str, request: Request):
    """Start an A/B test for an update."""
    if not kernel:
        raise HTTPException(status_code=500, detail="Kernel not initialized")
    try:
        data = await request.json()
        duration = data.get("duration", 3600)  # Default to 1 hour
        test_id = kernel.self_evolution_engine.initiate_ab_test(update_id, test_duration=duration)
        
        if test_id:
            await ws_manager.broadcast_to_topic("self_evolution.abtests", {"event": "abtest_started", "test_id": test_id})
            await ws_manager.broadcast_to_topic("self_evolution.abtests", {"event": "abtest_started", "test_id": test_id})
            return {"success": True, "test_id": test_id}
        else:
            return {"success": False, "message": "Failed to start A/B test"}
    except Exception as e:
        logging.error(f"Error starting A/B test: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/self-evolution/agent-reflection/{agent_id}")
async def trigger_agent_reflection(agent_id: str):
    """Trigger an agent to reflect on its performance and suggest improvements."""
    if not kernel:
        raise HTTPException(status_code=500, detail="Kernel not initialized")
    
    try:
        result = kernel.self_evolution_engine.trigger_agent_self_reflection(agent_id)
        
        if result.get("success", False):
            await ws_manager.broadcast_to_topic("self_evolution.agents", {"event": "agent_reflection_completed", "agent_id": agent_id})
            await ws_manager.broadcast_to_topic("self_evolution.agents", {"event": "agent_reflection_completed", "agent_id": agent_id})
            return {"success": True, "result": result}
        else:
            return {"success": False, "message": result.get("error", "Unknown error")}
    except Exception as e:
        logging.error(f"Error triggering agent reflection: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/self-evolution/check-updates")
async def check_for_updates(request: Request):
    """Check for available framework updates."""
    if not kernel:
        raise HTTPException(status_code=500, detail="Kernel not initialized")
    
    try:
        data = await request.json()
        source = data.get("source", "repository")
        branch = data.get("branch", "main")
        
        updates = kernel.self_evolution_engine.check_for_updates(source=source, branch=branch)
        
        return {"success": True, "updates": updates}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/system/start")
async def start_system():
    """Start the EvoGenesis kernel."""
    if not kernel:
        raise HTTPException(status_code=500, detail="Kernel not initialized")
    try:
        kernel.start()
        await ws_manager.broadcast_to_topic("system", {"event": "system_started"})
        return {"status": "started"}
    except Exception as e:
        logging.error(f"Error starting system: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/system/stop")
async def stop_system():
    """Stop the EvoGenesis kernel."""
    if not kernel:
        raise HTTPException(status_code=500, detail="Kernel not initialized")
    try:
        kernel.stop()
        await ws_manager.broadcast_to_topic("system", {"event": "system_stopped"})
        return {"status": "stopped"}
    except Exception as e:
        logging.error(f"Error stopping system: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/system/pause")
async def pause_system():
    """Pause the EvoGenesis kernel (placeholder)."""
    if not kernel:
        raise HTTPException(status_code=500, detail="Kernel not initialized")
    try:
        # This would need implementation in the kernel
        # For now, just broadcast the event
        await ws_manager.broadcast_to_topic("system", {"event": "system_paused"})
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
                # Send a heartbeat to confirm server is alive
                await ws_manager.broadcast_to_topic("system", {
                    "event": "heartbeat",
                    "timestamp": time.time(),
                    "server_time": datetime.now().isoformat()
                })
                
                # Get latest system status and broadcast
                status = get_component_status(kernel)
                await ws_manager.broadcast_to_topic("system.status", status)
                
                # Get and broadcast memory manager status
                try:
                    memory_status = kernel.memory_manager.get_status()
                    await ws_manager.broadcast_to_topic("memory.status", memory_status)
                except Exception as e:
                    logging.error(f"Error getting memory status: {str(e)}")
                
                # Get and broadcast adapter status if changed
                try:
                    adapter_status = {
                        "available": list(kernel.framework_adapter_manager.available_adapters.keys()),
                        "initialized": list(kernel.framework_adapter_manager.initialized_adapters.keys()),
                        "health": kernel.framework_adapter_manager.get_adapter_health() if hasattr(kernel.framework_adapter_manager, "get_adapter_health") else {}
                    }
                    await ws_manager.broadcast_to_topic("adapters.status", adapter_status)
                except Exception as e:
                    logging.error(f"Error getting adapter status: {str(e)}")
                
                # Get and broadcast recent activities
                try:
                    recent_activities = kernel.get_recent_activities(5)
                    await ws_manager.broadcast_to_topic("system.activities", {
                        "event": "activities_updated",
                        "activities": recent_activities
                    })
                except Exception as e:
                    logging.error(f"Error getting recent activities: {str(e)}")
                
                # Get and broadcast tool status
                try:
                    tools_status = await get_tools()
                    await ws_manager.broadcast_to_topic("tools.status", {
                        "event": "tools_updated",
                        "tools": tools_status
                    })
                except Exception as e:
                    logging.error(f"Error getting tools status: {str(e)}")
                
                # Get WebSocket connection statistics
                ws_stats = ws_manager.get_connection_stats()
                await ws_manager.broadcast_to_topic("system.metrics", {
                    "event": "websocket_stats",
                    "stats": ws_stats,
                    "timestamp": time.time()
                })
            
            await asyncio.sleep(5)  # Check every 5 seconds
        except Exception as e:
            logging.error(f"Error in event monitor: {str(e)}")
            await asyncio.sleep(10)  # Back off on error

async def metrics_update_task():
    """Background task to update system metrics and broadcast them."""
    while True:
        try:
            if kernel:
                # Gather system metrics
                metrics = {}
                  # 1. Get agent metrics
                agent_factory = kernel.get_module("agent_factory")
                if agent_factory:
                    agents = agent_factory.list_agents() if hasattr(agent_factory, "list_agents") else {}
                    if not agents and hasattr(agent_factory, "agents"):
                        agents = agent_factory.agents
                    active_agents = sum(1 for a in agents.values() 
                                        if hasattr(a, "status") and a.status == "active")
                    total_agents = len(agents)
                    metrics["agents"] = {
                        "active": active_agents,
                        "total": total_agents
                    }
                
                # 2. Get task metrics
                task_planner = kernel.get_module("task_planner")
                if task_planner:
                    tasks = task_planner.list_tasks() if hasattr(task_planner, "list_tasks") else {}
                    running_tasks = sum(1 for t in tasks.values() 
                                       if hasattr(t, "status") and t.status == "in_progress")
                    completed_tasks = sum(1 for t in tasks.values() 
                                         if hasattr(t, "status") and t.status == "completed")
                    total_tasks = len(tasks)
                    metrics["tasks"] = {
                        "running": running_tasks,
                        "completed": completed_tasks,
                        "total": total_tasks
                    }
                
                # 3. Get memory usage
                try:
                    import psutil
                    process = psutil.Process()
                    memory_usage = process.memory_info().rss / (1024 * 1024)  # MB
                    metrics["memory"] = {
                        "usage_mb": memory_usage,
                        "usage_percent": (memory_usage / 1000) * 100  # Assuming 1GB is 100%
                    }
                except ImportError:
                    metrics["memory"] = {
                        "usage_mb": 0,
                        "usage_percent": 0
                    }
                  # 4. Get team metrics
                teams = {}
                if hasattr(kernel, "agent_factory") and hasattr(kernel.agent_factory, "teams"):
                    teams = kernel.agent_factory.teams
                metrics["teams"] = {
                    "active": sum(1 for t in teams.values() 
                                 if hasattr(t, "status") and t.status == "active"),
                    "total": len(teams)
                }
                
                # 5. Get activities
                recent_activities = kernel.get_recent_activities(10) if hasattr(kernel, "get_recent_activities") else []
                metrics["activities"] = recent_activities
                
                # Broadcast metrics
                await ws_manager.broadcast_to_topic("metrics", metrics)
            
            # Update every 5 seconds
            await asyncio.sleep(5)
        except Exception as e:
            logging.error(f"Error in metrics update task: {str(e)}")
            await asyncio.sleep(10)  # Back off on error

# Start the server
def start_server(kernel_instance, host="0.0.0.0", port=5000):
    """Start the web UI server."""
    global kernel
    kernel = kernel_instance
    
    # Initialize swarm coordinator if available
    if hasattr(kernel, "swarm_module") and kernel.swarm_module is not None:
        from evogenesis_core.swarm.coordinator import SwarmCoordinator
        kernel.swarm_coordinator = SwarmCoordinator(kernel)
    
    # Initialize WebSocket handlers for all components
    ws_handlers = init_ws_handlers(kernel, ws_manager)
    
    # Register all API routes
    setup_routes(app)
    
    # Start the background tasks
    @app.on_event("startup")
    async def startup_event():
        asyncio.create_task(event_monitor())
        asyncio.create_task(metrics_update_task())
        
        # Log system startup
        kernel.log_activity(
            activity_type="system.startup",
            title="Web UI Server Started",
            message=f"Web UI server started on http://{host}:{port}",
            data={"host": host, "port": port}
        )
        
        # Broadcast startup event
        broadcast_system_event(ws_manager, "system.started", {
            "host": host,
            "port": port,
            "server_time": datetime.now().isoformat()
        })
    
    # Attach the kernel instance and WebSocket manager to the FastAPI app state
    app.state.kernel = kernel_instance
    app.state.ws_manager = ws_manager
    app.state.ws_handlers = ws_handlers
    
    # Run the server
    # Setting log_level to prevent duplicate logs
    uvicorn.run(app, host=host, port=port, log_level="info")
