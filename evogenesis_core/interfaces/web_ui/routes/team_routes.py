"""
Team Routes for EvoGenesis Web UI

This module provides API routes for managing teams in the EvoGenesis system.
"""

import json
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel


# Define models
class TeamBase(BaseModel):
    name: str
    goal: Optional[str] = None
    is_system: bool = False


class TeamCreate(TeamBase):
    pass


class TeamUpdate(TeamBase):
    pass


class TeamResponse(TeamBase):
    id: str
    status: str  # active, paused, inactive
    agent_count: int
    task_count: int
    completion: int  # percentage


# Create router
router = APIRouter(
    prefix="/api/teams",
    tags=["teams"],
)


# Define routes
@router.get("/", response_model=List[TeamResponse])
async def get_teams(request: Request):
    """Get all teams in the system."""
    kernel = request.app.state.kernel
    
    try:
        # Note: The current API design assumes there's a team_manager module.
        # You might need to adapt this to your actual architecture.
        agent_factory = kernel.get_module("agent_factory")
        task_planner = kernel.get_module("task_planner")
        
        # Get teams
        teams = agent_factory.list_teams()
        
        # Transform to response format
        team_responses = []
        for team_id, team_data in teams.items():
            # Get agent count for this team
            agents = agent_factory.list_agents(team_id=team_id)
            agent_count = len(agents)
            
            # Get task count for this team
            tasks = task_planner.list_tasks(team_id=team_id)
            task_count = len(tasks)
            
            # Calculate completion based on task progress
            completion = 0
            if task_count > 0:
                total_progress = sum(task.progress for task in tasks.values() if hasattr(task, "progress"))
                completion = int(total_progress / task_count) if task_count > 0 else 0
            
            team_responses.append({
                "id": team_id,
                "name": team_data.name if hasattr(team_data, "name") else f"Team-{team_id[:6]}",
                "goal": team_data.goal if hasattr(team_data, "goal") else "",
                "status": team_data.status if hasattr(team_data, "status") else "inactive",
                "is_system": team_data.is_system if hasattr(team_data, "is_system") else False,
                "agent_count": agent_count,
                "task_count": task_count,
                "completion": completion
            })
        
        return team_responses
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get teams: {str(e)}")


@router.get("/{team_id}", response_model=Dict[str, Any])
async def get_team(team_id: str, request: Request):
    """Get details of a specific team."""
    kernel = request.app.state.kernel
    
    try:
        agent_factory = kernel.get_module("agent_factory")
        task_planner = kernel.get_module("task_planner")
        
        team = agent_factory.get_team(team_id)
        
        if not team:
            raise HTTPException(status_code=404, detail=f"Team {team_id} not found")
        
        # Get agents in this team
        agents = agent_factory.list_agents(team_id=team_id)
        agent_list = []
        for agent_id, agent_data in agents.items():
            agent_list.append({
                "id": agent_id,
                "name": agent_data.name if hasattr(agent_data, "name") else f"Agent-{agent_id[:6]}",
                "type": agent_data.type if hasattr(agent_data, "type") else "unknown",
                "status": agent_data.status if hasattr(agent_data, "status") else "inactive"
            })
        
        # Get tasks for this team
        tasks = task_planner.list_tasks(team_id=team_id)
        task_list = []
        for task_id, task_data in tasks.items():
            task_list.append({
                "id": task_id,
                "title": task_data.title if hasattr(task_data, "title") else f"Task-{task_id[:6]}",
                "status": task_data.status if hasattr(task_data, "status") else "pending",
                "progress": task_data.progress if hasattr(task_data, "progress") else 0
            })
        
        # Calculate completion based on task progress
        completion = 0
        if task_list:
            total_progress = sum(task["progress"] for task in task_list)
            completion = int(total_progress / len(task_list))
        
        return {
            "id": team_id,
            "name": team.name if hasattr(team, "name") else f"Team-{team_id[:6]}",
            "goal": team.goal if hasattr(team, "goal") else "",
            "status": team.status if hasattr(team, "status") else "inactive",
            "is_system": team.is_system if hasattr(team, "is_system") else False,
            "completion": completion,
            "agents": agent_list,
            "tasks": task_list
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get team {team_id}: {str(e)}")


@router.post("/", response_model=Dict[str, Any])
async def create_team(team: TeamCreate, request: Request):
    """Create a new team."""
    kernel = request.app.state.kernel
    
    try:
        agent_factory = kernel.get_module("agent_factory")
        
        # Create the team
        team_id = agent_factory.create_team(
            name=team.name,
            goal=team.goal,
            is_system=team.is_system
        )
        
        return {
            "success": True,
            "message": f"Team {team.name} created successfully",
            "team_id": team_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create team: {str(e)}")


@router.put("/{team_id}", response_model=Dict[str, Any])
async def update_team(team_id: str, team: TeamUpdate, request: Request):
    """Update a team."""
    kernel = request.app.state.kernel
    
    try:
        agent_factory = kernel.get_module("agent_factory")
        success = agent_factory.update_team(
            team_id=team_id,
            name=team.name,
            goal=team.goal,
            is_system=team.is_system
        )
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Team {team_id} not found")
        
        return {
            "success": True,
            "message": f"Team {team_id} updated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update team {team_id}: {str(e)}")


@router.delete("/{team_id}", response_model=Dict[str, Any])
async def delete_team(team_id: str, request: Request):
    """Delete a team."""
    kernel = request.app.state.kernel
    
    try:
        agent_factory = kernel.get_module("agent_factory")
        success = agent_factory.delete_team(team_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Team {team_id} not found")
        
        return {
            "success": True,
            "message": f"Team {team_id} deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete team {team_id}: {str(e)}")


@router.post("/{team_id}/status", response_model=Dict[str, Any])
async def update_team_status(team_id: str, status: str, request: Request):
    """Update the status of a team."""
    kernel = request.app.state.kernel
    
    if status not in ["active", "paused", "inactive"]:
        raise HTTPException(status_code=400, detail="Status must be one of: active, paused, inactive")
    
    try:
        agent_factory = kernel.get_module("agent_factory")
        success = agent_factory.update_team_status(team_id, status)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Team {team_id} not found")
        
        return {
            "success": True,
            "message": f"Team {team_id} status updated to {status}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update team status: {str(e)}")


@router.post("/{team_id}/add-agent/{agent_id}", response_model=Dict[str, Any])
async def add_agent_to_team(team_id: str, agent_id: str, request: Request):
    """Add an agent to a team."""
    kernel = request.app.state.kernel
    
    try:
        agent_factory = kernel.get_module("agent_factory")
        success = agent_factory.add_agent_to_team(agent_id, team_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Agent or team not found")
        
        return {
            "success": True,
            "message": f"Agent {agent_id} added to team {team_id} successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add agent to team: {str(e)}")


@router.post("/{team_id}/remove-agent/{agent_id}", response_model=Dict[str, Any])
async def remove_agent_from_team(team_id: str, agent_id: str, request: Request):
    """Remove an agent from a team."""
    kernel = request.app.state.kernel
    
    try:
        agent_factory = kernel.get_module("agent_factory")
        success = agent_factory.remove_agent_from_team(agent_id, team_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Agent or team not found")
        
        return {
            "success": True,
            "message": f"Agent {agent_id} removed from team {team_id} successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove agent from team: {str(e)}")


@router.post("/auto-create", response_model=Dict[str, Any])
async def auto_create_team(request: Request, goal: str):
    """Auto-create a team based on a goal description."""
    kernel = request.app.state.kernel
    
    try:
        agent_factory = kernel.get_module("agent_factory")
        llm_orchestrator = kernel.get_module("llm_orchestrator")
        task_planner = kernel.get_module("task_planner")
        
        # Use LLM to design the team
        prompt = f"""
        Based on the following goal, design a team of AI agents:
        
        Goal: {goal}
        
        Generate a JSON response with the following structure:
        {{
            "team_name": "Appropriate name for the team",
            "team_goal": "Refined goal statement",
            "agents": [
                {{
                    "name": "Agent 1 name",
                    "type": "One of: assistant, user_proxy, coordinator, planner, critic",
                    "description": "Agent's purpose within the team",
                    "capabilities": ["list", "of", "capabilities"]
                }},
                // More agents...
            ],
            "initial_tasks": [
                {{
                    "title": "Task 1 title",
                    "description": "Detailed description of task 1",
                    "priority": "high|medium|low"
                }},
                // More tasks...
            ]
        }}
        """
        
        llm_response = await llm_orchestrator.generate(prompt)
        team_design = json.loads(llm_response)
        
        # Create the team
        team_id = agent_factory.create_team(
            name=team_design["team_name"],
            goal=team_design["team_goal"]
        )
        
        # Create the agents and add them to the team
        created_agents = []
        for agent_config in team_design["agents"]:
            agent_id = agent_factory.create_agent(
                name=agent_config["name"],
                agent_type=agent_config["type"],
                description=agent_config["description"],
                capabilities=agent_config["capabilities"],
                team_id=team_id
            )
            created_agents.append({
                "agent_id": agent_id,
                "name": agent_config["name"]
            })
        
        # Create initial tasks for the team
        created_tasks = []
        for task_config in team_design["initial_tasks"]:
            task_id = task_planner.create_task(
                title=task_config["title"],
                description=task_config["description"],
                priority=task_config["priority"],
                team_id=team_id
            )
            created_tasks.append({
                "task_id": task_id,
                "title": task_config["title"]
            })
        
        return {
            "success": True,
            "message": f"Team '{team_design['team_name']}' auto-created successfully with {len(created_agents)} agents and {len(created_tasks)} tasks",
            "team_id": team_id,
            "agents": created_agents,
            "tasks": created_tasks,
            "design": team_design
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to auto-create team: {str(e)}")


def add_routes(app):
    """Add team routes to the main app."""
    app.include_router(router)
