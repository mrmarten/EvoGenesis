"""
Agent Routes for EvoGenesis Web UI

This module provides API routes for managing agents in the EvoGenesis system.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel


# Define models
class AgentBase(BaseModel):
    name: str
    type: str
    description: Optional[str] = None
    capabilities: List[str] = []
    team_id: Optional[str] = None


class AgentCreate(AgentBase):
    pass


class AgentUpdate(AgentBase):
    pass


class AgentResponse(AgentBase):
    id: str
    status: str
    created_at: str


# Create router
router = APIRouter(
    prefix="/api/agents",
    tags=["agents"],
)


# Define routes
@router.get("/", response_model=List[AgentResponse])
async def get_agents(request: Request):
    """Get all agents in the system."""
    kernel = request.app.state.kernel
    
    try:
        agent_manager = kernel.get_module("agent_factory")
        agents = agent_manager.list_agents()
        
        # Transform to response format
        agent_responses = []
        for agent_id, agent_data in agents.items():
            agent_responses.append({
                "id": agent_id,
                "name": agent_data.name if hasattr(agent_data, "name") else f"Agent-{agent_id[:6]}",
                "type": agent_data.type if hasattr(agent_data, "type") else "unknown",
                "description": agent_data.description if hasattr(agent_data, "description") else "",
                "capabilities": agent_data.capabilities if hasattr(agent_data, "capabilities") else [],
                "status": agent_data.status if hasattr(agent_data, "status") else "inactive",
                "team_id": agent_data.team_id if hasattr(agent_data, "team_id") else None,
                "created_at": agent_data.created_at if hasattr(agent_data, "created_at") else ""
            })
        
        return agent_responses
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agents: {str(e)}")


@router.get("/{agent_id}", response_model=Dict[str, Any])
async def get_agent(agent_id: str, request: Request):
    """Get details of a specific agent."""
    kernel = request.app.state.kernel
    
    try:
        agent_manager = kernel.get_module("agent_factory")
        agent = agent_manager.get_agent(agent_id)
        
        if not agent:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Get additional information like activities, tasks, etc.
        agent_details = {
            "id": agent_id,
            "name": agent.name if hasattr(agent, "name") else f"Agent-{agent_id[:6]}",
            "type": agent.type if hasattr(agent, "type") else "unknown",
            "description": agent.description if hasattr(agent, "description") else "",
            "capabilities": agent.capabilities if hasattr(agent, "capabilities") else [],
            "status": agent.status if hasattr(agent, "status") else "inactive",
            "team_id": agent.team_id if hasattr(agent, "team_id") else None,
            "created_at": agent.created_at if hasattr(agent, "created_at") else "",
            "activities": [],  # Would be populated from a log/activity system
            "current_tasks": []  # Would be populated from the task system
        }
        
        return agent_details
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agent {agent_id}: {str(e)}")


@router.post("/", response_model=Dict[str, Any])
async def create_agent(agent: AgentCreate, request: Request):
    """Create a new agent."""
    kernel = request.app.state.kernel
    
    try:
        agent_manager = kernel.get_module("agent_factory")
        
        # Create the agent
        agent_id = agent_manager.create_agent(
            name=agent.name,
            agent_type=agent.type,
            description=agent.description,
            capabilities=agent.capabilities,
            team_id=agent.team_id
        )
        
        return {
            "success": True,
            "message": f"Agent {agent.name} created successfully",
            "agent_id": agent_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create agent: {str(e)}")


@router.delete("/{agent_id}", response_model=Dict[str, Any])
async def delete_agent(agent_id: str, request: Request):
    """Delete an agent."""
    kernel = request.app.state.kernel
    
    try:
        agent_manager = kernel.get_module("agent_factory")
        success = agent_manager.delete_agent(agent_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        return {
            "success": True,
            "message": f"Agent {agent_id} deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete agent {agent_id}: {str(e)}")


@router.post("/auto-create", response_model=Dict[str, Any])
async def auto_create_agent(request: Request, description: str):
    """Auto-create an agent based on a natural language description."""
    kernel = request.app.state.kernel
    
    try:
        agent_manager = kernel.get_module("agent_factory")
        llm_orchestrator = kernel.get_module("llm_orchestrator")
        
        # Use LLM to determine the agent type, capabilities, etc.
        prompt = f"""
        Based on the following description, determine the appropriate agent configuration:
        
        Description: {description}
        
        Generate a JSON response with the following structure:
        {{
            "name": "Appropriate name for the agent",
            "type": "One of: assistant, user_proxy, coordinator, planner, critic",
            "description": "Refined description of the agent's purpose",
            "capabilities": ["list", "of", "capabilities", "the", "agent", "should", "have"]
        }}
        """
        
        llm_response = await llm_orchestrator.generate(prompt)
        agent_config = json.loads(llm_response)
        
        # Create the agent with the generated configuration
        agent_id = agent_manager.create_agent(
            name=agent_config["name"],
            agent_type=agent_config["type"],
            description=agent_config["description"],
            capabilities=agent_config["capabilities"]
        )
        
        return {
            "success": True,
            "message": f"Agent {agent_config['name']} auto-created successfully",
            "agent_id": agent_id,
            "agent_config": agent_config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to auto-create agent: {str(e)}")


def add_routes(app):
    """Add agent routes to the main app."""
    app.include_router(router)
