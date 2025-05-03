"""
Tool Routes for EvoGenesis Web UI

This module provides API routes for managing tools in the EvoGenesis system.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, File, UploadFile
from pydantic import BaseModel


# Define models
class ToolBase(BaseModel):
    name: str
    description: str
    category: str
    type: str  # external, internal, script
    config: Dict[str, Any]


class ToolCreate(ToolBase):
    pass


class ToolUpdate(ToolBase):
    pass


class ToolResponse(ToolBase):
    id: str
    active: bool
    stats: Optional[Dict[str, Any]] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None


# Create router
router = APIRouter(
    prefix="/api/tools",
    tags=["tools"],
)


# Define routes
@router.get("/", response_model=List[ToolResponse])
async def get_tools(request: Request):
    """Get all tools in the system."""
    kernel = request.app.state.kernel
    
    try:
        tooling_system = kernel.get_module("tooling_system")
        tools = tooling_system.list_tools()
        
        # Transform to response format
        tool_responses = []
        for tool_id, tool_data in tools.items():
            tool_responses.append({
                "id": tool_id,
                "name": tool_data.name if hasattr(tool_data, "name") else f"Tool-{tool_id[:6]}",
                "description": tool_data.description if hasattr(tool_data, "description") else "",
                "category": tool_data.category if hasattr(tool_data, "category") else "utils",
                "type": tool_data.type if hasattr(tool_data, "type") else "internal",
                "config": tool_data.config if hasattr(tool_data, "config") else {},
                "active": tool_data.active if hasattr(tool_data, "active") else True,
                "stats": tool_data.stats if hasattr(tool_data, "stats") else None,
                "input_schema": tool_data.input_schema if hasattr(tool_data, "input_schema") else None,
                "output_schema": tool_data.output_schema if hasattr(tool_data, "output_schema") else None
            })
        
        return tool_responses
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tools: {str(e)}")


@router.get("/{tool_id}", response_model=ToolResponse)
async def get_tool(tool_id: str, request: Request):
    """Get details of a specific tool."""
    kernel = request.app.state.kernel
    
    try:
        tooling_system = kernel.get_module("tooling_system")
        tool = tooling_system.get_tool(tool_id)
        
        if not tool:
            raise HTTPException(status_code=404, detail=f"Tool {tool_id} not found")
        
        return {
            "id": tool_id,
            "name": tool.name if hasattr(tool, "name") else f"Tool-{tool_id[:6]}",
            "description": tool.description if hasattr(tool, "description") else "",
            "category": tool.category if hasattr(tool, "category") else "utils",
            "type": tool.type if hasattr(tool, "type") else "internal",
            "config": tool.config if hasattr(tool, "config") else {},
            "active": tool.active if hasattr(tool, "active") else True,
            "stats": tool.stats if hasattr(tool, "stats") else None,
            "input_schema": tool.input_schema if hasattr(tool, "input_schema") else None,
            "output_schema": tool.output_schema if hasattr(tool, "output_schema") else None
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get tool {tool_id}: {str(e)}")


@router.post("/", response_model=Dict[str, Any])
async def create_tool(tool: ToolCreate, request: Request):
    """Create a new tool."""
    kernel = request.app.state.kernel
    
    try:
        tooling_system = kernel.get_module("tooling_system")
        
        # Create the tool
        tool_id = tooling_system.register_tool(
            name=tool.name,
            description=tool.description,
            category=tool.category,
            tool_type=tool.type,
            config=tool.config
        )
        
        return {
            "success": True,
            "message": f"Tool {tool.name} registered successfully",
            "tool_id": tool_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register tool: {str(e)}")


@router.put("/{tool_id}", response_model=Dict[str, Any])
async def update_tool(tool_id: str, tool: ToolUpdate, request: Request):
    """Update a tool."""
    kernel = request.app.state.kernel
    
    try:
        tooling_system = kernel.get_module("tooling_system")
        success = tooling_system.update_tool(
            tool_id=tool_id,
            name=tool.name,
            description=tool.description,
            category=tool.category,
            tool_type=tool.type,
            config=tool.config
        )
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Tool {tool_id} not found")
        
        return {
            "success": True,
            "message": f"Tool {tool_id} updated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update tool {tool_id}: {str(e)}")


@router.delete("/{tool_id}", response_model=Dict[str, Any])
async def delete_tool(tool_id: str, request: Request):
    """Delete a tool."""
    kernel = request.app.state.kernel
    
    try:
        tooling_system = kernel.get_module("tooling_system")
        success = tooling_system.unregister_tool(tool_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Tool {tool_id} not found")
        
        return {
            "success": True,
            "message": f"Tool {tool_id} deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete tool {tool_id}: {str(e)}")


@router.post("/{tool_id}/toggle", response_model=Dict[str, Any])
async def toggle_tool(tool_id: str, active: bool, request: Request):
    """Toggle a tool's active state."""
    kernel = request.app.state.kernel
    
    try:
        tooling_system = kernel.get_module("tooling_system")
        success = tooling_system.set_tool_status(tool_id, active)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Tool {tool_id} not found")
        
        return {
            "success": True,
            "message": f"Tool {tool_id} {'enabled' if active else 'disabled'} successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to toggle tool {tool_id}: {str(e)}")


@router.post("/upload-script", response_model=Dict[str, Any])
async def upload_tool_script(
    request: Request,
    name: str,
    description: str,
    category: str,
    file: UploadFile = File(...)
):
    """Upload a script file as a new tool."""
    kernel = request.app.state.kernel
    
    try:
        tooling_system = kernel.get_module("tooling_system")
        
        # Read the script content
        script_content = await file.read()
        
        # Register the tool with the script content
        tool_id = tooling_system.register_script_tool(
            name=name,
            description=description,
            category=category,
            script_content=script_content.decode('utf-8'),
            script_filename=file.filename
        )
        
        return {
            "success": True,
            "message": f"Tool script {name} uploaded successfully",
            "tool_id": tool_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload tool script: {str(e)}")


@router.post("/auto-create", response_model=Dict[str, Any])
async def auto_create_tool(request: Request, description: str):
    """Auto-create a tool based on a natural language description."""
    kernel = request.app.state.kernel
    
    try:
        tooling_system = kernel.get_module("tooling_system")
        llm_orchestrator = kernel.get_module("llm_orchestrator")
        
        # Use LLM to design the tool
        prompt = f"""
        Based on the following description, design a tool configuration:
        
        Description: {description}
        
        Generate a JSON response with the following structure:
        {{
            "name": "Appropriate name for the tool",
            "description": "Refined description of the tool's purpose",
            "category": "One of: data, web, file, api, ai, utils",
            "type": "One of: external, internal, script",
            "config": {{
                // Configuration parameters appropriate for this tool
            }},
            "input_schema": {{
                // JSON schema describing the tool's input
            }},
            "output_schema": {{
                // JSON schema describing the tool's output
            }}
        }}
        """
        
        llm_response = await llm_orchestrator.generate(prompt)
        tool_config = json.loads(llm_response)
        
        # Create the tool with the generated configuration
        tool_id = tooling_system.register_tool(
            name=tool_config["name"],
            description=tool_config["description"],
            category=tool_config["category"],
            tool_type=tool_config["type"],
            config=tool_config["config"],
            input_schema=tool_config.get("input_schema"),
            output_schema=tool_config.get("output_schema")
        )
        
        return {
            "success": True,
            "message": f"Tool {tool_config['name']} auto-created successfully",
            "tool_id": tool_id,
            "tool_config": tool_config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to auto-create tool: {str(e)}")


@router.post("/{tool_id}/execute", response_model=Dict[str, Any])
async def execute_tool(tool_id: str, parameters: Dict[str, Any], request: Request):
    """Execute a tool with the given parameters."""
    kernel = request.app.state.kernel
    
    try:
        tooling_system = kernel.get_module("tooling_system")
        result = await tooling_system.execute_tool(tool_id, parameters)
        
        return {
            "success": True,
            "tool_id": tool_id,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to execute tool {tool_id}: {str(e)}")


def add_routes(app):
    """Add tool routes to the main app."""
    app.include_router(router)
