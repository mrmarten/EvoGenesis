"""
LLM Orchestrator Routes for EvoGenesis Web UI

This module provides API routes for managing language models in the EvoGenesis system.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel


# Define models
class ModelBase(BaseModel):
    name: str
    provider: str
    context_window: int
    api_key_required: bool = True
    default_parameters: Dict[str, Any] = {}


class ModelCreate(ModelBase):
    api_key: Optional[str] = None


class ModelUpdate(ModelBase):
    api_key: Optional[str] = None


class ModelResponse(ModelBase):
    id: str
    available: bool
    usage_stats: Optional[Dict[str, Any]] = None


class GenerationRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    parameters: Dict[str, Any] = {}
    stream: bool = False
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


# Create router
router = APIRouter(
    prefix="/api/llm",
    tags=["llm"],
)


# Define routes
@router.get("/models", response_model=List[ModelResponse])
async def get_models(request: Request):
    """Get all language models in the system."""
    kernel = request.app.state.kernel
    
    try:
        llm_orchestrator = kernel.get_module("llm_orchestrator")
        models = llm_orchestrator.list_models()
        
        # Transform to response format
        model_responses = []
        for model_id, model_data in models.items():
            model_responses.append({
                "id": model_id,
                "name": model_data.name if hasattr(model_data, "name") else model_id,
                "provider": model_data.provider if hasattr(model_data, "provider") else "unknown",
                "context_window": model_data.context_window if hasattr(model_data, "context_window") else 0,
                "api_key_required": model_data.api_key_required if hasattr(model_data, "api_key_required") else True,
                "default_parameters": model_data.default_parameters if hasattr(model_data, "default_parameters") else {},
                "available": model_data.available if hasattr(model_data, "available") else False,
                "usage_stats": model_data.usage_stats if hasattr(model_data, "usage_stats") else None
            })
        
        return model_responses
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get models: {str(e)}")


@router.get("/models/{model_id}", response_model=ModelResponse)
async def get_model(model_id: str, request: Request):
    """Get details of a specific language model."""
    kernel = request.app.state.kernel
    
    try:
        llm_orchestrator = kernel.get_module("llm_orchestrator")
        model = llm_orchestrator.get_model(model_id)
        
        if not model:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        return {
            "id": model_id,
            "name": model.name if hasattr(model, "name") else model_id,
            "provider": model.provider if hasattr(model, "provider") else "unknown",
            "context_window": model.context_window if hasattr(model, "context_window") else 0,
            "api_key_required": model.api_key_required if hasattr(model, "api_key_required") else True,
            "default_parameters": model.default_parameters if hasattr(model, "default_parameters") else {},
            "available": model.available if hasattr(model, "available") else False,
            "usage_stats": model.usage_stats if hasattr(model, "usage_stats") else None
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model {model_id}: {str(e)}")


@router.post("/models", response_model=Dict[str, Any])
async def register_model(model: ModelCreate, request: Request):
    """Register a new language model."""
    kernel = request.app.state.kernel
    
    try:
        llm_orchestrator = kernel.get_module("llm_orchestrator")
        
        # Register the model
        model_id = llm_orchestrator.register_model(
            name=model.name,
            provider=model.provider,
            context_window=model.context_window,
            api_key=model.api_key,
            api_key_required=model.api_key_required,
            default_parameters=model.default_parameters
        )
        
        return {
            "success": True,
            "message": f"Model {model.name} registered successfully",
            "model_id": model_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to register model: {str(e)}")


@router.put("/models/{model_id}", response_model=Dict[str, Any])
async def update_model(model_id: str, model: ModelUpdate, request: Request):
    """Update a language model."""
    kernel = request.app.state.kernel
    
    try:
        llm_orchestrator = kernel.get_module("llm_orchestrator")
        success = llm_orchestrator.update_model(
            model_id=model_id,
            name=model.name,
            provider=model.provider,
            context_window=model.context_window,
            api_key=model.api_key,
            api_key_required=model.api_key_required,
            default_parameters=model.default_parameters
        )
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        return {
            "success": True,
            "message": f"Model {model_id} updated successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update model {model_id}: {str(e)}")


@router.delete("/models/{model_id}", response_model=Dict[str, Any])
async def unregister_model(model_id: str, request: Request):
    """Unregister a language model."""
    kernel = request.app.state.kernel
    
    try:
        llm_orchestrator = kernel.get_module("llm_orchestrator")
        success = llm_orchestrator.unregister_model(model_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        return {
            "success": True,
            "message": f"Model {model_id} unregistered successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unregister model {model_id}: {str(e)}")


@router.post("/generate", response_model=Dict[str, Any])
async def generate_text(generation_request: GenerationRequest, request: Request):
    """Generate text using a language model."""
    kernel = request.app.state.kernel
    
    try:
        llm_orchestrator = kernel.get_module("llm_orchestrator")
        
        # Prepare parameters
        params = generation_request.parameters.copy()
        if generation_request.max_tokens:
            params["max_tokens"] = generation_request.max_tokens
        if generation_request.temperature is not None:
            params["temperature"] = generation_request.temperature
        
        # Generate text
        result = await llm_orchestrator.generate(
            prompt=generation_request.prompt,
            model=generation_request.model,
            parameters=params,
            stream=generation_request.stream
        )
        
        return {
            "success": True,
            "response": result,
            "model_used": llm_orchestrator.get_last_model_used()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate text: {str(e)}")


@router.post("/models/{model_id}/test", response_model=Dict[str, Any])
async def test_model(model_id: str, request: Request):
    """Test a language model to verify it's working."""
    kernel = request.app.state.kernel
    
    try:
        llm_orchestrator = kernel.get_module("llm_orchestrator")
        
        # Simple test prompt
        test_prompt = "Please respond with 'Hello, EvoGenesis!' to confirm you're working correctly."
        
        result = await llm_orchestrator.generate(
            prompt=test_prompt,
            model=model_id
        )
        
        return {
            "success": True,
            "model_id": model_id,
            "working": True,
            "response": result
        }
    except Exception as e:
        return {
            "success": False,
            "model_id": model_id,
            "working": False,
            "error": str(e)
        }


@router.get("/stats", response_model=Dict[str, Any])
async def get_llm_stats(request: Request):
    """Get usage statistics for language models."""
    kernel = request.app.state.kernel
    
    try:
        llm_orchestrator = kernel.get_module("llm_orchestrator")
        stats = llm_orchestrator.get_stats()
        
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get LLM stats: {str(e)}")


@router.post("/default-model", response_model=Dict[str, Any])
async def set_default_model(model_id: str, request: Request):
    """Set the default language model."""
    kernel = request.app.state.kernel
    
    try:
        llm_orchestrator = kernel.get_module("llm_orchestrator")
        success = llm_orchestrator.set_default_model(model_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        
        return {
            "success": True,
            "message": f"Default model set to {model_id}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set default model: {str(e)}")


def add_routes(app):
    """Add LLM orchestrator routes to the main app."""
    app.include_router(router)
