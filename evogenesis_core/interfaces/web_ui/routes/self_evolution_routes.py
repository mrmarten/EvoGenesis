"""
Self Evolution Routes for EvoGenesis Web UI

This module provides API routes for managing the Self-Evolution Engine.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, File, UploadFile
from pydantic import BaseModel


# Define models
class EvolutionRequest(BaseModel):
    goals: List[str]
    constraints: List[str] = []
    optimization_metrics: List[str] = []
    max_iterations: Optional[int] = 1
    risk_level: str = "medium"  # low, medium, high


class OpportunityBase(BaseModel):
    title: str
    description: str
    category: str
    impact: str  # low, medium, high
    risk: str  # low, medium, high
    effort: str  # low, medium, high


class OpportunityCreate(OpportunityBase):
    pass


class OpportunityResponse(OpportunityBase):
    id: str
    status: str  # pending, approved, rejected, implemented
    created_at: str
    implementation_plan: Optional[Dict[str, Any]] = None


# Create router
router = APIRouter(
    prefix="/api/self-evolution",
    tags=["self-evolution"],
)


# Define routes
@router.get("/status", response_model=Dict[str, Any])
async def get_evolution_status(request: Request):
    """Get the current status of the self-evolution engine."""
    kernel = request.app.state.kernel
    
    try:
        self_evolution = kernel.get_module("self_evolution_engine")
        
        if not self_evolution:
            raise HTTPException(status_code=404, detail="Self-evolution engine module not found")
        
        status = self_evolution.get_status()
        
        return {
            "active": status.get("active", False),
            "current_cycle": status.get("current_cycle", 0),
            "total_cycles": status.get("total_cycles", 0),
            "last_run": status.get("last_run", ""),
            "next_scheduled_run": status.get("next_scheduled_run", ""),
            "metrics": status.get("metrics", {}),
            "current_focus": status.get("current_focus", []),
            "health": status.get("health", 0.0)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get evolution status: {str(e)}")


@router.get("/history", response_model=List[Dict[str, Any]])
async def get_evolution_history(request: Request):
    """Get the history of evolution cycles."""
    kernel = request.app.state.kernel
    
    try:
        self_evolution = kernel.get_module("self_evolution_engine")
        
        if not self_evolution:
            raise HTTPException(status_code=404, detail="Self-evolution engine module not found")
        
        history = self_evolution.get_evolution_history()
        
        # Transform to response format if needed
        history_items = []
        for item in history:
            history_items.append({
                "cycle_id": item.get("cycle_id", ""),
                "timestamp": item.get("timestamp", ""),
                "goals": item.get("goals", []),
                "changes_implemented": item.get("changes_implemented", []),
                "metrics_before": item.get("metrics_before", {}),
                "metrics_after": item.get("metrics_after", {}),
                "success": item.get("success", False)
            })
        
        return history_items
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get evolution history: {str(e)}")


@router.post("/trigger", response_model=Dict[str, Any])
async def trigger_evolution_cycle(
    request: Request,
    evolution_request: EvolutionRequest
):
    """Trigger a self-evolution cycle with specific goals and constraints."""
    kernel = request.app.state.kernel
    
    try:
        self_evolution = kernel.get_module("self_evolution_engine")
        
        if not self_evolution:
            raise HTTPException(status_code=404, detail="Self-evolution engine module not found")
        
        # Start the evolution cycle
        cycle_id = await self_evolution.start_evolution_cycle(
            goals=evolution_request.goals,
            constraints=evolution_request.constraints,
            optimization_metrics=evolution_request.optimization_metrics,
            max_iterations=evolution_request.max_iterations,
            risk_level=evolution_request.risk_level
        )
        
        return {
            "success": True,
            "message": "Evolution cycle triggered successfully",
            "cycle_id": cycle_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger evolution cycle: {str(e)}")


@router.get("/opportunities", response_model=List[OpportunityResponse])
async def get_improvement_opportunities(
    request: Request,
    status: Optional[str] = None,
    category: Optional[str] = None
):
    """Get improvement opportunities identified by the self-evolution engine."""
    kernel = request.app.state.kernel
    
    try:
        self_evolution = kernel.get_module("self_evolution_engine")
        
        if not self_evolution:
            raise HTTPException(status_code=404, detail="Self-evolution engine module not found")
        
        # Get opportunities
        opportunities = self_evolution.get_opportunities()
        
        # Apply filters
        filtered_opportunities = []
        for opp in opportunities:
            if status and hasattr(opp, "status") and opp.status != status:
                continue
            if category and hasattr(opp, "category") and opp.category != category:
                continue
            
            filtered_opportunities.append({
                "id": opp.id if hasattr(opp, "id") else "",
                "title": opp.title if hasattr(opp, "title") else "",
                "description": opp.description if hasattr(opp, "description") else "",
                "category": opp.category if hasattr(opp, "category") else "",
                "impact": opp.impact if hasattr(opp, "impact") else "medium",
                "risk": opp.risk if hasattr(opp, "risk") else "medium",
                "effort": opp.effort if hasattr(opp, "effort") else "medium",
                "status": opp.status if hasattr(opp, "status") else "pending",
                "created_at": opp.created_at if hasattr(opp, "created_at") else "",
                "implementation_plan": opp.implementation_plan if hasattr(opp, "implementation_plan") else None
            })
        
        return filtered_opportunities
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get improvement opportunities: {str(e)}")


@router.get("/opportunities/{opportunity_id}", response_model=OpportunityResponse)
async def get_opportunity_details(opportunity_id: str, request: Request):
    """Get details of a specific improvement opportunity."""
    kernel = request.app.state.kernel
    
    try:
        self_evolution = kernel.get_module("self_evolution_engine")
        
        if not self_evolution:
            raise HTTPException(status_code=404, detail="Self-evolution engine module not found")
        
        opp = self_evolution.get_opportunity(opportunity_id)
        
        if not opp:
            raise HTTPException(status_code=404, detail=f"Opportunity {opportunity_id} not found")
        
        return {
            "id": opp.id if hasattr(opp, "id") else opportunity_id,
            "title": opp.title if hasattr(opp, "title") else "",
            "description": opp.description if hasattr(opp, "description") else "",
            "category": opp.category if hasattr(opp, "category") else "",
            "impact": opp.impact if hasattr(opp, "impact") else "medium",
            "risk": opp.risk if hasattr(opp, "risk") else "medium",
            "effort": opp.effort if hasattr(opp, "effort") else "medium",
            "status": opp.status if hasattr(opp, "status") else "pending",
            "created_at": opp.created_at if hasattr(opp, "created_at") else "",
            "implementation_plan": opp.implementation_plan if hasattr(opp, "implementation_plan") else None
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get opportunity details: {str(e)}")


@router.post("/opportunities/{opportunity_id}/approve", response_model=Dict[str, Any])
async def approve_opportunity(opportunity_id: str, request: Request):
    """Approve an improvement opportunity for implementation."""
    kernel = request.app.state.kernel
    
    try:
        self_evolution = kernel.get_module("self_evolution_engine")
        
        if not self_evolution:
            raise HTTPException(status_code=404, detail="Self-evolution engine module not found")
        
        success = self_evolution.approve_opportunity(opportunity_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Opportunity {opportunity_id} not found or cannot be approved")
        
        return {
            "success": True,
            "message": f"Opportunity {opportunity_id} approved successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to approve opportunity: {str(e)}")


@router.post("/opportunities/{opportunity_id}/reject", response_model=Dict[str, Any])
async def reject_opportunity(opportunity_id: str, reason: str, request: Request):
    """Reject an improvement opportunity."""
    kernel = request.app.state.kernel
    
    try:
        self_evolution = kernel.get_module("self_evolution_engine")
        
        if not self_evolution:
            raise HTTPException(status_code=404, detail="Self-evolution engine module not found")
        
        success = self_evolution.reject_opportunity(opportunity_id, reason)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Opportunity {opportunity_id} not found or cannot be rejected")
        
        return {
            "success": True,
            "message": f"Opportunity {opportunity_id} rejected successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reject opportunity: {str(e)}")


@router.post("/opportunities", response_model=Dict[str, Any])
async def create_opportunity(
    request: Request,
    opportunity: OpportunityCreate
):
    """Create a custom improvement opportunity."""
    kernel = request.app.state.kernel
    
    try:
        self_evolution = kernel.get_module("self_evolution_engine")
        
        if not self_evolution:
            raise HTTPException(status_code=404, detail="Self-evolution engine module not found")
        
        opportunity_id = self_evolution.create_opportunity(
            title=opportunity.title,
            description=opportunity.description,
            category=opportunity.category,
            impact=opportunity.impact,
            risk=opportunity.risk,
            effort=opportunity.effort
        )
        
        return {
            "success": True,
            "message": "Opportunity created successfully",
            "opportunity_id": opportunity_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create opportunity: {str(e)}")


@router.post("/generate-opportunity", response_model=Dict[str, Any])
async def generate_improvement_opportunity(
    request: Request,
    description: str
):
    """Generate an improvement opportunity using the LLM."""
    kernel = request.app.state.kernel
    
    try:
        self_evolution = kernel.get_module("self_evolution_engine")
        llm_orchestrator = kernel.get_module("llm_orchestrator")
        
        if not self_evolution:
            raise HTTPException(status_code=404, detail="Self-evolution engine module not found")
        
        if not llm_orchestrator:
            raise HTTPException(status_code=404, detail="LLM orchestrator module not found")
        
        # Use LLM to generate a detailed opportunity
        prompt = f"""
        Based on the following description, generate a detailed improvement opportunity for an AI system:
        
        Description: {description}
        
        Generate a JSON response with the following structure:
        {{
            "title": "Concise title of the opportunity",
            "description": "Detailed description of the improvement opportunity",
            "category": "One of: performance, reliability, capability, usability, security",
            "impact": "One of: low, medium, high",
            "risk": "One of: low, medium, high",
            "effort": "One of: low, medium, high",
            "implementation_plan": [
                "Step 1 of implementation",
                "Step 2 of implementation",
                "..."
            ]
        }}
        """
        
        llm_response = await llm_orchestrator.generate(prompt)
        opportunity_config = json.loads(llm_response)
        
        # Create the opportunity
        opportunity_id = self_evolution.create_opportunity(
            title=opportunity_config["title"],
            description=opportunity_config["description"],
            category=opportunity_config["category"],
            impact=opportunity_config["impact"],
            risk=opportunity_config["risk"],
            effort=opportunity_config["effort"],
            implementation_plan={"steps": opportunity_config["implementation_plan"]}
        )
        
        return {
            "success": True,
            "message": "Improvement opportunity generated successfully",
            "opportunity_id": opportunity_id,
            "opportunity": opportunity_config
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate improvement opportunity: {str(e)}")


@router.get("/metrics", response_model=Dict[str, Any])
async def get_evolution_metrics(request: Request):
    """Get metrics tracked by the self-evolution engine."""
    kernel = request.app.state.kernel
    
    try:
        self_evolution = kernel.get_module("self_evolution_engine")
        
        if not self_evolution:
            raise HTTPException(status_code=404, detail="Self-evolution engine module not found")
        
        metrics = self_evolution.get_metrics()
        
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get evolution metrics: {str(e)}")


def add_routes(app):
    """Add self-evolution routes to the main app."""
    app.include_router(router)
