"""
Strategic Opportunity Observatory API routes.

This module defines the API routes for the Strategic Opportunity Observatory
to be used by the web interface.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body
from pydantic import BaseModel

class OpportunityFeedback(BaseModel):
    """Model for opportunity feedback submission."""
    approved: bool
    feedback: str

# Create the API router
router = APIRouter(prefix="/soo", tags=["Strategic Observatory"])

def get_observatory(request):
    """Get the Strategic Opportunity Observatory instance from the request state."""
    kernel = request.app.state.kernel
    if not hasattr(kernel, "strategic_observatory"):
        raise HTTPException(status_code=503, detail="Strategic Opportunity Observatory not available")
    return kernel.strategic_observatory

@router.get("/status")
async def get_observatory_status(request: Any):
    """Get the current status of the Strategic Opportunity Observatory."""
    observatory = get_observatory(request)
    return observatory.get_status()

@router.get("/opportunities")
async def get_opportunities(
    request: Any,
    status: Optional[str] = Query(None, description="Filter by opportunity status"),
    type: Optional[str] = Query(None, description="Filter by opportunity type"),
    confidence: Optional[str] = Query(None, description="Minimum confidence level"),
    limit: int = Query(100, description="Maximum number of opportunities to return")
):
    """Get a list of opportunities, optionally filtered."""
    observatory = get_observatory(request)
    
    # Convert string parameters to enum values if needed
    status_enum = None
    type_enum = None
    confidence_enum = None
    
    if status:
        try:
            from evogenesis_core.modules.strategic_opportunity_observatory import OpportunityStatus
            status_enum = OpportunityStatus(status)
        except (ValueError, ImportError):
            pass
    
    if type:
        try:
            from evogenesis_core.modules.strategic_opportunity_observatory import OpportunityType
            type_enum = OpportunityType(type)
        except (ValueError, ImportError):
            pass
    
    if confidence:
        try:
            from evogenesis_core.modules.strategic_opportunity_observatory import ConfidenceLevel
            confidence_enum = ConfidenceLevel(confidence)
        except (ValueError, ImportError):
            pass
    
    return observatory.get_opportunities(
        status=status_enum,
        opportunity_type=type_enum,
        min_confidence=confidence_enum,
        limit=limit
    )

@router.get("/opportunities/{opportunity_id}")
async def get_opportunity_details(
    request: Any,
    opportunity_id: str = Path(..., description="ID of the opportunity to retrieve")
):
    """Get detailed information about a specific opportunity."""
    observatory = get_observatory(request)
    
    opportunities = observatory.get_opportunities(limit=1000)
    for opp in opportunities:
        if opp["id"] == opportunity_id:
            return opp
    
    raise HTTPException(status_code=404, detail="Opportunity not found")

@router.post("/opportunities/{opportunity_id}/simulate")
async def simulate_opportunity(
    request: Any,
    opportunity_id: str = Path(..., description="ID of the opportunity to simulate")
):
    """Run a scenario simulation for an opportunity."""
    observatory = get_observatory(request)
    
    result = observatory.run_scenario_simulation(opportunity_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

@router.post("/opportunities/{opportunity_id}/value")
async def value_opportunity(
    request: Any,
    opportunity_id: str = Path(..., description="ID of the opportunity to value")
):
    """Perform financial valuation for an opportunity."""
    observatory = get_observatory(request)
    
    result = observatory.perform_valuation(opportunity_id)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    
    return result

@router.post("/opportunities/{opportunity_id}/feedback")
async def record_opportunity_feedback(
    request: Any,
    opportunity_id: str = Path(..., description="ID of the opportunity to provide feedback for"),
    feedback: OpportunityFeedback = Body(..., description="Feedback data")
):
    """Record feedback (approve/reject) for an opportunity."""
    observatory = get_observatory(request)
    
    # Get the user ID from the session
    user_id = "admin"  # Replace with actual user ID from session when available
    
    success = observatory.record_opportunity_feedback(
        opportunity_id=opportunity_id,
        approved=feedback.approved,
        user_id=user_id,
        feedback=feedback.feedback
    )
    
    if not success:
        raise HTTPException(status_code=400, detail="Failed to record feedback")
    
    return {"success": True}

@router.post("/evolve-heuristics")
async def evolve_heuristics(request: Any):
    """Trigger evolution of miner heuristics."""
    observatory = get_observatory(request)
    
    new_heuristics = observatory.evolve_heuristics(force=True)
    
    return {
        "success": True,
        "new_heuristics": new_heuristics
    }

def add_routes(app):
    """Add the SOO routes to the given FastAPI app."""
    app.include_router(router)
