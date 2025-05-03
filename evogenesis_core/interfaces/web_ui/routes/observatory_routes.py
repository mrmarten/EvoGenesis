"""
Strategic Opportunity Observatory Routes for EvoGenesis Web UI

This module provides API routes for the Strategic Opportunity Observatory.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel


# Define models
class SignalSourceBase(BaseModel):
    name: str
    type: str
    description: str
    config: Dict[str, Any]
    active: bool = True


class SignalSourceCreate(SignalSourceBase):
    pass


class SignalSourceResponse(SignalSourceBase):
    id: str
    last_update: Optional[str] = None
    total_signals: int = 0


class OpportunityBase(BaseModel):
    title: str
    description: str
    category: str
    confidence: float
    source_signals: List[str] = []
    tags: List[str] = []


class OpportunityResponse(OpportunityBase):
    id: str
    created_at: str
    status: str
    impact_score: float


# Create router
router = APIRouter(
    prefix="/api/observatory",
    tags=["observatory"],
)


# Define routes
@router.get("/status", response_model=Dict[str, Any])
async def get_observatory_status(request: Request):
    """Get the current status of the Strategic Opportunity Observatory."""
    kernel = request.app.state.kernel
    
    try:
        soo = kernel.get_module("strategic_opportunity_observatory")
        
        if not soo:
            raise HTTPException(status_code=404, detail="Strategic Opportunity Observatory module not found")
        
        status = soo.get_status()
        
        return {
            "active": status.get("active", False),
            "signal_sources": status.get("signal_sources", 0),
            "total_signals": status.get("total_signals", 0),
            "total_opportunities": status.get("total_opportunities", 0),
            "last_scan": status.get("last_scan", ""),
            "next_scan": status.get("next_scan", ""),
            "health": status.get("health", 0.0)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get observatory status: {str(e)}")


@router.get("/sources", response_model=List[SignalSourceResponse])
async def get_signal_sources(request: Request):
    """Get all signal sources in the observatory."""
    kernel = request.app.state.kernel
    
    try:
        soo = kernel.get_module("strategic_opportunity_observatory")
        
        if not soo:
            raise HTTPException(status_code=404, detail="Strategic Opportunity Observatory module not found")
        
        sources = soo.get_signal_sources()
        
        # Transform to response format
        source_responses = []
        for source_id, source_data in sources.items():
            source_responses.append({
                "id": source_id,
                "name": source_data.name if hasattr(source_data, "name") else f"Source-{source_id[:6]}",
                "type": source_data.type if hasattr(source_data, "type") else "unknown",
                "description": source_data.description if hasattr(source_data, "description") else "",
                "config": source_data.config if hasattr(source_data, "config") else {},
                "active": source_data.active if hasattr(source_data, "active") else True,
                "last_update": source_data.last_update if hasattr(source_data, "last_update") else None,
                "total_signals": source_data.total_signals if hasattr(source_data, "total_signals") else 0
            })
        
        return source_responses
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get signal sources: {str(e)}")


@router.get("/sources/{source_id}", response_model=SignalSourceResponse)
async def get_signal_source(source_id: str, request: Request):
    """Get details of a specific signal source."""
    kernel = request.app.state.kernel
    
    try:
        soo = kernel.get_module("strategic_opportunity_observatory")
        
        if not soo:
            raise HTTPException(status_code=404, detail="Strategic Opportunity Observatory module not found")
        
        source = soo.get_signal_source(source_id)
        
        if not source:
            raise HTTPException(status_code=404, detail=f"Signal source {source_id} not found")
        
        return {
            "id": source_id,
            "name": source.name if hasattr(source, "name") else f"Source-{source_id[:6]}",
            "type": source.type if hasattr(source, "type") else "unknown",
            "description": source.description if hasattr(source, "description") else "",
            "config": source.config if hasattr(source, "config") else {},
            "active": source.active if hasattr(source, "active") else True,
            "last_update": source.last_update if hasattr(source, "last_update") else None,
            "total_signals": source.total_signals if hasattr(source, "total_signals") else 0
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get signal source {source_id}: {str(e)}")


@router.post("/sources", response_model=Dict[str, Any])
async def create_signal_source(
    request: Request,
    source: SignalSourceCreate
):
    """Create a new signal source."""
    kernel = request.app.state.kernel
    
    try:
        soo = kernel.get_module("strategic_opportunity_observatory")
        
        if not soo:
            raise HTTPException(status_code=404, detail="Strategic Opportunity Observatory module not found")
        
        source_id = soo.add_signal_source(
            name=source.name,
            source_type=source.type,
            description=source.description,
            config=source.config,
            active=source.active
        )
        
        return {
            "success": True,
            "message": f"Signal source {source.name} created successfully",
            "source_id": source_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create signal source: {str(e)}")


@router.delete("/sources/{source_id}", response_model=Dict[str, Any])
async def delete_signal_source(source_id: str, request: Request):
    """Delete a signal source."""
    kernel = request.app.state.kernel
    
    try:
        soo = kernel.get_module("strategic_opportunity_observatory")
        
        if not soo:
            raise HTTPException(status_code=404, detail="Strategic Opportunity Observatory module not found")
        
        success = soo.remove_signal_source(source_id)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Signal source {source_id} not found")
        
        return {
            "success": True,
            "message": f"Signal source {source_id} deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete signal source {source_id}: {str(e)}")


@router.post("/sources/{source_id}/toggle", response_model=Dict[str, Any])
async def toggle_signal_source(source_id: str, active: bool, request: Request):
    """Toggle a signal source's active state."""
    kernel = request.app.state.kernel
    
    try:
        soo = kernel.get_module("strategic_opportunity_observatory")
        
        if not soo:
            raise HTTPException(status_code=404, detail="Strategic Opportunity Observatory module not found")
        
        success = soo.set_signal_source_status(source_id, active)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Signal source {source_id} not found")
        
        return {
            "success": True,
            "message": f"Signal source {source_id} {'activated' if active else 'deactivated'} successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to toggle signal source {source_id}: {str(e)}")


@router.get("/opportunities", response_model=List[OpportunityResponse])
async def get_strategic_opportunities(
    request: Request,
    category: Optional[str] = None,
    min_confidence: Optional[float] = None,
    status: Optional[str] = None
):
    """Get strategic opportunities identified by the observatory."""
    kernel = request.app.state.kernel
    
    try:
        soo = kernel.get_module("strategic_opportunity_observatory")
        
        if not soo:
            raise HTTPException(status_code=404, detail="Strategic Opportunity Observatory module not found")
        
        opportunities = soo.get_opportunities()
        
        # Apply filters
        filtered_opportunities = []
        for opp in opportunities:
            if category and hasattr(opp, "category") and opp.category != category:
                continue
            if min_confidence and hasattr(opp, "confidence") and opp.confidence < min_confidence:
                continue
            if status and hasattr(opp, "status") and opp.status != status:
                continue
            
            filtered_opportunities.append({
                "id": opp.id if hasattr(opp, "id") else "",
                "title": opp.title if hasattr(opp, "title") else "",
                "description": opp.description if hasattr(opp, "description") else "",
                "category": opp.category if hasattr(opp, "category") else "",
                "confidence": opp.confidence if hasattr(opp, "confidence") else 0.0,
                "source_signals": opp.source_signals if hasattr(opp, "source_signals") else [],
                "tags": opp.tags if hasattr(opp, "tags") else [],
                "created_at": opp.created_at if hasattr(opp, "created_at") else "",
                "status": opp.status if hasattr(opp, "status") else "new",
                "impact_score": opp.impact_score if hasattr(opp, "impact_score") else 0.0
            })
        
        return filtered_opportunities
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get strategic opportunities: {str(e)}")


@router.get("/opportunities/{opportunity_id}", response_model=Dict[str, Any])
async def get_opportunity_details(opportunity_id: str, request: Request):
    """Get detailed information about a specific strategic opportunity."""
    kernel = request.app.state.kernel
    
    try:
        soo = kernel.get_module("strategic_opportunity_observatory")
        
        if not soo:
            raise HTTPException(status_code=404, detail="Strategic Opportunity Observatory module not found")
        
        opp = soo.get_opportunity(opportunity_id)
        
        if not opp:
            raise HTTPException(status_code=404, detail=f"Opportunity {opportunity_id} not found")
        
        # Get detailed information including signal details
        detailed_signals = []
        if hasattr(opp, "source_signals"):
            for signal_id in opp.source_signals:
                signal = soo.get_signal(signal_id)
                if signal:
                    detailed_signals.append({
                        "id": signal_id,
                        "content": signal.content if hasattr(signal, "content") else "",
                        "source": signal.source_id if hasattr(signal, "source_id") else "",
                        "timestamp": signal.timestamp if hasattr(signal, "timestamp") else "",
                        "relevance": signal.relevance if hasattr(signal, "relevance") else 0.0
                    })
        
        # Get related opportunities
        related = soo.get_related_opportunities(opportunity_id) if hasattr(soo, "get_related_opportunities") else []
        related_opportunities = []
        for rel_opp in related:
            related_opportunities.append({
                "id": rel_opp.id if hasattr(rel_opp, "id") else "",
                "title": rel_opp.title if hasattr(rel_opp, "title") else "",
                "similarity": rel_opp.similarity if hasattr(rel_opp, "similarity") else 0.0
            })
        
        return {
            "id": opp.id if hasattr(opp, "id") else opportunity_id,
            "title": opp.title if hasattr(opp, "title") else "",
            "description": opp.description if hasattr(opp, "description") else "",
            "category": opp.category if hasattr(opp, "category") else "",
            "confidence": opp.confidence if hasattr(opp, "confidence") else 0.0,
            "tags": opp.tags if hasattr(opp, "tags") else [],
            "created_at": opp.created_at if hasattr(opp, "created_at") else "",
            "status": opp.status if hasattr(opp, "status") else "new",
            "impact_score": opp.impact_score if hasattr(opp, "impact_score") else 0.0,
            "metrics": opp.metrics if hasattr(opp, "metrics") else {},
            "detailed_signals": detailed_signals,
            "related_opportunities": related_opportunities,
            "recommended_actions": opp.recommended_actions if hasattr(opp, "recommended_actions") else []
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get opportunity details: {str(e)}")


@router.post("/scan", response_model=Dict[str, Any])
async def trigger_opportunity_scan(request: Request):
    """Trigger a scan for new strategic opportunities."""
    kernel = request.app.state.kernel
    
    try:
        soo = kernel.get_module("strategic_opportunity_observatory")
        
        if not soo:
            raise HTTPException(status_code=404, detail="Strategic Opportunity Observatory module not found")
        
        scan_id = await soo.scan_for_opportunities()
        
        return {
            "success": True,
            "message": "Opportunity scan triggered successfully",
            "scan_id": scan_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to trigger opportunity scan: {str(e)}")


@router.post("/opportunities/{opportunity_id}/status", response_model=Dict[str, Any])
async def update_opportunity_status(
    opportunity_id: str,
    status: str,
    request: Request
):
    """Update the status of a strategic opportunity."""
    kernel = request.app.state.kernel
    
    if status not in ["new", "investigating", "actionable", "actioned", "rejected"]:
        raise HTTPException(status_code=400, detail="Invalid status value")
    
    try:
        soo = kernel.get_module("strategic_opportunity_observatory")
        
        if not soo:
            raise HTTPException(status_code=404, detail="Strategic Opportunity Observatory module not found")
        
        success = soo.update_opportunity_status(opportunity_id, status)
        
        if not success:
            raise HTTPException(status_code=404, detail=f"Opportunity {opportunity_id} not found")
        
        return {
            "success": True,
            "message": f"Opportunity {opportunity_id} status updated to {status} successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update opportunity status: {str(e)}")


@router.post("/opportunities/{opportunity_id}/export-to-evolution", response_model=Dict[str, Any])
async def export_to_evolution_engine(opportunity_id: str, request: Request):
    """Export a strategic opportunity to the self-evolution engine."""
    kernel = request.app.state.kernel
    
    try:
        soo = kernel.get_module("strategic_opportunity_observatory")
        self_evolution = kernel.get_module("self_evolution_engine")
        
        if not soo:
            raise HTTPException(status_code=404, detail="Strategic Opportunity Observatory module not found")
        
        if not self_evolution:
            raise HTTPException(status_code=404, detail="Self-evolution engine module not found")
        
        opp = soo.get_opportunity(opportunity_id)
        
        if not opp:
            raise HTTPException(status_code=404, detail=f"Opportunity {opportunity_id} not found")
        
        # Convert to evolution opportunity
        evolution_opportunity_id = self_evolution.create_opportunity(
            title=opp.title if hasattr(opp, "title") else f"Opportunity-{opportunity_id[:6]}",
            description=opp.description if hasattr(opp, "description") else "",
            category="capability",  # Default category for SOO opportunities
            impact="medium",  # Default impact
            risk="medium",    # Default risk
            effort="medium",  # Default effort
            metadata={"source": "soo", "opportunity_id": opportunity_id}
        )
        
        # Update the status of the SOO opportunity
        soo.update_opportunity_status(opportunity_id, "exported")
        
        return {
            "success": True,
            "message": f"Opportunity {opportunity_id} exported to evolution engine successfully",
            "evolution_opportunity_id": evolution_opportunity_id
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export opportunity to evolution engine: {str(e)}")


@router.get("/signals", response_model=Dict[str, Any])
async def get_signals(
    request: Request,
    source_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """Get signals collected by the observatory."""
    kernel = request.app.state.kernel
    
    try:
        soo = kernel.get_module("strategic_opportunity_observatory")
        
        if not soo:
            raise HTTPException(status_code=404, detail="Strategic Opportunity Observatory module not found")
        
        signals = soo.get_signals(source_id=source_id, limit=limit, offset=offset)
        
        # Transform to response format
        signal_items = []
        for signal_id, signal_data in signals.items():
            signal_items.append({
                "id": signal_id,
                "content": signal_data.content if hasattr(signal_data, "content") else "",
                "source_id": signal_data.source_id if hasattr(signal_data, "source_id") else "",
                "timestamp": signal_data.timestamp if hasattr(signal_data, "timestamp") else "",
                "tags": signal_data.tags if hasattr(signal_data, "tags") else [],
                "metadata": signal_data.metadata if hasattr(signal_data, "metadata") else {}
            })
        
        # Get total count
        total = soo.get_signal_count(source_id)
        
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "items": signal_items
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get signals: {str(e)}")


def add_routes(app):
    """Add observatory routes to the main app."""
    app.include_router(router)
