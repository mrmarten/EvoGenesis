"""
Activity Routes for EvoGenesis Control Panel

This module provides routes for tracking and displaying system activities.
"""

import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Request

router = APIRouter()

# Kernel reference (will be set when routes are added)
kernel = None

@router.get("/api/activities")
async def get_activities(limit: int = 10, activity_type: Optional[str] = None):
    """
    Get recent system activities.
    
    Args:
        limit: Maximum number of activities to return
        activity_type: Optional filter by activity type
        
    Returns:
        List of activity records
    """
    if not kernel:
        return []
    
    try:
        activities = kernel.get_recent_activities(limit=limit, activity_type=activity_type)
        
        # Format timestamps for better display
        for activity in activities:
            if 'timestamp' in activity:
                import datetime
                timestamp = activity['timestamp']
                if isinstance(timestamp, (int, float)):
                    # Convert to ISO format if it's a numeric timestamp
                    activity['formatted_time'] = datetime.datetime.fromtimestamp(timestamp).isoformat()
                    
                    # Add relative time (e.g., "2 minutes ago")
                    now = datetime.datetime.now()
                    activity_time = datetime.datetime.fromtimestamp(timestamp)
                    delta = now - activity_time
                    
                    if delta.days > 0:
                        activity['relative_time'] = f"{delta.days} days ago"
                    elif delta.seconds >= 3600:
                        activity['relative_time'] = f"{delta.seconds // 3600} hours ago"
                    elif delta.seconds >= 60:
                        activity['relative_time'] = f"{delta.seconds // 60} minutes ago"
                    else:
                        activity['relative_time'] = f"{delta.seconds} seconds ago"
        
        return activities
    except Exception as e:
        logging.error(f"Error getting activities: {str(e)}")
        return []

@router.get("/api/activities/types")
async def get_activity_types():
    """
    Get all available activity types.
    
    Returns:
        List of activity types
    """
    if not kernel:
        return []
    
    try:
        # Get all activities
        all_activities = kernel.get_recent_activities(limit=100)
        
        # Extract unique activity types
        activity_types = set()
        for activity in all_activities:
            if 'type' in activity:
                activity_types.add(activity['type'])
                
        return sorted(list(activity_types))
    except Exception as e:
        logging.error(f"Error getting activity types: {str(e)}")
        return []

@router.post("/api/activities/log")
async def log_activity(request: Request):
    """
    Log a new activity.
    
    Returns:
        Success message
    """
    if not kernel:
        raise HTTPException(status_code=500, detail="Kernel not initialized")
    
    try:
        data = await request.json()
        
        # Required fields
        activity_type = data.get("type")
        title = data.get("title")
        message = data.get("message")
        
        if not activity_type or not title or not message:
            raise HTTPException(status_code=400, detail="Type, title, and message are required")
        
        # Optional data
        activity_data = data.get("data", {})
        
        # Log the activity
        kernel.log_activity(
            activity_type=activity_type,
            title=title,
            message=message,
            data=activity_data
        )
        
        return {"success": True, "message": "Activity logged successfully"}
    except Exception as e:
        logging.error(f"Error logging activity: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

def add_routes(app, kernel_instance=None):
    """Add activity routes to the FastAPI app."""
    global kernel
    kernel = kernel_instance
    
    app.include_router(router, tags=["activities"])
    
    return app
