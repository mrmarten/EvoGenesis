"""
Log Routes for EvoGenesis Web UI

This module provides API routes for accessing system logs.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from pydantic import BaseModel


# Create router
router = APIRouter(
    prefix="/api/logs",
    tags=["logs"],
)


# Define routes
@router.get("/", response_model=Dict[str, Any])
async def get_logs(
    request: Request,
    level: Optional[str] = Query(None, description="Filter by log level (debug, info, warning, error)"),
    module: Optional[str] = Query(None, description="Filter by module name"),
    search: Optional[str] = Query(None, description="Search term within log messages"),
    start_time: Optional[datetime] = Query(None, description="Start time for log entries"),
    end_time: Optional[datetime] = Query(None, description="End time for log entries"),
    limit: int = Query(100, description="Maximum number of log entries to return"),
    offset: int = Query(0, description="Offset for pagination")
):
    """Get system logs with optional filtering."""
    try:
        # Implement log retrieval logic
        # This would typically involve reading from a log file or database
        
        # Set default time range if not specified
        if not start_time:
            start_time = datetime.now() - timedelta(days=1)
        if not end_time:
            end_time = datetime.now()
        
        # Example of retrieving logs from a file
        log_entries = []
        
        # Try to use system logging handler if available
        logger = logging.getLogger()
        for handler in logger.handlers:
            if hasattr(handler, 'get_logs'):
                log_entries = handler.get_logs(
                    level=level,
                    module=module,
                    search=search,
                    start_time=start_time,
                    end_time=end_time,
                    limit=limit,
                    offset=offset
                )
                break
        
        # If no special handler found, try to read logs from file
        if not log_entries:
            # Get log file path from settings
            log_file = "evogenesis.log"  # Default, should be configured
            
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                
                # Parse log lines into structured data
                for line in lines:
                    try:
                        # Example format: 2023-04-26 10:15:30,123 - INFO - module_name - Message
                        parts = line.strip().split(' - ', 3)
                        if len(parts) >= 3:
                            timestamp_str = parts[0]
                            log_level = parts[1].strip()
                            module_name = parts[2].strip() if len(parts) > 3 else ""
                            message = parts[3] if len(parts) > 3 else parts[2]
                            
                            # Parse timestamp
                            timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                            
                            # Apply filters
                            if level and log_level.lower() != level.lower():
                                continue
                            if module and module_name.lower() != module.lower():
                                continue
                            if search and search.lower() not in message.lower():
                                continue
                            if timestamp < start_time or timestamp > end_time:
                                continue
                            
                            log_entries.append({
                                "id": str(len(log_entries)),
                                "timestamp": timestamp.isoformat(),
                                "level": log_level,
                                "module": module_name,
                                "message": message,
                                "context": {}
                            })
                            
                            # Apply limit and offset
                            if len(log_entries) >= offset + limit:
                                break
                    except Exception as parsing_error:
                        # Skip lines that can't be parsed
                        continue
                
                # Apply offset
                log_entries = log_entries[offset:offset+limit]
            except Exception as file_error:
                # If file reading fails, return empty list
                log_entries = []
        
        # Get total count and unique modules for filtering UI
        unique_modules = set(entry["module"] for entry in log_entries if "module" in entry)
        
        return {
            "total": len(log_entries),
            "modules": list(unique_modules),
            "entries": log_entries
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get logs: {str(e)}")


@router.get("/{log_id}", response_model=Dict[str, Any])
async def get_log_details(log_id: str, request: Request):
    """Get detailed information about a specific log entry."""
    try:
        # Implement log detail retrieval logic
        # This would typically involve reading the specific log entry with context
        
        # Try to use system logging handler if available
        logger = logging.getLogger()
        for handler in logger.handlers:
            if hasattr(handler, 'get_log_detail'):
                log_detail = handler.get_log_detail(log_id)
                if log_detail:
                    return log_detail
        
        # If no special handler found or log not found, return 404
        raise HTTPException(status_code=404, detail=f"Log entry {log_id} not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get log details: {str(e)}")


@router.post("/clear", response_model=Dict[str, Any])
async def clear_logs(request: Request):
    """Clear all logs."""
    try:
        # Implement log clearing logic
        success = False
        
        # Try to use system logging handler if available
        logger = logging.getLogger()
        for handler in logger.handlers:
            if hasattr(handler, 'clear_logs'):
                success = handler.clear_logs()
                break
        
        # If no special handler found, try to clear log file
        if not success:
            # Get log file path from settings
            log_file = "evogenesis.log"  # Default, should be configured
            
            try:
                with open(log_file, 'w') as f:
                    f.write('')  # Truncate file
                success = True
            except Exception as file_error:
                success = False
        
        if success:
            return {
                "success": True,
                "message": "Logs cleared successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to clear logs")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear logs: {str(e)}")


@router.get("/download", response_model=Dict[str, Any])
async def prepare_logs_download(
    request: Request,
    format: str = "csv",
    level: Optional[str] = None,
    module: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None
):
    """Prepare logs for download in specified format."""
    try:
        # Implement log download preparation logic
        # This would typically involve formatting logs for download
        
        # Set default time range if not specified
        if not start_time:
            start_time = datetime.now() - timedelta(days=7)
        if not end_time:
            end_time = datetime.now()
        
        # Generate a unique download ID
        import uuid
        download_id = str(uuid.uuid4())
        
        # Store download parameters for later retrieval
        # This could be in a database or in-memory store
        download_params = {
            "id": download_id,
            "format": format,
            "level": level,
            "module": module,
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
            "created_at": datetime.now().isoformat()
        }
        
        # Return the download ID and URL
        return {
            "success": True,
            "download_id": download_id,
            "download_url": f"/api/logs/download/{download_id}",
            "format": format
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to prepare logs download: {str(e)}")


def add_routes(app):
    """Add log routes to the main app."""
    app.include_router(router)
