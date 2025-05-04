"""
Settings Routes for EvoGenesis Web UI

This module provides API routes for managing system settings in the EvoGenesis system.
"""

import os
import json
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel


# Define models
class SettingCategory(BaseModel):
    general: Dict[str, Any] = {}
    llm_orchestrator: Dict[str, Any] = {}
    agent_factory: Dict[str, Any] = {}  # Changed from agent_manager
    memory_manager: Dict[str, Any] = {}
    tooling_system: Dict[str, Any] = {}
    hitl_interface: Dict[str, Any] = {}
    self_evolution: Dict[str, Any] = {}
    system: Dict[str, Any] = {}
    api_keys: Optional[Dict[str, Any]] = None
    notifications: Dict[str, Any] = {}


# Create router
router = APIRouter(
    prefix="/api/settings",
    tags=["settings"],
)


# Define routes
@router.get("/", response_model=SettingCategory)
async def get_settings(request: Request):
    """Get all system settings."""
    kernel = request.app.state.kernel
    
    try:
        # Collect settings from various modules
        settings = {
            "general": {
                "system_name": "EvoGenesis",
                "ui_theme": "light",
                "log_level": "info",
                "enable_telemetry": True
            },
            "llm_orchestrator": {},
            "agent_factory": {},  # Changed from agent_manager
            "memory_manager": {},
            "tooling_system": {},
            "hitl_interface": {},
            "self_evolution": {},
            "system": {},
            "api_keys": {},
            "notifications": {}
        }
        
        # Get LLM orchestrator settings
        llm_orchestrator = kernel.get_module("llm_orchestrator")
        if llm_orchestrator:
            llm_settings = llm_orchestrator.get_settings()
            settings["llm_orchestrator"] = {
                "default_model": llm_settings.get("default_model", "gpt-3.5-turbo"),
                "fallback_model": llm_settings.get("fallback_model", "llama-2-70b"),
                "rate_limiting": llm_settings.get("rate_limiting", {
                    "enabled": True,
                    "requests_per_minute": 20
                }),
                "models": llm_settings.get("models", {})
            }
            
            # Check for API keys
            for provider, provider_info in llm_settings.get("providers", {}).items():
                if "api_key" in provider_info:
                    settings["api_keys"][provider] = True  # Just indicate presence, don't expose key
        
        # Get agent factory settings (changed from agent manager)
        agent_factory = kernel.get_module("agent_factory") # Changed from agent_manager
        if agent_factory:
            agent_settings = agent_factory.get_settings()
            settings["agent_factory"] = { # Changed from agent_manager
                "max_concurrent_agents": agent_settings.get("max_concurrent_agents", 10),
                "agent_timeout_seconds": agent_settings.get("agent_timeout_seconds", 300),
                "default_prompt_template": agent_settings.get("default_prompt_template", "standard"),
                "prompt_templates": agent_settings.get("prompt_templates", {
                    "standard": "You are an AI assistant built with EvoGenesis. Answer the user's questions helpfully and accurately."
                })
            }
        
        # Get memory manager settings
        memory_manager = kernel.get_module("memory_manager")
        if memory_manager:
            memory_settings = memory_manager.get_settings()
            settings["memory_manager"] = {
                "vector_db": memory_settings.get("vector_db", {
                    "type": "chroma",
                    "path": "./data/vector_db"
                }),
                "ttl_days": memory_settings.get("ttl_days", 30),
                "chunk_size": memory_settings.get("chunk_size", 1000),
                "chunk_overlap": memory_settings.get("chunk_overlap", 200)
            }
        
        # Get tooling system settings
        tooling_system = kernel.get_module("tooling_system")
        if tooling_system:
            tool_settings = tooling_system.get_settings()
            settings["tooling_system"] = {
                "max_concurrent_tools": tool_settings.get("max_concurrent_tools", 5),
                "tool_timeout_seconds": tool_settings.get("tool_timeout_seconds", 60),
                "auto_reload_tools": tool_settings.get("auto_reload_tools", True),
                "tool_directories": tool_settings.get("tool_directories", ["./tools", "./custom_tools"])
            }
        
        # Get HITL interface settings
        hitl_interface = kernel.get_module("hitl_interface")
        if hitl_interface:
            hitl_settings = hitl_interface.get_settings()
            settings["hitl_interface"] = {
                "control_thresholds": hitl_settings.get("control_thresholds", {
                    "risk_approval_threshold": "medium",
                    "cost_approval_threshold": 0.5,
                    "auto_approve_below_threshold": True
                }),
                "feedback_collection": hitl_settings.get("feedback_collection", {
                    "enabled": True,
                    "prompt_frequency": "medium"
                })
            }
        
        # Get self-evolution settings
        self_evolution = kernel.get_module("self_evolution_engine")
        if self_evolution:
            evolution_settings = self_evolution.get_settings()
            settings["self_evolution"] = {
                "auto_evolution": evolution_settings.get("auto_evolution", {
                    "enabled": True,
                    "interval_hours": 24,
                    "max_changes_per_cycle": 3,
                    "risk_threshold": "medium"
                }),
                "optimization_targets": evolution_settings.get("optimization_targets", [
                    "performance", "reliability", "versatility"
                ])
            }
        
        # Get system settings
        settings["system"] = {
            "debug_mode": kernel.debug_mode if hasattr(kernel, "debug_mode") else False,
            "enable_backup_kernel": kernel.enable_backup if hasattr(kernel, "enable_backup") else False,
            "heartbeat_interval": kernel.heartbeat_interval if hasattr(kernel, "heartbeat_interval") else 5,
            "heartbeat_timeout": kernel.heartbeat_timeout if hasattr(kernel, "heartbeat_timeout") else 15
        }
        
        # Get notification settings
        notification_settings = kernel.get_notification_settings() if hasattr(kernel, "get_notification_settings") else {}
        settings["notifications"] = {
            "email": notification_settings.get("email", {
                "enabled": False,
                "server": "",
                "port": 587,
                "username": "",
                "recipients": []
            }),
            "slack": notification_settings.get("slack", {
                "enabled": False,
                "webhook_url": ""
            })
        }
        
        return settings
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get settings: {str(e)}")


@router.post("/", response_model=Dict[str, Any])
async def update_settings(settings: SettingCategory, request: Request):
    """Update system settings."""
    kernel = request.app.state.kernel
    
    try:
        # Update settings in each module
        
        # Update LLM orchestrator settings
        if settings.llm_orchestrator:
            llm_orchestrator = kernel.get_module("llm_orchestrator")
            if llm_orchestrator:
                llm_orchestrator.update_settings(settings.llm_orchestrator)
        
        # Update agent factory settings
        if settings.agent_factory:
            agent_factory = kernel.get_module("agent_factory")
            if agent_factory:
                agent_factory.update_settings(settings.agent_factory)
        
        # Update memory manager settings
        if settings.memory_manager:
            memory_manager = kernel.get_module("memory_manager")
            if memory_manager:
                memory_manager.update_settings(settings.memory_manager)
        
        # Update tooling system settings
        if settings.tooling_system:
            tooling_system = kernel.get_module("tooling_system")
            if tooling_system:
                tooling_system.update_settings(settings.tooling_system)
        
        # Update HITL interface settings
        if settings.hitl_interface:
            hitl_interface = kernel.get_module("hitl_interface")
            if hitl_interface:
                hitl_interface.update_settings(settings.hitl_interface)
        
        # Update self-evolution settings
        if settings.self_evolution:
            self_evolution = kernel.get_module("self_evolution_engine")
            if self_evolution:
                self_evolution.update_settings(settings.self_evolution)
        
        # Update system settings
        if settings.system:
            if "debug_mode" in settings.system and hasattr(kernel, "set_debug_mode"):
                kernel.set_debug_mode(settings.system["debug_mode"])
            if "enable_backup_kernel" in settings.system and hasattr(kernel, "set_backup_kernel"):
                kernel.set_backup_kernel(settings.system["enable_backup_kernel"])
            if "heartbeat_interval" in settings.system and hasattr(kernel, "set_heartbeat_interval"):
                kernel.set_heartbeat_interval(settings.system["heartbeat_interval"])
            if "heartbeat_timeout" in settings.system and hasattr(kernel, "set_heartbeat_timeout"):
                kernel.set_heartbeat_timeout(settings.system["heartbeat_timeout"])
        
        # Update notification settings
        if settings.notifications and hasattr(kernel, "update_notification_settings"):
            kernel.update_notification_settings(settings.notifications)
        
        # Update API keys if provided
        if settings.api_keys:
            llm_orchestrator = kernel.get_module("llm_orchestrator")
            if llm_orchestrator and hasattr(llm_orchestrator, "update_api_keys"):
                llm_orchestrator.update_api_keys(settings.api_keys)
        
        return {
            "success": True,
            "message": "Settings updated successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update settings: {str(e)}")


@router.post("/optimize-db", response_model=Dict[str, Any])
async def optimize_database(request: Request):
    """Optimize the vector database."""
    kernel = request.app.state.kernel
    
    try:
        memory_manager = kernel.get_module("memory_manager")
        if not memory_manager:
            raise HTTPException(status_code=500, detail="Memory manager module not found")
        
        result = memory_manager.optimize_database()
        
        return {
            "success": True,
            "message": "Database optimized successfully",
            "details": result
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to optimize database: {str(e)}")


@router.post("/clear-db", response_model=Dict[str, Any])
async def clear_database(request: Request):
    """Clear the database (warning: this deletes all data)."""
    kernel = request.app.state.kernel
    
    try:
        memory_manager = kernel.get_module("memory_manager")
        if not memory_manager:
            raise HTTPException(status_code=500, detail="Memory manager module not found")
        
        result = memory_manager.clear_database()
        
        return {
            "success": True,
            "message": "Database cleared successfully",
            "details": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {str(e)}")


@router.post("/create-backup", response_model=Dict[str, Any])
async def create_backup(request: Request):
    """Create a backup of the current system state."""
    kernel = request.app.state.kernel
    
    try:
        # Create a timestamp for the backup
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = f"./backups/{timestamp}"
        
        # Ensure the backup directory exists
        os.makedirs(backup_dir, exist_ok=True)
        
        # Create a settings JSON file
        settings = await get_settings(request)
        with open(f"{backup_dir}/settings.json", "w") as f:
            json.dump(settings.dict(), f, indent=2)
        
        # Backup vector database
        memory_manager = kernel.get_module("memory_manager")
        if memory_manager:
            vector_db_path = memory_manager.get_vector_db_path()
            if os.path.exists(vector_db_path):
                shutil.copytree(vector_db_path, f"{backup_dir}/vector_db")
        
        # Back up any other important data
        # TODO: Add more backup logic as needed
        
        return {
            "success": True,
            "message": "Backup created successfully",
            "backup_path": backup_dir
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create backup: {str(e)}")


@router.post("/restore-backup/{backup_id}", response_model=Dict[str, Any])
async def restore_backup(backup_id: str, request: Request):
    """Restore the system from a backup."""
    kernel = request.app.state.kernel
    
    try:
        backup_dir = f"./backups/{backup_id}"
        
        if not os.path.exists(backup_dir):
            raise HTTPException(status_code=404, detail=f"Backup {backup_id} not found")
        
        # Load settings from backup
        with open(f"{backup_dir}/settings.json", "r") as f:
            settings = json.load(f)
        
        # Restore settings
        await update_settings(SettingCategory(**settings), request)
        
        # Restore vector database
        memory_manager = kernel.get_module("memory_manager")
        if memory_manager:
            vector_db_path = memory_manager.get_vector_db_path()
            if os.path.exists(vector_db_path):
                shutil.rmtree(vector_db_path)
            if os.path.exists(f"{backup_dir}/vector_db"):
                shutil.copytree(f"{backup_dir}/vector_db", vector_db_path)
        
        # Restore any other important data
        # TODO: Add more restore logic as needed
        
        return {
            "success": True,
            "message": f"System restored from backup {backup_id} successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restore from backup: {str(e)}")


@router.get("/backups", response_model=List[Dict[str, Any]])
async def list_backups():
    """List all available backups."""
    try:
        backups_dir = "./backups"
        if not os.path.exists(backups_dir):
            return []
        
        backups = []
        for backup_id in os.listdir(backups_dir):
            backup_path = os.path.join(backups_dir, backup_id)
            if os.path.isdir(backup_path):
                # Get creation time of the backup
                created_at = datetime.fromtimestamp(os.path.getctime(backup_path))
                
                # Get some basic info about the backup
                settings_path = os.path.join(backup_path, "settings.json")
                settings_info = {}
                if os.path.exists(settings_path):
                    with open(settings_path, "r") as f:
                        settings_info = json.load(f)
                
                backup_info = {
                    "id": backup_id,
                    "created_at": created_at.isoformat(),
                    "size": sum(os.path.getsize(os.path.join(dirpath, filename)) 
                               for dirpath, _, filenames in os.walk(backup_path) 
                               for filename in filenames)
                }
                
                # Add some key info from settings
                if "general" in settings_info:
                    backup_info["system_name"] = settings_info["general"].get("system_name", "EvoGenesis")
                
                backups.append(backup_info)
        
        # Sort backups by creation time (newest first)
        backups.sort(key=lambda x: x["created_at"], reverse=True)
        
        return backups
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list backups: {str(e)}")


def add_routes(app):
    """Add settings routes to the main app."""
    app.include_router(router)
