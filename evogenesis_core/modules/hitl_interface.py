"""
Human-in-the-Loop (HiTL) Interface Module - Facilitates safe human interaction and control.

This module provides the critical safety, control, and collaboration layer for the EvoGenesis framework.
It enables human approval for sensitive actions, collects feedback, provides system control,
and offers transparency into agent operations.
"""

from typing import Dict, Any, List, Optional, Tuple # Union, Callable removed
import time
import uuid
import json
import logging
import asyncio
import threading
from enum import Enum
# from datetime import datetime, timedelta # Removed
import websockets
import queue


class PermissionStatus(str, Enum):
    """Status of permission requests."""
    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"
    CANCELED = "canceled"


class AgentStatus(str, Enum):
    """Status of agents in the system."""
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    WAITING = "waiting"  # Waiting for human input
    ERROR = "error"


class FeedbackType(str, Enum):
    """Types of feedback that can be provided."""
    RATING = "rating"  # Numerical rating (e.g., 1-5)
    CORRECTION = "correction"  # Correcting an output
    COMMENT = "comment"  # General comment
    APPROVAL = "approval"  # Approving an action
    REJECTION = "rejection"  # Rejecting an action
    SUGGESTION = "suggestion"  # Suggesting an improvement


class PermissionRequest:
    """Represents a request for permission from an agent to perform an action."""
    
    def __init__(self, agent_id: str, action_description: str, details: Dict[str, Any],
                rationale: str = None, risk_level: str = "low", 
                timeout: int = 300, category: str = "general"):
        """
        Initialize a permission request.
        
        Args:
            agent_id: ID of the agent requesting permission
            action_description: Description of the action requiring permission
            details: Additional details about the action
            rationale: Agent's reasoning for why this action is necessary
            risk_level: Estimated risk level of the action (low, medium, high)
            timeout: Timeout in seconds (default: 5 minutes)
            category: Category of permission (file_access, network, execution, etc.)
        """
        self.request_id = str(uuid.uuid4())
        self.agent_id = agent_id
        self.action_description = action_description
        self.details = details
        self.rationale = rationale or "No rationale provided"
        self.risk_level = risk_level
        self.category = category
        self.status = PermissionStatus.PENDING
        self.created_at = time.time()
        self.timeout = timeout
        self.response_time = None
        self.responder_id = None
        self.response_notes = None
        self.alternative_options = []  # List of alternative actions the agent could take
        self.context_data = {}  # Additional context that explains the request
        self.priority = self._calculate_priority()

    def approve(self, responder_id: str, notes: Optional[str] = None):
        """Approve the permission request."""
        # notes parameter is not used, but kept for interface compatibility
        if self.status == PermissionStatus.PENDING:
            self.status = PermissionStatus.APPROVED
            self.response_time = time.time()
            self.responder_id = responder_id
            return True
        return False

    def deny(self, responder_id: str, notes: Optional[str] = None):
        """Deny the permission request."""
        if self.status == PermissionStatus.PENDING:
            self.status = PermissionStatus.DENIED
            self.status = PermissionStatus.DENIED
            self.response_time = time.time()
            self.responder_id = responder_id
            self.response_notes = notes
            return True
        return False
    
    def cancel(self):
        """Cancel the permission request."""
        if self.status == PermissionStatus.PENDING:
            self.status = PermissionStatus.CANCELED
            self.response_time = time.time()
            return True
        return False
    
    def check_timeout(self):
        """Check if the permission request has timed out."""
        if (self.status == PermissionStatus.PENDING and 
            time.time() - self.created_at > self.timeout):
            self.status = PermissionStatus.EXPIRED
            self.response_time = time.time()
            return True
        return False
    
    def as_dict(self) -> Dict[str, Any]:
        """Get the permission request as a dictionary."""
        return {
            "request_id": self.request_id,
            "agent_id": self.agent_id,
            "action_description": self.action_description,
            "details": self.details,
            "rationale": self.rationale,
            "risk_level": self.risk_level,
            "category": self.category,
            "status": self.status,
            "created_at": self.created_at,
            "timeout": self.timeout,
            "response_time": self.response_time,
            "responder_id": self.responder_id,
            "response_notes": self.response_notes,
            "alternative_options": self.alternative_options,
            "priority": self.priority
        }
    
    def _calculate_priority(self) -> str:
        """Calculate the priority of the request based on risk level and category."""
        # High risk actions are always high priority
        if self.risk_level == "high":
            return "high"
        
        # Certain categories have elevated priority
        critical_categories = ["security", "financial", "data_deletion", "external_communication"]
        if self.category in critical_categories:
            return "high"
        
        if self.risk_level == "medium":
            return "medium"
            
        return "low"


class Feedback:
    """Represents feedback provided by a user."""
    
    def __init__(self, user_id: str, target_id: str, feedback_type: FeedbackType,
                feedback_data: Dict[str, Any]):
        """
        Initialize feedback.
        
        Args:
            user_id: ID of the user providing feedback
            target_id: ID of the target (agent, task, output, etc.)
            feedback_type: Type of feedback
            feedback_data: Feedback data (varies by type)
        """
        self.feedback_id = str(uuid.uuid4())
        self.user_id = user_id
        self.target_id = target_id
        self.feedback_type = feedback_type
        self.feedback_data = feedback_data
        self.created_at = time.time()
        self.target_type = feedback_data.get("target_type", "unknown")  # agent, task, system, etc.
        self.reference_id = feedback_data.get("reference_id")  # E.g., specific output/action ID
    
    def as_dict(self) -> Dict[str, Any]:
        """Get the feedback as a dictionary."""
        return {
            "feedback_id": self.feedback_id,
            "user_id": self.user_id,
            "target_id": self.target_id,
            "target_type": self.target_type,
            "feedback_type": self.feedback_type,
            "feedback_data": self.feedback_data,
            "created_at": self.created_at,
            "reference_id": self.reference_id
        }


class AgentView:
    """Provides a view into an agent's state and operations."""
    
    def __init__(self, agent_id: str):
        """
        Initialize an agent view.
        
        Args:
            agent_id: ID of the agent
        """
        self.agent_id = agent_id
        self.status = AgentStatus.STOPPED
        self.current_action = None
        self.plan = []
        self.reasoning_log = []
        self.cost_accumulator = 0.0
        self.resource_usage = {}
        self.external_view_url = None
        self.last_updated = time.time()
        self.metadata = {}
        self.performance_metrics = {}
    
    def update(self, update_data: Dict[str, Any]):
        """Update the agent view with new data."""
        for key, value in update_data.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        self.last_updated = time.time()
    
    def as_dict(self) -> Dict[str, Any]:
        """Get the agent view as a dictionary."""
        return {
            "agent_id": self.agent_id,
            "status": self.status,
            "current_action": self.current_action,
            "plan": self.plan,
            "reasoning_log": self.reasoning_log,
            "cost_accumulator": self.cost_accumulator,
            "resource_usage": self.resource_usage,
            "external_view_url": self.external_view_url,
            "last_updated": self.last_updated,
            "metadata": self.metadata,
            "performance_metrics": self.performance_metrics
        }


class HiTLInterface:
    """
    Human-in-the-Loop Interface for the EvoGenesis framework.
    
    Responsible for:
    - Requesting permission for sensitive actions
    - Providing system control (play/pause/stop)
    - Offering transparency into agent operations
    """
    def __init__(self, kernel, config_path: Optional[str] = None):
        """
        Initialize the HiTL Interface.
        Initialize the HiTL Interface.
        
        Args:
            kernel: The EvoGenesis kernel instance
            config_path: Optional path to configuration file
        """
        self.kernel = kernel
        self.permission_requests = {}  # request_id -> PermissionRequest
        self.feedback_store = []  # List of Feedback objects
        self.agent_views = {}  # agent_id -> AgentView
        
        # Configuration with defaults
        self.config = {
            "urgent_notifications": {
                "enabled": False,
                "methods": []
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "hitl_interface.log",
                "rotation": "1 MB",
                "backup_count": 5,
                "console_output": True,
                "json_output": False
            },
            "transparency": {
                "record_chain_of_thought": True,
                "record_llm_calls": True,
                "record_tool_usage": True,
                "record_data_provenance": True,
                "visualization_enabled": True,
                "max_reasoning_log_length": 100,
                "broadcast_reasoning": True
            },
            "control_thresholds": {
                "cost_alert_threshold": 5.0,  # USD
                "auto_pause_threshold": 10.0,  # USD
                "risk_approval_threshold": "medium"  # Actions with this risk level or higher require approval
            }
        }
        
        # Load configuration if provided
        if config_path:
            self.initialize_config(config_path)
        
        # Setup logging
        self.logger = self.setup_logging()
        
        # Websocket server for live updates
        self.clients = set()
        self.ws_server = None
        self.ws_loop = None
        self.ws_thread = None
        
        # Queue for permission requests awaiting response
        self.request_queue = queue.Queue()
        
        # Authorization
        self.authorized_users = {}  # user_id -> {role, permissions}
        
        # CLI interface state
        self.cli_active = False
        self.cli_thread = None
    
    def initialize_config(self, config_path: Optional[str] = None):
        """Initialize the HiTL interface configuration."""
        import os
        
        # Default configuration already set in __init__
        
        # Load user configuration if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    self._merge_configs(self.config, user_config)
                logging.info(f"Loaded HiTL configuration from {config_path}")
            except Exception as e:
                logging.error(f"Failed to load configuration from {config_path}: {str(e)}")
    
    def _merge_configs(self, base_config, user_config):
        """Recursively merge user configuration into base configuration."""
        for key, value in user_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._merge_configs(base_config[key], value)
            else:
                base_config[key] = value
                
    def setup_logging(self):
        """Set up logging for the HiTL interface."""
        log_config = self.config.get("logging", {})
        log_level = getattr(logging, log_config.get("level", "INFO"))
        log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        log_file = log_config.get("file", "hitl_interface.log")
        
        # Create logger
        logger = logging.getLogger("hitl_interface")
        logger.setLevel(log_level)
        
        # Create file handler with rotation
        if log_config.get("rotation"):
            from logging.handlers import RotatingFileHandler
            # Parse size string like "1 MB" to bytes
            def parse_size(size_str):
                units = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
                size_parts = size_str.split()
                if len(size_parts) == 2:
                    try:
                        num = float(size_parts[0])
                        unit = size_parts[1].upper()
                        return int(num * units.get(unit, 1))
                    except (ValueError, KeyError):
                        pass
                return 1024 * 1024  # Default to 1 MB
                
            max_bytes = parse_size(log_config.get("rotation", "1 MB"))
            backup_count = log_config.get("backup_count", 5)
            file_handler = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count
            )
        else:
            file_handler = logging.FileHandler(log_file)
        
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        logger.addHandler(file_handler)
        
        # Add console handler if enabled
        if log_config.get("console_output", True):
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(logging.Formatter(log_format))
            logger.addHandler(console_handler)
        
        # Add structured JSON handler for searchable logs if enabled
        if log_config.get("json_output"):
            import json
            class JSONFormatter(logging.Formatter):
                def format(self, record):
                    log_record = {
                        "timestamp": self.formatTime(record),
                        "level": record.levelname,
                        "name": record.name,
                        "message": record.getMessage()
                    }
                    # Add extra fields from record
                    for key, value in record.__dict__.items():
                        if key not in ["args", "exc_info", "exc_text", "msg", "message", "levelname", "name"] and not key.startswith("_"):
                            log_record[key] = value
                    return json.dumps(log_record)
            
            json_handler = logging.FileHandler(log_config.get("json_file", "hitl_interface_structured.log"))
            json_handler.setLevel(log_level)
            json_handler.setFormatter(JSONFormatter())
            logger.addHandler(json_handler)
        return logger
    async def _ws_handler(self, websocket, path): # noqa: F841
        """Handle websocket connections."""
        _ = path  # Mark as used to avoid unused variable warning
        client_id = str(uuid.uuid4())
        self.clients.add((client_id, websocket))
        self.clients.add((client_id, websocket))
        
        try:
            # Send initial state
            await websocket.send(json.dumps({
                "type": "initial_state",
                "data": {
                    "system_status": self.get_system_status(),
                    "agent_views": {agent_id: view.as_dict() for agent_id, view in self.agent_views.items()},
                    "permission_requests": [req.as_dict() for req in self.permission_requests.values() 
                                        if req.status == PermissionStatus.PENDING]
                }
            }))
            
            # Listen for commands
            async for message in websocket:
                try:
                    data = json.loads(message)
                    command = data.get("command")
                    
                    if command == "agent_control":
                        agent_id = data.get("agent_id")
                        action = data.get("action")  # play, pause, stop
                        await self._handle_agent_control(agent_id, action)
                    
                    elif command == "permission_response":
                        request_id = data.get("request_id")
                        approved = data.get("approved")
                        user_id = data.get("user_id")
                        notes = data.get("notes")
                        
                        if approved:
                            self.approve_permission(request_id, user_id, notes)
                        else:
                            self.deny_permission(request_id, user_id, notes)
                    
                    elif command == "submit_feedback":
                        await self._handle_feedback_submission(data)
                    
                    elif command == "get_agent_details":
                        agent_id = data.get("agent_id")
                        agent_details = self.get_agent_details(agent_id)
                        await websocket.send(json.dumps({
                            "type": "agent_details",
                            "data": agent_details
                        }))
                    
                except json.JSONDecodeError:
                    logging.error(f"Invalid JSON received: {message}")
                except Exception as e:
                    logging.error(f"Error handling websocket message: {str(e)}")
        
        finally:
            # Remove client on disconnect
            self.clients = {c for c in self.clients if c[0] != client_id}
    
    async def _handle_agent_control(self, agent_id, action):
        """Handle agent control commands."""
        if action == "play":
            self.resume_agent(agent_id)
        elif action == "pause":
            self.pause_agent(agent_id)
        elif action == "stop":
            self.stop_agent(agent_id)
        
        # Broadcast updated status
        await self._broadcast_update("agent_status_change", {
            "agent_id": agent_id,
            "status": self.get_agent_details(agent_id).get("status")
        })
    
    async def _handle_feedback_submission(self, data):
        """Handle feedback submission."""
        user_id = data.get("user_id")
        target_id = data.get("target_id")
        feedback_type = data.get("feedback_type")
        feedback_data = data.get("feedback_data", {})
        
        # Record the feedback
        self.record_feedback(
            user_id=user_id,
            target_id=target_id,
            feedback_type=FeedbackType(feedback_type),
            feedback_data=feedback_data
        )
        
        # Broadcast feedback received
        await self._broadcast_update("feedback_received", {
            "target_id": target_id,
            "feedback_type": feedback_type
        })
    
    async def _broadcast_update(self, update_type, data):
        """Broadcast an update to all connected clients."""
        message = json.dumps({
            "type": update_type,
            "data": data,
            "timestamp": time.time()
        })
        
        for _, websocket in self.clients:
            try:
                await websocket.send(message)
            except websockets.exceptions.ConnectionClosed:
                continue
    
    def start_websocket_server(self, host="0.0.0.0", port=8765):
        """Start the websocket server for real-time updates."""
        async def _run_server():
            self.ws_server = await websockets.serve(self._ws_handler, host, port)
            logging.info(f"HiTL websocket server started on {host}:{port}")
            await self.ws_server.wait_closed()
        
        # Run in a separate thread
        def run_loop():
            self.ws_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.ws_loop)
            self.ws_loop.run_until_complete(_run_server())
            self.ws_loop.run_forever()
        
        self.ws_thread = threading.Thread(target=run_loop, daemon=True)
        self.ws_thread.start()
    
    def start_cli(self):
        """Start a CLI interface for the HiTL Interface."""
        # This would implement a command-line interface using a library like `rich`
        # For now, just a placeholder
        self.cli_active = True
        
        def run_cli():
            try:
                # In a real implementation, this would use rich or similar
                # to provide an interactive CLI
                while self.cli_active:
                    while not self.request_queue.empty():
                        _ = self.request_queue.get()
                        # Display request and get response
                        # ...
                        # ...
                    time.sleep(1)
            except KeyboardInterrupt:
                self.cli_active = False
            finally:
                logging.info("HiTL CLI interface stopped")
        
        self.cli_thread = threading.Thread(target=run_cli, daemon=True)
        self.cli_thread.start()
        logging.info("HiTL CLI interface started")
    
    def start(self):
        """Start the HiTL Interface."""
        # Start websocket server for UI
        self.start_websocket_server()
        
        # Start background task to check for expired permission requests
        def check_timeouts():
            while True:
                expired = []
                for request_id, request in self.permission_requests.items():
                    if request.check_timeout():
                        expired.append(request_id)
                
                # Notify of expired requests
                for request_id in expired:
                    request = self.permission_requests[request_id]
                    logging.warning(f"Permission request {request_id} from agent {request.agent_id} expired")
                    # Notify the agent
                    if hasattr(self.kernel, "agent_manager"):
                        agent = self.kernel.agent_manager.get_agent(request.agent_id)
                        if agent:
                            # Notify agent that request expired
                            pass
                
                time.sleep(10)  # Check every 10 seconds
        
        timeout_thread = threading.Thread(target=check_timeouts, daemon=True)
        timeout_thread.start()
    
    def stop(self):
        """Stop the HiTL Interface."""
        self.cli_active = False
        
        # Stop websocket server
        if self.ws_server:
            self.ws_loop.call_soon_threadsafe(self.ws_server.close)
            self.ws_loop.call_soon_threadsafe(self.ws_loop.stop)

    def request_permission(self, agent_id: str, action_description: str, 
                           details: Dict[str, Any], rationale: Optional[str] = None,
                           risk_level: str = "low", category: str = "general",
                           timeout: int = 300, alternative_options: Optional[List[str]] = None,
                           context_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Request permission for an agent to perform an action.

        Args:
            agent_id: ID of the agent requesting permission
            action_description: Description of the action requiring permission
            details: Additional details about the action
            rationale: Agent's reasoning for why this action is necessary
            risk_level: Estimated risk level (low, medium, high)
            category: Category of permission (file_access, network, execution, etc.)
            timeout: Timeout in seconds
            alternative_options: List of alternative actions if this is denied
            context_data: Additional context that explains the request

        Returns:
            ID of the permission request
        """
        # Create permission request
        request = PermissionRequest(
            agent_id=agent_id,
            action_description=action_description,
            details=details,
            rationale=rationale,
            risk_level=risk_level,
            timeout=timeout,
            category=category
        )

        # Add optional data
        if alternative_options:
            request.alternative_options = alternative_options
        if context_data:
            request.context_data = context_data

        # Store request
        self.permission_requests[request.request_id] = request
        self.request_queue.put(request)

        # Send urgent notifications if necessary
        if risk_level == "high" or request.priority == "high":
            self._send_urgent_notification(request)

        logging.info(f"Permission request {request.request_id} created for agent {agent_id}: {action_description}")

        return request.request_id
    
    def check_permission(self, request_id: str) -> Tuple[bool, Optional[str]]:
        """
        Check the status of a permission request.
        
        Args:
            request_id: ID of the permission request
            
        Returns:
            Tuple of (approved, reason)
        """
        if request_id not in self.permission_requests:
            return False, "Request not found"
        
        request = self.permission_requests[request_id]
        
        # Check for timeout
        request.check_timeout()
        
        if request.status == PermissionStatus.APPROVED:
            return True, request.response_notes
        
        elif request.status == PermissionStatus.DENIED:
            return False, request.response_notes
        
        elif request.status == PermissionStatus.EXPIRED:
            return False, "Request expired"
        
        elif request.status == PermissionStatus.CANCELED:
            return False, "Request canceled"

    def deny_permission(self, request_id: str, user_id: str, notes: Optional[str] = None) -> bool:
        """
        Deny a permission request.

        Args:
            request_id: ID of the permission request
            user_id: ID of the user denying the request
            notes: Additional notes

        Returns:
            True if successful, False otherwise
        """
        if request_id not in self.permission_requests:
            return False

        request = self.permission_requests[request_id]
        success = request.deny(user_id, notes)

        if success and self.ws_loop:
            asyncio.run_coroutine_threadsafe(
                self._broadcast_update("permission_response", {
                    "request_id": request_id,
                    "approved": False,
                    "user_id": user_id,
                    "notes": notes
                }),
                self.ws_loop
            )
            logging.info(f"Permission request {request_id} denied by user {user_id}")

        return success
    
    def record_feedback(self, user_id: str, target_id: str, 
                      feedback_type: FeedbackType, feedback_data: Dict[str, Any]) -> str:
        """
        Record feedback from a user.
        
        Args:
            user_id: ID of the user providing feedback
            target_id: ID of the target (agent, task, output, etc.)
            feedback_type: Type of feedback
            feedback_data: Feedback data
            
        Returns:
            ID of the feedback
        """
        # Create feedback
        feedback = Feedback(
            user_id=user_id,
            target_id=target_id,
            feedback_type=feedback_type,
            feedback_data=feedback_data
        )
        
        # Store feedback
        self.feedback_store.append(feedback)
        
        # Store in memory manager if available
        if hasattr(self.kernel, "memory_manager"):
            # Convert to storable format and store
            feedback_dict = feedback.as_dict()
            self.kernel.memory_manager.store_long_term(
                namespace="feedback",
                metadata={
                    "user_id": user_id,
                    "target_id": target_id,
                    "target_type": feedback.target_type,
                    "feedback_type": feedback_type.value,
                    "created_at": feedback.created_at
                },
                content=json.dumps(feedback_dict)
            )
        
        # Process feedback via the evolution engine if available
        if hasattr(self.kernel, "evolution_engine"):
            if feedback.target_type == "agent" and feedback_type in [
                FeedbackType.CORRECTION, FeedbackType.SUGGESTION, FeedbackType.RATING
            ]:
                self.kernel.evolution_engine.process_agent_feedback(
                    agent_id=target_id,
                    feedback=feedback.as_dict()
                )
        
        logging.info(f"Feedback {feedback.feedback_id} recorded from user {user_id} for {feedback.target_type} {target_id}")
        
        return feedback.feedback_id
    
    def get_feedback(self, target_id: Optional[str] = None, 
                   feedback_type: Optional[FeedbackType] = None,
                   limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get feedback, optionally filtered by target and type.
        
        Args:
            target_id: Optional target ID to filter by
            feedback_type: Optional feedback type to filter by
            limit: Maximum number of feedbacks to return
            
        Returns:
            List of feedback dictionaries
        """
        # Filter feedbacks
        filtered = self.feedback_store
        
        if target_id:
            filtered = [f for f in filtered if f.target_id == target_id]
        
        if feedback_type:
            filtered = [f for f in filtered if f.feedback_type == feedback_type]
        
        # Sort by creation time (newest first)
        filtered.sort(key=lambda f: f.created_at, reverse=True)
        
        # Return limited number
        return [f.as_dict() for f in filtered[:limit]]
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get the current status of the system.
        
        Returns:
            Dictionary with system status information
        """
        agents_status = {}
        
        # Get agent statuses
        if hasattr(self.kernel, "agent_manager"):
            for agent_id, agent in self.kernel.agent_manager.agents.items():
                agents_status[agent_id] = {
                    "status": agent.status if hasattr(agent, "status") else "unknown",
                    "type": agent.type if hasattr(agent, "type") else "unknown",
                    "name": agent.name if hasattr(agent, "name") else "unknown"
                }
        
        # Get resource usage
        resource_usage = {}
        # In a real implementation, this would collect CPU, memory, etc.
        
        # Check subsystems
        subsystems = {}
        for module_name in ["llm_orchestrator", "tooling_system", "memory_manager", "evolution_engine"]:
            if hasattr(self.kernel, module_name):
                module = getattr(self.kernel, module_name)
                if hasattr(module, "get_status"):
                    subsystems[module_name] = module.get_status()
                else:
                    subsystems[module_name] = {"status": "unknown"}
        
        return {
            "agents": agents_status,
            "resource_usage": resource_usage,
            "subsystems": subsystems,
            "pending_permissions": len([r for r in self.permission_requests.values() 
                                    if r.status == PermissionStatus.PENDING]),
            "timestamp": time.time()
        }
    
    def get_agent_details(self, agent_id: str) -> Dict[str, Any]:
        """
        Get detailed information about an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Dictionary with agent details
        """
        # Check if we have a view for this agent
        if agent_id in self.agent_views:
            view = self.agent_views[agent_id]
            return view.as_dict()
        
        # Try to get from agent manager
        if hasattr(self.kernel, "agent_manager"):
            agent = self.kernel.agent_manager.get_agent(agent_id)
            if agent:
                # Create a new view
                view = AgentView(agent_id)
                
                # Extract information from agent
                view.status = agent.status if hasattr(agent, "status") else AgentStatus.UNKNOWN
                view.current_action = agent.current_action if hasattr(agent, "current_action") else None
                view.plan = agent.plan if hasattr(agent, "plan") else []
                view.reasoning_log = agent.reasoning_log if hasattr(agent, "reasoning_log") else []
                view.cost_accumulator = agent.cost_accumulator if hasattr(agent, "cost_accumulator") else 0.0
                
                # Add to agent views
                self.agent_views[agent_id] = view
                
                return view.as_dict()
        
        return {"error": f"Agent {agent_id} not found"}
    
    def update_agent_view(self, agent_id: str, update_data: Dict[str, Any]):
        """
        Update an agent view with new data.
        
        Args:
            agent_id: ID of the agent
            update_data: Data to update the view with
        """
        if agent_id not in self.agent_views:
            self.agent_views[agent_id] = AgentView(agent_id)
        
        self.agent_views[agent_id].update(update_data)
        
        # Broadcast to websocket clients
        if self.ws_loop:
            asyncio.run_coroutine_threadsafe(
                self._broadcast_update("agent_view_update", {
                    "agent_id": agent_id,
                    "view": self.agent_views[agent_id].as_dict()
                }),
                self.ws_loop
            )
    
    def pause_agent(self, agent_id: str) -> bool:
        """
        Pause an agent.
        
        Args:
            agent_id: ID of the agent to pause
            
        Returns:
            True if successful, False otherwise
        """
        if hasattr(self.kernel, "agent_manager"):
            return self.kernel.agent_manager.pause_agent(agent_id)
        return False
    
    def resume_agent(self, agent_id: str) -> bool:
        """
        Resume a paused agent.
        
        Args:
            agent_id: ID of the agent to resume
            
        Returns:
            True if successful, False otherwise
        """
        if hasattr(self.kernel, "agent_manager"):
            return self.kernel.agent_manager.resume_agent(agent_id)
        return False
    
    def stop_agent(self, agent_id: str) -> bool:
        """
        Stop an agent.
        
        Args:
            agent_id: ID of the agent to stop
            
        Returns:
            True if successful, False otherwise
        """
        if hasattr(self.kernel, "agent_manager"):
            return self.kernel.agent_manager.stop_agent(agent_id)
        return False
    
    def pause_all_agents(self) -> bool:
        """
        Pause all agents.
        
        Returns:
            True if successful, False otherwise
        """
        if hasattr(self.kernel, "agent_manager"):
            return self.kernel.agent_manager.pause_all_agents()
        return False
    
    def resume_all_agents(self) -> bool:
        """
        Resume all paused agents.
        
        Returns:
            True if successful, False otherwise
        """
        if hasattr(self.kernel, "agent_manager"):
            return self.kernel.agent_manager.resume_all_agents()
        return False
    
    def stop_all_agents(self) -> bool:
        """
        Stop all agents.
        
        Returns:
            True if successful, False otherwise
        """
        if hasattr(self.kernel, "agent_manager"):
            return self.kernel.agent_manager.stop_all_agents()
        return False
    
    def stream_agent_view(self, agent_id: str) -> Dict[str, Any]:
        """
        Get a streaming view of an agent.
        
        Args:
            agent_id: ID of the agent
            
        Returns:
            Dictionary with streaming view information
        """
        # In a real implementation, this would set up a streaming connection
        # For now, just return basic info
        
        if agent_id in self.agent_views:
            view = self.agent_views[agent_id]
            if view.external_view_url:
                return {
                    "agent_id": agent_id,
                    "stream_url": view.external_view_url,
                    "stream_type": "external"
                }
            else:
                return {
                    "agent_id": agent_id,
                    "stream_type": "log",
                    "connect_via": f"ws://localhost:8765/stream/{agent_id}"
                }
        
        return {"error": f"Agent {agent_id} not found or not streaming"}
    def _send_urgent_notification(self, request: PermissionRequest) -> None:
        """
        Send urgent notifications for high-priority permission requests.
        
        Args:
            request: The permission request requiring urgent attention
        """
        # request is used in this function
        # Check if urgent notifications are configured
        if not self.config.get("urgent_notifications", {}).get("enabled", False):
            return
            return
            
        notification_methods = self.config.get("urgent_notifications", {}).get("methods", [])
        
        for method in notification_methods:
            if method["type"] == "email" and method.get("enabled", False):
                self._send_email_notification(
                    email=method["address"],
                    subject=f"URGENT: EvoGenesis Permission Request #{request.request_id[:8]}",
                    body=self._format_notification_body(request)
                )
            
            elif method["type"] == "slack" and method.get("enabled", False):
                self._send_slack_notification(
                    webhook_url=method["webhook_url"],
                    text=self._format_notification_body(request),
                    channel=method.get("channel")
                )
                
            elif method["type"] == "teams" and method.get("enabled", False):
                self._send_teams_notification(
                    webhook_url=method["webhook_url"],
                    title=f"URGENT: EvoGenesis Permission Request #{request.request_id[:8]}",
                    text=self._format_notification_body(request)
                )
                
            elif method["type"] == "sms" and method.get("enabled", False):
                self._send_sms_notification(
                    phone_number=method["phone_number"],
                    text=f"URGENT: EvoGenesis permission needed. {request.action_description}"
                )
                
        logging.info(f"Sent urgent notification for permission request {request.request_id}")
    
    def _format_notification_body(self, request: PermissionRequest) -> str:
        """Format a permission request for notification messages."""
        return f"""
URGENT: Permission Required from Agent {request.agent_id}
Action: {request.action_description}
Risk Level: {request.risk_level.upper()}
Category: {request.category}
Rationale: {request.rationale}

Please log in to the EvoGenesis control panel to respond.
Request ID: {request.request_id}
"""

    def notify_team_created(self, team_id: str, team_name: str, goal: str) -> None:
        """
        Called when a new team is created.
        
        Args:
            team_id: ID of the created team
            team_name: Name of the created team
            goal: Goal of the created team
        """
        logging.info(f"Team created: {team_id} ({team_name}) with goal: {goal}")
        
        # Broadcast to websocket clients if available
        if self.ws_loop:
            asyncio.run_coroutine_threadsafe(
                self._broadcast_update("team_created", {
                    "team_id": team_id,
                    "name": team_name,
                    "goal": goal
                }),
                self.ws_loop
            )
    
    def notify_agent_terminated(self, agent_id: str, agent_name: str) -> None:
        """
        Called when an agent is terminated.
        
        Args:
            agent_id: ID of the terminated agent
            agent_name: Name of the terminated agent
        """
        logging.info(f"Agent terminated: {agent_id} ({agent_name})")
        
        # Broadcast to websocket clients if available
        if self.ws_loop:
            asyncio.run_coroutine_threadsafe(
                self._broadcast_update("agent_terminated", {
                    "agent_id": agent_id,
                    "name": agent_name
                }),
                self.ws_loop
            )
